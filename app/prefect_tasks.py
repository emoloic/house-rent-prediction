import os
import time
import shutil
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


@task(name="Download Data", retries=3, retry_delay_seconds=120)
def download_data(kaggle_dataset, download_path, file_name):
    """
    Downloads training data from Kaggle
    """

    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")

    if KAGGLE_USERNAME and KAGGLE_KEY:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(kaggle_dataset, path=download_path, force=True)

        datetime = time.strftime("%Y%m%d-%H%M%S")

        # Unzip our dataset
        dataset_name = kaggle_dataset.split("/")[1]
        shutil.unpack_archive(f"{dataset_name}.zip", download_path)
        # Remove the zip file
        os.remove(f"{dataset_name}.zip")

        data_file = f"data-{datetime}.csv"

        # Rename data with current datetime
        os.rename(
            f"{download_path}/{file_name}",
            f"{download_path}/{data_file}",
        )

        data_file_path = f"{download_path}/{data_file}"
        status = True

        return data_file_path, status
    else:
        return None, False


@task(name="Data Loading")
def load_data(filename):
    """
    Loading our dataset downloaded from Kaggle with some preprocessing applied to our columns.
    """
    df = pd.read_csv(filename)

    # Feature engineering of the column 'floor'
    df['Total Floors'] = [i.split()[-1] for i in df['Floor']]
    df['Floor'] = [i.split()[0] for i in df['Floor']]
    df['Total Floors'] = df['Total Floors'].replace({'Ground': '1'})
    df['Total Floors'] = df['Total Floors'].astype(int)
    df['Floor'].replace({'Ground': '0', 'Lower': '-1'}, inplace=True)

    for i, floor in zip(range(df.shape[0]), df['Floor']):
        if floor == 'Upper':
            df.at[i, 'Floor'] = df.at[i, 'Total Floors']
    df['Floor'] = df['Floor'].astype(int)

    # Remove outliers
    df = df[df['Rent'] < 3000000]
    df = df[df['Bathroom'] < 10]
    df = df[~df['Area Type'].str.contains("Built Area")]
    df = df[~df['Point of Contact'].str.contains("Contact Builder")]

    return df


@task(name="Data Splitting")
def split_data(df):
    """
    Split our data for training and validation with a ratio of 80/20
    """
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[['City', 'Furnishing Status', 'Area Type', 'Point of Contact']],
    )

    return df_train, df_val


@task(name="Data Preparation")
def prepare_data(df_train, df_val, selected_columns, target_column):
    """
    Data preparation for modeling (Feature Engineering)
    """
    df_train = df_train[selected_columns]
    df_val = df_val[selected_columns]

    dv = DictVectorizer(sparse=False)

    target_index = selected_columns.index(target_column)

    features = [
        selected_columns[i] for i in range(len(selected_columns)) if i != target_index
    ]

    train_dicts = df_train[features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[features].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = target_column
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


@task(name="Train XGBoost Model with HyperOpt")
def train_model_xgboost_search(
    train, valid, y_val, data_file, dv, sc, model_search_iterations
):
    """
    Hyperparameter Search for xgboost model.
    """
    mlflow.xgboost.autolog()

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_param("data_file_path", data_file)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50,
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2score = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2score)

            with open("preprocessor/preprocessor.b", "wb") as f_out:
                pickle.dump((dv, sc), f_out)

            mlflow.log_artifact(
                "preprocessor/preprocessor.b", artifact_path="preprocessor"
            )

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.uniform('max_depth', 1, 20)),
        'learning_rate': hp.loguniform('learning_rate', 0.01, 0.2),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=model_search_iterations,
        trials=Trials(),
        rstate=rstate,
    )
    return


@task(name="Train Random Forest Model with HyperOpt")
def train_model_rf_search(
    X_train, X_val, y_train, y_val, data_file, dv, sc, model_search_iterations
):
    """
    Hyperparameters Search for Random Forest Model.
    """
    mlflow.sklearn.autolog()

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "random_forest")
            mlflow.log_param("data_file_path", data_file)
            rf_model = RandomForestRegressor(**params)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2score = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2score)

            with open("preprocessor/preprocessor.b", "wb") as f_out:
                pickle.dump((dv, sc), f_out)

            mlflow.log_artifact(
                "preprocessor/preprocessor.b", artifact_path="preprocessor"
            )

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'n_estimators': scope.int(hp.uniform('n_estimators', 10, 50)),
        'max_depth': scope.int(hp.uniform('max_depth', 1, 20)),
        'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 5)),
        'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 6)),
        'random_state': 42,
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=model_search_iterations,
        trials=Trials(),
        rstate=rstate,
    )
    return


@task(name="Register Best Model", retries=3, retry_delay_seconds=120)
def register_best_model(tracking_uri, experiment_name, model_registry_name):
    """
    Register our best model (with the minimum RMSE) in mlflow model registry.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"],
    )[0]

    # register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_rmse = best_run.data.metrics['rmse']
    model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
    model_version = model_details.version

    date = datetime.today().date()

    # transition of our best model in "Production"
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )

    client.update_model_version(
        name=model_details.name,
        version=model_version,
        description=f"The model version {model_version} was transitioned to Production on {date}",
    )

    client.update_registered_model(
        name=model_details.name,
        description=f"Current model version in production: {model_version} with rmse: {model_rmse}",
    )