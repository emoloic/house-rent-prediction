from prefect_tasks import *

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST", "localhost")
TRACKING_SERVER_PORT = os.getenv("TRACKING_SERVER_PORT", "5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "house-price-prediction")
MODEL_REGISTRY_NAME = os.getenv("EXPERIMENT_NAME", "house-price-prediction-model")
MODEL_SEARCH_ITERATIONS = int(os.getenv("MODEL_SEARCH_ITERATIONS", "60"))


@flow(task_runner=SequentialTaskRunner())
def main():
    """
    Executes the training workflow
    """
    kaggle_dataset = "iamsouravbanerjee/house-rent-prediction-dataset"
    file_name = "House_Rent_Dataset.csv"
    download_path = "./data"

    tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger = get_run_logger()
    data_file_path, download_status = download_data(
        kaggle_dataset, download_path, file_name
    )

    if download_status:
        df = load_data.submit(f"{data_file_path}").result()
        df_train, df_val = split_data.submit(df).result()
        selected_columns = [
            'BHK',
            'Rent',
            'Size',
            'Area Type',
            'City',
            'Furnishing Status',
            'Tenant Preferred',
            'Bathroom',
            'Floor',
            'Total Floors',
            'Point of Contact',
        ]
        target_column = 'Rent'
        X_train, X_val, y_train, y_val, dv, sc = prepare_data.submit(
            df_train, df_val, selected_columns, target_column
        ).result()
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        train_model_xgboost_search(
            train, valid, y_val, data_file_path, dv, sc, MODEL_SEARCH_ITERATIONS
        )
        train_model_rf_search(
            X_train,
            X_val,
            y_train,
            y_val,
            data_file_path,
            dv,
            sc,
            MODEL_SEARCH_ITERATIONS,
        )
        register_best_model(tracking_uri, EXPERIMENT_NAME, MODEL_REGISTRY_NAME)
        logger.info("Successfully executed our flow !!!")
    else:
        logger.info(
            "Unabled to run the flow because the task for downloading data failed !!!"
        )


if __name__ == "__main__":
    main()
