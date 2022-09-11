import os
import pickle

import mlflow
import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, request
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pymongo import MongoClient

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "house-price-prediction")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "False") == "True"
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST", "localhost")
TRACKING_SERVER_PORT = os.getenv("TRACKING_SERVER_PORT", "5000")
DEFAULT_MODEL_ENABLED = os.getenv("DEFAULT_MODEL_ENABLED", "True") == "True"
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "False") == "True"
EVIDENTLY_SERVICE_HOST = os.getenv("EVIDENTLY_SERVICE_HOST", "localhost")
EVIDENTLY_SERVICE_PORT = os.getenv("EVIDENTLY_SERVICE_PORT", "8085")
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "mongoAdmin")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER_URL = os.getenv("MONGODB_CLUSTER_URL")
PREDICTION_DATABASE_NAME = os.getenv("PREDICTION_DATABASE_NAME", "prediction_service")


if not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

if MONITORING_ENABLED:
    mongodb_uri = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER_URL}/?retryWrites=true&w=majority"
    mongodb_client = MongoClient(mongodb_uri)
    db = mongodb_client.get_database(f"{PREDICTION_DATABASE_NAME}")
    collection = db.get_collection(EXPERIMENT_NAME)


def load_preprocessor_from_mlflow():
    """
    Loads the preprocessor from mlflow.
    """
    client = MlflowClient(tracking_uri=f"{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
    experiment = client.get_experiment_by_name(f"{EXPERIMENT_NAME}")
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"],
    )[0]
    run_id = best_run.info.run_id
    client.download_artifacts(run_id=run_id, path='preprocessor', dst_path='.')

    with open(f"preprocessor/preprocessor.b", "rb") as f_in:
        dv, sc = pickle.load(f_in)

    return dv, sc


def load_default_preprocessor():
    """
    Loads the preprocessor
    """
    with open(f"preprocessor/preprocessor.b", "rb") as f_in:
        dv, sc = pickle.load(f_in)

    return dv, sc


def load_model_from_registry():
    """
    Loads the ML model from the MLFlow registry
    """
    tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{EXPERIMENT_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Loaded model from S3 Bucket")

    return loaded_model


def load_default_model():
    """
    Loads the default ML model from disk
    """
    with open(f"model/model.pkl", "rb") as f_in:
        loaded_model = pickle.load(f_in)
    print("Loaded default model from disk")

    return loaded_model


def load_preprocessor():
    """
    Loads the preprocessor
    """
    try:
        if MLFLOW_ENABLED:
            return load_preprocessor_from_mlflow()

        if DEFAULT_MODEL_ENABLED:
            return load_default_preprocessor()
    except:
        if DEFAULT_MODEL_ENABLED:
            return load_default_preprocessor()

    return None


def load_model():
    """
    Loads the ML model
    """
    try:
        if MLFLOW_ENABLED:
            return load_model_from_registry()

        if DEFAULT_MODEL_ENABLED:
            return load_default_model()
    except:
        if DEFAULT_MODEL_ENABLED:
            return load_default_model()

    return None


def predict(record):
    """
    Predicts the house rent
    """
    dv, sc = load_preprocessor()
    X = dv.transform([record])
    X = sc.transform(X)
    prediction = model.predict(X)

    return np.int(prediction[0])


def save_to_db(record, house_rent_prediction):
    """
    Saves the prediction data to the Mongo database.
    """
    rec = record.copy()
    rec["prediction"] = house_rent_prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, house_rent_prediction):
    """
    Sends the prediction data to the Evidently monitoring service
    """
    rec = record.copy()
    # We need to name the prediction column "Rent"
    # because our monitoring service expects to have this column as the target to calculate NumTargetDrift.
    rec["Rent"] = house_rent_prediction
    evidently_service_uri = f"http://{EVIDENTLY_SERVICE_HOST}:{EVIDENTLY_SERVICE_PORT}"
    requests.post(f"{evidently_service_uri}/iterate/house-rent-prediction", json=[rec])


def calculate_house_rent(record):
    """
    Calculates the house rent
    """
    pred = predict(record)
    if MONITORING_ENABLED:
        save_to_db(record, pred)
        send_to_evidently_service(record, pred)

    return pred


app = Flask(EXPERIMENT_NAME)
app.secret_key = os.urandom(24)

model = load_model()


@app.route("/predict", methods=["POST"])
def predict_json_endpoint():
    """
    Prediction API endpoint
    """
    record = request.get_json()
    house_rent_prediction = calculate_house_rent(record)
    result = {
        "house_rent_prediction": house_rent_prediction
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
