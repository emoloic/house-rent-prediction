"""Prediction database cleaning module"""

import os

from pymongo import MongoClient

MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "mongoAdmin")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_CLUSTER_URL = os.getenv("MONGODB_CLUSTER_URL")
MONGO_DATABASE = "prediction_service"

if __name__ == "__main__":
    mongodb_uri = f"mongodb://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER_URL}/?retryWrites=true&w=majority"
    mongodb_client = MongoClient(mongodb_uri)
    mongodb_client.drop_database(MONGO_DATABASE)
