#!/usr/bin/env bash

cd "$(dirname "$0")"

function print_info {
    RESET="\e[0m"
    BOLD="\e[1m"
    YELLOW="\e[33m"
    echo -e "$YELLOW$BOLD [+] $1 $RESET"
}

# DockerHub Credentials
export DOCKERHUB_USERNAME=emoloic
export DOCKERHUB_REPOSITORY=house-rent-prediction
export IMAGE_TAG=latest

# Tracking Server Credentials
export TRACKING_SERVER_HOST=test-mlflow-server
export TRACKING_SERVER_PORT=5000

# Tracking server database credentials
export POSTGRES_SERVER_HOST=test-mlflow-db
export POSTGRES_SERVER_PORT=5432
export POSTGRES_USER=mlflow
export POSTGRES_PASSWORD=mlflow
export POSTGRES_DB=test-house-rent-prediction-db

# Experiment data
export EXPERIMENT_NAME=test-house-rent-prediction
export MODEL_REGISTRY_NAME=test-house-rent-prediction-model
export MODEL_SEARCH_ITERATIONS=300
export DEFAULT_MODEL_ENABLED=True

print_info "Creating MLOps test pipeline"
docker-compose up -d

print_info "Waiting for test pipeline to be ready"
sleep 10

print_info "Executing test training workflow"
docker exec -t test-prefect python prefect_flow.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

sleep 5

print_info "Prediction test"
python test_predict.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

print_info "Clean-up"
docker-compose down