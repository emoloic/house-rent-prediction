FROM python:3.9-slim-buster

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "*.py", "./" ]

ADD ["models", "./models"]