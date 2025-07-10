FROM apache/airflow:3.0.2

RUN pip install --no-cache-dir poetry

COPY ../pyproject.toml ../poetry.lock README.md /opt/airflow/app/

WORKDIR /opt/airflow/app

COPY ../src /opt/airflow/app/src

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main

COPY ../.config /opt/airflow/app/.config

ENV PYTHONPATH="/opt/airflow/app/src:${PYTHONPATH}"
