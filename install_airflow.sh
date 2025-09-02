#!/usr/bin/env bash


AIRFLOW_VERSION=3.0.2

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

case "$PYTHON_VERSION" in
  3.8|3.9|3.10|3.11) ;;
  *)
    echo "Unsupported Python version: $PYTHON_VERSION for Airflow $AIRFLOW_VERSION"
    exit 1
    ;;
esac

CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

if uv pip show apache-airflow >/dev/null 2>&1; then
  echo "apache-airflow is already installed. Removing it..."
  uv pip uninstall -y apache-airflow
fi

CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

echo "Installing apache-airflow==$AIRFLOW_VERSION for Python $PYTHON_VERSION"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
