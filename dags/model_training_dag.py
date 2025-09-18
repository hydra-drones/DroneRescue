from __future__ import annotations

from airflow import DAG
import pendulum
from airflow.providers.standard.operators.bash import BashOperator


default_args = {
    "owner": "airflow",
    "start_date": pendulum.datetime(2025, 9, 17, tz="UTC"),
}

dag = DAG(
    "model_training_pipeline",
    start_date=pendulum.datetime(2025, 9, 17, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["model-training"],
)

# Task to start MLflow server
start_mlflow_task = BashOperator(
    task_id="start_mlflow_server",
    bash_command="""
    source /home/evgenii-iurin/miniconda3/bin/activate drone-rescue-env && \
    cd /home/evgenii-iurin/work/DroneRescue && \
    if ! nc -z localhost 5000; then
        echo "Starting MLflow server..."
        nohup python -m mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
        sleep 15  # Wait for server to start
        echo "MLflow server started"
    else
        echo "MLflow server is already running"
    fi
    """,
    dag=dag,
)

# Task to run model training
train_model_task = BashOperator(
    task_id="train_model",
    bash_command="""
    source /home/evgenii-iurin/miniconda3/bin/activate drone-rescue-env && \
    cd /home/evgenii-iurin/work/DroneRescue && \
    export AIRFLOW_HOME=/home/evgenii-iurin/work/DroneRescue/.airflow && \
    export PYTHONPATH=/home/evgenii-iurin/work/DroneRescue:$PYTHONPATH && \
    python -m src.pipeline.train
    """,
    dag=dag,
)

# Set task dependencies
start_mlflow_task >> train_model_task
