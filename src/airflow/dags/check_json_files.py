from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from pathlib import Path
from datetime import datetime


def list_json_files():
    folder = Path("/datasamples")
    json_files = list(folder.glob("*.json"))

    if not json_files:
        print("🟡 No JSON files found.")
    else:
        print("🟢 Found JSON files:")
        for f in json_files:
            print(f" - {f.name}")


with DAG(
    dag_id="list_json_files",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["test"],
) as dag:

    list_files = PythonOperator(
        task_id="list_json_files_in_folder",
        python_callable=list_json_files,
    )
