from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator

from airflow.sdk import DAG

with DAG(
    "my_first_dag",
    default_args={
        "depends_on_past": False,
        "retries": 1,
    },
    description="Download Raw Data",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    is_paused_upon_creation=False,
    catchup=False,
    tags=["Main pipeline"],
) as dag:
    t1 = BashOperator(
        task_id="load_raw_data",
        bash_command=(
            "drone-rescue data download-folder-to-temp "
            "1GTl4Tg2CdOTktPkQm5oT8cOzpUEFvgdF "
            "-c /.config/credentials.json "
            "-t /.config/token.json"
        ),
    )
