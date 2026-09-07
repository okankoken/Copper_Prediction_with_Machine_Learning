from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = "/opt/airflow"

default_args = {
    "owner": "copper_prediction",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="copper_07_mlflow_logging",
    description="Log the production copper forecast run to MLflow",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "copper",
        "mlflow",
        "production",
    ],
) as dag:

    log_production_run_mlflow = BashOperator(
        task_id="log_production_run_mlflow",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.models.log_production_run_mlflow"
        ),
    )
