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
    dag_id="copper_06_model_forecast",
    description="Generate the production H1-H12 copper forecast",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "copper",
        "forecast",
        "production",
    ],
) as dag:

    generate_production_forecast = BashOperator(
        task_id="generate_production_forecast",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.models.generate_production_forecast"
        ),
    )
