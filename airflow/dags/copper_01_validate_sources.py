from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = "/opt/airflow"


default_args = {
    "owner": "copper_prediction",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="copper_01_validate_sources",
    description=(
        "Validate manual and periodic local source files "
        "before copper production pipeline"
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=[
        "copper",
        "validation",
        "production",
    ],
) as dag:

    validate_manual_sources = BashOperator(
        task_id="validate_manual_sources",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.quality.validate_manual_sources"
        ),
    )
