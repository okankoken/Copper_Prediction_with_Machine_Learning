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
    dag_id="copper_04_build_master",
    description=(
        "Build and validate the monthly copper master dataset"
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "copper",
        "master",
        "production",
    ],
) as dag:

    build_master_feature_plan = BashOperator(
        task_id="build_master_feature_plan",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.features.build_master_feature_plan"
        ),
    )

    build_copper_monthly_master = BashOperator(
        task_id="build_copper_monthly_master",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.features.build_copper_monthly_master"
        ),
    )

    quality_master = BashOperator(
        task_id="quality_master",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.quality.quality_master"
        ),
    )

    (
        build_master_feature_plan
        >> build_copper_monthly_master
        >> quality_master
    )
