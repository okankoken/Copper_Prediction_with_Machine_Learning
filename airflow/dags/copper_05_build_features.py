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
    dag_id="copper_05_build_features",
    description=(
        "Build model features, multi-horizon dataset, "
        "and archive production model inputs"
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "copper",
        "features",
        "production",
    ],
) as dag:

    build_model_features = BashOperator(
        task_id="build_model_features",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.features.build_model_features"
        ),
    )

    build_multi_horizon_dataset = BashOperator(
        task_id="build_multi_horizon_dataset",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.features.build_multi_horizon_dataset"
        ),
    )

    archive_model_inputs = BashOperator(
        task_id="archive_model_inputs",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m src.utils.archive_model_inputs"
        ),
    )

    (
        build_model_features
        >> build_multi_horizon_dataset
        >> archive_model_inputs
    )
