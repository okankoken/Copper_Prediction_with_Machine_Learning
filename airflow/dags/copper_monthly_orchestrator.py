from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.trigger_dagrun import (
    TriggerDagRunOperator,
)


default_args = {
    "owner": "copper_prediction",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="copper_monthly_orchestrator",
    description=(
        "Run the complete monthly copper production pipeline "
        "from source validation through MLflow logging"
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "copper",
        "orchestrator",
        "production",
    ],
) as dag:

    run_01_validate_sources = TriggerDagRunOperator(
        task_id="run_01_validate_sources",
        trigger_dag_id="copper_01_validate_sources",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_02_ingestion = TriggerDagRunOperator(
        task_id="run_02_ingestion",
        trigger_dag_id="copper_02_ingestion",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_03_source_quality = TriggerDagRunOperator(
        task_id="run_03_source_quality",
        trigger_dag_id="copper_03_source_quality",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_04_build_master = TriggerDagRunOperator(
        task_id="run_04_build_master",
        trigger_dag_id="copper_04_build_master",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_05_build_features = TriggerDagRunOperator(
        task_id="run_05_build_features",
        trigger_dag_id="copper_05_build_features",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_06_model_forecast = TriggerDagRunOperator(
        task_id="run_06_model_forecast",
        trigger_dag_id="copper_06_model_forecast",
        wait_for_completion=True,
        poke_interval=5,
    )

    run_07_mlflow_logging = TriggerDagRunOperator(
        task_id="run_07_mlflow_logging",
        trigger_dag_id="copper_07_mlflow_logging",
        wait_for_completion=True,
        poke_interval=5,
    )

    (
        run_01_validate_sources
        >> run_02_ingestion
        >> run_03_source_quality
        >> run_04_build_master
        >> run_05_build_features
        >> run_06_model_forecast
        >> run_07_mlflow_logging
    )
