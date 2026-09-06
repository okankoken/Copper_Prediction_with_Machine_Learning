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

QUALITY_MODULES = [
    "quality_bdi",
    "quality_chile_cochilco",
    "quality_china",
    "quality_copper_company_stocks",
    "quality_country_stock_indices",
    "quality_economic_policy_uncertainty",
    "quality_energy_transition",
    "quality_fred",
    "quality_geopolitical_risk",
    "quality_global_leading_indicators",
    "quality_global_macro",
    "quality_icsg",
    "quality_lme",
    "quality_metals",
    "quality_peru_copper_cost_drivers",
    "quality_peru_copper_mining",
    "quality_portwatch",
    "quality_shfe",
    "quality_turkey",
    "quality_usgs",
    "quality_worldbank_commodities",
]


with DAG(
    dag_id="copper_03_source_quality",
    description="Run source-level quality checks for copper ingestion outputs",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["copper", "quality", "production"],
) as dag:

    previous_task = None

    for module_name in QUALITY_MODULES:
        task = BashOperator(
            task_id=module_name,
            bash_command=(
                f"cd {PROJECT_ROOT} && "
                f"python -m src.quality.{module_name}"
            ),
        )

        if previous_task is not None:
            previous_task >> task

        previous_task = task
