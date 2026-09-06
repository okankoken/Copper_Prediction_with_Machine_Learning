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

INGESTION_MODULES = [
    "ingest_bdi",
    "ingest_chile_cochilco",
    "ingest_china",
    "ingest_copper_company_stocks",
    "ingest_country_stock_indices",
    "ingest_economic_policy_uncertainty",
    "ingest_energy_transition",
    "ingest_fred",
    "ingest_geopolitical_risk",
    "ingest_global_leading_indicators",
    "ingest_global_macro",
    "ingest_icsg",
    "ingest_lme",
    "ingest_metals",
    "ingest_peru_copper_cost_drivers",
    "ingest_peru_copper_mining",
    "ingest_portwatch",
    "ingest_shfe",
    "ingest_turkey",
    "ingest_usgs",
    "ingest_worldbank_commodities",
]


with DAG(
    dag_id="copper_02_ingestion",
    description="Run all copper project ingestion pipelines",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["copper", "ingestion", "production"],
) as dag:

    previous_task = None

    for module_name in INGESTION_MODULES:
        task = BashOperator(
            task_id=module_name,
            bash_command=(
                f"cd {PROJECT_ROOT} && "
                f"python -m src.ingestion.{module_name}"
            ),
        )

        if previous_task is not None:
            previous_task >> task

        previous_task = task
