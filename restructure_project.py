from pathlib import Path
import shutil


ROOT = Path.cwd()


# ---------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------

DIRECTORIES = [
    "src/ingestion",
    "src/features",
    "src/models",
    "src/utils",

    "data/raw/market",
    "data/raw/macro",
    "data/raw/mining",
    "data/raw/shipping",
    "data/raw/risk",
    "data/raw/equities",
    "data/raw/energy_transition",

    "data/processed/quality",
    "data/processed/diagnostics",
    "data/processed/features",
]


# ---------------------------------------------------------------------
# File moves
# ---------------------------------------------------------------------

MOVES = {
    # Python ingestion
    "src/ingest_bdi.py":
        "src/ingestion/ingest_bdi.py",

    "src/ingest_chile_cochilco.py":
        "src/ingestion/ingest_chile_cochilco.py",

    "src/ingest_china.py":
        "src/ingestion/ingest_china.py",

    "src/ingest_copper_company_stocks.py":
        "src/ingestion/ingest_copper_company_stocks.py",

    "src/ingest_country_stock_indices.py":
        "src/ingestion/ingest_country_stock_indices.py",

    "src/ingest_economic_policy_uncertainty.py":
        "src/ingestion/ingest_economic_policy_uncertainty.py",

    "src/ingest_energy_transition.py":
        "src/ingestion/ingest_energy_transition.py",

    "src/ingest_fred.py":
        "src/ingestion/ingest_fred.py",

    "src/ingest_geopolitical_risk.py":
        "src/ingestion/ingest_geopolitical_risk.py",

    "src/ingest_global_leading_indicators.py":
        "src/ingestion/ingest_global_leading_indicators.py",

    "src/ingest_global_macro.py":
        "src/ingestion/ingest_global_macro.py",

    "src/ingest_icsg.py":
        "src/ingestion/ingest_icsg.py",

    "src/ingest_lme.py":
        "src/ingestion/ingest_lme.py",

    "src/ingest_metals.py":
        "src/ingestion/ingest_metals.py",

    "src/ingest_peru_copper_cost_drivers.py":
        "src/ingestion/ingest_peru_copper_cost_drivers.py",

    "src/ingest_peru_copper_mining.py":
        "src/ingestion/ingest_peru_copper_mining.py",

    "src/ingest_portwatch.py":
        "src/ingestion/ingest_portwatch.py",

    "src/ingest_shfe.py":
        "src/ingestion/ingest_shfe.py",

    "src/ingest_turkey.py":
        "src/ingestion/ingest_turkey.py",

    "src/ingest_usgs.py":
        "src/ingestion/ingest_usgs.py",

    "src/ingest_worldbank_commodities.py":
        "src/ingestion/ingest_worldbank_commodities.py",

    # Feature / model code
    "src/feature_engineering.py":
        "src/features/feature_engineering.py",

    "src/train_models.py":
        "src/models/train_models.py",

    "src/evaluate_models.py":
        "src/models/evaluate_models.py",

    # Market data
    "data/raw/lme_copper_daily.csv":
        "data/raw/market/lme_copper_daily.csv",

    "data/raw/lme_other_metals_daily.csv":
        "data/raw/market/lme_other_metals_daily.csv",

    "data/raw/shfe_copper_daily.csv":
        "data/raw/market/shfe_copper_daily.csv",

    "data/raw/palladium_daily.csv":
        "data/raw/market/palladium_daily.csv",

    "data/raw/worldbank_commodities_monthly.csv":
        "data/raw/market/worldbank_commodities_monthly.csv",

    "data/raw/china_refined_copper_monthly.csv":
        "data/raw/market/china_refined_copper_monthly.csv",

    # Macro data
    "data/raw/fred_macro_monthly.csv":
        "data/raw/macro/fred_macro_monthly.csv",

    "data/raw/global_macro_monthly.csv":
        "data/raw/macro/global_macro_monthly.csv",

    "data/raw/global_leading_indicators_monthly.csv":
        "data/raw/macro/global_leading_indicators_monthly.csv",

    "data/raw/turkey_monthly.csv":
        "data/raw/macro/turkey_monthly.csv",

    "data/raw/china_electricity_generation_monthly.csv":
        "data/raw/macro/china_electricity_generation_monthly.csv",

    "data/raw/china_fixed_asset_investment_monthly.csv":
        "data/raw/macro/china_fixed_asset_investment_monthly.csv",

    "data/raw/china_industrial_output_monthly.csv":
        "data/raw/macro/china_industrial_output_monthly.csv",

    "data/raw/china_ppi_monthly.csv":
        "data/raw/macro/china_ppi_monthly.csv",

    "data/raw/china_real_estate_investment_monthly.csv":
        "data/raw/macro/china_real_estate_investment_monthly.csv",

    # Mining / supply
    "data/raw/usgs_copper_annual.csv":
        "data/raw/mining/usgs_copper_annual.csv",

    "data/raw/icsg_copper_annual.csv":
        "data/raw/mining/icsg_copper_annual.csv",

    "data/raw/chile_cochilco_copper_cost_annual.csv":
        "data/raw/mining/chile_cochilco_copper_cost_annual.csv",

    "data/raw/peru_copper_mining_annual.csv":
        "data/raw/mining/peru_copper_mining_annual.csv",

    "data/raw/peru_copper_cost_drivers_monthly.csv":
        "data/raw/mining/peru_copper_cost_drivers_monthly.csv",

    # Shipping
    "data/raw/baltic_dry_index_daily.csv":
        "data/raw/shipping/baltic_dry_index_daily.csv",

    "data/raw/portwatch_shipping_activity_daily.csv":
        "data/raw/shipping/portwatch_shipping_activity_daily.csv",

    # Risk
    "data/raw/geopolitical_risk_monthly.csv":
        "data/raw/risk/geopolitical_risk_monthly.csv",

    "data/raw/economic_policy_uncertainty_monthly.csv":
        "data/raw/risk/economic_policy_uncertainty_monthly.csv",

    # Equities
    "data/raw/country_stock_indices_monthly.csv":
        "data/raw/equities/country_stock_indices_monthly.csv",

    "data/raw/copper_company_stocks_monthly.csv":
        "data/raw/equities/copper_company_stocks_monthly.csv",

    "data/raw/Chile_IPSA_Historical_Data.csv":
        "data/raw/equities/Chile_IPSA_Historical_Data.csv",

    "data/raw/China_CSI300_Historical_Data.csv":
        "data/raw/equities/China_CSI300_Historical_Data.csv",

    "data/raw/Poland_WIG20_Historical_Data.csv":
        "data/raw/equities/Poland_WIG20_Historical_Data.csv",

    # Energy transition
    "data/raw/energy_transition_annual.csv":
        "data/raw/energy_transition/energy_transition_annual.csv",

    # Diagnostics
    "data/raw/fred_global_macro_recent_candidates.csv":
        "data/processed/diagnostics/fred_global_macro_recent_candidates.csv",

    "data/raw/fred_global_macro_search_results.csv":
        "data/processed/diagnostics/fred_global_macro_search_results.csv",

    "data/processed/fred_country_macro_scan.csv":
        "data/processed/diagnostics/fred_country_macro_scan.csv",

    # Existing quality outputs
    "data/processed/chile_cochilco_data_quality_anomalies.csv":
        "data/processed/quality/chile_cochilco_data_quality_anomalies.csv",

    "data/processed/chile_cochilco_data_quality_summary.csv":
        "data/processed/quality/chile_cochilco_data_quality_summary.csv",

    "data/processed/fred_data_quality_anomalies.csv":
        "data/processed/quality/fred_data_quality_anomalies.csv",

    "data/processed/fred_data_quality_summary.csv":
        "data/processed/quality/fred_data_quality_summary.csv",

    "data/processed/icsg_data_quality_anomalies.csv":
        "data/processed/quality/icsg_data_quality_anomalies.csv",

    "data/processed/icsg_data_quality_summary.csv":
        "data/processed/quality/icsg_data_quality_summary.csv",

    "data/processed/lme_data_quality_anomalies.csv":
        "data/processed/quality/lme_data_quality_anomalies.csv",

    "data/processed/lme_data_quality_summary.csv":
        "data/processed/quality/lme_data_quality_summary.csv",

    "data/processed/lme_monthly_observation_counts.csv":
        "data/processed/quality/lme_monthly_observation_counts.csv",

    "data/processed/peru_copper_cost_drivers_quality_anomalies.csv":
        "data/processed/quality/peru_copper_cost_drivers_quality_anomalies.csv",

    "data/processed/peru_copper_cost_drivers_quality_summary.csv":
        "data/processed/quality/peru_copper_cost_drivers_quality_summary.csv",

    "data/processed/peru_copper_mining_quality_anomalies.csv":
        "data/processed/quality/peru_copper_mining_quality_anomalies.csv",

    "data/processed/peru_copper_mining_quality_summary.csv":
        "data/processed/quality/peru_copper_mining_quality_summary.csv",

    "data/processed/usgs_data_quality_anomalies.csv":
        "data/processed/quality/usgs_data_quality_anomalies.csv",

    "data/processed/usgs_data_quality_summary.csv":
        "data/processed/quality/usgs_data_quality_summary.csv",
}


TEXT_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".sh",
    ".txt",
}


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
}


def create_directories():
    for relative in DIRECTORIES:
        path = ROOT / relative
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        print(
            "[DIR]",
            relative,
        )


def create_packages():
    package_dirs = [
        "src",
        "src/ingestion",
        "src/quality",
        "src/features",
        "src/models",
        "src/utils",
    ]

    for relative in package_dirs:
        init_file = ROOT / relative / "__init__.py"

        if not init_file.exists():
            init_file.write_text(
                "",
                encoding="utf-8",
            )

            print(
                "[PACKAGE]",
                init_file.relative_to(ROOT),
            )


def move_files():
    for old_relative, new_relative in MOVES.items():
        old_path = ROOT / old_relative
        new_path = ROOT / new_relative

        if not old_path.exists():
            print(
                "[SKIP]",
                old_relative,
            )
            continue

        if new_path.exists():
            print(
                "[EXISTS]",
                new_relative,
            )
            continue

        new_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(old_path),
            str(new_path),
        )

        print(
            "[MOVE]",
            old_relative,
            "->",
            new_relative,
        )


def should_scan(path):
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False

    if path.name == "restructure_project.py":
        return False

    relative_parts = path.relative_to(ROOT).parts

    if any(
        part in SKIP_DIRS
        for part in relative_parts
    ):
        return False

    return True


def update_references():
    replacements = dict(MOVES)

    changed_files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if not should_scan(path):
            continue

        try:
            original = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue

        updated = original

        for old_relative, new_relative in replacements.items():
            updated = updated.replace(
                old_relative,
                new_relative,
            )

        if updated != original:
            path.write_text(
                updated,
                encoding="utf-8",
            )

            changed_files.append(
                str(
                    path.relative_to(ROOT)
                )
            )

    print()
    print(
        "[INFO] Updated references in",
        len(changed_files),
        "files",
    )

    for item in changed_files:
        print(
            "[UPDATE]",
            item,
        )


def remove_pycache():
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(
                path,
                ignore_errors=True,
            )

            print(
                "[REMOVE]",
                path.relative_to(ROOT),
            )


def main():
    print(
        "[INFO] Starting project restructure"
    )

    create_directories()

    print()
    move_files()

    print()
    create_packages()

    print()
    update_references()

    print()
    remove_pycache()

    print()
    print(
        "[DONE] Project restructure completed"
    )


if __name__ == "__main__":
    main()
