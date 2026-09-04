from pathlib import Path

import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    ENERGY_TRANSITION_RAW_DIR,
    EQUITIES_RAW_DIR,
    MACRO_RAW_DIR,
    MARKET_RAW_DIR,
    MINING_RAW_DIR,
    RISK_RAW_DIR,
    SHIPPING_RAW_DIR,
)


OUTPUT_FILE = (
    CONFIG_DIR
    / "master_feature_plan.csv"
)


rows = []


def add(
    category,
    source_file,
    source_column,
    output_column,
    frequency,
    aggregation,
    availability_rule,
    role="feature",
    include=True,
    reason="",
):
    rows.append(
        {
            "category": category,
            "source_file": str(source_file),
            "source_column": source_column,
            "output_column": output_column,
            "frequency": frequency,
            "aggregation": aggregation,
            "availability_rule": availability_rule,
            "role": role,
            "include": include,
            "reason": reason,
        }
    )


# ------------------------------------------------------------------
# LME COPPER
# ------------------------------------------------------------------

lme_file = (
    MARKET_RAW_DIR
    / "lme_copper_daily.csv"
)

add(
    "copper_market",
    lme_file,
    "cash_settlement_usd_per_ton",
    "cash_settlement_usd_per_ton",
    "daily",
    "monthly_mean",
    "same_month",
    role="target",
)

add(
    "copper_market",
    lme_file,
    "copper_stock_ton",
    "copper_stock_ton",
    "daily",
    "month_end_last",
    "same_month",
)

add(
    "copper_market",
    lme_file,
    "copper_3_month_usd_per_ton",
    "copper_3_month_usd_per_ton",
    "daily",
    "exclude",
    "same_month",
    include=False,
    reason="Excluded by project decision; LME cash settlement is target.",
)


# ------------------------------------------------------------------
# SHFE COPPER
# ------------------------------------------------------------------

shfe_file = (
    MARKET_RAW_DIR
    / "shfe_copper_daily.csv"
)

shfe_rules = {
    "close_cny_per_ton": "monthly_mean",
    "settlement_cny_per_ton": "monthly_mean",
    "pre_settlement_cny_per_ton": "monthly_mean",
    "volume_lots": "monthly_sum",
    "open_interest_lots": "month_end_last",
    "open_interest_change": "monthly_sum",
    "turnover": "monthly_sum",
}

for column, aggregation in shfe_rules.items():
    add(
        "copper_market",
        shfe_file,
        column,
        f"shfe_{column}",
        "daily",
        aggregation,
        "same_month",
    )

for column in [
    "delivery_month",
    "open_cny_per_ton",
    "high_cny_per_ton",
    "low_cny_per_ton",
]:
    add(
        "copper_market",
        shfe_file,
        column,
        f"shfe_{column}",
        "daily",
        "exclude",
        "same_month",
        include=False,
        reason="Excluded by project decision.",
    )


# ------------------------------------------------------------------
# OTHER LME METALS
# ------------------------------------------------------------------

metals_file = (
    MARKET_RAW_DIR
    / "lme_other_metals_daily.csv"
)

for metal in [
    "aluminum",
    "nickel",
    "zinc",
    "lead",
    "tin",
]:
    add(
        "other_metals",
        metals_file,
        f"{metal}_cash_usd_per_ton",
        f"{metal}_cash_usd_per_ton",
        "daily",
        "monthly_mean",
        "same_month",
    )

    add(
        "other_metals",
        metals_file,
        f"{metal}_stock_ton",
        f"{metal}_stock_ton",
        "daily",
        "month_end_last",
        "same_month",
    )

    add(
        "other_metals",
        metals_file,
        f"{metal}_3_month_usd_per_ton",
        f"{metal}_3_month_usd_per_ton",
        "daily",
        "exclude",
        "same_month",
        include=False,
        reason="Excluded by project decision.",
    )


# ------------------------------------------------------------------
# PALLADIUM
# ------------------------------------------------------------------

add(
    "other_metals",
    MARKET_RAW_DIR / "palladium_daily.csv",
    "palladium_usd_per_troy_ounce",
    "palladium_usd_per_troy_ounce",
    "daily",
    "monthly_mean",
    "same_month",
)


# ------------------------------------------------------------------
# WORLD BANK COMMODITIES
# ------------------------------------------------------------------

worldbank_file = (
    MARKET_RAW_DIR
    / "worldbank_commodities_monthly.csv"
)

for column in [
    "gold_usd_per_troy_ounce",
    "silver_usd_per_troy_ounce",
    "platinum_usd_per_troy_ounce",
    "iron_ore_usd_per_dmtu",
    "wti_usd_per_barrel",
    "coal_australia_usd_per_ton",
]:
    add(
        "commodities",
        worldbank_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# BALTIC DRY INDEX
# ------------------------------------------------------------------

add(
    "shipping",
    SHIPPING_RAW_DIR / "baltic_dry_index_daily.csv",
    "baltic_dry_index",
    "baltic_dry_index",
    "daily",
    "monthly_mean",
    "same_month",
)


# ------------------------------------------------------------------
# PORTWATCH
# ------------------------------------------------------------------

portwatch_file = (
    SHIPPING_RAW_DIR
    / "portwatch_shipping_activity_daily.csv"
)

portwatch_columns = [
    "global_portcalls",
    "global_container_portcalls",
    "global_dry_bulk_portcalls",
    "global_cargo_portcalls",
    "global_import",
    "global_export",
    "global_dry_bulk_import",
    "global_dry_bulk_export",
    "global_container_import",
    "global_container_export",
    "copper_supply_portcalls",
    "copper_supply_dry_bulk_portcalls",
    "copper_supply_dry_bulk_import",
    "copper_supply_dry_bulk_export",
    "china_copper_related_portcalls",
    "china_copper_related_dry_bulk_portcalls",
    "china_copper_related_dry_bulk_import",
    "china_copper_related_dry_bulk_export",
]

for column in portwatch_columns:
    add(
        "shipping",
        portwatch_file,
        column,
        column,
        "daily",
        "monthly_sum",
        "same_month",
    )


# ------------------------------------------------------------------
# FRED
# ------------------------------------------------------------------

fred_file = (
    MACRO_RAW_DIR
    / "fred_macro_monthly.csv"
)

fred_df = pd.read_csv(
    fred_file,
    nrows=5,
)

for column in fred_df.columns:
    if column in {
        "date",
        "month",
    }:
        continue

    if column == "copper_fred_usd_per_ton":
        add(
            "macro_fred",
            fred_file,
            column,
            column,
            "monthly",
            "exclude",
            "native_month",
            include=False,
            reason="Excluded because LME cash settlement is the copper target.",
        )
        continue

    add(
        "macro_fred",
        fred_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# GLOBAL MACRO
# ------------------------------------------------------------------

global_macro_file = (
    MACRO_RAW_DIR
    / "global_macro_monthly.csv"
)

global_macro_df = pd.read_csv(
    global_macro_file,
    nrows=5,
)

for column in global_macro_df.columns:
    if column in {
        "date",
        "month",
    }:
        continue

    add(
        "global_macro",
        global_macro_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# CHINA MONTHLY
# ------------------------------------------------------------------

china_monthly_files = [
    MARKET_RAW_DIR / "china_refined_copper_monthly.csv",
    MACRO_RAW_DIR / "china_industrial_output_monthly.csv",
    MACRO_RAW_DIR / "china_ppi_monthly.csv",
    MACRO_RAW_DIR / "china_electricity_generation_monthly.csv",
    MACRO_RAW_DIR / "china_fixed_asset_investment_monthly.csv",
    MACRO_RAW_DIR / "china_real_estate_investment_monthly.csv",
]

for file_path in china_monthly_files:
    df = pd.read_csv(
        file_path,
        nrows=5,
    )

    for column in df.columns:
        if column in {
            "date",
            "month",
            "source",
            "source_url",
            "unit",
            "source_value_10000_ton",
        }:
            continue

        add(
            "china",
            file_path,
            column,
            column,
            "monthly",
            "as_is",
            "native_month",
        )


# ------------------------------------------------------------------
# TURKEY
# ------------------------------------------------------------------

turkey_file = (
    MACRO_RAW_DIR
    / "turkey_monthly.csv"
)

turkey_df = pd.read_csv(
    turkey_file,
    nrows=5,
)

for column in turkey_df.columns:
    if column in {
        "date",
        "month",
    }:
        continue

    add(
        "turkey",
        turkey_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# PERU COPPER COST DRIVERS
# ------------------------------------------------------------------

peru_cost_file = (
    MINING_RAW_DIR
    / "peru_copper_cost_drivers_monthly.csv"
)

peru_cost_df = pd.read_csv(
    peru_cost_file,
    nrows=5,
)

for column in peru_cost_df.columns:
    if column in {
        "date",
        "month",
        "source",
        "source_url",
        "unit",
    }:
        continue

    add(
        "peru_cost_drivers",
        peru_cost_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )

# ------------------------------------------------------------------
# GLOBAL LEADING INDICATOR
# ------------------------------------------------------------------

leading_file = (
    MACRO_RAW_DIR
    / "global_leading_indicators_monthly.csv"
)

leading_df = pd.read_csv(
    leading_file,
    nrows=5,
)

for column in leading_df.columns:
    if column in {
        "date",
        "month",
    }:
        continue

    add(
        "leading_indicators",
        leading_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# RISK
# ------------------------------------------------------------------

for risk_file in [
    RISK_RAW_DIR / "geopolitical_risk_monthly.csv",
    RISK_RAW_DIR / "economic_policy_uncertainty_monthly.csv",
]:
    df = pd.read_csv(
        risk_file,
        nrows=5,
    )

    for column in df.columns:
        if column in {
            "date",
            "month",
            "source",
            "source_url",
            "unit",
        }:
            continue

        add(
            "risk",
            risk_file,
            column,
            column,
            "monthly",
            "as_is",
            "native_month",
        )


# ------------------------------------------------------------------
# COUNTRY STOCK INDICES
# ------------------------------------------------------------------

indices_file = (
    EQUITIES_RAW_DIR
    / "country_stock_indices_monthly.csv"
)

indices_df = pd.read_csv(
    indices_file,
    nrows=5,
)

for column in indices_df.columns:
    if column == "date":
        continue

    add(
        "equities_indices",
        indices_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# COPPER COMPANY STOCKS
# ------------------------------------------------------------------

companies_file = (
    EQUITIES_RAW_DIR
    / "copper_company_stocks_monthly.csv"
)

companies_df = pd.read_csv(
    companies_file,
    nrows=5,
)

for column in companies_df.columns:
    if column == "date":
        continue

    add(
        "equities_companies",
        companies_file,
        column,
        column,
        "monthly",
        "as_is",
        "native_month",
    )


# ------------------------------------------------------------------
# ANNUAL DATA
# ------------------------------------------------------------------

annual_files = [
    (
        "usgs",
        MINING_RAW_DIR / "usgs_copper_annual.csv",
    ),
    (
        "icsg",
        MINING_RAW_DIR / "icsg_copper_annual.csv",
    ),
    (
        "chile_cochilco",
        MINING_RAW_DIR / "chile_cochilco_copper_cost_annual.csv",
    ),
    (
        "peru_mining",
        MINING_RAW_DIR / "peru_copper_mining_annual.csv",
    ),
    (
        "energy_transition",
        ENERGY_TRANSITION_RAW_DIR / "energy_transition_annual.csv",
    ),
]

metadata_columns = {
    "year",
    "date",
    "month",
    "observation_year",
    "report_year",
    "source",
    "source_url",
    "unit",
    "employment_source",
    "production_source",
}

for category, file_path in annual_files:
    df = pd.read_csv(
        file_path,
        nrows=5,
    )

    for column in df.columns:
        if column in metadata_columns:
            continue

        add(
            category,
            file_path,
            column,
            column,
            "annual",
            "repeat_within_year",
            "observation_year",
        )


# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

plan = pd.DataFrame(
    rows
)

plan = plan.sort_values(
    [
        "category",
        "source_file",
        "source_column",
    ]
).reset_index(
    drop=True
)

CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

plan.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"[OK] Saved: {OUTPUT_FILE}"
)

print(
    f"[INFO] Total rules: {len(plan)}"
)

print(
    "[INFO] Included:",
    int(plan["include"].sum()),
)

print(
    "[INFO] Excluded:",
    int((~plan["include"]).sum()),
)

print()
print(
    plan.groupby(
        [
            "category",
            "frequency",
            "aggregation",
            "include",
        ]
    )
    .size()
    .to_string()
)
