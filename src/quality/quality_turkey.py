from pathlib import Path

import pandas as pd
from src.utils.paths import MACRO_RAW_DIR


INPUT_PATH = (
    MACRO_RAW_DIR
    / "turkey_monthly.csv"
)


def add_result(
    results,
    check_name,
    status,
    details,
):
    results.append(
        {
            "check": check_name,
            "status": status,
            "details": details,
        }
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    results = []

    expected_columns = [
        "date",
        "turkey_usd_try_monthly_average",
        "turkey_eur_try_monthly_average",
        "turkey_cpi_index",
        "turkey_domestic_ppi_index",
        "turkey_unemployment_rate_percent",
        "turkey_real_sector_confidence_index",
        "turkey_consumer_confidence_index",
        "turkey_manufacturing_capacity_utilization_percent",
        "turkey_ppi_metal_ores_index",
        "turkey_ppi_nonferrous_metal_ores_index",
        "turkey_ppi_basic_metals_index",
        "turkey_ppi_nonferrous_metals_index",
        "turkey_ppi_electrical_equipment_index",
        "turkey_ppi_wire_cable_index",
        "turkey_ppi_household_appliances_index",
        "turkey_ppi_motor_vehicles_index",
        "turkey_capacity_intermediate_goods_percent",
        "turkey_capacity_investment_goods_percent",
        "turkey_capacity_basic_metals_percent",
        "turkey_capacity_fabricated_metals_percent",
        "turkey_capacity_electrical_equipment_percent",
        "turkey_capacity_machinery_percent",
        "turkey_capacity_motor_vehicles_percent",
        "turkey_exports_usd",
        "turkey_imports_usd",
        "turkey_foreign_trade_balance_usd",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    add_result(
        results,
        "required_columns",
        "FAIL" if missing_columns else "PASS",
        (
            f"Missing columns: {missing_columns}"
            if missing_columns
            else "All required columns are present"
        ),
    )

    if df["date"].isna().any():
        add_result(
            results,
            "valid_dates",
            "FAIL",
            "Invalid dates detected",
        )
    else:
        add_result(
            results,
            "valid_dates",
            "PASS",
            "All dates are valid",
        )

    if df["date"].duplicated().any():
        add_result(
            results,
            "duplicate_dates",
            "FAIL",
            (
                f"Duplicate dates: "
                f"{int(df['date'].duplicated().sum())}"
            ),
        )
    else:
        add_result(
            results,
            "duplicate_dates",
            "PASS",
            "No duplicate dates",
        )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    expected_dates = pd.date_range(
        start="2015-01-01",
        end=df["date"].max(),
        freq="MS",
    )

    observed_dates = pd.DatetimeIndex(
        df["date"]
    )

    missing_dates = expected_dates.difference(
        observed_dates
    )

    if len(missing_dates) == 0:
        add_result(
            results,
            "monthly_coverage",
            "PASS",
            (
                f"Continuous monthly coverage from "
                f"{df['date'].min().date()} to "
                f"{df['date'].max().date()}"
            ),
        )
    else:
        add_result(
            results,
            "monthly_coverage",
            "FAIL",
            f"Missing months: {list(missing_dates)}",
        )

    numeric_columns = [
        column
        for column in expected_columns
        if column != "date"
        and column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    allowed_latest_missing = {
        "turkey_unemployment_rate_percent": 1,
        "turkey_exports_usd": 1,
        "turkey_imports_usd": 1,
        "turkey_foreign_trade_balance_usd": 1,
    }

    for column in numeric_columns:
        missing_count = int(
            df[column].isna().sum()
        )

        allowed = allowed_latest_missing.get(
            column,
            0,
        )

        if missing_count == 0:
            status = "PASS"
            details = "No missing values"

        elif missing_count <= allowed:
            status = "WARNING"
            details = (
                f"Missing values: {missing_count}. "
                f"Accepted as publication lag"
            )

        else:
            status = "FAIL"
            details = (
                f"Missing values: {missing_count}"
            )

        add_result(
            results,
            f"{column}_missing",
            status,
            details,
        )

    positive_columns = [
        "turkey_usd_try_monthly_average",
        "turkey_eur_try_monthly_average",
        "turkey_cpi_index",
        "turkey_domestic_ppi_index",
        "turkey_real_sector_confidence_index",
        "turkey_consumer_confidence_index",
        "turkey_ppi_metal_ores_index",
        "turkey_ppi_nonferrous_metal_ores_index",
        "turkey_ppi_basic_metals_index",
        "turkey_ppi_nonferrous_metals_index",
        "turkey_ppi_electrical_equipment_index",
        "turkey_ppi_wire_cable_index",
        "turkey_ppi_household_appliances_index",
        "turkey_ppi_motor_vehicles_index",
        "turkey_exports_usd",
        "turkey_imports_usd",
    ]

    for column in positive_columns:
        invalid = int(
            (
                df[column].dropna()
                <= 0
            ).sum()
        )

        add_result(
            results,
            f"{column}_positive",
            "PASS" if invalid == 0 else "FAIL",
            (
                "All available values are positive"
                if invalid == 0
                else f"Non-positive values: {invalid}"
            ),
        )

    percent_columns = [
        "turkey_unemployment_rate_percent",
        "turkey_manufacturing_capacity_utilization_percent",
        "turkey_capacity_intermediate_goods_percent",
        "turkey_capacity_investment_goods_percent",
        "turkey_capacity_basic_metals_percent",
        "turkey_capacity_fabricated_metals_percent",
        "turkey_capacity_electrical_equipment_percent",
        "turkey_capacity_machinery_percent",
        "turkey_capacity_motor_vehicles_percent",
    ]

    for column in percent_columns:
        values = df[
            column
        ].dropna()

        valid = values.between(
            0,
            100,
        ).all()

        add_result(
            results,
            f"{column}_range",
            "PASS" if valid else "FAIL",
            (
                "All available values are between 0 and 100"
                if valid
                else "Values outside 0-100 range detected"
            ),
        )

    calculated_balance = (
        df["turkey_exports_usd"]
        - df["turkey_imports_usd"]
    )

    mask = (
        calculated_balance.notna()
        & df[
            "turkey_foreign_trade_balance_usd"
        ].notna()
    )

    difference = (
        calculated_balance[mask]
        - df.loc[
            mask,
            "turkey_foreign_trade_balance_usd",
        ]
    ).abs()

    if len(difference) == 0:
        add_result(
            results,
            "foreign_trade_balance_reconciliation",
            "WARNING",
            "No overlapping trade observations available",
        )
    elif (
        difference <= 1
    ).all():
        add_result(
            results,
            "foreign_trade_balance_reconciliation",
            "PASS",
            "Trade balance equals exports minus imports",
        )
    else:
        add_result(
            results,
            "foreign_trade_balance_reconciliation",
            "FAIL",
            f"Maximum difference: {difference.max()}",
        )

    latest_date = df[
        "date"
    ].max()

    latest_row = df.loc[
        df["date"] == latest_date
    ].iloc[0]

    core_columns = [
        "turkey_usd_try_monthly_average",
        "turkey_eur_try_monthly_average",
        "turkey_cpi_index",
        "turkey_domestic_ppi_index",
        "turkey_real_sector_confidence_index",
        "turkey_consumer_confidence_index",
        "turkey_manufacturing_capacity_utilization_percent",
    ]

    missing_core_latest = [
        column
        for column in core_columns
        if pd.isna(
            latest_row[column]
        )
    ]

    add_result(
        results,
        "latest_core_data",
        (
            "PASS"
            if not missing_core_latest
            else "WARNING"
        ),
        (
            f"Core data available through {latest_date.date()}"
            if not missing_core_latest
            else (
                "Latest month missing core series: "
                f"{missing_core_latest}"
            )
        ),
    )

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 120)
    print("TURKEY DATA QUALITY")
    print("=" * 120)

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    summary = (
        result_df[
            "status"
        ]
        .value_counts()
        .reindex(
            [
                "PASS",
                "WARNING",
                "FAIL",
            ],
            fill_value=0,
        )
    )

    for status, count in summary.items():
        print(
            f"{status}: {count}"
        )

    fail_count = int(
        (
            result_df[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
