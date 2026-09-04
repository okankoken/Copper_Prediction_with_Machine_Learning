from pathlib import Path

import pandas as pd
from src.utils.paths import ENERGY_TRANSITION_RAW_DIR


INPUT_PATH = (
    ENERGY_TRANSITION_RAW_DIR
    / "energy_transition_annual.csv"
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
        "global_ev_sales_units",
        "china_ev_sales_units",
        "europe_ev_sales_units",
        "usa_ev_sales_units",
        "rest_of_world_ev_sales_units",
        "global_ev_sales_share_percent",
        "global_ev_stock_units",
        "china_ev_stock_units",
        "europe_ev_stock_units",
        "usa_ev_stock_units",
        "rest_of_world_ev_stock_units",
        "global_renewable_capacity_mw",
        "global_solar_capacity_mw",
        "global_wind_capacity_mw",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            "required_columns",
            "FAIL",
            f"Missing columns: {missing_columns}",
        )
    else:
        add_result(
            results,
            "required_columns",
            "PASS",
            "All required columns are present",
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
        duplicates = int(
            df["date"].duplicated().sum()
        )

        add_result(
            results,
            "duplicate_dates",
            "FAIL",
            f"Duplicate dates: {duplicates}",
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

    years = df[
        "date"
    ].dt.year.tolist()

    expected_years = list(
        range(
            2015,
            2026,
        )
    )

    if years == expected_years:
        add_result(
            results,
            "annual_coverage",
            "PASS",
            "Continuous annual coverage from 2015 to 2025",
        )
    else:
        add_result(
            results,
            "annual_coverage",
            "FAIL",
            f"Observed years: {years}",
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

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            add_result(
                results,
                f"{column}_missing",
                "PASS",
                "No missing values",
            )
        else:
            add_result(
                results,
                f"{column}_missing",
                "FAIL",
                f"Missing values: {missing_count}",
            )

        negative_count = int(
            (
                df[column] < 0
            ).sum()
        )

        if negative_count == 0:
            add_result(
                results,
                f"{column}_non_negative",
                "PASS",
                "No negative values",
            )
        else:
            add_result(
                results,
                f"{column}_non_negative",
                "FAIL",
                f"Negative values: {negative_count}",
            )

    sales_components = [
        "china_ev_sales_units",
        "europe_ev_sales_units",
        "usa_ev_sales_units",
        "rest_of_world_ev_sales_units",
    ]

    calculated_sales = df[
        sales_components
    ].sum(
        axis=1
    )

    sales_difference = (
        calculated_sales
        - df["global_ev_sales_units"]
    ).abs()

    if (
        sales_difference <= 1
    ).all():
        add_result(
            results,
            "global_ev_sales_reconciliation",
            "PASS",
            "Global EV sales equal regional components",
        )
    else:
        add_result(
            results,
            "global_ev_sales_reconciliation",
            "FAIL",
            (
                "Maximum difference: "
                f"{sales_difference.max()}"
            ),
        )

    stock_components = [
        "china_ev_stock_units",
        "europe_ev_stock_units",
        "usa_ev_stock_units",
        "rest_of_world_ev_stock_units",
    ]

    calculated_stock = df[
        stock_components
    ].sum(
        axis=1
    )

    stock_difference = (
        calculated_stock
        - df["global_ev_stock_units"]
    ).abs()

    if (
        stock_difference <= 1
    ).all():
        add_result(
            results,
            "global_ev_stock_reconciliation",
            "PASS",
            "Global EV stock equals regional components",
        )
    else:
        add_result(
            results,
            "global_ev_stock_reconciliation",
            "FAIL",
            (
                "Maximum difference: "
                f"{stock_difference.max()}"
            ),
        )

    share_valid = (
        df[
            "global_ev_sales_share_percent"
        ].between(
            0,
            100,
        )
    ).all()

    if share_valid:
        add_result(
            results,
            "ev_sales_share_range",
            "PASS",
            "EV sales share is between 0 and 100 percent",
        )
    else:
        add_result(
            results,
            "ev_sales_share_range",
            "FAIL",
            "EV sales share outside valid range",
        )

    renewable_ge_solar = (
        df[
            "global_renewable_capacity_mw"
        ]
        >= df[
            "global_solar_capacity_mw"
        ]
    ).all()

    add_result(
        results,
        "renewable_capacity_ge_solar",
        (
            "PASS"
            if renewable_ge_solar
            else "FAIL"
        ),
        (
            "Renewable capacity is always >= solar capacity"
            if renewable_ge_solar
            else "Solar capacity exceeds total renewable capacity"
        ),
    )

    renewable_ge_wind = (
        df[
            "global_renewable_capacity_mw"
        ]
        >= df[
            "global_wind_capacity_mw"
        ]
    ).all()

    add_result(
        results,
        "renewable_capacity_ge_wind",
        (
            "PASS"
            if renewable_ge_wind
            else "FAIL"
        ),
        (
            "Renewable capacity is always >= wind capacity"
            if renewable_ge_wind
            else "Wind capacity exceeds total renewable capacity"
        ),
    )

    capacity_columns = [
        "global_renewable_capacity_mw",
        "global_solar_capacity_mw",
        "global_wind_capacity_mw",
    ]

    for column in capacity_columns:
        differences = df[
            column
        ].diff().dropna()

        decreasing = int(
            (
                differences < 0
            ).sum()
        )

        if decreasing == 0:
            add_result(
                results,
                f"{column}_trend",
                "PASS",
                "No annual capacity decline detected",
            )
        else:
            add_result(
                results,
                f"{column}_trend",
                "WARNING",
                f"Annual declines detected: {decreasing}",
            )

    stock_difference = df[
        "global_ev_stock_units"
    ].diff().dropna()

    if (
        stock_difference >= 0
    ).all():
        add_result(
            results,
            "global_ev_stock_trend",
            "PASS",
            "Global EV stock is non-decreasing",
        )
    else:
        add_result(
            results,
            "global_ev_stock_trend",
            "WARNING",
            "Global EV stock contains an annual decline",
        )

    latest_year = int(
        df["date"].dt.year.max()
    )

    if latest_year >= 2025:
        add_result(
            results,
            "freshness",
            "PASS",
            f"Latest annual observation: {latest_year}",
        )
    else:
        add_result(
            results,
            "freshness",
            "WARNING",
            f"Latest annual observation: {latest_year}",
        )

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 120)
    print("ENERGY TRANSITION DATA QUALITY")
    print("=" * 120)

    print(
        result_df.to_string(
            index=False
        )
    )

    print()

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

    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

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
