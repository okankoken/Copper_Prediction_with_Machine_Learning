from pathlib import Path

import pandas as pd

from src.utils.paths import MACRO_RAW_DIR, MARKET_RAW_DIR


FILES = {
    "refined_copper": (
        MARKET_RAW_DIR
        / "china_refined_copper_monthly.csv"
    ),
    "industrial_output": (
        MACRO_RAW_DIR
        / "china_industrial_output_monthly.csv"
    ),
    "ppi": (
        MACRO_RAW_DIR
        / "china_ppi_monthly.csv"
    ),
    
    "electricity_generation": (
        MACRO_RAW_DIR
        / "china_electricity_generation_monthly.csv"
    ),
    
    "fixed_asset_investment": (
        MACRO_RAW_DIR
        / "china_fixed_asset_investment_monthly.csv"
    ),
    "real_estate_investment": (
        MACRO_RAW_DIR
        / "china_real_estate_investment_monthly.csv"
    ),
}

def check_investment_growth(
    results,
    dataset_name,
    file_key,
    value_column,
):
    path = FILES[
        file_key
    ]

    if not path.exists():
        add_result(
            results,
            dataset_name,
            "FAIL",
            "file_exists",
            f"File not found: {path}",
        )
        return

    df = pd.read_csv(
        path
    )

    required_columns = [
        "month",
        value_column,
        "unit",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(
                    missing_columns
                )
            ),
        )
        return

    add_result(
        results,
        dataset_name,
        "PASS",
        "required_columns",
        "All required columns are present",
    )

    df, missing_months = (
        check_common_monthly(
            df,
            dataset_name,
            value_column,
            results,
        )
    )

    extreme_level_count = int(
        (
            df[
                value_column
            ].abs()
            > 50
        ).sum()
    )

    if extreme_level_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_yoy_level",
            (
                f"{extreme_level_count} values "
                "have absolute YoY above 50%"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_yoy_level",
            "No absolute YoY values above 50%",
        )

    df = df.sort_values(
        "month"
    ).copy()

    df["change_pp"] = (
        df[
            value_column
        ].diff()
    )

    extreme_change_count = int(
        (
            df[
                "change_pp"
            ].abs()
            > 15
        ).sum()
    )

    if extreme_change_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_monthly_change",
            (
                f"{extreme_change_count} changes "
                "exceed 15 percentage points"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_monthly_change",
            (
                "No changes exceed "
                "15 percentage points"
            ),
        )

def check_electricity_generation(results):
    dataset_name = "china_electricity_generation"

    path = FILES[
        "electricity_generation"
    ]

    if not path.exists():
        add_result(
            results,
            dataset_name,
            "FAIL",
            "file_exists",
            f"File not found: {path}",
        )
        return

    df = pd.read_csv(
        path
    )

    required_columns = [
        "month",
        "china_electricity_generation_100m_kwh",
        "unit",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        )
        return

    add_result(
        results,
        dataset_name,
        "PASS",
        "required_columns",
        "All required columns are present",
    )

    df, missing_months = (
        check_common_monthly(
            df,
            dataset_name,
            "china_electricity_generation_100m_kwh",
            results,
        )
    )

    non_positive_count = int(
        (
            df[
                "china_electricity_generation_100m_kwh"
            ]
            <= 0
        ).sum()
    )

    if non_positive_count > 0:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "non_positive_values",
            (
                f"{non_positive_count} values "
                "are <= 0"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "non_positive_values",
            "All generation values are positive",
        )

    df = df.sort_values(
        "month"
    ).copy()

    df["mom_change"] = (
        df[
            "china_electricity_generation_100m_kwh"
        ]
        .pct_change(
            fill_method=None
        )
    )

    extreme_count = int(
        (
            df["mom_change"]
            .abs()
            > 0.35
        ).sum()
    )

    if extreme_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_mom_change",
            (
                f"{extreme_count} monthly changes "
                "exceed 35%"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_mom_change",
            "No monthly changes exceed 35%",
        )

def add_result(
    results,
    dataset,
    level,
    check,
    detail,
):
    results.append(
        {
            "dataset": dataset,
            "level": level,
            "check": check,
            "detail": detail,
        }
    )


def check_common_monthly(
    df,
    dataset_name,
    value_column,
    results,
):
    df = df.copy()

    df["month"] = (
        df["month"]
        .astype(str)
    )

    parsed_month = pd.to_datetime(
        df["month"],
        format="%Y-%m",
        errors="coerce",
    )

    invalid_count = int(
        parsed_month.isna().sum()
    )

    if invalid_count > 0:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "invalid_month",
            f"{invalid_count} invalid month values",
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "invalid_month",
            "All month values are valid",
        )

    duplicate_count = int(
        df.duplicated(
            subset=["month"]
        ).sum()
    )

    if duplicate_count > 0:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "duplicate_month",
            f"{duplicate_count} duplicate month rows",
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "duplicate_month",
            "No duplicate months",
        )

    df[value_column] = pd.to_numeric(
        df[value_column],
        errors="coerce",
    )

    missing_value_count = int(
        df[value_column]
        .isna()
        .sum()
    )

    if missing_value_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "missing_values",
            f"{missing_value_count} missing values",
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "missing_values",
            "No missing values",
        )

    valid_dates = (
        parsed_month
        .dropna()
    )

    missing_months = []

    if not valid_dates.empty:
        first_month = (
            valid_dates
            .min()
            .to_period("M")
        )

        last_month = (
            valid_dates
            .max()
            .to_period("M")
        )

        expected_months = pd.period_range(
            first_month,
            last_month,
            freq="M",
        )

        actual_months = set(
            valid_dates
            .dt
            .to_period("M")
        )

        missing_months = [
            str(month)
            for month in expected_months
            if month not in actual_months
        ]

        if missing_months:
            add_result(
                results,
                dataset_name,
                "WARNING",
                "missing_months",
                (
                    f"{len(missing_months)} missing "
                    "calendar months"
                ),
            )
        else:
            add_result(
                results,
                dataset_name,
                "PASS",
                "missing_months",
                "No missing calendar months",
            )

        add_result(
            results,
            dataset_name,
            "PASS",
            "coverage",
            (
                f"{first_month} -> {last_month}, "
                f"{len(df)} observations"
            ),
        )

    return df, missing_months


def check_refined_copper(results):
    dataset_name = "china_refined_copper"

    path = FILES["refined_copper"]

    if not path.exists():
        add_result(
            results,
            dataset_name,
            "FAIL",
            "file_exists",
            f"File not found: {path}",
        )
        return

    df = pd.read_csv(
        path
    )

    required_columns = [
        "month",
        "china_refined_copper_production_ton",
        "source_value_10000_ton",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        )
        return

    add_result(
        results,
        dataset_name,
        "PASS",
        "required_columns",
        "All required columns are present",
    )

    df, missing_months = (
        check_common_monthly(
            df,
            dataset_name,
            "china_refined_copper_production_ton",
            results,
        )
    )

    non_positive_count = int(
        (
            df[
                "china_refined_copper_production_ton"
            ]
            <= 0
        ).sum()
    )

    if non_positive_count > 0:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "non_positive_values",
            (
                f"{non_positive_count} values "
                "are <= 0"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "non_positive_values",
            "All production values are positive",
        )

    df[
        "source_value_10000_ton"
    ] = pd.to_numeric(
        df[
            "source_value_10000_ton"
        ],
        errors="coerce",
    )

    expected_value = (
        df[
            "source_value_10000_ton"
        ]
        * 10000
    )

    conversion_difference = (
        df[
            "china_refined_copper_production_ton"
        ]
        - expected_value
    ).abs()

    conversion_error_count = int(
        (
            conversion_difference
            > 0.01
        ).sum()
    )

    if conversion_error_count > 0:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "unit_conversion",
            (
                f"{conversion_error_count} rows "
                "have inconsistent conversion"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "unit_conversion",
            "All rows satisfy source_value * 10000",
        )

    df = df.sort_values(
        "month"
    ).copy()

    df["mom_change"] = (
        df[
            "china_refined_copper_production_ton"
        ]
        .pct_change(
            fill_method=None
        )
    )

    extreme_count = int(
        (
            df["mom_change"]
            .abs()
            > 0.30
        ).sum()
    )

    if extreme_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_mom_change",
            (
                f"{extreme_count} monthly changes "
                "exceed 30%"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_mom_change",
            "No monthly changes exceed 30%",
        )


def check_industrial_output(results):
    dataset_name = "china_industrial_output"

    path = FILES["industrial_output"]

    if not path.exists():
        add_result(
            results,
            dataset_name,
            "FAIL",
            "file_exists",
            f"File not found: {path}",
        )
        return

    df = pd.read_csv(
        path
    )

    required_columns = [
        "month",
        "china_industrial_value_added_yoy_pct",
        "unit",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        )
        return

    add_result(
        results,
        dataset_name,
        "PASS",
        "required_columns",
        "All required columns are present",
    )

    df, missing_months = (
        check_common_monthly(
            df,
            dataset_name,
            "china_industrial_value_added_yoy_pct",
            results,
        )
    )

    extreme_level_count = int(
        (
            df[
                "china_industrial_value_added_yoy_pct"
            ]
            .abs()
            > 30
        ).sum()
    )

    if extreme_level_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_yoy_level",
            (
                f"{extreme_level_count} values "
                "have absolute YoY above 30%"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_yoy_level",
            "No absolute YoY values above 30%",
        )

    df = df.sort_values(
        "month"
    ).copy()

    df["change_pp"] = (
        df[
            "china_industrial_value_added_yoy_pct"
        ]
        .diff()
    )

    extreme_change_count = int(
        (
            df["change_pp"]
            .abs()
            > 10
        ).sum()
    )

    if extreme_change_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_monthly_change",
            (
                f"{extreme_change_count} changes "
                "exceed 10 percentage points"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_monthly_change",
            (
                "No changes exceed "
                "10 percentage points"
            ),
        )


def check_ppi(results):
    dataset_name = "china_ppi"

    path = FILES["ppi"]

    if not path.exists():
        add_result(
            results,
            dataset_name,
            "FAIL",
            "file_exists",
            f"File not found: {path}",
        )
        return

    df = pd.read_csv(
        path
    )

    required_columns = [
        "month",
        "china_ppi_yoy_pct",
        "unit",
        "source",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            dataset_name,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        )
        return

    add_result(
        results,
        dataset_name,
        "PASS",
        "required_columns",
        "All required columns are present",
    )

    df, missing_months = (
        check_common_monthly(
            df,
            dataset_name,
            "china_ppi_yoy_pct",
            results,
        )
    )

    extreme_level_count = int(
        (
            df[
                "china_ppi_yoy_pct"
            ]
            .abs()
            > 20
        ).sum()
    )

    if extreme_level_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_yoy_level",
            (
                f"{extreme_level_count} values "
                "have absolute YoY above 20%"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_yoy_level",
            "No absolute YoY values above 20%",
        )

    df = df.sort_values(
        "month"
    ).copy()

    df["change_pp"] = (
        df[
            "china_ppi_yoy_pct"
        ]
        .diff()
    )

    extreme_change_count = int(
        (
            df["change_pp"]
            .abs()
            > 8
        ).sum()
    )

    if extreme_change_count > 0:
        add_result(
            results,
            dataset_name,
            "WARNING",
            "extreme_monthly_change",
            (
                f"{extreme_change_count} changes "
                "exceed 8 percentage points"
            ),
        )
    else:
        add_result(
            results,
            dataset_name,
            "PASS",
            "extreme_monthly_change",
            (
                "No changes exceed "
                "8 percentage points"
            ),
        )


def main():
    print(
        "=" * 80
    )

    print(
        "CHINA DATA QUALITY"
    )

    print(
        "=" * 80
    )

    results = []

    check_refined_copper(
        results
    )

    check_industrial_output(
        results
    )

    check_ppi(
        results
    )
    
    check_electricity_generation(
        results
    )

    check_investment_growth(
        results,
        "china_fixed_asset_investment",
        "fixed_asset_investment",
        "china_fixed_asset_investment_yoy_pct",
    )
    
    check_investment_growth(
        results,
        "china_real_estate_investment",
        "real_estate_investment",
        "china_real_estate_investment_yoy_pct",
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n[INFO] Quality results:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    pass_count = int(
        (
            results_df["level"]
            == "PASS"
        ).sum()
    )

    warning_count = int(
        (
            results_df["level"]
            == "WARNING"
        ).sum()
    )

    fail_count = int(
        (
            results_df["level"]
            == "FAIL"
        ).sum()
    )

    print(
        "\n[INFO] Summary:"
    )

    print(
        f"PASS={pass_count} "
        f"WARNING={warning_count} "
        f"FAIL={fail_count}"
    )

    if fail_count > 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
