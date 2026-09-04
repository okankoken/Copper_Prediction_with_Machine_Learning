from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    MONTHLY_DIR,
    QUALITY_DIR,
)


MASTER_FILE = (
    MONTHLY_DIR
    / "copper_monthly_master.csv"
)

PLAN_FILE = (
    CONFIG_DIR
    / "master_feature_plan.csv"
)

SUMMARY_FILE = (
    QUALITY_DIR
    / "master_data_quality_summary.csv"
)

COLUMN_FILE = (
    QUALITY_DIR
    / "master_data_quality_columns.csv"
)


TARGET_COLUMN = "cash_settlement_usd_per_ton"


FORBIDDEN_COLUMNS = {
    "delivery_month",
    "shfe_delivery_month",
    "shfe_open_cny_per_ton",
    "shfe_high_cny_per_ton",
    "shfe_low_cny_per_ton",
    "copper_3_month_usd_per_ton",
    "copper_fred_usd_per_ton",
}


def get_last_full_month():
    today = pd.Timestamp.today().normalize()

    return (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )


def add_result(
    rows,
    check,
    status,
    value,
    detail,
):
    rows.append(
        {
            "check": check,
            "status": status,
            "value": value,
            "detail": detail,
        }
    )


def main():
    print("=" * 80)
    print("MASTER DATA QUALITY")
    print("=" * 80)

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ------------------------------------------------------------
    # FILE CHECKS
    # ------------------------------------------------------------

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master file not found: {MASTER_FILE}"
        )

    if not PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Feature plan not found: {PLAN_FILE}"
        )

    df = pd.read_csv(
        MASTER_FILE,
        parse_dates=["date"],
    )

    plan = pd.read_csv(
        PLAN_FILE
    )

    plan["include"] = (
        plan["include"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    included = plan[
        plan["include"] == True
    ].copy()

    # ------------------------------------------------------------
    # BASIC STRUCTURE
    # ------------------------------------------------------------

    add_result(
        results,
        "row_count",
        "PASS",
        len(df),
        "Total monthly rows",
    )

    add_result(
        results,
        "column_count",
        "PASS",
        len(df.columns),
        "Total columns including date",
    )

    duplicate_dates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    add_result(
        results,
        "duplicate_dates",
        "PASS" if duplicate_dates == 0 else "FAIL",
        duplicate_dates,
        "Duplicate monthly dates",
    )

    duplicate_columns = int(
        df.columns
        .duplicated()
        .sum()
    )

    add_result(
        results,
        "duplicate_columns",
        "PASS" if duplicate_columns == 0 else "FAIL",
        duplicate_columns,
        "Duplicate column names",
    )

    sorted_dates = bool(
        df["date"]
        .is_monotonic_increasing
    )

    add_result(
        results,
        "date_order",
        "PASS" if sorted_dates else "FAIL",
        sorted_dates,
        "Dates must be sorted ascending",
    )

    # ------------------------------------------------------------
    # MONTH CONTINUITY
    # ------------------------------------------------------------

    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="ME",
    )

    missing_months = (
        expected_dates
        .difference(
            pd.DatetimeIndex(
                df["date"]
            )
        )
    )

    add_result(
        results,
        "missing_calendar_months",
        "PASS" if len(missing_months) == 0 else "FAIL",
        len(missing_months),
        (
            "Missing months: "
            + ", ".join(
                d.strftime("%Y-%m")
                for d in missing_months
            )
            if len(missing_months) > 0
            else "No missing months"
        ),
    )

    # ------------------------------------------------------------
    # LAST FULL MONTH
    # ------------------------------------------------------------

    expected_last = (
        get_last_full_month()
        .to_period("M")
        .to_timestamp("M")
    )

    actual_last = (
        df["date"]
        .max()
    )

    last_month_status = (
        "PASS"
        if actual_last == expected_last
        else "FAIL"
    )

    add_result(
        results,
        "last_full_month",
        last_month_status,
        actual_last.strftime("%Y-%m-%d"),
        (
            "Expected last full month: "
            f"{expected_last.strftime('%Y-%m-%d')}"
        ),
    )

    # ------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        add_result(
            results,
            "target_exists",
            "FAIL",
            0,
            f"Missing target: {TARGET_COLUMN}",
        )
    else:
        add_result(
            results,
            "target_exists",
            "PASS",
            1,
            TARGET_COLUMN,
        )

        target_missing = int(
            df[TARGET_COLUMN]
            .isna()
            .sum()
        )

        add_result(
            results,
            "target_missing",
            "PASS" if target_missing == 0 else "FAIL",
            target_missing,
            "Target missing values",
        )

    # ------------------------------------------------------------
    # PLAN VS MASTER
    # ------------------------------------------------------------

    expected_features = set(
        included[
            "output_column"
        ]
    )

    actual_features = set(
        df.columns
    ) - {"date"}

    missing_features = sorted(
        expected_features
        - actual_features
    )

    extra_features = sorted(
        actual_features
        - expected_features
    )

    add_result(
        results,
        "plan_missing_features",
        "PASS" if not missing_features else "FAIL",
        len(missing_features),
        (
            str(missing_features)
            if missing_features
            else "None"
        ),
    )

    add_result(
        results,
        "plan_extra_features",
        "PASS" if not extra_features else "FAIL",
        len(extra_features),
        (
            str(extra_features)
            if extra_features
            else "None"
        ),
    )

    # ------------------------------------------------------------
    # FORBIDDEN COLUMNS
    # ------------------------------------------------------------

    forbidden_present = sorted(
        FORBIDDEN_COLUMNS
        .intersection(
            df.columns
        )
    )

    add_result(
        results,
        "forbidden_columns",
        "PASS" if not forbidden_present else "FAIL",
        len(forbidden_present),
        (
            str(forbidden_present)
            if forbidden_present
            else "None"
        ),
    )

    # ------------------------------------------------------------
    # NUMERIC QUALITY
    # ------------------------------------------------------------

    numeric_df = (
        df.drop(
            columns=["date"]
        )
        .select_dtypes(
            include=[
                np.number
            ]
        )
    )

    inf_count = int(
        np.isinf(
            numeric_df
        )
        .sum()
        .sum()
    )

    add_result(
        results,
        "infinite_values",
        "PASS" if inf_count == 0 else "FAIL",
        inf_count,
        "Positive or negative infinity values",
    )

    all_null_columns = [
        column
        for column in df.columns
        if column != "date"
        and df[column].isna().all()
    ]

    add_result(
        results,
        "all_null_columns",
        "PASS" if not all_null_columns else "FAIL",
        len(all_null_columns),
        (
            str(all_null_columns)
            if all_null_columns
            else "None"
        ),
    )

    # ------------------------------------------------------------
    # COLUMN PROFILE
    # ------------------------------------------------------------

    rows = []

    plan_lookup = (
        included
        .drop_duplicates(
            subset=[
                "output_column"
            ]
        )
        .set_index(
            "output_column"
        )
    )

    for column in df.columns:
        if column == "date":
            continue

        series = df[
            column
        ]

        non_null = (
            series.notna()
        )

        unique_count = int(
            series
            .dropna()
            .nunique()
        )

        constant = (
            unique_count <= 1
        )

        if column in plan_lookup.index:
            rule = plan_lookup.loc[
                column
            ]

            category = rule[
                "category"
            ]

            frequency = rule[
                "frequency"
            ]

            aggregation = rule[
                "aggregation"
            ]
        else:
            category = None
            frequency = None
            aggregation = None

        first_valid = (
            df.loc[
                non_null,
                "date"
            ].min()
            if non_null.any()
            else pd.NaT
        )

        last_valid = (
            df.loc[
                non_null,
                "date"
            ].max()
            if non_null.any()
            else pd.NaT
        )

        rows.append(
            {
                "column": column,
                "category": category,
                "frequency": frequency,
                "aggregation": aggregation,
                "dtype": str(
                    series.dtype
                ),
                "non_null_count": int(
                    non_null.sum()
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_ratio": float(
                    series.isna().mean()
                ),
                "unique_count": unique_count,
                "is_constant": constant,
                "first_valid_month": first_valid,
                "last_valid_month": last_valid,
            }
        )

    column_report = pd.DataFrame(
        rows
    )

    constant_columns = (
        column_report.loc[
            column_report[
                "is_constant"
            ],
            "column",
        ]
        .tolist()
    )

    add_result(
        results,
        "constant_columns",
        "WARNING" if constant_columns else "PASS",
        len(constant_columns),
        (
            str(constant_columns)
            if constant_columns
            else "None"
        ),
    )

    # ------------------------------------------------------------
    # ANNUAL CONSISTENCY
    # ------------------------------------------------------------

    annual_columns = (
        included.loc[
            included[
                "frequency"
            ] == "annual",
            "output_column",
        ]
        .tolist()
    )

    annual_violations = []

    temp = df.copy()

    temp["year"] = (
        temp["date"]
        .dt.year
    )

    for column in annual_columns:
        if column not in temp.columns:
            continue

        counts = (
            temp.groupby(
                "year"
            )[column]
            .nunique(
                dropna=True
            )
        )

        bad_years = (
            counts[
                counts > 1
            ]
            .index
            .tolist()
        )

        if bad_years:
            annual_violations.append(
                {
                    "column": column,
                    "years": bad_years,
                }
            )

    add_result(
        results,
        "annual_within_year_consistency",
        (
            "PASS"
            if not annual_violations
            else "FAIL"
        ),
        len(
            annual_violations
        ),
        (
            str(
                annual_violations
            )
            if annual_violations
            else "All annual values are constant within year"
        ),
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    column_report.to_csv(
        COLUMN_FILE,
        index=False,
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "[OK] Summary:",
        SUMMARY_FILE,
    )

    print(
        "[OK] Columns:",
        COLUMN_FILE,
    )

    fail_count = int(
        (
            summary["status"]
            == "FAIL"
        )
        .sum()
    )

    warning_count = int(
        (
            summary["status"]
            == "WARNING"
        )
        .sum()
    )

    print()
    print(
        f"[INFO] FAIL: {fail_count}"
    )

    print(
        f"[INFO] WARNING: {warning_count}"
    )

    if fail_count > 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
