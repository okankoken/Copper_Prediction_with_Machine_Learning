# -*- coding: utf-8 -*-

"""
FRED monthly data quality checks.

V1 scope:
- Invalid dates
- Duplicate months
- Missing calendar months
- Null values
- Latest available month by series
- Basic logical checks
"""

from pathlib import Path

import pandas as pd
from src.utils.paths import MACRO_RAW_DIR, QUALITY_DIR



# =========================
# PROJECT PATHS
# =========================


RAW_FILE = (
    MACRO_RAW_DIR
    / "fred_macro_monthly.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "fred_data_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "fred_data_quality_anomalies.csv"
)


# =========================
# HELPERS
# =========================

def add_summary(
    rows,
    check_name,
    status,
    value,
    detail,
):

    rows.append(
        {
            "check": check_name,
            "status": status,
            "value": value,
            "detail": detail,
        }
    )


# =========================
# LOAD DATA
# =========================

def load_fred_data():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"FRED raw file not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m",
        errors="coerce",
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


# =========================
# QUALITY CHECKS
# =========================

def validate_fred_data(df):

    summary = []
    anomalies = []

    # -------------------------
    # Basic information
    # -------------------------

    add_summary(
        summary,
        "row_count",
        "INFO",
        len(df),
        "Total number of monthly observations",
    )

    first_date = df["date"].min()
    last_date = df["date"].max()

    add_summary(
        summary,
        "first_month",
        "INFO",
        first_date.strftime("%Y-%m"),
        "First available FRED month",
    )

    add_summary(
        summary,
        "last_month",
        "INFO",
        last_date.strftime("%Y-%m"),
        "Latest row in FRED dataset",
    )

    # -------------------------
    # Invalid dates
    # -------------------------

    invalid_date_count = int(
        df["date"].isna().sum()
    )

    add_summary(
        summary,
        "invalid_dates",
        "PASS"
        if invalid_date_count == 0
        else "FAIL",
        invalid_date_count,
        "Dates that could not be parsed",
    )

    # -------------------------
    # Duplicate months
    # -------------------------

    duplicate_mask = df["date"].duplicated(
        keep=False
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    add_summary(
        summary,
        "duplicate_months",
        "PASS"
        if duplicate_count == 0
        else "FAIL",
        duplicate_count,
        "Duplicate monthly observations",
    )

    # -------------------------
    # Missing calendar months
    # -------------------------

    valid_dates = df["date"].dropna()

    expected_months = pd.date_range(
        start=valid_dates.min(),
        end=valid_dates.max(),
        freq="MS",
    )

    existing_months = pd.DatetimeIndex(
        valid_dates
        .dt.to_period("M")
        .dt.to_timestamp()
        .unique()
    )

    missing_months = expected_months.difference(
        existing_months
    )

    add_summary(
        summary,
        "missing_calendar_months",
        "PASS"
        if len(missing_months) == 0
        else "WARNING",
        len(missing_months),
        "Entire months missing from dataset",
    )

    for month in missing_months:

        anomalies.append(
            {
                "date": month.strftime("%Y-%m"),
                "type": "missing_month",
                "column": "date",
                "value": None,
                "detail": "Entire month missing",
            }
        )

    # -------------------------
    # Column checks
    # -------------------------

    value_columns = [
        column
        for column in df.columns
        if column != "date"
    ]

    for column in value_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        null_count = int(
            df[column].isna().sum()
        )

        add_summary(
            summary,
            f"null_{column}",
            "PASS"
            if null_count == 0
            else "WARNING",
            null_count,
            f"Missing values in {column}",
        )

        null_rows = df[
            df[column].isna()
        ]

        for _, row in null_rows.iterrows():

            anomalies.append(
                {
                    "date": (
                        row["date"].strftime("%Y-%m")
                        if pd.notna(row["date"])
                        else None
                    ),
                    "type": "null_value",
                    "column": column,
                    "value": None,
                    "detail": f"Missing value in {column}",
                }
            )

        non_null_dates = df.loc[
            df[column].notna(),
            "date",
        ]

        if len(non_null_dates) > 0:

            latest_date = non_null_dates.max()

            add_summary(
                summary,
                f"latest_{column}",
                "INFO",
                latest_date.strftime("%Y-%m"),
                f"Latest available month for {column}",
            )

    # -------------------------
    # Logical checks
    # -------------------------

    non_negative_columns = [
        "us_cpi_index",
        "us_industrial_production",
        "us_10y_treasury_yield",
        "m2_money_supply",
        "brent_oil_usd",
        "copper_fred_usd_per_ton",
    ]

    for column in non_negative_columns:

        if column not in df.columns:
            continue

        invalid_mask = (
            df[column].notna()
            & (df[column] < 0)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        add_summary(
            summary,
            f"negative_{column}",
            "PASS"
            if invalid_count == 0
            else "WARNING",
            invalid_count,
            f"Negative values found in {column}",
        )

    return (
        pd.DataFrame(summary),
        pd.DataFrame(anomalies),
    )


# =========================
# SAVE REPORTS
# =========================

def save_reports(
    summary_df,
    anomaly_df,
):

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    anomaly_df.to_csv(
        ANOMALY_FILE,
        index=False,
    )

    print(
        f"[OK] Summary report: {SUMMARY_FILE}"
    )

    print(
        f"[OK] Anomaly report: {ANOMALY_FILE}"
    )


# =========================
# MAIN
# =========================

def main():

    print(
        "[INFO] Starting FRED data quality checks..."
    )

    df = load_fred_data()

    summary_df, anomaly_df = (
        validate_fred_data(df)
    )

    save_reports(
        summary_df,
        anomaly_df,
    )

    print(
        "\n========== FRED DATA QUALITY SUMMARY =========="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n==============================================="
    )

    fail_count = int(
        summary_df["status"]
        .eq("FAIL")
        .sum()
    )

    warning_count = int(
        summary_df["status"]
        .eq("WARNING")
        .sum()
    )

    print(
        f"[RESULT] FAIL checks: {fail_count}"
    )

    print(
        f"[RESULT] WARNING checks: {warning_count}"
    )

    if fail_count > 0:

        print(
            "[RESULT] Overall status: FAIL"
        )

    elif warning_count > 0:

        print(
            "[RESULT] Overall status: WARNING"
        )

    else:

        print(
            "[RESULT] Overall status: PASS"
        )


if __name__ == "__main__":
    main()
