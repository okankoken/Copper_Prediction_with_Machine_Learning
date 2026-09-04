# -*- coding: utf-8 -*-

"""
Copper project data quality module.

V1 scope:
- Validate daily LME Copper data
- Detect missing, duplicate and suspicious observations
- Check data freshness
- Check price and stock anomalies
- Check monthly observation coverage
- Save quality reports under data/processed

Bu script veriyi otomatik olarak silmez veya degistirmez.
Sadece kalite problemlerini raporlar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import MARKET_RAW_DIR, QUALITY_DIR



# =========================
# PROJECT PATHS
# =========================


RAW_FILE = (
    MARKET_RAW_DIR
    / "lme_copper_daily.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "lme_data_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "lme_data_quality_anomalies.csv"
)

MONTHLY_COUNT_FILE = (
    QUALITY_DIR
    / "lme_monthly_observation_counts.csv"
)


# =========================
# QUALITY SETTINGS
# =========================

PRICE_COLUMNS = [
    "cash_settlement_usd_per_ton",
    "copper_3_month_usd_per_ton",
]

STOCK_COLUMN = "copper_stock_ton"

MAX_FRESHNESS_DAYS_PASS = 3
MAX_FRESHNESS_DAYS_WARNING = 7

PRICE_JUMP_WARNING_PCT = 10.0
STOCK_JUMP_WARNING_PCT = 30.0

SPREAD_RATIO_WARNING = 0.15

MIN_FULL_MONTH_OBSERVATIONS = 15


# =========================
# HELPERS
# =========================

def add_summary(
    rows: list,
    check_name: str,
    status: str,
    value,
    detail: str,
) -> None:
    """
    Kalite sonucunu summary listesine ekler.
    """

    rows.append(
        {
            "check": check_name,
            "status": status,
            "value": value,
            "detail": detail,
        }
    )


def get_last_completed_month() -> pd.Period:
    """
    Son tamamlanmis ayi YYYY-MM Period olarak dondurur.
    """

    today = pd.Timestamp.today().normalize()

    last_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_month_end.to_period("M")


# =========================
# LOAD DATA
# =========================

def load_lme_data() -> pd.DataFrame:
    """
    LME raw CSV dosyasini okur.
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"LME raw file not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    required_columns = [
        "date",
        "cash_settlement_usd_per_ton",
        "copper_3_month_usd_per_ton",
        "copper_stock_ton",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for column in required_columns[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


# =========================
# MAIN QUALITY CHECK
# =========================

def validate_lme_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    summary = []
    anomalies = []

    today = pd.Timestamp.today().normalize()

    # -------------------------
    # Basic dataset information
    # -------------------------

    row_count = len(df)

    add_summary(
        summary,
        "row_count",
        "INFO",
        row_count,
        "Total number of daily observations",
    )

    first_date = df["date"].min()
    last_date = df["date"].max()

    add_summary(
        summary,
        "first_date",
        "INFO",
        first_date.date()
        if pd.notna(first_date)
        else None,
        "First available LME observation",
    )

    add_summary(
        summary,
        "last_date",
        "INFO",
        last_date.date()
        if pd.notna(last_date)
        else None,
        "Latest available LME observation",
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
    # Duplicate dates
    # -------------------------

    duplicate_mask = df["date"].duplicated(
        keep=False
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    add_summary(
        summary,
        "duplicate_dates",
        "PASS"
        if duplicate_count == 0
        else "FAIL",
        duplicate_count,
        "Duplicate daily dates",
    )

    if duplicate_count > 0:

        duplicate_rows = df[
            duplicate_mask
        ].copy()

        for _, row in duplicate_rows.iterrows():

            anomalies.append(
                {
                    "date": row["date"],
                    "type": "duplicate_date",
                    "column": "date",
                    "value": row["date"],
                    "detail": "Duplicate date detected",
                }
            )

    # -------------------------
    # Null values
    # -------------------------

    value_columns = [
        "cash_settlement_usd_per_ton",
        "copper_3_month_usd_per_ton",
        "copper_stock_ton",
    ]

    for column in value_columns:

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
                    "date": row["date"],
                    "type": "null_value",
                    "column": column,
                    "value": np.nan,
                    "detail": (
                        f"Missing value in {column}"
                    ),
                }
            )

    # -------------------------
    # Zero or negative prices
    # -------------------------

    for column in PRICE_COLUMNS:

        invalid_mask = (
            df[column].notna()
            & (df[column] <= 0)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        add_summary(
            summary,
            f"non_positive_{column}",
            "PASS"
            if invalid_count == 0
            else "FAIL",
            invalid_count,
            "Price must be greater than zero",
        )

        for _, row in df[
            invalid_mask
        ].iterrows():

            anomalies.append(
                {
                    "date": row["date"],
                    "type": "non_positive_price",
                    "column": column,
                    "value": row[column],
                    "detail": (
                        "Price is zero or negative"
                    ),
                }
            )

    # -------------------------
    # Negative stock
    # -------------------------

    negative_stock_mask = (
        df[STOCK_COLUMN].notna()
        & (df[STOCK_COLUMN] < 0)
    )

    negative_stock_count = int(
        negative_stock_mask.sum()
    )

    add_summary(
        summary,
        "negative_stock",
        "PASS"
        if negative_stock_count == 0
        else "FAIL",
        negative_stock_count,
        "LME stock cannot be negative",
    )

    # -------------------------
    # Data freshness
    # -------------------------

    if pd.notna(last_date):

        freshness_days = (
            today - last_date.normalize()
        ).days

        if freshness_days <= MAX_FRESHNESS_DAYS_PASS:
            freshness_status = "PASS"

        elif freshness_days <= MAX_FRESHNESS_DAYS_WARNING:
            freshness_status = "WARNING"

        else:
            freshness_status = "FAIL"

        add_summary(
            summary,
            "data_freshness_days",
            freshness_status,
            freshness_days,
            (
                "Calendar days between today "
                "and latest LME observation"
            ),
        )

    # -------------------------
    # Large price changes
    # -------------------------

    for column in PRICE_COLUMNS:

        change_column = (
            f"{column}_daily_change_pct"
        )

        df[change_column] = (
            df[column]
            .pct_change(
                fill_method=None
            )
            * 100
        )

        jump_mask = (
            df[change_column]
            .abs()
            > PRICE_JUMP_WARNING_PCT
        )

        jump_count = int(
            jump_mask.sum()
        )

        add_summary(
            summary,
            f"large_change_{column}",
            "PASS"
            if jump_count == 0
            else "WARNING",
            jump_count,
            (
                f"Daily absolute change above "
                f"{PRICE_JUMP_WARNING_PCT}%"
            ),
        )

        for _, row in df[
            jump_mask
        ].iterrows():

            anomalies.append(
                {
                    "date": row["date"],
                    "type": "large_price_change",
                    "column": column,
                    "value": row[change_column],
                    "detail": (
                        "Daily price change "
                        "exceeded warning threshold"
                    ),
                }
            )

    # -------------------------
    # Large stock changes
    # -------------------------

    df["stock_daily_change_pct"] = (
        df[STOCK_COLUMN]
        .pct_change(
            fill_method=None
        )
        * 100
    )

    stock_jump_mask = (
        df["stock_daily_change_pct"]
        .abs()
        > STOCK_JUMP_WARNING_PCT
    )

    stock_jump_count = int(
        stock_jump_mask.sum()
    )

    add_summary(
        summary,
        "large_stock_change",
        "PASS"
        if stock_jump_count == 0
        else "WARNING",
        stock_jump_count,
        (
            f"Daily stock change above "
            f"{STOCK_JUMP_WARNING_PCT}%"
        ),
    )

    for _, row in df[
        stock_jump_mask
    ].iterrows():

        anomalies.append(
            {
                "date": row["date"],
                "type": "large_stock_change",
                "column": STOCK_COLUMN,
                "value": row[
                    "stock_daily_change_pct"
                ],
                "detail": (
                    "Daily stock change "
                    "exceeded warning threshold"
                ),
            }
        )

    # -------------------------
    # Cash vs 3M spread
    # -------------------------

    df["cash_3m_spread_usd"] = (
        df[
            "cash_settlement_usd_per_ton"
        ]
        - df[
            "copper_3_month_usd_per_ton"
        ]
    )

    df["cash_3m_spread_ratio"] = (
        df["cash_3m_spread_usd"].abs()
        / df[
            "cash_settlement_usd_per_ton"
        ]
    )

    spread_mask = (
        df["cash_3m_spread_ratio"]
        > SPREAD_RATIO_WARNING
    )

    spread_count = int(
        spread_mask.sum()
    )

    add_summary(
        summary,
        "extreme_cash_3m_spread",
        "PASS"
        if spread_count == 0
        else "WARNING",
        spread_count,
        (
            "Absolute Cash-3M spread exceeds "
            f"{SPREAD_RATIO_WARNING:.0%} "
            "of cash price"
        ),
    )

    for _, row in df[
        spread_mask
    ].iterrows():

        anomalies.append(
            {
                "date": row["date"],
                "type": "cash_3m_spread",
                "column": "cash_3m_spread_usd",
                "value": row[
                    "cash_3m_spread_usd"
                ],
                "detail": (
                    "Unusually large Cash-3M spread"
                ),
            }
        )

    # -------------------------
    # Calendar gaps
    # -------------------------

    df["calendar_gap_days"] = (
        df["date"]
        .diff()
        .dt.days
    )

    gap_mask = (
        df["calendar_gap_days"] > 4
    )

    gap_count = int(
        gap_mask.sum()
    )

    add_summary(
        summary,
        "calendar_gaps_over_4_days",
        "PASS"
        if gap_count == 0
        else "INFO",
        gap_count,
        (
            "Calendar gaps may include weekends "
            "and LME market holidays."
        ),
    )

    for _, row in df[
        gap_mask
    ].iterrows():

        anomalies.append(
            {
                "date": row["date"],
                "type": "calendar_gap",
                "column": "date",
                "value": row[
                    "calendar_gap_days"
                ],
                "detail": (
                    "More than 4 calendar days "
                    "since previous observation"
                ),
            }
        )

    # -------------------------
    # Monthly observation count
    # -------------------------

    df["month"] = (
        df["date"]
        .dt.to_period("M")
    )

    monthly_counts = (
        df.groupby("month")
        .size()
        .reset_index(
            name="observation_count"
        )
    )

    monthly_counts[
        "month"
    ] = monthly_counts[
        "month"
    ].astype(str)

    last_completed_month = (
        get_last_completed_month()
    )

    completed_months = (
        df[
            df["date"].dt.to_period("M")
            <= last_completed_month
        ]
        .groupby(
            df["date"].dt.to_period("M")
        )
        .size()
    )

    low_months = completed_months[
        completed_months
        < MIN_FULL_MONTH_OBSERVATIONS
    ]

    add_summary(
        summary,
        "low_observation_months",
        "PASS"
        if len(low_months) == 0
        else "WARNING",
        len(low_months),
        (
            "Completed months with fewer than "
            f"{MIN_FULL_MONTH_OBSERVATIONS} observations"
        ),
    )

    for month, count in low_months.items():

        anomalies.append(
            {
                "date": str(month),
                "type": "low_monthly_coverage",
                "column": "date",
                "value": int(count),
                "detail": (
                    "Low number of daily "
                    "observations in completed month"
                ),
            }
        )

    # -------------------------
    # Final result
    # -------------------------

    summary_df = pd.DataFrame(
        summary
    )

    anomaly_df = pd.DataFrame(
        anomalies
    )

    return (
        summary_df,
        anomaly_df,
        monthly_counts,
    )


# =========================
# SAVE REPORTS
# =========================

def save_reports(
    summary_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    monthly_counts: pd.DataFrame,
) -> None:

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    anomaly_df.to_csv(
        ANOMALY_FILE,
        index=False,
    )

    monthly_counts.to_csv(
        MONTHLY_COUNT_FILE,
        index=False,
    )

    print(
        f"[OK] Summary report: {SUMMARY_FILE}"
    )

    print(
        f"[OK] Anomaly report: {ANOMALY_FILE}"
    )

    print(
        f"[OK] Monthly counts: {MONTHLY_COUNT_FILE}"
    )


# =========================
# MAIN
# =========================

def main():

    print(
        "[INFO] Starting LME data quality checks..."
    )

    df = load_lme_data()

    summary_df, anomaly_df, monthly_counts = (
        validate_lme_data(df)
    )

    save_reports(
        summary_df,
        anomaly_df,
        monthly_counts,
    )

    print(
        "\n========== DATA QUALITY SUMMARY =========="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n=========================================="
    )

    fail_count = (
        summary_df["status"]
        .eq("FAIL")
        .sum()
    )

    warning_count = (
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
