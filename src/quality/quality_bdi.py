from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import SHIPPING_RAW_DIR



BDI_FILE = (
    SHIPPING_RAW_DIR
    / "baltic_dry_index_daily.csv"
)

EXPECTED_COLUMNS = [
    "date",
    "baltic_dry_index",
]


def load_data():

    if not BDI_FILE.exists():

        print(
            "[FAIL] File not found:",
            BDI_FILE,
        )

        raise SystemExit(
            1
        )

    print(
        "[PASS] File exists:",
        BDI_FILE,
    )

    df = pd.read_csv(
        BDI_FILE
    )

    return df


def check_expected_columns(df):

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "[FAIL] Missing columns:",
            missing_columns,
        )

        return 1

    print(
        "[PASS] All expected columns exist"
    )

    return 0


def prepare_data(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["baltic_dry_index"] = pd.to_numeric(
        df["baltic_dry_index"],
        errors="coerce",
    )

    return df


def check_dates(df):

    fail_count = 0

    invalid_dates = int(
        df["date"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:

        print(
            "[FAIL] Invalid dates:",
            invalid_dates,
        )

        fail_count += 1

    else:

        print(
            "[PASS] No invalid dates"
        )

    duplicate_dates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    if duplicate_dates > 0:

        print(
            "[FAIL] Duplicate dates:",
            duplicate_dates,
        )

        fail_count += 1

    else:

        print(
            "[PASS] No duplicate dates"
        )

    valid_dates = (
        df["date"]
        .dropna()
    )

    if not valid_dates.is_monotonic_increasing:

        print(
            "[FAIL] Dates are not sorted"
        )

        fail_count += 1

    else:

        print(
            "[PASS] Dates are sorted"
        )

    if not valid_dates.empty:

        print(
            "[INFO] Dataset range:",
            valid_dates.min().date(),
            "->",
            valid_dates.max().date(),
        )

    return fail_count


def check_numeric_values(df):

    fail_count = 0

    series = df[
        "baltic_dry_index"
    ]

    infinite_count = int(
        np.isinf(
            series
        ).sum()
    )

    if infinite_count > 0:

        print(
            "[FAIL] Infinite values:",
            infinite_count,
        )

        fail_count += 1

    else:

        print(
            "[PASS] No infinite values"
        )

    missing_count = int(
        series
        .isna()
        .sum()
    )

    if missing_count > 0:

        print(
            "[FAIL] Missing BDI values:",
            missing_count,
        )

        fail_count += 1

    else:

        print(
            "[PASS] No missing BDI values"
        )

    non_positive_count = int(
        (
            series <= 0
        ).sum()
    )

    if non_positive_count > 0:

        print(
            "[FAIL] Non-positive BDI values:",
            non_positive_count,
        )

        fail_count += 1

    else:

        print(
            "[PASS] All BDI values are positive"
        )

    return fail_count


def print_series_summary(df):

    series = df[
        "baltic_dry_index"
    ]

    print(
        "\n"
        + "=" * 120
    )

    print(
        "SERIES QUALITY SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        "[INFO] Valid observations:",
        int(
            series.notna().sum()
        ),
    )

    print(
        "[INFO] Minimum BDI:",
        series.min(),
    )

    print(
        "[INFO] Maximum BDI:",
        series.max(),
    )

    print(
        "[INFO] Mean BDI:",
        round(
            series.mean(),
            2,
        ),
    )


def check_daily_jumps(df):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "DAILY BDI JUMP SCREEN"
    )

    print(
        "=" * 120
    )

    warning_count = 0

    gap_days = (
        df["date"]
        .diff()
        .dt.days
    )

    change_pct = (
        df[
            "baltic_dry_index"
        ]
        .pct_change(
            fill_method=None
        )
        * 100
    )

    # Ignore percentage changes across long data gaps
    change_pct = change_pct.where(
        gap_days <= 7
    )

    suspicious = (
        change_pct.abs()
        > 25
    )

    count = int(
        suspicious.sum()
    )

    if count == 0:

        print(
            "[PASS] No daily BDI moves above 25%"
        )

        return 0

    warning_count += count

    print(
        f"[WARNING] BDI: "
        f"{count} daily moves above 25%"
    )

    temp = pd.DataFrame(
        {
            "date": df["date"],
            "value": df[
                "baltic_dry_index"
            ],
            "change_pct": change_pct,
        }
    )

    print(
        temp.loc[
            suspicious
        ]
        .tail(20)
        .to_string(
            index=False
        )
    )

    return warning_count


def check_large_gaps(df):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "DATE GAP SCREEN"
    )

    print(
        "=" * 120
    )

    gap_days = (
        df["date"]
        .diff()
        .dt.days
    )

    large_gaps = (
        gap_days > 14
    )

    gap_count = int(
        large_gaps.sum()
    )

    if gap_count == 0:

        print(
            "[PASS] No gaps above 14 days"
        )

        return 0

    print(
        f"[WARNING] BDI: "
        f"{gap_count} gaps above 14 days"
    )

    temp = pd.DataFrame(
        {
            "date": df["date"],
            "gap_days": gap_days,
        }
    )

    print(
        temp.loc[
            large_gaps
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    return gap_count


def check_freshness(df):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FRESHNESS CHECK"
    )

    print(
        "=" * 120
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    last_full_month_end = (
        today.replace(
            day=1
        )
        - pd.Timedelta(
            days=1
        )
    )

    last_date = (
        df["date"]
        .dropna()
        .max()
    )

    if pd.isna(
        last_date
    ):

        print(
            "[FAIL] No valid BDI date available"
        )

        return 1

    lag_days = (
        last_full_month_end
        - last_date
    ).days

    print(
        "[INFO] Last BDI date:",
        last_date.date(),
    )

    print(
        "[INFO] Last full month end:",
        last_full_month_end.date(),
    )

    print(
        "[INFO] Lag days:",
        lag_days,
    )

    if lag_days > 31:

        print(
            "[FAIL] BDI data is stale"
        )

        print(
            "[INFO] Download a fresh "
            "'Baltic Dry Index Gecmis Verileri.csv' "
            "file and run ingest_bdi.py again"
        )

        return 1

    print(
        "[PASS] BDI data freshness is acceptable"
    )

    return 0


def check_current_partial_month(df):

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    current_month_start = (
        today.replace(
            day=1
        )
    )

    current_month_rows = int(
        (
            df["date"]
            >= current_month_start
        ).sum()
    )

    print(
        "[INFO] Current partial month rows:",
        current_month_rows,
    )

    if current_month_rows > 0:

        print(
            "[INFO] Current partial month will be "
            "excluded during monthly feature creation"
        )


def main():

    print(
        "=" * 120
    )

    print(
        "BALTIC DRY INDEX DATA QUALITY"
    )

    print(
        "=" * 120
    )

    df = load_data()

    fail_count = 0
    warning_count = 0

    fail_count += (
        check_expected_columns(
            df
        )
    )

    if fail_count > 0:

        print(
            "[FAIL] Cannot continue without "
            "required columns"
        )

        raise SystemExit(
            1
        )

    df = prepare_data(
        df
    )

    fail_count += (
        check_dates(
            df
        )
    )

    fail_count += (
        check_numeric_values(
            df
        )
    )

    print_series_summary(
        df
    )

    warning_count += (
        check_daily_jumps(
            df
        )
    )

    warning_count += (
        check_large_gaps(
            df
        )
    )

    fail_count += (
        check_freshness(
            df
        )
    )

    check_current_partial_month(
        df
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 120
    )

    print(
        "[INFO] FAIL:",
        fail_count,
    )

    print(
        "[INFO] WARNING:",
        warning_count,
    )

    if fail_count > 0:

        print(
            "[FAIL] BDI quality check failed"
        )

        raise SystemExit(
            1
        )

    print(
        "[PASS] BDI quality check completed"
    )


if __name__ == "__main__":
    main()