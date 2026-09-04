from pathlib import Path
import sys

import pandas as pd
from src.utils.paths import MACRO_RAW_DIR



DATA_FILE = (
    MACRO_RAW_DIR
    / "global_leading_indicators_monthly.csv"
)


EXPECTED_COLUMNS = [
    "date",
    "g20_composite_leading_indicator",
]


EXPECTED_START_DATE = pd.Timestamp(
    "2015-01-01"
)

MAX_LAG_MONTHS = 3
MAX_MISSING_RATIO = 0.01


results = []


def add_result(
    status,
    check,
    detail,
):
    results.append(
        {
            "status": status,
            "check": check,
            "detail": detail,
        }
    )


def check_file():
    if DATA_FILE.exists():

        add_result(
            "PASS",
            "File exists",
            str(DATA_FILE),
        )

        return True

    add_result(
        "FAIL",
        "File exists",
        f"Missing file: {DATA_FILE}",
    )

    return False


def check_schema(df):
    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing:

        add_result(
            "FAIL",
            "Expected columns",
            f"Missing columns: {missing}",
        )

    else:

        add_result(
            "PASS",
            "Expected columns",
            "All expected columns exist",
        )

    if extra:

        add_result(
            "WARNING",
            "Unexpected columns",
            f"Unexpected columns: {extra}",
        )

    else:

        add_result(
            "PASS",
            "Unexpected columns",
            "No unexpected columns",
        )


def check_dates(df):
    null_dates = int(
        df["date"]
        .isna()
        .sum()
    )

    if null_dates > 0:

        add_result(
            "FAIL",
            "Date parsing",
            f"Null dates: {null_dates}",
        )

        return

    add_result(
        "PASS",
        "Date parsing",
        "All dates parsed successfully",
    )

    duplicate_dates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    if duplicate_dates > 0:

        add_result(
            "FAIL",
            "Duplicate dates",
            f"Duplicate dates: {duplicate_dates}",
        )

    else:

        add_result(
            "PASS",
            "Duplicate dates",
            "No duplicate dates",
        )

    if df["date"].is_monotonic_increasing:

        add_result(
            "PASS",
            "Date sorting",
            "Dates are sorted ascending",
        )

    else:

        add_result(
            "WARNING",
            "Date sorting",
            "Dates are not sorted ascending",
        )

    start_date = df["date"].min()

    if start_date == EXPECTED_START_DATE:

        add_result(
            "PASS",
            "Start date",
            f"Start date: {start_date.date()}",
        )

    else:

        add_result(
            "FAIL",
            "Start date",
            (
                f"Expected {EXPECTED_START_DATE.date()}, "
                f"found {start_date.date()}"
            ),
        )


def check_monthly_continuity(df):
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS",
    )

    actual_dates = pd.DatetimeIndex(
        df["date"]
    )

    missing_dates = expected_dates.difference(
        actual_dates
    )

    if len(missing_dates) == 0:

        add_result(
            "PASS",
            "Monthly continuity",
            "No missing months",
        )

    else:

        add_result(
            "FAIL",
            "Monthly continuity",
            f"Missing months: {len(missing_dates)}",
        )


def check_values(df):
    for column in EXPECTED_COLUMNS[1:]:

        missing_ratio = (
            df[column]
            .isna()
            .mean()
        )

        if missing_ratio == 0:

            add_result(
                "PASS",
                f"Missing values - {column}",
                "0.00%",
            )

        elif missing_ratio <= MAX_MISSING_RATIO:

            add_result(
                "WARNING",
                f"Missing values - {column}",
                f"{missing_ratio:.2%}",
            )

        else:

            add_result(
                "FAIL",
                f"Missing values - {column}",
                f"{missing_ratio:.2%}",
            )

        non_positive = int(
            (
                df[column] <= 0
            )
            .sum()
        )

        if non_positive == 0:

            add_result(
                "PASS",
                f"Positive values - {column}",
                "All observed values are positive",
            )

        else:

            add_result(
                "FAIL",
                f"Positive values - {column}",
                f"Non-positive rows: {non_positive}",
            )


def check_freshness(df):
    current_period = (
        pd.Timestamp.today()
        .normalize()
        .to_period("M")
    )

    for column in EXPECTED_COLUMNS[1:]:

        valid = (
            df[
                [
                    "date",
                    column,
                ]
            ]
            .dropna()
        )

        if valid.empty:

            add_result(
                "FAIL",
                f"Freshness - {column}",
                "No valid observations",
            )

            continue

        latest_date = valid[
            "date"
        ].max()

        latest_period = (
            latest_date
            .to_period("M")
        )

        lag_months = (
            current_period.ordinal
            - latest_period.ordinal
        )

        if lag_months <= MAX_LAG_MONTHS:

            add_result(
                "PASS",
                f"Freshness - {column}",
                (
                    f"Latest month: {latest_date:%Y-%m}, "
                    f"lag: {lag_months} month(s)"
                ),
            )

        else:

            add_result(
                "FAIL",
                f"Freshness - {column}",
                (
                    f"Latest month: {latest_date:%Y-%m}, "
                    f"lag: {lag_months} month(s)"
                ),
            )


def check_cli_range(df):
    column = (
        "g20_composite_leading_indicator"
    )

    minimum = df[column].min()
    maximum = df[column].max()

    if (
        minimum > 0
        and maximum < 200
    ):

        add_result(
            "PASS",
            "G20 CLI plausible range",
            (
                f"Min: {minimum:.4f}, "
                f"Max: {maximum:.4f}"
            ),
        )

    else:

        add_result(
            "WARNING",
            "G20 CLI plausible range",
            (
                f"Min: {minimum:.4f}, "
                f"Max: {maximum:.4f}"
            ),
        )


def check_recent_variation(df):
    column = (
        "g20_composite_leading_indicator"
    )

    recent = df.tail(
        24
    )

    unique_count = (
        recent[column]
        .nunique(
            dropna=True
        )
    )

    if unique_count >= 12:

        add_result(
            "PASS",
            "Recent variation",
            (
                f"{unique_count} unique values "
                "in recent 24 months"
            ),
        )

    else:

        add_result(
            "WARNING",
            "Recent variation",
            (
                f"Only {unique_count} unique values "
                "in recent 24 months"
            ),
        )


def print_summary(df):
    print(
        "=" * 110
    )

    print(
        "GLOBAL LEADING INDICATORS DATA QUALITY REPORT"
    )

    print(
        "=" * 110
    )

    print(
        "[INFO] File:",
        DATA_FILE,
    )

    print(
        "[INFO] Rows:",
        len(df),
    )

    print(
        "[INFO] Columns:",
        len(df.columns),
    )

    if (
        not df.empty
        and "date" in df.columns
        and df["date"].notna().any()
    ):

        print(
            "[INFO] Date range:",
            df["date"].min().date(),
            "->",
            df["date"].max().date(),
        )

    print()

    result_df = pd.DataFrame(
        results
    )

    if not result_df.empty:

        print(
            result_df.to_string(
                index=False
            )
        )

    counts = (
        result_df["status"]
        .value_counts()
        .to_dict()
    )

    pass_count = counts.get(
        "PASS",
        0,
    )

    warning_count = counts.get(
        "WARNING",
        0,
    )

    fail_count = counts.get(
        "FAIL",
        0,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        "[PASS]:",
        pass_count,
    )

    print(
        "[WARNING]:",
        warning_count,
    )

    print(
        "[FAIL]:",
        fail_count,
    )

    print()

    if fail_count > 0:

        print(
            "[RESULT] FAIL"
        )

        return 1

    if warning_count > 0:

        print(
            "[RESULT] PASS WITH WARNINGS"
        )

        return 0

    print(
        "[RESULT] PASS"
    )

    return 0


def main():
    if not check_file():

        return print_summary(
            pd.DataFrame()
        )

    df = pd.read_csv(
        DATA_FILE
    )

    check_schema(
        df
    )

    if "date" not in df.columns:

        add_result(
            "FAIL",
            "Date column",
            "date column is missing",
        )

        return print_summary(
            df
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    check_dates(
        df
    )

    if df["date"].notna().any():

        check_monthly_continuity(
            df
        )

    check_values(
        df
    )

    check_freshness(
        df
    )

    check_cli_range(
        df
    )

    check_recent_variation(
        df
    )

    return print_summary(
        df
    )


if __name__ == "__main__":
    exit_code = main()

    sys.exit(
        exit_code
    )
