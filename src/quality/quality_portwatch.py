from pathlib import Path
import sys
import pandas as pd
from src.utils.paths import SHIPPING_RAW_DIR



DATA_FILE = (
    SHIPPING_RAW_DIR
    / "portwatch_shipping_activity_daily.csv"
)


EXPECTED_COLUMNS = [
    "date",
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


MIN_EXPECTED_DATE = pd.Timestamp(
    "2019-01-01"
)

MAX_ACCEPTABLE_SOURCE_LAG_DAYS = 14

MAX_MISSING_RATIO = 0.01

MAX_ZERO_RATIO_GLOBAL = 0.01

MAX_ZERO_RATIO_BASKET = 0.20


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


def check_file_exists():
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


def check_columns(df):
    actual_columns = list(
        df.columns
    )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        add_result(
            "FAIL",
            "Expected columns",
            f"Missing columns: {missing_columns}",
        )
    else:
        add_result(
            "PASS",
            "Expected columns",
            f"All {len(EXPECTED_COLUMNS)} expected columns exist",
        )

    if unexpected_columns:
        add_result(
            "WARNING",
            "Unexpected columns",
            f"Unexpected columns: {unexpected_columns}",
        )
    else:
        add_result(
            "PASS",
            "Unexpected columns",
            "No unexpected columns",
        )


def check_date_quality(df):
    if df["date"].isna().any():
        add_result(
            "FAIL",
            "Date parsing",
            f"Null dates: {df['date'].isna().sum()}",
        )

        return

    add_result(
        "PASS",
        "Date parsing",
        "All dates parsed successfully",
    )

    duplicate_dates = int(
        df["date"].duplicated().sum()
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

    min_date = df["date"].min()

    if min_date <= MIN_EXPECTED_DATE:
        add_result(
            "PASS",
            "Start date",
            f"Start date: {min_date.date()}",
        )
    else:
        add_result(
            "FAIL",
            "Start date",
            (
                f"Expected start on or before "
                f"{MIN_EXPECTED_DATE.date()}, "
                f"found {min_date.date()}"
            ),
        )

    max_date = df["date"].max()

    today = pd.Timestamp.today().normalize()

    source_lag_days = (
        today - max_date
    ).days

    if source_lag_days < 0:
        add_result(
            "WARNING",
            "Source freshness",
            (
                f"Latest source date {max_date.date()} "
                f"is ahead of local date {today.date()}"
            ),
        )

    elif (
        source_lag_days
        <= MAX_ACCEPTABLE_SOURCE_LAG_DAYS
    ):
        add_result(
            "PASS",
            "Source freshness",
            (
                f"Latest date: {max_date.date()}, "
                f"lag: {source_lag_days} days"
            ),
        )

    else:
        add_result(
            "FAIL",
            "Source freshness",
            (
                f"Latest date: {max_date.date()}, "
                f"lag: {source_lag_days} days"
            ),
        )


def check_calendar_continuity(df):
    min_date = df["date"].min()
    max_date = df["date"].max()

    expected_dates = pd.date_range(
        start=min_date,
        end=max_date,
        freq="D",
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
            "Daily continuity",
            (
                f"Complete daily coverage: "
                f"{min_date.date()} -> {max_date.date()}"
            ),
        )
    else:
        sample = [
            str(date.date())
            for date in missing_dates[:10]
        ]

        add_result(
            "WARNING",
            "Daily continuity",
            (
                f"Missing calendar days: "
                f"{len(missing_dates)}. "
                f"Sample: {sample}"
            ),
        )


def check_numeric_quality(df):
    numeric_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column != "date"
        and column in df.columns
    ]

    for column in numeric_columns:
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

        negative_count = int(
            (
                df[column] < 0
            ).sum()
        )

        if negative_count == 0:
            add_result(
                "PASS",
                f"Negative values - {column}",
                "No negative values",
            )
        else:
            add_result(
                "FAIL",
                f"Negative values - {column}",
                f"Negative rows: {negative_count}",
            )


def check_zero_ratios(df):
    global_columns = [
        "global_portcalls",
        "global_dry_bulk_portcalls",
        "global_import",
        "global_export",
    ]

    basket_columns = [
        "copper_supply_portcalls",
        "copper_supply_dry_bulk_portcalls",
        "copper_supply_dry_bulk_import",
        "copper_supply_dry_bulk_export",
        "china_copper_related_portcalls",
        "china_copper_related_dry_bulk_portcalls",
        "china_copper_related_dry_bulk_import",
        "china_copper_related_dry_bulk_export",
    ]

    for column in global_columns:
        if column not in df.columns:
            continue

        zero_ratio = (
            df[column]
            .eq(0)
            .mean()
        )

        if zero_ratio <= MAX_ZERO_RATIO_GLOBAL:
            add_result(
                "PASS",
                f"Zero ratio - {column}",
                f"{zero_ratio:.2%}",
            )
        else:
            add_result(
                "WARNING",
                f"Zero ratio - {column}",
                f"{zero_ratio:.2%}",
            )

    for column in basket_columns:
        if column not in df.columns:
            continue

        zero_ratio = (
            df[column]
            .eq(0)
            .mean()
        )

        if zero_ratio <= MAX_ZERO_RATIO_BASKET:
            add_result(
                "PASS",
                f"Zero ratio - {column}",
                f"{zero_ratio:.2%}",
            )
        else:
            add_result(
                "WARNING",
                f"Zero ratio - {column}",
                f"{zero_ratio:.2%}",
            )


def check_logical_relations(df):
    checks = [
        (
            "global_container_portcalls",
            "global_portcalls",
        ),
        (
            "global_dry_bulk_portcalls",
            "global_portcalls",
        ),
        (
            "global_cargo_portcalls",
            "global_portcalls",
        ),
        (
            "copper_supply_dry_bulk_portcalls",
            "copper_supply_portcalls",
        ),
        (
            "china_copper_related_dry_bulk_portcalls",
            "china_copper_related_portcalls",
        ),
    ]

    for child, parent in checks:
        if (
            child not in df.columns
            or parent not in df.columns
        ):
            continue

        invalid_count = int(
            (
                df[child]
                > df[parent]
            ).sum()
        )

        if invalid_count == 0:
            add_result(
                "PASS",
                f"{child} <= {parent}",
                "Logical relation holds",
            )
        else:
            add_result(
                "FAIL",
                f"{child} <= {parent}",
                f"Invalid rows: {invalid_count}",
            )


def check_recent_activity(df):
    latest_date = df["date"].max()

    recent_start = (
        latest_date
        - pd.Timedelta(
            days=29
        )
    )

    recent = df[
        df["date"] >= recent_start
    ]

    important_columns = [
        "global_portcalls",
        "global_dry_bulk_portcalls",
        "copper_supply_portcalls",
        "china_copper_related_portcalls",
    ]

    for column in important_columns:
        if column not in recent.columns:
            continue

        unique_values = recent[
            column
        ].nunique(
            dropna=True
        )

        if unique_values >= 5:
            add_result(
                "PASS",
                f"Recent variation - {column}",
                (
                    f"{unique_values} unique values "
                    f"in recent {len(recent)} days"
                ),
            )
        else:
            add_result(
                "WARNING",
                f"Recent variation - {column}",
                (
                    f"Only {unique_values} unique values "
                    f"in recent {len(recent)} days"
                ),
            )


def print_summary(df):
    print(
        "=" * 110
    )

    print(
        "IMF PORTWATCH DATA QUALITY REPORT"
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

    if not df.empty:
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

    print()

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
    if not check_file_exists():
        dummy = pd.DataFrame()

        return print_summary(
            dummy
        )

    df = pd.read_csv(
        DATA_FILE
    )

    check_columns(
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

    check_date_quality(
        df
    )

    if not df["date"].isna().all():
        check_calendar_continuity(
            df
        )

    check_numeric_quality(
        df
    )

    check_zero_ratios(
        df
    )

    check_logical_relations(
        df
    )

    check_recent_activity(
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
