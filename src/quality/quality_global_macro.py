from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import MACRO_RAW_DIR


INPUT_FILE = (
    MACRO_RAW_DIR
    / "global_macro_monthly.csv"
)


EXPECTED_COLUMNS = [
    "euro_area_ppi",
    "euro_area_copper_production_ppi",
    "euro_area_industrial_production",
    "euro_area_construction_output",
    "ecb_policy_rate",
    "boj_call_rate",
    "japan_ppi",
    "japan_cpi",
    "japan_industrial_production",
    "indonesia_inflation_mom",
    "drc_policy_rate",
    "usd_cdf",
    "drc_inflation_mom",
]


NON_NEGATIVE_COLUMNS = [
    "euro_area_ppi",
    "euro_area_copper_production_ppi",
    "euro_area_industrial_production",
    "euro_area_construction_output",
    "japan_ppi",
    "japan_cpi",
    "japan_industrial_production",
    "usd_cdf",
]


MAX_ALLOWED_LAG_MONTHS = {
    "euro_area_ppi": 3,
    "euro_area_copper_production_ppi": 3,
    "euro_area_industrial_production": 3,
    "euro_area_construction_output": 3,
    "ecb_policy_rate": 1,
    "boj_call_rate": 1,
    "japan_ppi": 2,
    "japan_cpi": 2,
    "japan_industrial_production": 3,
    "indonesia_inflation_mom": 1,
    "drc_policy_rate": 2,
    "usd_cdf": 1,
    "drc_inflation_mom": 3,
}


def get_last_full_month():
    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_full_month_end.to_period("M")


def month_difference(
    latest_month,
    reference_month,
):
    return (
        (
            reference_month.year
            - latest_month.year
        )
        * 12
        + (
            reference_month.month
            - latest_month.month
        )
    )


def check_file_exists():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(
        f"[PASS] File exists: {INPUT_FILE}"
    )


def load_data():
    df = pd.read_csv(
        INPUT_FILE
    )

    if "date" not in df.columns:
        raise ValueError(
            "Missing required date column"
        )

    df["date"] = pd.PeriodIndex(
        df["date"],
        freq="M",
    )

    return df


def check_expected_columns(
    df,
):
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


def check_duplicate_dates(
    df,
):
    duplicate_count = int(
        df["date"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        print(
            "[FAIL] Duplicate dates:",
            duplicate_count,
        )

        return 1

    print(
        "[PASS] No duplicate dates"
    )

    return 0


def check_date_order(
    df,
):
    if not df["date"].is_monotonic_increasing:
        print(
            "[FAIL] Dates are not sorted"
        )

        return 1

    print(
        "[PASS] Dates are sorted"
    )

    return 0


def check_month_continuity(
    df,
):
    expected = pd.period_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="M",
    )

    missing_months = (
        expected
        .difference(
            df["date"]
        )
    )

    if len(missing_months) > 0:
        print(
            "[FAIL] Missing dataset months:",
            list(missing_months),
        )

        return 1

    print(
        "[PASS] Dataset month sequence is continuous"
    )

    return 0


def check_date_range(
    df,
):
    last_full_month = (
        get_last_full_month()
    )

    first_month = df[
        "date"
    ].min()

    last_month = df[
        "date"
    ].max()

    print(
        "[INFO] Dataset range:",
        first_month,
        "->",
        last_month,
    )

    if first_month != pd.Period(
        "2005-01",
        freq="M",
    ):
        print(
            "[WARNING] Dataset does not start at 2005-01"
        )

    if last_month != last_full_month:
        print(
            "[WARNING] Dataset does not end at "
            "the last completed month"
        )

        return 1

    print(
        "[PASS] Dataset ends at last completed month"
    )

    return 0


def check_infinite_values(
    df,
):
    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_count = int(
        np.isinf(
            numeric_df
        ).sum().sum()
    )

    if infinite_count > 0:
        print(
            "[FAIL] Infinite values:",
            infinite_count,
        )

        return 1

    print(
        "[PASS] No infinite values"
    )

    return 0


def check_negative_values(
    df,
):
    fail_count = 0

    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue

        negative_count = int(
            (
                df[column]
                < 0
            ).sum()
        )

        if negative_count > 0:
            print(
                f"[FAIL] {column}: "
                f"{negative_count} negative values"
            )

            fail_count += 1

    if fail_count == 0:
        print(
            "[PASS] No unexpected negative values"
        )

    return fail_count


def print_series_quality(
    df,
):
    last_full_month = (
        get_last_full_month()
    )

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

    rows = []

    warnings = 0

    for column in EXPECTED_COLUMNS:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid_mask = (
            series.notna()
        )

        valid_count = int(
            valid_mask.sum()
        )

        missing_count = int(
            series.isna().sum()
        )

        missing_ratio = (
            missing_count
            / len(df)
        )

        if valid_count > 0:

            first_valid = df.loc[
                valid_mask,
                "date",
            ].min()

            last_valid = df.loc[
                valid_mask,
                "date",
            ].max()

            lag_months = month_difference(
                last_valid,
                last_full_month,
            )

        else:

            first_valid = None
            last_valid = None
            lag_months = None

        max_lag = (
            MAX_ALLOWED_LAG_MONTHS
            .get(
                column,
                3,
            )
        )

        if (
            lag_months is not None
            and lag_months > max_lag
        ):
            status = "WARNING"
            warnings += 1

        else:
            status = "PASS"

        rows.append(
            {
                "series": column,
                "status": status,
                "valid": valid_count,
                "missing": missing_count,
                "missing_pct": round(
                    missing_ratio * 100,
                    2,
                ),
                "first_valid": first_valid,
                "last_valid": last_valid,
                "lag_months": lag_months,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return warnings


def check_large_monthly_changes(
    df,
):
    print(
        "\n"
        + "=" * 120
    )

    print(
        "LARGE MONTHLY CHANGE SCREEN"
    )

    print(
        "=" * 120
    )

    warning_count = 0

    # Level and index series:
    # percentage change is meaningful.
    pct_change_columns = {
        "euro_area_ppi": 20.0,
        "euro_area_copper_production_ppi": 30.0,
        "euro_area_industrial_production": 20.0,
        "euro_area_construction_output": 20.0,
        "japan_ppi": 20.0,
        "japan_cpi": 20.0,
        "japan_industrial_production": 20.0,
        "usd_cdf": 30.0,
    }

    # Rate series:
    # use absolute percentage-point changes instead
    # of percentage changes around zero.
    absolute_change_columns = {
        "ecb_policy_rate": 2.0,
        "boj_call_rate": 1.0,
        "indonesia_inflation_mom": 3.0,
        "drc_policy_rate": 20.0,
        "drc_inflation_mom": 5.0,
    }

    for column, threshold in (
        pct_change_columns.items()
    ):

        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        previous = series.shift(1)

        valid_previous = (
            previous.notna()
            & (previous != 0)
        )

        pct_change = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        pct_change.loc[
            valid_previous
        ] = (
            (
                series.loc[
                    valid_previous
                ]
                / previous.loc[
                    valid_previous
                ]
            )
            - 1
        ) * 100

        suspicious = (
            pct_change.abs()
            > threshold
        )

        count = int(
            suspicious.sum()
        )

        if count == 0:
            continue

        warning_count += count

        print(
            f"\n[WARNING] {column}: "
            f"{count} changes above "
            f"{threshold:.1f}%"
        )

        temp = pd.DataFrame(
            {
                "date": df["date"],
                "value": series,
                "change_pct": pct_change,
            }
        )

        print(
            temp.loc[
                suspicious
            ]
            .tail(10)
            .to_string(
                index=False
            )
        )

    for column, threshold in (
        absolute_change_columns.items()
    ):

        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        absolute_change = (
            series
            - series.shift(1)
        )

        suspicious = (
            absolute_change.abs()
            > threshold
        )

        count = int(
            suspicious.sum()
        )

        if count == 0:
            continue

        warning_count += count

        print(
            f"\n[WARNING] {column}: "
            f"{count} absolute changes above "
            f"{threshold:.2f} points"
        )

        temp = pd.DataFrame(
            {
                "date": df["date"],
                "value": series,
                "change_points": absolute_change,
            }
        )

        print(
            temp.loc[
                suspicious
            ]
            .tail(10)
            .to_string(
                index=False
            )
        )

    if warning_count == 0:
        print(
            "[PASS] No suspicious monthly changes"
        )

    return warning_count


def main():

    print(
        "=" * 120
    )

    print(
        "GLOBAL MACRO DATA QUALITY"
    )

    print(
        "=" * 120
    )

    check_file_exists()

    df = load_data()

    fail_count = 0

    fail_count += (
        check_expected_columns(
            df
        )
    )

    fail_count += (
        check_duplicate_dates(
            df
        )
    )

    fail_count += (
        check_date_order(
            df
        )
    )

    fail_count += (
        check_month_continuity(
            df
        )
    )

    fail_count += (
        check_date_range(
            df
        )
    )

    fail_count += (
        check_infinite_values(
            df
        )
    )

    fail_count += (
        check_negative_values(
            df
        )
    )

    warning_count = (
        print_series_quality(
            df
        )
    )

    warning_count += (
        check_large_monthly_changes(
            df
        )
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
            "[FAIL] Global macro quality check failed"
        )

        raise SystemExit(
            1
        )

    print(
        "[PASS] Global macro quality check completed"
    )


if __name__ == "__main__":
    main()
