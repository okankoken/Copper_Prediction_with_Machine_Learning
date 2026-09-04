from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import MARKET_RAW_DIR


INPUT_FILE = (
    MARKET_RAW_DIR
    / "worldbank_commodities_monthly.csv"
)


EXPECTED_COLUMNS = [
    "gold_usd_per_troy_ounce",
    "silver_usd_per_troy_ounce",
    "platinum_usd_per_troy_ounce",
    "iron_ore_usd_per_dmtu",
    "wti_usd_per_barrel",
    "coal_australia_usd_per_ton",
]


MAX_MONTHLY_CHANGE_PCT = {
    "gold_usd_per_troy_ounce": 30.0,
    "silver_usd_per_troy_ounce": 50.0,
    "platinum_usd_per_troy_ounce": 40.0,
    "iron_ore_usd_per_dmtu": 50.0,
    "wti_usd_per_barrel": 80.0,
    "coal_australia_usd_per_ton": 80.0,
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


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(
        f"[PASS] File exists: {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    if "date" not in df.columns:
        raise ValueError(
            "Missing date column"
        )

    df["date"] = pd.PeriodIndex(
        df["date"],
        freq="M",
    )

    return df


def check_structure(
    df,
):
    fail_count = 0

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

        fail_count += 1
    else:
        print(
            "[PASS] All expected columns exist"
        )

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

        fail_count += 1
    else:
        print(
            "[PASS] No duplicate dates"
        )

    if not df["date"].is_monotonic_increasing:
        print(
            "[FAIL] Dates are not sorted"
        )

        fail_count += 1
    else:
        print(
            "[PASS] Dates are sorted"
        )

    expected_months = pd.period_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="M",
    )

    missing_months = expected_months.difference(
        df["date"]
    )

    if len(missing_months) > 0:
        print(
            "[FAIL] Missing dataset months:",
            list(missing_months),
        )

        fail_count += 1
    else:
        print(
            "[PASS] Dataset month sequence is continuous"
        )

    return fail_count


def check_numeric_values(
    df,
):
    fail_count = 0

    numeric_df = df[
        EXPECTED_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
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

        fail_count += 1
    else:
        print(
            "[PASS] No infinite values"
        )

    non_positive_count = int(
        (
            numeric_df <= 0
        ).sum().sum()
    )

    if non_positive_count > 0:
        print(
            "[FAIL] Non-positive commodity prices:",
            non_positive_count,
        )

        fail_count += 1
    else:
        print(
            "[PASS] All commodity prices are positive"
        )

    return fail_count


def print_series_summary(
    df,
):
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

    last_full_month = (
        get_last_full_month()
    )

    rows = []

    warning_count = 0

    for column in EXPECTED_COLUMNS:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = series.notna()

        first_valid = df.loc[
            valid,
            "date",
        ].min()

        last_valid = df.loc[
            valid,
            "date",
        ].max()

        lag_months = month_difference(
            last_valid,
            last_full_month,
        )

        status = "PASS"

        if lag_months > 2:
            status = "WARNING"
            warning_count += 1

        rows.append(
            {
                "series": column,
                "status": status,
                "valid": int(
                    valid.sum()
                ),
                "missing": int(
                    series.isna().sum()
                ),
                "missing_pct": round(
                    series.isna().mean()
                    * 100,
                    2,
                ),
                "first_valid": first_valid,
                "last_valid": last_valid,
                "lag_months": lag_months,
                "min": series.min(),
                "max": series.max(),
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

    return warning_count


def check_monthly_changes(
    df,
):
    print(
        "\n"
        + "=" * 120
    )

    print(
        "MONTHLY PRICE CHANGE SCREEN"
    )

    print(
        "=" * 120
    )

    warning_count = 0

    for column, threshold in (
        MAX_MONTHLY_CHANGE_PCT.items()
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        pct_change = (
            series.pct_change(
                fill_method=None
            )
            * 100
        )

        suspicious = (
            pct_change.abs()
            > threshold
        )

        count = int(
            suspicious.sum()
        )

        if count == 0:
            print(
                f"[PASS] {column}: "
                f"no changes above {threshold:.0f}%"
            )

            continue

        warning_count += count

        print(
            f"\n[WARNING] {column}: "
            f"{count} changes above "
            f"{threshold:.0f}%"
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

    return warning_count


def main():

    print(
        "=" * 120
    )

    print(
        "WORLD BANK COMMODITY DATA QUALITY"
    )

    print(
        "=" * 120
    )

    df = load_data()

    print(
        "[INFO] Dataset range:",
        df["date"].min(),
        "->",
        df["date"].max(),
    )

    fail_count = 0

    fail_count += check_structure(
        df
    )

    fail_count += check_numeric_values(
        df
    )

    warning_count = print_series_summary(
        df
    )

    warning_count += check_monthly_changes(
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
            "[FAIL] World Bank commodity "
            "quality check failed"
        )

        raise SystemExit(1)

    print(
        "[PASS] World Bank commodity "
        "quality check completed"
    )


if __name__ == "__main__":
    main()
