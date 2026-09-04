from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import MARKET_RAW_DIR


INPUT_FILE = (
    MARKET_RAW_DIR
    / "lme_other_metals_daily.csv"
)


METALS = [
    "aluminum",
    "nickel",
    "zinc",
    "lead",
    "tin",
]

PALLADIUM_FILE = (
    MARKET_RAW_DIR
    / "palladium_daily.csv"
)

def check_palladium():
    print(
        "\n"
        + "=" * 120
    )

    print(
        "PALLADIUM DATA QUALITY"
    )

    print(
        "=" * 120
    )

    fail_count = 0
    warning_count = 0

    if not PALLADIUM_FILE.exists():
        print(
            "[FAIL] Palladium file not found:",
            PALLADIUM_FILE,
        )

        return 1, 0

    df = pd.read_csv(
        PALLADIUM_FILE
    )

    expected_columns = [
        "date",
        "palladium_usd_per_troy_ounce",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            "[FAIL] Missing Palladium columns:",
            missing_columns,
        )

        return 1, 0

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    df[
        "palladium_usd_per_troy_ounce"
    ] = pd.to_numeric(
        df[
            "palladium_usd_per_troy_ounce"
        ],
        errors="coerce",
    )

    invalid_dates = int(
        df["date"].isna().sum()
    )

    if invalid_dates > 0:
        print(
            "[FAIL] Invalid Palladium dates:",
            invalid_dates,
        )

        fail_count += 1
    else:
        print(
            "[PASS] No invalid Palladium dates"
        )

    duplicates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    if duplicates > 0:
        print(
            "[FAIL] Duplicate Palladium dates:",
            duplicates,
        )

        fail_count += 1
    else:
        print(
            "[PASS] No duplicate Palladium dates"
        )

    valid = (
        df[
            "palladium_usd_per_troy_ounce"
        ]
        .notna()
    )

    non_positive = int(
        (
            df[
                "palladium_usd_per_troy_ounce"
            ]
            <= 0
        ).sum()
    )

    if non_positive > 0:
        print(
            "[FAIL] Non-positive Palladium prices:",
            non_positive,
        )

        fail_count += 1
    else:
        print(
            "[PASS] All Palladium prices are positive"
        )

    print(
        "[INFO] Palladium observations:",
        int(valid.sum()),
    )

    print(
        "[INFO] Palladium range:",
        df.loc[
            valid,
            "date",
        ].min().date(),
        "->",
        df.loc[
            valid,
            "date",
        ].max().date(),
    )

    print(
        "[INFO] Palladium min:",
        df.loc[
            valid,
            "palladium_usd_per_troy_ounce",
        ].min(),
    )

    print(
        "[INFO] Palladium max:",
        df.loc[
            valid,
            "palladium_usd_per_troy_ounce",
        ].max(),
    )

    series = df[
        "palladium_usd_per_troy_ounce"
    ]

    gap_days = (
        df["date"]
        .sort_values()
        .diff()
        .dt.days
    )
    
    pct_change = (
        series.pct_change(
            fill_method=None
        )
        * 100
    )
    
    pct_change = pct_change.where(
        gap_days <= 7
    )

    suspicious = (
        pct_change.abs()
        > 20
    )

    count = int(
        suspicious.sum()
    )

    if count > 0:

        warning_count += count

        print(
            f"[WARNING] Palladium: "
            f"{count} daily moves above 20%"
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

    else:

        print(
            "[PASS] No Palladium daily moves above 20%"
        )

    gap_days = (
        df["date"]
        .sort_values()
        .diff()
        .dt.days
    )

    large_gaps = (
        gap_days > 45
    )

    gap_count = int(
        large_gaps.sum()
    )

    if gap_count > 0:

        warning_count += gap_count

        print(
            f"[WARNING] Palladium: "
            f"{gap_count} gaps above 45 days"
        )

        gap_df = pd.DataFrame(
            {
                "date": df["date"],
                "gap_days": gap_days,
            }
        )

        print(
            gap_df.loc[
                large_gaps
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "[PASS] No Palladium gaps above 45 days"
        )

    return (
        fail_count,
        warning_count,
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

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    return df


def check_expected_columns(
    df,
):
    expected = [
        "date",
    ]

    for metal in METALS:
        expected.extend(
            [
                f"{metal}_cash_usd_per_ton",
                f"{metal}_3_month_usd_per_ton",
                f"{metal}_stock_ton",
            ]
        )

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:
        print(
            "[FAIL] Missing columns:",
            missing,
        )

        return 1

    print(
        "[PASS] All expected columns exist"
    )

    return 0


def check_dates(
    df,
):
    fail_count = 0

    invalid_dates = int(
        df["date"].isna().sum()
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

    duplicates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    if duplicates > 0:
        print(
            "[FAIL] Duplicate dates:",
            duplicates,
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

    print(
        "[INFO] Dataset range:",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    return fail_count


def check_numeric_values(
    df,
):
    fail_count = 0

    numeric_columns = [
        column
        for column in df.columns
        if column != "date"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    numeric_df = df[
        numeric_columns
    ]

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

    negative_count = int(
        (
            numeric_df < 0
        ).sum().sum()
    )

    if negative_count > 0:
        print(
            "[FAIL] Negative values:",
            negative_count,
        )

        fail_count += 1
    else:
        print(
            "[PASS] No negative values"
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

    rows = []

    for metal in METALS:

        for field in [
            "cash_usd_per_ton",
            "3_month_usd_per_ton",
            "stock_ton",
        ]:

            column = (
                f"{metal}_{field}"
            )

            series = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            valid = series.notna()

            if valid.any():

                first_valid = df.loc[
                    valid,
                    "date",
                ].min()

                last_valid = df.loc[
                    valid,
                    "date",
                ].max()

            else:

                first_valid = pd.NaT
                last_valid = pd.NaT

            rows.append(
                {
                    "series": column,
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


def check_cash_3m_relationship(
    df,
):
    print(
        "\n"
        + "=" * 120
    )

    print(
        "CASH VS 3-MONTH SCREEN"
    )

    print(
        "=" * 120
    )

    warning_count = 0

    for metal in METALS:

        cash_column = (
            f"{metal}_cash_usd_per_ton"
        )

        month3_column = (
            f"{metal}_3_month_usd_per_ton"
        )

        cash = pd.to_numeric(
            df[cash_column],
            errors="coerce",
        )

        month3 = pd.to_numeric(
            df[month3_column],
            errors="coerce",
        )

        valid = (
            cash.notna()
            & month3.notna()
            & (cash > 0)
        )

        spread_pct = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        spread_pct.loc[
            valid
        ] = (
            (
                month3.loc[valid]
                - cash.loc[valid]
            )
            / cash.loc[valid]
        ) * 100

        suspicious = (
            spread_pct.abs()
            > 20
        )

        count = int(
            suspicious.sum()
        )

        if count == 0:
            print(
                f"[PASS] {metal}: "
                "no cash/3M spread above 20%"
            )

            continue

        warning_count += count

        print(
            f"[WARNING] {metal}: "
            f"{count} cash/3M spreads above 20%"
        )

        temp = pd.DataFrame(
            {
                "date": df["date"],
                "cash": cash,
                "three_month": month3,
                "spread_pct": spread_pct,
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


def check_daily_price_jumps(
    df,
):
    print(
        "\n"
        + "=" * 120
    )

    print(
        "DAILY PRICE JUMP SCREEN"
    )

    print(
        "=" * 120
    )

    warning_count = 0

    for metal in METALS:

        column = (
            f"{metal}_cash_usd_per_ton"
        )

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
            > 15
        )

        count = int(
            suspicious.sum()
        )

        if count == 0:
            print(
                f"[PASS] {metal}: "
                "no daily moves above 15%"
            )

            continue

        warning_count += count

        print(
            f"\n[WARNING] {metal}: "
            f"{count} daily moves above 15%"
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


def check_current_partial_month(
    df,
):
    today = pd.Timestamp.today().normalize()

    current_month_start = (
        today.replace(
            day=1
        )
    )

    current_month_rows = df[
        df["date"]
        >= current_month_start
    ]

    if current_month_rows.empty:
        print(
            "[INFO] No current partial month "
            "observations in raw data"
        )

        return

    print(
        "[INFO] Current partial month rows:",
        len(current_month_rows),
    )

    print(
        "[INFO] Current partial month will be "
        "excluded during monthly feature creation"
    )


def main():

    print(
        "=" * 120
    )

    print(
        "METALS DATA QUALITY"
    )

    print(
        "=" * 120
    )

    df = load_data()

    fail_count = 0

    fail_count += (
        check_expected_columns(
            df
        )
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

    warning_count = (
        check_cash_3m_relationship(
            df
        )
    )

    warning_count += (
        check_daily_price_jumps(
            df
        )
    )

    check_current_partial_month(
        df
    )

    # Check Borsa Istanbul Palladium data
    (
        palladium_fail_count,
        palladium_warning_count,
    ) = check_palladium()

    fail_count += (
        palladium_fail_count
    )

    warning_count += (
        palladium_warning_count
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
            "[FAIL] Metals quality check failed"
        )

        raise SystemExit(
            1
        )

    print(
        "[PASS] Metals quality check completed"
    )


if __name__ == "__main__":
    main()