from pathlib import Path

import pandas as pd
from src.utils.paths import MARKET_RAW_DIR



INPUT_PATH = (
    MARKET_RAW_DIR
    / "shfe_copper_daily.csv"
)


PRICE_COLUMNS = [
    "open_cny_per_ton",
    "high_cny_per_ton",
    "low_cny_per_ton",
    "close_cny_per_ton",
    "settlement_cny_per_ton",
    "pre_settlement_cny_per_ton",
]

REQUIRED_COLUMNS = [
    "date",
    "delivery_month",
    "open_cny_per_ton",
    "high_cny_per_ton",
    "low_cny_per_ton",
    "close_cny_per_ton",
    "settlement_cny_per_ton",
    "pre_settlement_cny_per_ton",
    "volume_lots",
    "open_interest_lots",
    "open_interest_change",
    "turnover",
]


def add_result(
    results,
    level,
    check,
    detail,
):
    results.append(
        {
            "level": level,
            "check": check,
            "detail": detail,
        }
    )


def main():
    print(
        "=" * 80
    )

    print(
        "SHFE COPPER DAILY DATA QUALITY"
    )

    print(
        "=" * 80
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    results = []

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        add_result(
            results,
            "FAIL",
            "required_columns",
            (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "required_columns",
            "All required columns are present",
        )

    if missing_columns:
        results_df = pd.DataFrame(
            results
        )

        print(
            results_df.to_string(
                index=False
            )
        )

        raise SystemExit(
            1
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_date_count = int(
        df["date"]
        .isna()
        .sum()
    )

    if invalid_date_count > 0:
        add_result(
            results,
            "FAIL",
            "invalid_date",
            (
                f"{invalid_date_count} invalid "
                "date values"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "invalid_date",
            "All dates are valid",
        )

    duplicate_count = int(
        df.duplicated(
            subset=["date"]
        ).sum()
    )

    if duplicate_count > 0:
        add_result(
            results,
            "FAIL",
            "duplicate_date",
            (
                f"{duplicate_count} duplicate "
                "date rows"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "duplicate_date",
            "No duplicate dates",
        )

    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    numeric_columns = [
        "volume_lots",
        "open_interest_lots",
        "open_interest_change",
        "turnover",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_price_count = int(
        df[
            PRICE_COLUMNS
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing_price_count > 0:
        add_result(
            results,
            "WARNING",
            "missing_price_values",
            (
                f"{missing_price_count} rows have "
                "at least one missing price"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "missing_price_values",
            "No missing price values",
        )

    non_positive_price_count = int(
        (
            df[
                PRICE_COLUMNS
            ]
            <= 0
        )
        .any(axis=1)
        .sum()
    )

    if non_positive_price_count > 0:
        add_result(
            results,
            "FAIL",
            "non_positive_prices",
            (
                f"{non_positive_price_count} rows "
                "have non-positive prices"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "non_positive_prices",
            "All prices are positive",
        )

    invalid_high_count = int(
        (
            df[
                "high_cny_per_ton"
            ]
            < df[
                [
                    "open_cny_per_ton",
                    "close_cny_per_ton",
                    "low_cny_per_ton",
                ]
            ].max(axis=1)
        ).sum()
    )

    if invalid_high_count > 0:
        add_result(
            results,
            "FAIL",
            "high_price_logic",
            (
                f"{invalid_high_count} rows have "
                "invalid high price logic"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "high_price_logic",
            "High price logic is valid",
        )

    invalid_low_count = int(
        (
            df[
                "low_cny_per_ton"
            ]
            > df[
                [
                    "open_cny_per_ton",
                    "close_cny_per_ton",
                    "high_cny_per_ton",
                ]
            ].min(axis=1)
        ).sum()
    )

    if invalid_low_count > 0:
        add_result(
            results,
            "FAIL",
            "low_price_logic",
            (
                f"{invalid_low_count} rows have "
                "invalid low price logic"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "low_price_logic",
            "Low price logic is valid",
        )

    negative_volume_count = int(
        (
            df[
                "volume_lots"
            ]
            < 0
        ).sum()
    )

    if negative_volume_count > 0:
        add_result(
            results,
            "FAIL",
            "negative_volume",
            (
                f"{negative_volume_count} rows "
                "have negative volume"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "negative_volume",
            "No negative volume values",
        )

    negative_open_interest_count = int(
        (
            df[
                "open_interest_lots"
            ]
            < 0
        ).sum()
    )

    if negative_open_interest_count > 0:
        add_result(
            results,
            "FAIL",
            "negative_open_interest",
            (
                f"{negative_open_interest_count} rows "
                "have negative open interest"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "negative_open_interest",
            "No negative open interest values",
        )

    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "settlement_return"
    ] = (
        df[
            "settlement_cny_per_ton"
        ]
        .pct_change(
            fill_method=None
        )
    )

    extreme_return = df[
        df[
            "settlement_return"
        ].abs() > 0.15
    ]

    if not extreme_return.empty:
        add_result(
            results,
            "WARNING",
            "extreme_daily_return",
            (
                f"{len(extreme_return)} daily "
                "settlement changes exceed 15%"
            ),
        )
    else:
        add_result(
            results,
            "PASS",
            "extreme_daily_return",
            (
                "No daily settlement changes "
                "exceed 15%"
            ),
        )

    valid_dates = (
        df[
            "date"
        ]
        .dropna()
    )

    if not valid_dates.empty:
        first_date = (
            valid_dates.min()
        )

        last_date = (
            valid_dates.max()
        )

        add_result(
            results,
            "PASS",
            "coverage",
            (
                f"{first_date.date()} -> "
                f"{last_date.date()}, "
                f"{len(df)} observations"
            ),
        )

        expected_business_days = pd.date_range(
            first_date,
            last_date,
            freq="B",
        )

        actual_dates = set(
            valid_dates.dt.normalize()
        )

        missing_business_days = [
            date
            for date in expected_business_days
            if date not in actual_dates
        ]

        if missing_business_days:
            add_result(
                results,
                "WARNING",
                "missing_business_days",
                (
                    f"{len(missing_business_days)} "
                    "weekday dates are absent; "
                    "includes exchange holidays"
                ),
            )
        else:
            add_result(
                results,
                "PASS",
                "missing_business_days",
                "No weekday dates are absent",
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

    if not extreme_return.empty:
        print(
            "\n[INFO] Extreme daily settlement changes:"
        )

        print(
            extreme_return[
                [
                    "date",
                    "settlement_cny_per_ton",
                    "settlement_return",
                ]
            ].to_string(
                index=False
            )
        )

    if (
        "missing_business_days"
        in locals()
        and missing_business_days
    ):
        print(
            "\n[INFO] First 30 missing weekday dates:"
        )

        print(
            [
                str(date.date())
                for date
                in missing_business_days[:30]
            ]
        )

    if fail_count > 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()