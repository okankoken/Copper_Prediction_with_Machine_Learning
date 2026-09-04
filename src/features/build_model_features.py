import numpy as np
import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    FEATURES_DIR,
    MONTHLY_DIR,
)


MASTER_FILE = (
    MONTHLY_DIR
    / "copper_monthly_master.csv"
)

PLAN_FILE = (
    CONFIG_DIR
    / "master_feature_plan.csv"
)

OUTPUT_FILE = (
    FEATURES_DIR
    / "copper_model_features.csv"
)

MANIFEST_FILE = (
    FEATURES_DIR
    / "copper_model_feature_manifest.csv"
)

TARGET_COLUMN = "cash_settlement_usd_per_ton"

TARGET_LAGS = [
    1,
    2,
    3,
    6,
    12,
]

EXOG_LAGS = [
    1,
    3,
    6,
    12,
]

ROLLING_WINDOWS = [
    3,
    6,
    12,
]

EPSILON = 1e-12


def load_data():
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

    plan = plan[
        plan["include"] == True
    ].copy()

    return df, plan


def safe_pct_change(
    series,
    periods,
):
    current = series

    previous = series.shift(
        periods
    )

    result = (
        current
        - previous
    ) / previous.abs()

    result = result.where(
        previous.abs() > EPSILON
    )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return result


def add_manifest_row(
    rows,
    feature,
    raw_feature,
    category,
    source_frequency,
    transform,
    lag_months,
    leakage_safe,
):
    rows.append(
        {
            "feature": feature,
            "raw_feature": raw_feature,
            "category": category,
            "source_frequency": source_frequency,
            "transform": transform,
            "lag_months": lag_months,
            "leakage_safe": leakage_safe,
        }
    )


def build_target_features(
    df,
    result,
    manifest_rows,
):
    target = df[
        TARGET_COLUMN
    ]

    for lag in TARGET_LAGS:
        column = (
            f"target_lag_{lag}"
        )

        result[column] = (
            target.shift(
                lag
            )
        )

        add_manifest_row(
            manifest_rows,
            column,
            TARGET_COLUMN,
            "target_history",
            "monthly",
            "lag",
            lag,
            True,
        )

    shifted_target = (
        target.shift(
            1
        )
    )

    for periods in [
        1,
        3,
        12,
    ]:
        column = (
            f"target_return_{periods}m"
        )

        result[column] = (
            safe_pct_change(
                shifted_target,
                periods,
            )
        )

        add_manifest_row(
            manifest_rows,
            column,
            TARGET_COLUMN,
            "target_history",
            "monthly",
            "pct_change",
            periods + 1,
            True,
        )

    for window in ROLLING_WINDOWS:
        mean_column = (
            f"target_rolling_mean_{window}"
        )

        std_column = (
            f"target_rolling_std_{window}"
        )

        result[
            mean_column
        ] = (
            shifted_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        result[
            std_column
        ] = (
            shifted_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
        )

        add_manifest_row(
            manifest_rows,
            mean_column,
            TARGET_COLUMN,
            "target_history",
            "monthly",
            "rolling_mean",
            1,
            True,
        )

        add_manifest_row(
            manifest_rows,
            std_column,
            TARGET_COLUMN,
            "target_history",
            "monthly",
            "rolling_std",
            1,
            True,
        )


def build_exogenous_features(
    df,
    plan,
    result,
    manifest_rows,
):
    plan_lookup = (
        plan
        .drop_duplicates(
            subset=[
                "output_column"
            ]
        )
        .set_index(
            "output_column"
        )
    )

    annual_skipped = []

    processed = 0

    for column in df.columns:
        if column in {
            "date",
            TARGET_COLUMN,
        }:
            continue

        if column not in plan_lookup.index:
            continue

        rule = plan_lookup.loc[
            column
        ]

        frequency = str(
            rule[
                "frequency"
            ]
        )

        category = str(
            rule[
                "category"
            ]
        )

        if frequency == "annual":
            annual_skipped.append(
                column
            )
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        shifted = (
            series.shift(
                1
            )
        )

        for lag in EXOG_LAGS:
            feature_name = (
                f"{column}_lag_{lag}"
            )

            result[
                feature_name
            ] = (
                series.shift(
                    lag
                )
            )

            add_manifest_row(
                manifest_rows,
                feature_name,
                column,
                category,
                frequency,
                "lag",
                lag,
                True,
            )

        change_column = (
            f"{column}_change_1m"
        )

        result[
            change_column
        ] = (
            shifted
            - series.shift(
                2
            )
        )

        add_manifest_row(
            manifest_rows,
            change_column,
            column,
            category,
            frequency,
            "absolute_change",
            1,
            True,
        )

        pct_1m_column = (
            f"{column}_pct_change_1m"
        )

        result[
            pct_1m_column
        ] = (
            safe_pct_change(
                shifted,
                1,
            )
        )

        add_manifest_row(
            manifest_rows,
            pct_1m_column,
            column,
            category,
            frequency,
            "pct_change",
            1,
            True,
        )

        pct_3m_column = (
            f"{column}_pct_change_3m"
        )

        result[
            pct_3m_column
        ] = (
            safe_pct_change(
                shifted,
                3,
            )
        )

        add_manifest_row(
            manifest_rows,
            pct_3m_column,
            column,
            category,
            frequency,
            "pct_change",
            3,
            True,
        )

        pct_12m_column = (
            f"{column}_pct_change_12m"
        )

        result[
            pct_12m_column
        ] = (
            safe_pct_change(
                shifted,
                12,
            )
        )

        add_manifest_row(
            manifest_rows,
            pct_12m_column,
            column,
            category,
            frequency,
            "pct_change",
            12,
            True,
        )

        processed += 1

    return processed, annual_skipped


def main():
    print(
        "=" * 80
    )
    print(
        "BUILD MODEL FEATURES"
    )
    print(
        "=" * 80
    )

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, plan = load_data()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target not found: {TARGET_COLUMN}"
        )

    result = pd.DataFrame(
        {
            "date": df["date"],
            TARGET_COLUMN: df[
                TARGET_COLUMN
            ],
        }
    )

    manifest_rows = []

    add_manifest_row(
        manifest_rows,
        TARGET_COLUMN,
        TARGET_COLUMN,
        "target",
        "monthly",
        "target",
        0,
        False,
    )

    build_target_features(
        df,
        result,
        manifest_rows,
    )

    (
        processed_exogenous,
        annual_skipped,
    ) = build_exogenous_features(
        df,
        plan,
        result,
        manifest_rows,
    )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    manifest = pd.DataFrame(
        manifest_rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    print()
    print(
        "[INFO] Source rows:",
        len(df),
    )

    print(
        "[INFO] Source master features:",
        len(df.columns) - 1,
    )

    print(
        "[INFO] Exogenous raw features processed:",
        processed_exogenous,
    )

    print(
        "[INFO] Annual raw features skipped:",
        len(annual_skipped),
    )

    print(
        "[INFO] Model feature columns:",
        len(result.columns) - 2,
    )

    print(
        "[INFO] Output shape:",
        result.shape,
    )

    print(
        "[INFO] First month:",
        result["date"]
        .min()
        .strftime("%Y-%m-%d"),
    )

    print(
        "[INFO] Last month:",
        result["date"]
        .max()
        .strftime("%Y-%m-%d"),
    )

    print()
    print(
        "[INFO] Target missing:",
        int(
            result[
                TARGET_COLUMN
            ]
            .isna()
            .sum()
        ),
    )

    print(
        "[INFO] Infinite values:",
        int(
            np.isinf(
                result
                .drop(
                    columns=[
                        "date"
                    ]
                )
                .select_dtypes(
                    include=[
                        np.number
                    ]
                )
            )
            .sum()
            .sum()
        ),
    )

    print()
    print(
        "[INFO] Annual features are intentionally not active yet."
    )

    print(
        "[INFO] Publication-safe annual logic will be added before modeling."
    )

    print()
    print(
        "[OK] Model features:",
        OUTPUT_FILE,
    )

    print(
        "[OK] Feature manifest:",
        MANIFEST_FILE,
    )


if __name__ == "__main__":
    main()
