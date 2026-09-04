from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    DIAGNOSTICS_DIR,
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

TARGET_COLUMN = "cash_settlement_usd_per_ton"

MIN_CORRELATION_OBSERVATIONS = 24

LAGS = [
    0,
    1,
    2,
    3,
    6,
    12,
]


OVERVIEW_FILE = (
    DIAGNOSTICS_DIR
    / "eda_master_overview.csv"
)

FEATURE_PROFILE_FILE = (
    DIAGNOSTICS_DIR
    / "eda_feature_profile.csv"
)

TARGET_CORRELATION_FILE = (
    DIAGNOSTICS_DIR
    / "eda_target_correlations.csv"
)

TARGET_LAG_CORRELATION_FILE = (
    DIAGNOSTICS_DIR
    / "eda_target_lag_correlations.csv"
)

TARGET_AUTOCORRELATION_FILE = (
    DIAGNOSTICS_DIR
    / "eda_target_autocorrelations.csv"
)

HIGH_FEATURE_CORRELATION_FILE = (
    DIAGNOSTICS_DIR
    / "eda_high_feature_correlations.csv"
)

TARGET_YEARLY_FILE = (
    DIAGNOSTICS_DIR
    / "eda_target_yearly_statistics.csv"
)

MISSING_BY_YEAR_FILE = (
    DIAGNOSTICS_DIR
    / "eda_missing_by_year.csv"
)


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


def convert_boolean_like_columns(df):
    result = df.copy()

    mapping = {
        True: 1.0,
        False: 0.0,
        "True": 1.0,
        "False": 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    }

    for column in result.columns:
        if column == "date":
            continue

        series = result[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):
            continue

        non_null_values = set(
            series
            .dropna()
            .astype(str)
            .unique()
        )

        allowed = {
            "True",
            "False",
            "true",
            "false",
            "1",
            "0",
        }

        if non_null_values.issubset(
            allowed
        ):
            result[column] = (
                series.map(
                    mapping
                )
            )

    return result


def build_plan_lookup(plan):
    return (
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


def get_rule_metadata(
    column,
    plan_lookup,
):
    if column not in plan_lookup.index:
        return {
            "category": None,
            "frequency": None,
            "aggregation": None,
            "source_file": None,
        }

    rule = plan_lookup.loc[
        column
    ]

    return {
        "category": rule["category"],
        "frequency": rule["frequency"],
        "aggregation": rule["aggregation"],
        "source_file": rule["source_file"],
    }


def build_overview(
    df,
    numeric_columns,
):
    rows = [
        {
            "metric": "row_count",
            "value": len(df),
        },
        {
            "metric": "column_count",
            "value": len(df.columns),
        },
        {
            "metric": "feature_count",
            "value": len(df.columns) - 1,
        },
        {
            "metric": "numeric_feature_count",
            "value": len(numeric_columns),
        },
        {
            "metric": "first_month",
            "value": df["date"].min(),
        },
        {
            "metric": "last_month",
            "value": df["date"].max(),
        },
        {
            "metric": "target_mean",
            "value": df[TARGET_COLUMN].mean(),
        },
        {
            "metric": "target_std",
            "value": df[TARGET_COLUMN].std(),
        },
        {
            "metric": "target_min",
            "value": df[TARGET_COLUMN].min(),
        },
        {
            "metric": "target_max",
            "value": df[TARGET_COLUMN].max(),
        },
    ]

    return pd.DataFrame(
        rows
    )


def build_feature_profile(
    df,
    plan_lookup,
):
    rows = []

    for column in df.columns:
        if column == "date":
            continue

        series = df[column]

        metadata = get_rule_metadata(
            column,
            plan_lookup,
        )

        valid = (
            series.notna()
        )

        unique_count = int(
            series
            .dropna()
            .nunique()
        )

        row = {
            "column": column,
            "category": metadata["category"],
            "frequency": metadata["frequency"],
            "aggregation": metadata["aggregation"],
            "source_file": metadata["source_file"],
            "dtype": str(series.dtype),
            "non_null_count": int(
                valid.sum()
            ),
            "missing_count": int(
                series.isna().sum()
            ),
            "missing_ratio": float(
                series.isna().mean()
            ),
            "unique_count": unique_count,
            "first_valid_month": (
                df.loc[
                    valid,
                    "date",
                ].min()
                if valid.any()
                else pd.NaT
            ),
            "last_valid_month": (
                df.loc[
                    valid,
                    "date",
                ].max()
                if valid.any()
                else pd.NaT
            ),
        }

        if pd.api.types.is_numeric_dtype(
            series
        ):
            row.update(
                {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "median": series.median(),
                    "max": series.max(),
                }
            )
        else:
            row.update(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                }
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def safe_correlation(
    target,
    feature,
    method,
):
    pair = pd.concat(
        [
            target.rename("target"),
            feature.rename("feature"),
        ],
        axis=1,
    ).dropna()

    observation_count = len(
        pair
    )

    if (
        observation_count
        < MIN_CORRELATION_OBSERVATIONS
    ):
        return np.nan, observation_count

    if (
        pair["target"].nunique() <= 1
        or pair["feature"].nunique() <= 1
    ):
        return np.nan, observation_count

    value = pair[
        "target"
    ].corr(
        pair["feature"],
        method=method,
    )

    return value, observation_count


def build_target_correlations(
    df,
    numeric_features,
    plan_lookup,
):
    rows = []

    target = df[
        TARGET_COLUMN
    ]

    for column in numeric_features:
        if column == TARGET_COLUMN:
            continue

        metadata = get_rule_metadata(
            column,
            plan_lookup,
        )

        pearson, count = (
            safe_correlation(
                target,
                df[column],
                "pearson",
            )
        )

        spearman, _ = (
            safe_correlation(
                target,
                df[column],
                "spearman",
            )
        )

        rows.append(
            {
                "feature": column,
                "category": metadata["category"],
                "frequency": metadata["frequency"],
                "aggregation": metadata["aggregation"],
                "paired_observations": count,
                "pearson_correlation": pearson,
                "abs_pearson_correlation": (
                    abs(pearson)
                    if pd.notna(pearson)
                    else np.nan
                ),
                "spearman_correlation": spearman,
                "abs_spearman_correlation": (
                    abs(spearman)
                    if pd.notna(spearman)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        "abs_pearson_correlation",
        ascending=False,
        na_position="last",
    )


def build_lag_correlations(
    df,
    numeric_features,
    plan_lookup,
):
    rows = []

    target = df[
        TARGET_COLUMN
    ]

    for column in numeric_features:
        if column == TARGET_COLUMN:
            continue

        metadata = get_rule_metadata(
            column,
            plan_lookup,
        )

        for lag in LAGS:
            lagged_feature = (
                df[column]
                .shift(lag)
            )

            pearson, count = (
                safe_correlation(
                    target,
                    lagged_feature,
                    "pearson",
                )
            )

            spearman, _ = (
                safe_correlation(
                    target,
                    lagged_feature,
                    "spearman",
                )
            )

            rows.append(
                {
                    "feature": column,
                    "category": metadata["category"],
                    "frequency": metadata["frequency"],
                    "aggregation": metadata["aggregation"],
                    "feature_lag_months": lag,
                    "paired_observations": count,
                    "pearson_correlation": pearson,
                    "abs_pearson_correlation": (
                        abs(pearson)
                        if pd.notna(pearson)
                        else np.nan
                    ),
                    "spearman_correlation": spearman,
                    "abs_spearman_correlation": (
                        abs(spearman)
                        if pd.notna(spearman)
                        else np.nan
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "abs_pearson_correlation",
            "paired_observations",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )


def build_target_autocorrelations(
    df,
):
    rows = []

    target = df[
        TARGET_COLUMN
    ]

    for lag in [
        1,
        2,
        3,
        6,
        12,
        18,
        24,
    ]:
        correlation, count = (
            safe_correlation(
                target,
                target.shift(lag),
                "pearson",
            )
        )

        rows.append(
            {
                "target_lag_months": lag,
                "paired_observations": count,
                "autocorrelation": correlation,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_high_feature_correlations(
    df,
    numeric_features,
):
    usable = [
        column
        for column in numeric_features
        if column != TARGET_COLUMN
    ]

    corr = (
        df[
            usable
        ]
        .corr(
            method="pearson",
            min_periods=MIN_CORRELATION_OBSERVATIONS,
        )
    )

    rows = []

    for i, left in enumerate(
        usable
    ):
        for right in usable[
            i + 1:
        ]:
            value = corr.loc[
                left,
                right,
            ]

            if pd.isna(
                value
            ):
                continue

            if abs(
                value
            ) < 0.90:
                continue

            paired = (
                df[
                    [
                        left,
                        right,
                    ]
                ]
                .dropna()
            )

            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "paired_observations": len(
                        paired
                    ),
                    "pearson_correlation": value,
                    "abs_pearson_correlation": abs(
                        value
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    return result.sort_values(
        "abs_pearson_correlation",
        ascending=False,
    )


def build_target_yearly_statistics(
    df,
):
    working = df[
        [
            "date",
            TARGET_COLUMN,
        ]
    ].copy()

    working["year"] = (
        working["date"]
        .dt.year
    )

    working[
        "target_monthly_change_pct"
    ] = (
        working[
            TARGET_COLUMN
        ]
        .pct_change(
            fill_method=None
        )
        * 100.0
    )

    yearly = (
        working
        .groupby(
            "year"
        )
        .agg(
            months=(
                TARGET_COLUMN,
                "count",
            ),
            average_price=(
                TARGET_COLUMN,
                "mean",
            ),
            minimum_price=(
                TARGET_COLUMN,
                "min",
            ),
            maximum_price=(
                TARGET_COLUMN,
                "max",
            ),
            price_std=(
                TARGET_COLUMN,
                "std",
            ),
            average_monthly_change_pct=(
                "target_monthly_change_pct",
                "mean",
            ),
            monthly_change_std_pct=(
                "target_monthly_change_pct",
                "std",
            ),
        )
        .reset_index()
    )

    return yearly


def build_missing_by_year(
    df,
):
    working = df.copy()

    working["year"] = (
        working["date"]
        .dt.year
    )

    rows = []

    for year, group in working.groupby(
        "year"
    ):
        feature_group = group.drop(
            columns=[
                "date",
                "year",
            ]
        )

        rows.append(
            {
                "year": year,
                "months": len(group),
                "total_cells": (
                    feature_group.shape[0]
                    * feature_group.shape[1]
                ),
                "missing_cells": int(
                    feature_group
                    .isna()
                    .sum()
                    .sum()
                ),
                "missing_ratio": float(
                    feature_group
                    .isna()
                    .mean()
                    .mean()
                ),
                "fully_available_features": int(
                    (
                        feature_group
                        .notna()
                        .all(
                            axis=0
                        )
                    )
                    .sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "=" * 80
    )

    print(
        "COPPER MASTER EDA"
    )

    print(
        "=" * 80
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, plan = load_data()

    df = convert_boolean_like_columns(
        df
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target not found: {TARGET_COLUMN}"
        )

    plan_lookup = (
        build_plan_lookup(
            plan
        )
    )

    numeric_columns = (
        df.drop(
            columns=[
                "date"
            ]
        )
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
        .tolist()
    )

    overview = build_overview(
        df,
        numeric_columns,
    )

    feature_profile = (
        build_feature_profile(
            df,
            plan_lookup,
        )
    )

    target_correlations = (
        build_target_correlations(
            df,
            numeric_columns,
            plan_lookup,
        )
    )

    lag_correlations = (
        build_lag_correlations(
            df,
            numeric_columns,
            plan_lookup,
        )
    )

    target_autocorrelations = (
        build_target_autocorrelations(
            df
        )
    )

    high_feature_correlations = (
        build_high_feature_correlations(
            df,
            numeric_columns,
        )
    )

    target_yearly = (
        build_target_yearly_statistics(
            df
        )
    )

    missing_by_year = (
        build_missing_by_year(
            df
        )
    )

    overview.to_csv(
        OVERVIEW_FILE,
        index=False,
    )

    feature_profile.to_csv(
        FEATURE_PROFILE_FILE,
        index=False,
    )

    target_correlations.to_csv(
        TARGET_CORRELATION_FILE,
        index=False,
    )

    lag_correlations.to_csv(
        TARGET_LAG_CORRELATION_FILE,
        index=False,
    )

    target_autocorrelations.to_csv(
        TARGET_AUTOCORRELATION_FILE,
        index=False,
    )

    high_feature_correlations.to_csv(
        HIGH_FEATURE_CORRELATION_FILE,
        index=False,
    )

    target_yearly.to_csv(
        TARGET_YEARLY_FILE,
        index=False,
    )

    missing_by_year.to_csv(
        MISSING_BY_YEAR_FILE,
        index=False,
    )

    print()
    print(
        "[INFO] Shape:",
        df.shape,
    )

    print(
        "[INFO] Numeric columns:",
        len(
            numeric_columns
        ),
    )

    print()
    print(
        "TARGET AUTOCORRELATIONS"
    )

    print(
        target_autocorrelations
        .to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 20 CONTEMPORANEOUS TARGET CORRELATIONS"
    )

    print(
        target_correlations[
            [
                "feature",
                "category",
                "frequency",
                "paired_observations",
                "pearson_correlation",
                "spearman_correlation",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 30 LAGGED TARGET CORRELATIONS"
    )

    print(
        lag_correlations[
            [
                "feature",
                "category",
                "frequency",
                "feature_lag_months",
                "paired_observations",
                "pearson_correlation",
                "spearman_correlation",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "MISSING DATA BY YEAR"
    )

    print(
        missing_by_year
        .to_string(
            index=False
        )
    )

    print()
    print(
        "[OK] EDA files saved under:",
        DIAGNOSTICS_DIR,
    )


if __name__ == "__main__":
    main()
