from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    DIAGNOSTICS_DIR,
    MONTHLY_DIR,
)


PLAN_FILE = (
    CONFIG_DIR
    / "master_feature_plan.csv"
)

OUTPUT_FILE = (
    MONTHLY_DIR
    / "copper_monthly_master.csv"
)

COVERAGE_FILE = (
    DIAGNOSTICS_DIR
    / "copper_monthly_master_coverage.csv"
)


START_MONTH = "2015-01"

TARGET_COLUMN = (
    "cash_settlement_usd_per_ton"
)


CATEGORY_ORDER = [
    "copper_market",
    "other_metals",
    "commodities",
    "china",
    "macro_fred",
    "global_macro",
    "leading_indicators",
    "turkey",
    "risk",
    "shipping",
    "equities_indices",
    "equities_companies",
    "usgs",
    "icsg",
    "chile_cochilco",
    "peru_mining",
    "peru_cost_drivers",
    "energy_transition",
]


# ------------------------------------------------------------------
# CALENDAR
# ------------------------------------------------------------------

def get_last_full_month():
    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_full_month_end.to_period(
        "M"
    )


def build_master_calendar():
    start_period = pd.Period(
        START_MONTH,
        freq="M",
    )

    end_period = get_last_full_month()

    periods = pd.period_range(
        start=start_period,
        end=end_period,
        freq="M",
    )

    calendar = pd.DataFrame(
        {
            "month_key": periods,
        }
    )

    calendar["date"] = (
        calendar["month_key"]
        .dt.to_timestamp(
            "M"
        )
    )

    return calendar


# ------------------------------------------------------------------
# GENERIC HELPERS
# ------------------------------------------------------------------

def load_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    return pd.read_csv(
        path
    )


def detect_month_key(df):
    if "date" in df.columns:
        values = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        return values.dt.to_period(
            "M"
        )

    if "month" in df.columns:
        values = pd.to_datetime(
            df["month"].astype(str),
            errors="coerce",
        )

        return values.dt.to_period(
            "M"
        )

    raise ValueError(
        "No date or month column found"
    )


def validate_source_columns(
    df,
    rules,
    source_file,
):
    required = set(
        rules["source_column"]
        .tolist()
    )

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing source columns in "
            f"{source_file}: {missing}"
        )


def numeric_series(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ------------------------------------------------------------------
# DAILY -> MONTHLY
# ------------------------------------------------------------------

def aggregate_daily(
    source_file,
    rules,
):
    df = load_csv(
        source_file
    )

    validate_source_columns(
        df,
        rules,
        source_file,
    )

    if "date" not in df.columns:
        raise ValueError(
            f"Daily source has no date column: {source_file}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    df = df.sort_values(
        "date"
    )

    df["month_key"] = (
        df["date"]
        .dt.to_period("M")
    )

    result = pd.DataFrame(
        {
            "month_key": sorted(
                df["month_key"]
                .dropna()
                .unique()
            )
        }
    )

    result = result.set_index(
        "month_key"
    )

    grouped = df.groupby(
        "month_key",
        sort=True,
    )

    for _, rule in rules.iterrows():
        source_column = (
            rule["source_column"]
        )

        output_column = (
            rule["output_column"]
        )

        aggregation = (
            rule["aggregation"]
        )

        working = df[
            [
                "date",
                "month_key",
                source_column,
            ]
        ].copy()

        working[source_column] = (
            numeric_series(
                working[source_column]
            )
        )

        if aggregation == "monthly_mean":
            series = (
                working
                .groupby(
                    "month_key"
                )[source_column]
                .mean()
            )

        elif aggregation == "monthly_sum":
            series = (
                working
                .groupby(
                    "month_key"
                )[source_column]
                .sum(
                    min_count=1
                )
            )

        elif aggregation == "month_end_last":
            non_null = (
                working
                .dropna(
                    subset=[
                        source_column
                    ]
                )
                .sort_values(
                    "date"
                )
            )

            series = (
                non_null
                .groupby(
                    "month_key"
                )[source_column]
                .last()
            )

        else:
            raise ValueError(
                "Unsupported daily aggregation: "
                f"{aggregation}"
            )

        result[
            output_column
        ] = series

    return (
        result
        .reset_index()
    )


# ------------------------------------------------------------------
# MONTHLY -> AS IS
# ------------------------------------------------------------------

def prepare_monthly(
    source_file,
    rules,
):
    df = load_csv(
        source_file
    )

    validate_source_columns(
        df,
        rules,
        source_file,
    )

    df["month_key"] = (
        detect_month_key(
            df
        )
    )

    df = df.dropna(
        subset=["month_key"]
    )

    duplicate_count = int(
        df["month_key"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        duplicate_months = (
            df.loc[
                df["month_key"]
                .duplicated(
                    keep=False
                ),
                "month_key",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate monthly rows found in "
            f"{source_file}: "
            f"{duplicate_months[:10]}"
        )

    result = df[
        ["month_key"]
    ].copy()

    for _, rule in rules.iterrows():
        source_column = (
            rule["source_column"]
        )

        output_column = (
            rule["output_column"]
        )

        result[
            output_column
        ] = df[
            source_column
        ].values

    return result


# ------------------------------------------------------------------
# ANNUAL -> PUBLICATION-AWARE MONTHLY
# ------------------------------------------------------------------

def detect_observation_year(df):
    if "observation_year" in df.columns:
        return pd.to_numeric(
            df["observation_year"],
            errors="coerce",
        )

    if "year" in df.columns:
        return pd.to_numeric(
            df["year"],
            errors="coerce",
        )

    if "date" in df.columns:
        dates = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        return dates.dt.year

    raise ValueError(
        "Annual source has no observation year field"
    )


def calculate_availability_year(
    df,
    observation_year,
):
    if "report_year" in df.columns:
        report_year = pd.to_numeric(
            df["report_year"],
            errors="coerce",
        )

        availability_year = pd.concat(
            [
                observation_year.rename(
                    "observation_year"
                ),
                report_year.rename(
                    "report_year"
                ),
            ],
            axis=1,
        ).max(
            axis=1,
            skipna=True,
        )

        fallback = (
            observation_year
            + 1
        )

        availability_year = (
            availability_year
            .fillna(
                fallback
            )
        )

        return availability_year

    return (
        observation_year
        + 1
    )


def prepare_annual(
    source_file,
    rules,
    calendar_periods,
):
    df = load_csv(
        source_file
    )

    validate_source_columns(
        df,
        rules,
        source_file,
    )

    observation_year = (
        detect_observation_year(
            df
        )
    )

    working = df.copy()

    working[
        "_observation_year"
    ] = observation_year

    working = working.dropna(
        subset=[
            "_observation_year"
        ]
    )

    working[
        "_observation_year"
    ] = (
        working[
            "_observation_year"
        ]
        .astype(int)
    )

    duplicate_years = (
        working[
            "_observation_year"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_years.any():
        duplicate_values = (
            working.loc[
                duplicate_years,
                "_observation_year",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate observation years found in "
            f"{source_file}: "
            f"{duplicate_values}"
        )

    calendar = pd.DataFrame(
        {
            "month_key": pd.PeriodIndex(
                calendar_periods,
                freq="M",
            )
        }
    )

    calendar[
        "_observation_year"
    ] = (
        calendar[
            "month_key"
        ]
        .dt.year
    )

    annual_values = working[
        [
            "_observation_year"
        ]
    ].copy()

    for _, rule in rules.iterrows():
        source_column = (
            rule["source_column"]
        )

        output_column = (
            rule["output_column"]
        )

        annual_values[
            output_column
        ] = working[
            source_column
        ].values

    result = calendar.merge(
        annual_values,
        on="_observation_year",
        how="left",
        validate="many_to_one",
    )

    result = result.drop(
        columns=[
            "_observation_year"
        ]
    )

    return result


# ------------------------------------------------------------------
# MERGE
# ------------------------------------------------------------------

def merge_source(
    master,
    source_df,
    source_name,
):
    before_rows = len(
        master
    )

    master = master.merge(
        source_df,
        on="month_key",
        how="left",
        validate="one_to_one",
    )

    after_rows = len(
        master
    )

    if before_rows != after_rows:
        raise RuntimeError(
            "Master row count changed after merging "
            f"{source_name}: "
            f"{before_rows} -> {after_rows}"
        )

    return master


# ------------------------------------------------------------------
# COLUMN ORDER
# ------------------------------------------------------------------

def build_column_order(
    plan,
):
    included = plan[
        plan["include"]
    ].copy()

    category_rank = {
        category: index
        for index, category
        in enumerate(
            CATEGORY_ORDER
        )
    }

    included[
        "_category_rank"
    ] = (
        included[
            "category"
        ]
        .map(
            category_rank
        )
        .fillna(
            len(
                CATEGORY_ORDER
            )
        )
    )

    included = included.sort_values(
        [
            "_category_rank",
            "category",
            "source_file",
            "output_column",
        ]
    )

    columns = (
        included[
            "output_column"
        ]
        .drop_duplicates()
        .tolist()
    )

    if TARGET_COLUMN in columns:
        columns.remove(
            TARGET_COLUMN
        )

        columns.insert(
            0,
            TARGET_COLUMN,
        )

    return columns


# ------------------------------------------------------------------
# COVERAGE DIAGNOSTICS
# ------------------------------------------------------------------

def build_coverage_report(
    master,
    plan,
):
    rule_lookup = (
        plan[
            plan["include"]
        ]
        .drop_duplicates(
            subset=[
                "output_column"
            ]
        )
        .set_index(
            "output_column"
        )
    )

    rows = []

    feature_columns = [
        column
        for column in master.columns
        if column not in {
            "date",
            "month_key",
        }
    ]

    for column in feature_columns:
        series = master[
            column
        ]

        non_null = (
            series.notna()
        )

        if non_null.any():
            first_valid = (
                master.loc[
                    non_null,
                    "date",
                ]
                .min()
            )

            last_valid = (
                master.loc[
                    non_null,
                    "date",
                ]
                .max()
            )
        else:
            first_valid = pd.NaT
            last_valid = pd.NaT

        if column in rule_lookup.index:
            rule = rule_lookup.loc[
                column
            ]

            category = rule[
                "category"
            ]

            source_file = rule[
                "source_file"
            ]

            aggregation = rule[
                "aggregation"
            ]

            frequency = rule[
                "frequency"
            ]
        else:
            category = None
            source_file = None
            aggregation = None
            frequency = None

        rows.append(
            {
                "column": column,
                "category": category,
                "frequency": frequency,
                "aggregation": aggregation,
                "source_file": source_file,
                "non_null_count": int(
                    non_null.sum()
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_ratio": float(
                    series.isna().mean()
                ),
                "first_valid_month": first_valid,
                "last_valid_month": last_valid,
            }
        )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------------

def validate_master(
    master,
    expected_features,
):
    if master[
        "date"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate dates found in final master"
        )

    if not master[
        "date"
    ].is_monotonic_increasing:
        raise RuntimeError(
            "Final master dates are not sorted"
        )

    actual_features = [
        column
        for column in master.columns
        if column != "date"
    ]

    missing_features = sorted(
        set(
            expected_features
        )
        - set(
            actual_features
        )
    )

    unexpected_features = sorted(
        set(
            actual_features
        )
        - set(
            expected_features
        )
    )

    if missing_features:
        raise RuntimeError(
            "Expected features missing from master: "
            f"{missing_features}"
        )

    if unexpected_features:
        raise RuntimeError(
            "Unexpected features in master: "
            f"{unexpected_features}"
        )

    if TARGET_COLUMN not in master.columns:
        raise RuntimeError(
            "Target column missing from master"
        )

    all_null_columns = [
        column
        for column in actual_features
        if master[
            column
        ].isna().all()
    ]

    if all_null_columns:
        print(
            "[WARNING] All-null columns:"
        )

        for column in all_null_columns:
            print(
                f"  - {column}"
            )


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print(
        "=" * 80
    )

    print(
        "BUILD COPPER MONTHLY MASTER"
    )

    print(
        "=" * 80
    )

    if not PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Feature plan not found: {PLAN_FILE}"
        )

    plan = pd.read_csv(
        PLAN_FILE
    )

    plan[
        "include"
    ] = (
        plan[
            "include"
        ]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if plan[
        "include"
    ].isna().any():
        raise ValueError(
            "Invalid include values in feature plan"
        )

    included = plan[
        plan["include"]
    ].copy()

    duplicate_outputs = included[
        included[
            "output_column"
        ].duplicated(
            keep=False
        )
    ]

    if not duplicate_outputs.empty:
        raise ValueError(
            "Duplicate output columns in feature plan:\n"
            + duplicate_outputs[
                [
                    "output_column",
                    "source_file",
                    "source_column",
                ]
            ].to_string(
                index=False
            )
        )

    calendar = (
        build_master_calendar()
    )

    master = (
        calendar.copy()
    )

    print(
        "[INFO] Calendar:",
        calendar["date"].min().date(),
        "->",
        calendar["date"].max().date(),
    )

    print(
        "[INFO] Calendar rows:",
        len(calendar),
    )

    for (
        frequency,
        source_file,
    ), rules in included.groupby(
        [
            "frequency",
            "source_file",
        ],
        sort=False,
    ):
        source_name = (
            Path(
                source_file
            ).name
        )

        print()
        print(
            f"[INFO] Processing: {source_name}"
        )

        print(
            f"       Frequency: {frequency}"
        )

        print(
            f"       Features: {len(rules)}"
        )

        if frequency == "daily":
            transformed = (
                aggregate_daily(
                    source_file,
                    rules,
                )
            )

        elif frequency == "monthly":
            transformed = (
                prepare_monthly(
                    source_file,
                    rules,
                )
            )

        elif frequency == "annual":
            transformed = (
                prepare_annual(
                    source_file,
                    rules,
                    calendar[
                        "month_key"
                    ],
                )
            )

        else:
            raise ValueError(
                "Unsupported frequency: "
                f"{frequency}"
            )

        master = merge_source(
            master,
            transformed,
            source_name,
        )

    expected_features = (
        included[
            "output_column"
        ]
        .tolist()
    )

    column_order = (
        build_column_order(
            plan
        )
    )

    master = master[
        [
            "date",
            "month_key",
        ]
        + column_order
    ]

    validate_master(
        master.drop(
            columns=[
                "month_key"
            ]
        ),
        expected_features,
    )

    coverage = (
        build_coverage_report(
            master,
            plan,
        )
    )

    MONTHLY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_master = (
        master
        .drop(
            columns=[
                "month_key"
            ]
        )
        .copy()
    )

    final_master.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "[OK] MASTER CREATED"
    )

    print(
        "=" * 80
    )

    print(
        "[INFO] Output:",
        OUTPUT_FILE,
    )

    print(
        "[INFO] Coverage:",
        COVERAGE_FILE,
    )

    print(
        "[INFO] Shape:",
        final_master.shape,
    )

    print(
        "[INFO] First month:",
        final_master[
            "date"
        ].min().date(),
    )

    print(
        "[INFO] Last month:",
        final_master[
            "date"
        ].max().date(),
    )

    print(
        "[INFO] Expected features:",
        len(
            expected_features
        ),
    )

    print(
        "[INFO] Actual features:",
        len(
            final_master.columns
        )
        - 1,
    )

    print(
        "[INFO] Target non-null:",
        int(
            final_master[
                TARGET_COLUMN
            ]
            .notna()
            .sum()
        ),
    )

    print(
        "[INFO] Target missing:",
        int(
            final_master[
                TARGET_COLUMN
            ]
            .isna()
            .sum()
        ),
    )


if __name__ == "__main__":
    main()
