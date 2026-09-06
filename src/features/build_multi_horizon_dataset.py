import pandas as pd

from src.utils.paths import (
    CONFIG_DIR,
    FEATURES_DIR,
    MONTHLY_DIR,
)


MODEL_FEATURE_FILE = (
    FEATURES_DIR
    / "copper_model_features.csv"
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
    / "copper_multi_horizon_features.csv"
)

TARGET_COLUMN = "cash_settlement_usd_per_ton"

# Production forecast horizons:
# H1 through H12.
HORIZONS = list(
    range(1, 13)
)


# These categories are considered known at the completed
# monthly forecast origin.
CURRENT_AVAILABLE_CATEGORIES = {
    "copper_market",
    "other_metals",
    "equities_companies",
    "equities_indices",
    "shipping",
}


def load_data():
    if not MODEL_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Model feature file not found: {MODEL_FEATURE_FILE}"
        )

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master file not found: {MASTER_FILE}"
        )

    if not PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Feature plan not found: {PLAN_FILE}"
        )

    model_features = pd.read_csv(
        MODEL_FEATURE_FILE,
        parse_dates=["date"],
    )

    master = pd.read_csv(
        MASTER_FILE,
        parse_dates=["date"],
    )

    plan = pd.read_csv(
        PLAN_FILE
    )

    plan["include"] = (
        plan["include"]
        .astype(str)
        .str.strip()
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

    return (
        model_features,
        master,
        plan,
    )


def validate_dates(
    model_features,
    master,
):
    if (
        model_features["date"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate dates found in model feature dataset"
        )

    if (
        master["date"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate dates found in master dataset"
        )

    if not model_features[
        "date"
    ].equals(
        master["date"]
    ):
        raise ValueError(
            "Model feature dates and master dates do not match"
        )

    if not master[
        "date"
    ].is_monotonic_increasing:
        raise ValueError(
            "Master dates are not sorted"
        )

    if not model_features[
        "date"
    ].is_monotonic_increasing:
        raise ValueError(
            "Model feature dates are not sorted"
        )


def get_current_available_features(
    plan,
):
    selected = plan[
        plan["category"].isin(
            CURRENT_AVAILABLE_CATEGORIES
        )
    ].copy()

    columns = (
        selected[
            "output_column"
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]


def add_current_origin_features(
    result,
    master,
    plan,
):
    result[
        "origin_copper_price_usd_per_ton"
    ] = master[
        TARGET_COLUMN
    ]

    current_features = (
        get_current_available_features(
            plan
        )
    )

    added_current_features = []
    missing_current_features = []

    for column in current_features:
        if column not in master.columns:
            missing_current_features.append(
                column
            )
            continue

        output_column = (
            f"origin_{column}"
        )

        result[
            output_column
        ] = master[
            column
        ]

        added_current_features.append(
            output_column
        )

    return (
        result,
        added_current_features,
        missing_current_features,
    )


def add_future_targets(
    result,
    master,
):
    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        result[
            target_column
        ] = (
            master[
                TARGET_COLUMN
            ]
            .shift(
                -horizon
            )
        )

        result[
            target_date_column
        ] = (
            result["date"]
            + pd.offsets.MonthEnd(
                horizon
            )
        )

    return result


def validate_horizon_targets(
    result,
):
    row_count = len(
        result
    )

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        if target_column not in result.columns:
            raise ValueError(
                f"Missing target column: {target_column}"
            )

        if target_date_column not in result.columns:
            raise ValueError(
                f"Missing target date column: {target_date_column}"
            )

        expected_available = max(
            row_count - horizon,
            0,
        )

        actual_available = int(
            result[
                target_column
            ]
            .notna()
            .sum()
        )

        if (
            actual_available
            != expected_available
        ):
            raise ValueError(
                f"H{horizon} target availability mismatch. "
                f"Expected={expected_available}, "
                f"actual={actual_available}"
            )

        expected_dates = (
            result["date"]
            + pd.offsets.MonthEnd(
                horizon
            )
        )

        if not result[
            target_date_column
        ].equals(
            expected_dates
        ):
            raise ValueError(
                f"H{horizon} target dates are misaligned"
            )


def print_target_availability(
    result,
):
    print()
    print("TARGET AVAILABILITY")

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        available = int(
            result[
                target_column
            ]
            .notna()
            .sum()
        )

        missing = int(
            result[
                target_column
            ]
            .isna()
            .sum()
        )

        print(
            f"H{horizon:02d}: "
            f"available={available}, "
            f"future_unlabeled={missing}"
        )


def print_latest_origin(
    result,
):
    latest = result.iloc[
        -1
    ]

    print()
    print("LATEST PRODUCTION ORIGIN")

    print(
        "Origin:",
        latest[
            "date"
        ].strftime(
            "%Y-%m-%d"
        ),
    )

    print(
        "Known copper:",
        latest[
            "origin_copper_price_usd_per_ton"
        ],
    )

    print()
    print("PRODUCTION FORECAST CALENDAR")

    for horizon in HORIZONS:
        target_date_column = (
            f"target_date_h{horizon}"
        )

        print(
            f"H{horizon:02d}:",
            latest[
                target_date_column
            ].strftime(
                "%Y-%m-%d"
            ),
        )


def main():
    print("=" * 80)
    print("BUILD FORECAST-ORIGIN MULTI-HORIZON DATASET")
    print("=" * 80)

    (
        model_features,
        master,
        plan,
    ) = load_data()

    model_features = (
        model_features
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    master = (
        master
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    validate_dates(
        model_features,
        master,
    )

    if TARGET_COLUMN not in master.columns:
        raise ValueError(
            f"Target not found: {TARGET_COLUMN}"
        )

    # Start from previously created leakage-safe
    # lagged and transformed model features.
    if TARGET_COLUMN in model_features.columns:
        result = model_features.drop(
            columns=[
                TARGET_COLUMN
            ]
        ).copy()
    else:
        result = model_features.copy()

    (
        result,
        added_current_features,
        missing_current_features,
    ) = add_current_origin_features(
        result=result,
        master=master,
        plan=plan,
    )

    # Add H1 through H12 realized future targets.
    result = add_future_targets(
        result=result,
        master=master,
    )

    validate_horizon_targets(
        result
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "[INFO] Shape:",
        result.shape,
    )

    print(
        "[INFO] First origin:",
        result[
            "date"
        ]
        .min()
        .strftime(
            "%Y-%m-%d"
        ),
    )

    print(
        "[INFO] Last origin:",
        result[
            "date"
        ]
        .max()
        .strftime(
            "%Y-%m-%d"
        ),
    )

    print()
    print(
        "[INFO] Forecast horizons:",
        len(
            HORIZONS
        ),
    )

    print(
        "[INFO] Horizons:",
        HORIZONS,
    )

    print()
    print(
        "[INFO] Current-origin market features added:",
        len(
            added_current_features
        ),
    )

    if missing_current_features:
        print(
            "[WARN] Planned current-origin features "
            "missing from master:",
            len(
                missing_current_features
            ),
        )

        for column in (
            missing_current_features
        ):
            print(
                "  -",
                column,
            )

    print()
    print("CURRENT-ORIGIN CATEGORIES")

    for category in sorted(
        CURRENT_AVAILABLE_CATEGORIES
    ):
        count = int(
            (
                plan[
                    "category"
                ]
                == category
            ).sum()
        )

        print(
            f"{category}: {count}"
        )

    print_target_availability(
        result
    )

    print_latest_origin(
        result
    )

    print()
    print(
        "[IMPORTANT] Monthly macro features remain lagged."
    )

    print(
        "[IMPORTANT] Current-origin values are limited "
        "to completed market data."
    )

    print(
        "[IMPORTANT] Future targets are labels only "
        "and must never be used as predictors."
    )

    print()
    print(
        "[OK] Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
