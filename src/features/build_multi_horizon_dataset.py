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

HORIZONS = [
    1,
    3,
    6,
    12,
]


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
        .drop_duplicates()
        .tolist()
    )

    return [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]


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
        .sort_values("date")
        .reset_index(drop=True)
    )

    master = (
        master
        .sort_values("date")
        .reset_index(drop=True)
    )

    if not model_features[
        "date"
    ].equals(
        master["date"]
    ):
        raise ValueError(
            "Model feature dates and master dates do not match"
        )

    if TARGET_COLUMN not in master.columns:
        raise ValueError(
            f"Target not found: {TARGET_COLUMN}"
        )

    # Start from previously created leakage-safe
    # lagged and transformed features.
    result = model_features.drop(
        columns=[
            TARGET_COLUMN
        ]
    ).copy()

    # Current completed-month copper price is known
    # at the forecast origin.
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

    for column in current_features:
        if column not in master.columns:
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

    # Create future labels from the realized target.
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
        result["date"]
        .min()
        .strftime("%Y-%m-%d"),
    )

    print(
        "[INFO] Last origin:",
        result["date"]
        .max()
        .strftime("%Y-%m-%d"),
    )

    print()
    print(
        "[INFO] Current-origin market features added:",
        len(
            added_current_features
        ),
    )

    print()
    print("CURRENT-ORIGIN CATEGORIES")

    for category in sorted(
        CURRENT_AVAILABLE_CATEGORIES
    ):
        count = int(
            (
                plan["category"]
                == category
            ).sum()
        )

        print(
            f"{category}: {count}"
        )

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

    for horizon in HORIZONS:
        target_date_column = (
            f"target_date_h{horizon}"
        )

        print(
            f"H{horizon:02d} forecast date:",
            latest[
                target_date_column
            ].strftime(
                "%Y-%m-%d"
            ),
        )

    print()
    print(
        "[IMPORTANT] Monthly macro features remain lagged."
    )

    print(
        "[IMPORTANT] Current-origin values are limited to completed market data."
    )

    print()
    print(
        "[OK] Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
