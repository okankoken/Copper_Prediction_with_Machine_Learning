import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.time_aware_feature_selector import (
    select_features,
)
from src.utils.paths import (
    DIAGNOSTICS_DIR,
    FEATURES_DIR,
)


INPUT_FILE = (
    FEATURES_DIR
    / "copper_multi_horizon_features.csv"
)

PREDICTION_FILE = (
    DIAGNOSTICS_DIR
    / "lightgbm_backtest_predictions.csv"
)

METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "lightgbm_backtest_metrics.csv"
)

FEATURE_USAGE_FILE = (
    DIAGNOSTICS_DIR
    / "lightgbm_feature_usage.csv"
)


HORIZONS = [
    1,
    3,
    6,
    12,
]

TRAINING_WINDOWS = [
    5,
    8,
    10,
]

BACKTEST_START = pd.Timestamp(
    "2025-01-31"
)

ORIGIN_PRICE_COLUMN = (
    "origin_copper_price_usd_per_ton"
)


def mae(actual, predicted):
    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(
            np.mean(
                (
                    actual - predicted
                ) ** 2
            )
        )
    )


def mape(actual, predicted):
    mask = (
        np.abs(actual)
        > 1e-12
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100.0
    )


def smape(actual, predicted):
    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    mask = (
        denominator
        > 1e-12
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            2.0
            * np.abs(
                actual[mask]
                - predicted[mask]
            )
            / denominator[mask]
        )
        * 100.0
    )


def bias(actual, predicted):
    return float(
        np.mean(
            predicted - actual
        )
    )


def directional_accuracy(
    origin,
    actual,
    predicted,
):
    actual_direction = np.sign(
        actual - origin
    )

    predicted_direction = np.sign(
        predicted - origin
    )

    return float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100.0
    )


def get_model():
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def prepare_training_data(
    df,
    forecast_origin,
    horizon,
    window_years,
):
    target_column = (
        f"target_h{horizon}"
    )

    target_date_column = (
        f"target_date_h{horizon}"
    )

    window_start = (
        forecast_origin
        - pd.DateOffset(
            years=window_years
        )
    )

    training = df[
        (
            df["date"]
            >= window_start
        )
        & (
            df["date"]
            < forecast_origin
        )
        & (
            df[target_date_column]
            <= forecast_origin
        )
        & (
            df[target_column]
            .notna()
        )
    ].copy()

    return training


def remove_other_targets(
    frame,
    current_target,
):
    drop_columns = []

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        if (
            target_column
            != current_target
            and target_column
            in frame.columns
        ):
            drop_columns.append(
                target_column
            )

        if (
            target_date_column
            in frame.columns
        ):
            drop_columns.append(
                target_date_column
            )

    return frame.drop(
        columns=drop_columns
    )


def main():
    print("=" * 80)
    print("LIGHTGBM WALK-FORWARD BACKTEST")
    print("=" * 80)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
            "target_date_h1",
            "target_date_h3",
            "target_date_h6",
            "target_date_h12",
        ],
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    prediction_rows = []
    feature_usage_rows = []

    start_time = time.time()

    for window_years in TRAINING_WINDOWS:
        print()
        print("=" * 80)
        print(
            f"TRAINING WINDOW: {window_years} YEARS"
        )
        print("=" * 80)

        for horizon in HORIZONS:
            target_column = (
                f"target_h{horizon}"
            )

            target_date_column = (
                f"target_date_h{horizon}"
            )

            test_origins = df[
                (
                    df["date"]
                    >= BACKTEST_START
                )
                & (
                    df[target_column]
                    .notna()
                )
            ][
                "date"
            ].tolist()

            print()
            print(
                f"[INFO] H{horizon}: "
                f"{len(test_origins)} backtest origins"
            )

            for index, forecast_origin in enumerate(
                test_origins,
                start=1,
            ):
                training = (
                    prepare_training_data(
                        df,
                        forecast_origin,
                        horizon,
                        window_years,
                    )
                )

                if len(training) < 36:
                    continue

                selector_data = (
                    remove_other_targets(
                        training,
                        target_column,
                    )
                )

                (
                    selected_features,
                    ranking,
                    decisions,
                ) = select_features(
                    selector_data,
                    target_column=target_column,
                )

                if not selected_features:
                    continue

                train_subset = training[
                    selected_features
                    + [
                        target_column
                    ]
                ].copy()

                train_subset = (
                    train_subset
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                )

                valid_target = (
                    train_subset[
                        target_column
                    ]
                    .notna()
                )

                train_subset = (
                    train_subset.loc[
                        valid_target
                    ]
                )

                X_train = train_subset[
                    selected_features
                ]

                y_train = train_subset[
                    target_column
                ]

                test_row = df[
                    df["date"]
                    == forecast_origin
                ].copy()

                if test_row.empty:
                    continue

                X_test = test_row[
                    selected_features
                ]

                model = get_model()

                model.fit(
                    X_train,
                    y_train,
                )

                prediction = float(
                    model.predict(
                        X_test
                    )[0]
                )

                actual = float(
                    test_row[
                        target_column
                    ].iloc[0]
                )

                origin_price = float(
                    test_row[
                        ORIGIN_PRICE_COLUMN
                    ].iloc[0]
                )

                target_date = test_row[
                    target_date_column
                ].iloc[0]

                prediction_rows.append(
                    {
                        "model": "lightgbm",
                        "training_window_years":
                            window_years,
                        "horizon": horizon,
                        "origin_date":
                            forecast_origin,
                        "target_date":
                            target_date,
                        "training_rows":
                            len(
                                training
                            ),
                        "selected_feature_count":
                            len(
                                selected_features
                            ),
                        "origin_price":
                            origin_price,
                        "actual":
                            actual,
                        "prediction":
                            prediction,
                        "error":
                            prediction
                            - actual,
                    }
                )

                importances = (
                    model.feature_importances_
                )

                for feature, importance in zip(
                    selected_features,
                    importances,
                ):
                    feature_usage_rows.append(
                        {
                            "training_window_years":
                                window_years,
                            "horizon":
                                horizon,
                            "origin_date":
                                forecast_origin,
                            "feature":
                                feature,
                            "importance":
                                float(
                                    importance
                                ),
                        }
                    )

                if (
                    index == 1
                    or index % 5 == 0
                    or index == len(
                        test_origins
                    )
                ):
                    print(
                        f"  H{horizon} "
                        f"{index}/{len(test_origins)} "
                        f"| origin={forecast_origin.date()} "
                        f"| train={len(training)} "
                        f"| features={len(selected_features)}"
                    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise RuntimeError(
            "No LightGBM backtest predictions were produced"
        )

    metric_rows = []

    for (
        window_years,
        horizon,
    ), group in predictions.groupby(
        [
            "training_window_years",
            "horizon",
        ]
    ):
        actual = group[
            "actual"
        ].to_numpy(
            dtype=float
        )

        predicted = group[
            "prediction"
        ].to_numpy(
            dtype=float
        )

        origin = group[
            "origin_price"
        ].to_numpy(
            dtype=float
        )

        metric_rows.append(
            {
                "model":
                    "lightgbm",
                "training_window_years":
                    window_years,
                "horizon":
                    horizon,
                "observations":
                    len(group),
                "mae":
                    mae(
                        actual,
                        predicted,
                    ),
                "rmse":
                    rmse(
                        actual,
                        predicted,
                    ),
                "mape_pct":
                    mape(
                        actual,
                        predicted,
                    ),
                "smape_pct":
                    smape(
                        actual,
                        predicted,
                    ),
                "bias":
                    bias(
                        actual,
                        predicted,
                    ),
                "directional_accuracy_pct":
                    directional_accuracy(
                        origin,
                        actual,
                        predicted,
                    ),
            }
        )

    metrics = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "horizon",
            "mape_pct",
        ]
    )

    feature_usage = pd.DataFrame(
        feature_usage_rows
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    feature_usage.to_csv(
        FEATURE_USAGE_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("LIGHTGBM RESULTS")
    print("=" * 80)
    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        f"[INFO] Runtime seconds: {elapsed:.2f}"
    )

    print()
    print(
        "[OK] Predictions:",
        PREDICTION_FILE,
    )

    print(
        "[OK] Metrics:",
        METRICS_FILE,
    )

    print(
        "[OK] Feature usage:",
        FEATURE_USAGE_FILE,
    )


if __name__ == "__main__":
    main()
