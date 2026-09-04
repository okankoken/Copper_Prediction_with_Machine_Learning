import time

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Ridge,
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    / "linear_models_ridge_elasticnet_backtest_predictions.csv"
)

METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "linear_models_ridge_elasticnet_backtest_metrics.csv"
)

FEATURE_USAGE_FILE = (
    DIAGNOSTICS_DIR
    / "linear_models_ridge_elasticnet_feature_usage.csv"
)

PARAMETER_FILE = (
    DIAGNOSTICS_DIR
    / "linear_models_ridge_elasticnet_parameters.csv"
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


RIDGE_ALPHA_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

ELASTICNET_ALPHA_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]

ELASTICNET_L1_RATIO_GRID = [
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]


def mae(
    actual,
    predicted,
):
    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


def rmse(
    actual,
    predicted,
):
    return float(
        np.sqrt(
            np.mean(
                (
                    actual
                    - predicted
                ) ** 2
            )
        )
    )


def mape(
    actual,
    predicted,
):
    mask = (
        np.abs(
            actual
        )
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


def smape(
    actual,
    predicted,
):
    denominator = (
        np.abs(
            actual
        )
        + np.abs(
            predicted
        )
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


def bias(
    actual,
    predicted,
):
    return float(
        np.mean(
            predicted
            - actual
        )
    )


def directional_accuracy(
    origin,
    actual,
    predicted,
):
    actual_direction = np.sign(
        actual
        - origin
    )

    predicted_direction = np.sign(
        predicted
        - origin
    )

    return float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100.0
    )


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


def prepare_training(
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
            df[
                target_date_column
            ]
            <= forecast_origin
        )
        & (
            df[
                target_column
            ]
            .notna()
        )
    ].copy()

    return training


def get_cv_splits(
    training_rows,
):
    if training_rows >= 90:
        return 5

    if training_rows >= 60:
        return 4

    return 3


def build_ridge_search(
    training_rows,
):
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )

    cv = TimeSeriesSplit(
        n_splits=get_cv_splits(
            training_rows
        )
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha":
                RIDGE_ALPHA_GRID,
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    return search


def build_elasticnet_search(
    training_rows,
):
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                ElasticNet(
                    max_iter=20000,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = TimeSeriesSplit(
        n_splits=get_cv_splits(
            training_rows
        )
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha":
                ELASTICNET_ALPHA_GRID,
            "model__l1_ratio":
                ELASTICNET_L1_RATIO_GRID,
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    return search


def main():
    print("=" * 80)
    print("RIDGE AND ELASTIC NET WALK-FORWARD BACKTEST")
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
        df.sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    prediction_rows = []
    feature_usage_rows = []
    parameter_rows = []

    start_time = time.time()

    for window_years in TRAINING_WINDOWS:
        print()
        print("=" * 80)
        print(
            f"TRAINING WINDOW: "
            f"{window_years} YEARS"
        )
        print("=" * 80)

        for horizon in HORIZONS:
            target_column = (
                f"target_h{horizon}"
            )

            target_date_column = (
                f"target_date_h{horizon}"
            )

            test_origins = (
                df.loc[
                    (
                        df["date"]
                        >= BACKTEST_START
                    )
                    & (
                        df[
                            target_column
                        ]
                        .notna()
                    ),
                    "date",
                ]
                .tolist()
            )

            print()
            print(
                f"[INFO] H{horizon}: "
                f"{len(test_origins)} "
                "backtest origins"
            )

            for index, forecast_origin in enumerate(
                test_origins,
                start=1,
            ):
                training = prepare_training(
                    df,
                    forecast_origin,
                    horizon,
                    window_years,
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

                X_train = (
                    training[
                        selected_features
                    ]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .astype(float)
                )

                y_train = (
                    training[
                        target_column
                    ]
                    .astype(float)
                )

                test_row = df[
                    df["date"]
                    == forecast_origin
                ]

                if test_row.empty:
                    continue

                X_test = (
                    test_row[
                        selected_features
                    ]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .astype(float)
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

                target_date = (
                    test_row[
                        target_date_column
                    ].iloc[0]
                )

                model_searches = {
                    "ridge":
                        build_ridge_search(
                            len(training)
                        ),
                    "elasticnet":
                        build_elasticnet_search(
                            len(training)
                        ),
                }

                for (
                    model_name,
                    search,
                ) in model_searches.items():
                    search.fit(
                        X_train,
                        y_train,
                    )

                    prediction = float(
                        search.predict(
                            X_test
                        )[0]
                    )

                    prediction_rows.append(
                        {
                            "model":
                                model_name,
                            "training_window_years":
                                window_years,
                            "horizon":
                                horizon,
                            "origin_date":
                                forecast_origin,
                            "target_date":
                                target_date,
                            "training_rows":
                                len(training),
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

                    parameter_row = {
                        "model":
                            model_name,
                        "training_window_years":
                            window_years,
                        "horizon":
                            horizon,
                        "origin_date":
                            forecast_origin,
                    }

                    for (
                        parameter,
                        value,
                    ) in search.best_params_.items():
                        parameter_row[
                            parameter
                        ] = value

                    parameter_rows.append(
                        parameter_row
                    )

                    best_model = (
                        search.best_estimator_
                        .named_steps[
                            "model"
                        ]
                    )

                    coefficients = (
                        best_model.coef_
                    )

                    for (
                        feature,
                        coefficient,
                    ) in zip(
                        selected_features,
                        coefficients,
                    ):
                        feature_usage_rows.append(
                            {
                                "model":
                                    model_name,
                                "training_window_years":
                                    window_years,
                                "horizon":
                                    horizon,
                                "origin_date":
                                    forecast_origin,
                                "feature":
                                    feature,
                                "coefficient":
                                    float(
                                        coefficient
                                    ),
                                "abs_coefficient":
                                    float(
                                        abs(
                                            coefficient
                                        )
                                    ),
                            }
                        )

                if (
                    index == 1
                    or index % 5 == 0
                    or index
                    == len(
                        test_origins
                    )
                ):
                    print(
                        f"  H{horizon} "
                        f"{index}/"
                        f"{len(test_origins)} "
                        f"| origin="
                        f"{forecast_origin.date()} "
                        f"| train="
                        f"{len(training)} "
                        f"| features="
                        f"{len(selected_features)}"
                    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise RuntimeError(
            "No linear model predictions were produced"
        )

    metric_rows = []

    for (
        model_name,
        window_years,
        horizon,
    ), group in predictions.groupby(
        [
            "model",
            "training_window_years",
            "horizon",
        ]
    ):
        actual = (
            group[
                "actual"
            ]
            .to_numpy(
                dtype=float
            )
        )

        predicted = (
            group[
                "prediction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        origin = (
            group[
                "origin_price"
            ]
            .to_numpy(
                dtype=float
            )
        )

        metric_rows.append(
            {
                "model":
                    model_name,
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

    metrics = (
        pd.DataFrame(
            metric_rows
        )
        .sort_values(
            [
                "horizon",
                "mape_pct",
            ]
        )
    )

    feature_usage = pd.DataFrame(
        feature_usage_rows
    )

    parameters = pd.DataFrame(
        parameter_rows
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

    parameters.to_csv(
        PARAMETER_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("RIDGE AND ELASTIC NET RESULTS")
    print("=" * 80)
    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        f"[INFO] Runtime seconds: "
        f"{elapsed:.2f}"
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

    print(
        "[OK] Parameters:",
        PARAMETER_FILE,
    )


if __name__ == "__main__":
    main()
