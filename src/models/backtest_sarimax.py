import time
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

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
    / "sarimax_backtest_predictions.csv"
)

METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "sarimax_backtest_metrics.csv"
)

FEATURE_USAGE_FILE = (
    DIAGNOSTICS_DIR
    / "sarimax_feature_usage.csv"
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


SARIMAX_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (0, 1, 2),
]


def get_max_exog(
    window_years,
):
    if window_years == 5:
        return 3

    if window_years == 8:
        return 4

    return 5


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


def select_exogenous_features(
    training,
    target_column,
    max_exog,
):
    selector_data = (
        remove_other_targets(
            training,
            target_column,
        )
    )

    (
        selected,
        ranking,
        decisions,
    ) = select_features(
        selector_data,
        target_column=target_column,
    )

    # Target-history features are not treated
    # as external regressors in SARIMAX.
    filtered = [
        feature
        for feature in selected
        if (
            not feature.startswith(
                "target_"
            )
            and feature
            != ORIGIN_PRICE_COLUMN
        )
    ]

    return filtered[
        :max_exog
    ]


def impute_exogenous(
    X_train,
    X_forecast,
):
    X_train = (
        X_train
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(float)
    )

    X_forecast = (
        X_forecast
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(float)
    )

    medians = (
        X_train
        .median(
            axis=0
        )
    )

    valid_columns = (
        medians[
            medians.notna()
        ]
        .index
        .tolist()
    )

    X_train = X_train[
        valid_columns
    ]

    X_forecast = X_forecast[
        valid_columns
    ]

    medians = medians[
        valid_columns
    ]

    X_train = (
        X_train.fillna(
            medians
        )
    )

    X_forecast = (
        X_forecast.fillna(
            medians
        )
    )

    return (
        X_train,
        X_forecast,
        valid_columns,
    )


def fit_best_sarimax(
    y_train,
    X_train,
):
    best_result = None
    best_order = None
    best_aic = np.inf

    y_array = np.asarray(
        y_train,
        dtype=float,
    )

    X_array = np.asarray(
        X_train,
        dtype=float,
    )

    for order in SARIMAX_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                model = SARIMAX(
                    endog=y_array,
                    exog=X_array,
                    order=order,
                    seasonal_order=(
                        0,
                        0,
                        0,
                        0,
                    ),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                result = model.fit(
                    disp=False,
                    maxiter=200,
                )

            if (
                np.isfinite(
                    result.aic
                )
                and result.aic
                < best_aic
            ):
                best_result = result
                best_order = order
                best_aic = float(
                    result.aic
                )

        except Exception:
            continue

    if best_result is None:
        raise RuntimeError(
            "No SARIMAX candidate could be fitted"
        )

    return (
        best_result,
        best_order,
        best_aic,
    )


def main():
    print("=" * 80)
    print("SARIMAX WALK-FORWARD BACKTEST")
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

    start_time = time.time()

    for window_years in TRAINING_WINDOWS:
        print()
        print("=" * 80)

        print(
            f"TRAINING WINDOW: "
            f"{window_years} YEARS"
        )

        print("=" * 80)

        max_exog = get_max_exog(
            window_years
        )

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
                training = (
                    prepare_training(
                        df,
                        forecast_origin,
                        horizon,
                        window_years,
                    )
                )

                if len(
                    training
                ) < 36:
                    continue

                selected_exog = (
                    select_exogenous_features(
                        training,
                        target_column,
                        max_exog,
                    )
                )

                if not selected_exog:
                    continue

                last_training_origin = (
                    training[
                        "date"
                    ]
                    .max()
                )

                forecast_path = df[
                    (
                        df["date"]
                        > last_training_origin
                    )
                    & (
                        df["date"]
                        <= forecast_origin
                    )
                ].copy()

                # For a direct H-horizon target,
                # the model series ends h months
                # before the forecast origin.
                if (
                    len(
                        forecast_path
                    )
                    != horizon
                ):
                    raise RuntimeError(
                        "Unexpected forecast path length: "
                        f"origin={forecast_origin}, "
                        f"horizon={horizon}, "
                        f"path={len(forecast_path)}"
                    )

                y_train = (
                    training[
                        target_column
                    ]
                    .astype(float)
                )

                X_train = training[
                    selected_exog
                ].copy()

                X_forecast = forecast_path[
                    selected_exog
                ].copy()

                (
                    X_train,
                    X_forecast,
                    selected_exog,
                ) = impute_exogenous(
                    X_train,
                    X_forecast,
                )

                if not selected_exog:
                    continue

                if len(
                    y_train
                ) != len(
                    X_train
                ):
                    raise RuntimeError(
                        "Training target and exogenous "
                        "row counts do not match"
                    )

                if len(
                    X_forecast
                ) != horizon:
                    raise RuntimeError(
                        "Forecast exogenous path "
                        "does not match horizon"
                    )

                (
                    model_result,
                    best_order,
                    best_aic,
                ) = fit_best_sarimax(
                    y_train,
                    X_train,
                )

                forecast = (
                    model_result
                    .get_forecast(
                        steps=horizon,
                        exog=np.asarray(
                            X_forecast,
                            dtype=float,
                        ),
                    )
                    .predicted_mean
                )

                prediction = float(
                    np.asarray(
                        forecast
                    )[
                        -1
                    ]
                )

                test_row = df[
                    df["date"]
                    == forecast_origin
                ]

                if test_row.empty:
                    continue

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

                prediction_rows.append(
                    {
                        "model":
                            "sarimax",
                        "training_window_years":
                            window_years,
                        "horizon":
                            horizon,
                        "origin_date":
                            forecast_origin,
                        "target_date":
                            target_date,
                        "training_rows":
                            len(
                                training
                            ),
                        "selected_exog_count":
                            len(
                                selected_exog
                            ),
                        "sarimax_order":
                            str(
                                best_order
                            ),
                        "aic":
                            best_aic,
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

                for feature in selected_exog:
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
                        f"| exog="
                        f"{len(selected_exog)} "
                        f"| steps="
                        f"{horizon} "
                        f"| order="
                        f"{best_order}"
                    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise RuntimeError(
            "No SARIMAX backtest "
            "predictions were produced"
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
                    "sarimax",
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
    print("SARIMAX RESULTS")
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


if __name__ == "__main__":
    main()
