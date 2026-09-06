import time
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from src.utils.paths import (
    DIAGNOSTICS_DIR,
    MONTHLY_DIR,
)


INPUT_FILE = (
    MONTHLY_DIR
    / "copper_monthly_master.csv"
)

PREDICTION_FILE = (
    DIAGNOSTICS_DIR
    / "arima_backtest_predictions.csv"
)

METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "arima_backtest_metrics.csv"
)

ORDER_USAGE_FILE = (
    DIAGNOSTICS_DIR
    / "arima_order_usage.csv"
)


TARGET_COLUMN = (
    "cash_settlement_usd_per_ton"
)

HORIZONS = list(
    range(1, 13)
)

TRAINING_WINDOWS = [
    5,
    8,
    10,
]

BACKTEST_START = pd.Timestamp(
    "2025-01-31"
)


ARIMA_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (0, 1, 2),
]


def mae(
    actual,
    predicted,
):
    return float(
        np.mean(
            np.abs(
                actual - predicted
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
                    actual - predicted
                ) ** 2
            )
        )
    )


def mape(
    actual,
    predicted,
):
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


def smape(
    actual,
    predicted,
):
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


def bias(
    actual,
    predicted,
):
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


def fit_best_arima(
    y_train,
):
    best_result = None
    best_order = None
    best_aic = np.inf

    for order in ARIMA_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                model = ARIMA(
                    y_train,
                    order=order,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                result = model.fit()

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
            "No ARIMA candidate could be fitted"
        )

    return (
        best_result,
        best_order,
        best_aic,
    )


def main():
    print("=" * 80)
    print("ARIMA H1-H12 WALK-FORWARD BACKTEST")
    print("=" * 80)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "date",
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

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target not found: {TARGET_COLUMN}"
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate monthly dates found"
        )

    prediction_rows = []
    order_rows = []

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
            valid_origins = []

            for index, row in df.iterrows():
                origin_date = row[
                    "date"
                ]

                if (
                    origin_date
                    < BACKTEST_START
                ):
                    continue

                target_index = (
                    index
                    + horizon
                )

                if (
                    target_index
                    >= len(df)
                ):
                    continue

                if pd.isna(
                    df.loc[
                        target_index,
                        TARGET_COLUMN,
                    ]
                ):
                    continue

                valid_origins.append(
                    (
                        index,
                        origin_date,
                        target_index,
                    )
                )

            print()
            print(
                f"[INFO] H{horizon:02d}: "
                f"{len(valid_origins)} "
                "backtest origins"
            )

            for counter, (
                origin_index,
                origin_date,
                target_index,
            ) in enumerate(
                valid_origins,
                start=1,
            ):
                window_start = (
                    origin_date
                    - pd.DateOffset(
                        years=window_years
                    )
                    + pd.offsets.MonthEnd(0)
                )

                training = df[
                    (
                        df["date"]
                        >= window_start
                    )
                    & (
                        df["date"]
                        <= origin_date
                    )
                ][
                    [
                        "date",
                        TARGET_COLUMN,
                    ]
                ].copy()

                training = (
                    training.dropna(
                        subset=[
                            TARGET_COLUMN
                        ]
                    )
                )

                if len(training) < 36:
                    continue

                y_train = (
                    training
                    .set_index(
                        "date"
                    )[
                        TARGET_COLUMN
                    ]
                    .astype(float)
                )

                # Give statsmodels a supported
                # regular monthly index.
                y_train.index = (
                    pd.DatetimeIndex(
                        y_train.index
                    )
                    .to_period("M")
                    .to_timestamp("M")
                )

                y_train = y_train.asfreq(
                    "ME"
                )

                if y_train.isna().any():
                    raise RuntimeError(
                        "Unexpected missing target "
                        "inside ARIMA training window"
                    )

                (
                    model_result,
                    best_order,
                    best_aic,
                ) = fit_best_arima(
                    y_train
                )

                forecast = (
                    model_result
                    .get_forecast(
                        steps=horizon
                    )
                    .predicted_mean
                )

                prediction = float(
                    forecast.iloc[
                        -1
                    ]
                )

                origin_price = float(
                    df.loc[
                        origin_index,
                        TARGET_COLUMN,
                    ]
                )

                actual = float(
                    df.loc[
                        target_index,
                        TARGET_COLUMN,
                    ]
                )

                target_date = df.loc[
                    target_index,
                    "date",
                ]

                prediction_rows.append(
                    {
                        "model":
                            "arima",
                        "training_window_years":
                            window_years,
                        "horizon":
                            horizon,
                        "origin_date":
                            origin_date,
                        "target_date":
                            target_date,
                        "training_rows":
                            len(
                                training
                            ),
                        "arima_order":
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

                order_rows.append(
                    {
                        "training_window_years":
                            window_years,
                        "horizon":
                            horizon,
                        "origin_date":
                            origin_date,
                        "arima_order":
                            str(
                                best_order
                            ),
                        "aic":
                            best_aic,
                    }
                )

                if (
                    counter == 1
                    or counter % 5 == 0
                    or counter
                    == len(
                        valid_origins
                    )
                ):
                    print(
                        f"  H{horizon:02d} "
                        f"{counter}/"
                        f"{len(valid_origins)} "
                        f"| origin="
                        f"{origin_date.date()} "
                        f"| train="
                        f"{len(training)} "
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
            "No ARIMA predictions were produced"
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
                    "arima",
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

    orders = pd.DataFrame(
        order_rows
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    orders.to_csv(
        ORDER_USAGE_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("ARIMA H1-H12 RESULTS")
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
        "[OK] Order usage:",
        ORDER_USAGE_FILE,
    )


if __name__ == "__main__":
    main()
