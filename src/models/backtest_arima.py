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


TARGET_COLUMN = "cash_settlement_usd_per_ton"

TRAINING_WINDOWS = [
    5,
    8,
    10,
]

HORIZONS = [
    1,
    3,
    6,
    12,
]

BACKTEST_START = pd.Timestamp(
    "2025-01-31"
)


# Small and controlled ARIMA search space.
# d=1 is appropriate for a non-stationary price level series.
ARIMA_CANDIDATES = [
    ((0, 1, 0), "n"),
    ((0, 1, 0), "t"),
    ((1, 1, 0), "n"),
    ((0, 1, 1), "n"),
    ((1, 1, 1), "n"),
    ((2, 1, 0), "n"),
    ((0, 1, 2), "n"),
    ((2, 1, 1), "n"),
]


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


def select_best_arima(
    series,
):
    best_result = None
    best_order = None
    best_trend = None
    best_aic = np.inf

    for order, trend in ARIMA_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                model = ARIMA(
                    series,
                    order=order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                result = model.fit()

            if (
                np.isfinite(
                    result.aic
                )
                and result.aic < best_aic
            ):
                best_result = result
                best_order = order
                best_trend = trend
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
        best_trend,
        best_aic,
    )


def main():
    print("=" * 80)
    print("ARIMA WALK-FORWARD BACKTEST")
    print("=" * 80)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    prediction_rows = []
    order_rows = []

    start_time = time.time()

    for window_years in TRAINING_WINDOWS:
        print()
        print("=" * 80)
        print(
            f"TRAINING WINDOW: {window_years} YEARS"
        )
        print("=" * 80)

        for horizon in HORIZONS:
            valid_origins = []

            for _, row in df.iterrows():
                origin_date = row[
                    "date"
                ]

                if origin_date < BACKTEST_START:
                    continue

                target_date = (
                    origin_date
                    + pd.offsets.MonthEnd(
                        horizon
                    )
                )

                if target_date > df[
                    "date"
                ].max():
                    continue

                valid_origins.append(
                    origin_date
                )

            print()
            print(
                f"[INFO] H{horizon}: "
                f"{len(valid_origins)} backtest origins"
            )

            for index, origin_date in enumerate(
                valid_origins,
                start=1,
            ):
                window_start = (
                    origin_date
                    - pd.DateOffset(
                        years=window_years
                    )
                    + pd.offsets.MonthEnd(1)
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
                ].dropna().copy()

                if len(training) < 36:
                    continue

                series = (
                    training[
                        TARGET_COLUMN
                    ]
                    .astype(float)
                )

                (
                    model_result,
                    best_order,
                    best_trend,
                    best_aic,
                ) = select_best_arima(
                    series
                )

                forecast = (
                    model_result
                    .forecast(
                        steps=horizon
                    )
                )

                prediction = float(
                    forecast.iloc[
                        -1
                    ]
                )

                origin_price = float(
                    training[
                        TARGET_COLUMN
                    ].iloc[-1]
                )

                target_date = (
                    origin_date
                    + pd.offsets.MonthEnd(
                        horizon
                    )
                )

                target_row = df[
                    df["date"]
                    == target_date
                ]

                if target_row.empty:
                    continue

                actual = float(
                    target_row[
                        TARGET_COLUMN
                    ].iloc[0]
                )

                prediction_rows.append(
                    {
                        "model": "arima",
                        "training_window_years":
                            window_years,
                        "horizon":
                            horizon,
                        "origin_date":
                            origin_date,
                        "target_date":
                            target_date,
                        "training_rows":
                            len(training),
                        "arima_order":
                            str(best_order),
                        "trend":
                            best_trend,
                        "aic":
                            best_aic,
                        "origin_price":
                            origin_price,
                        "actual":
                            actual,
                        "prediction":
                            prediction,
                        "error":
                            prediction - actual,
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
                            str(best_order),
                        "trend":
                            best_trend,
                        "aic":
                            best_aic,
                    }
                )

                if (
                    index == 1
                    or index % 5 == 0
                    or index
                    == len(
                        valid_origins
                    )
                ):
                    print(
                        f"  H{horizon} "
                        f"{index}/{len(valid_origins)} "
                        f"| origin={origin_date.date()} "
                        f"| train={len(training)} "
                        f"| order={best_order} "
                        f"| trend={best_trend}"
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

    order_usage = (
        pd.DataFrame(
            order_rows
        )
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    order_usage.to_csv(
        ORDER_USAGE_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("ARIMA RESULTS")
    print("=" * 80)
    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print("MOST USED ARIMA SPECIFICATIONS")
    print()

    usage = (
        order_usage
        .groupby(
            [
                "training_window_years",
                "horizon",
                "arima_order",
                "trend",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "training_window_years",
                "horizon",
                "count",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )

    print(
        usage.to_string(
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
        "[OK] Order usage:",
        ORDER_USAGE_FILE,
    )


if __name__ == "__main__":
    main()
