import numpy as np
import pandas as pd

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
    / "naive_backtest_predictions.csv"
)

METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "naive_backtest_metrics.csv"
)


HORIZONS = list(
    range(1, 13)
)

BACKTEST_START = pd.Timestamp(
    "2025-01-31"
)

ORIGIN_PRICE_COLUMN = (
    "origin_copper_price_usd_per_ton"
)


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


def main():
    print("=" * 80)
    print("NAIVE H1-H12 WALK-FORWARD BACKTEST")
    print("=" * 80)

    parse_date_columns = [
        "date",
    ] + [
        f"target_date_h{horizon}"
        for horizon in HORIZONS
    ]

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=parse_date_columns,
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

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        test = df[
            (
                df["date"]
                >= BACKTEST_START
            )
            & (
                df[
                    target_column
                ]
                .notna()
            )
        ].copy()

        print(
            f"[INFO] H{horizon:02d}: "
            f"{len(test)} observations"
        )

        for _, row in test.iterrows():
            origin_price = float(
                row[
                    ORIGIN_PRICE_COLUMN
                ]
            )

            actual = float(
                row[
                    target_column
                ]
            )

            # Random walk forecast:
            # future price equals current known price.
            prediction = origin_price

            prediction_rows.append(
                {
                    "model":
                        "naive",
                    "horizon":
                        horizon,
                    "origin_date":
                        row["date"],
                    "target_date":
                        row[
                            target_date_column
                        ],
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

    predictions = pd.DataFrame(
        prediction_rows
    )

    metric_rows = []

    for horizon, group in (
        predictions.groupby(
            "horizon"
        )
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
                    "naive",
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
            "horizon"
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

    print()
    print("=" * 80)
    print("NAIVE H1-H12 RESULTS")
    print("=" * 80)
    print()

    print(
        metrics.to_string(
            index=False
        )
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


if __name__ == "__main__":
    main()
