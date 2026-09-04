import numpy as np
import pandas as pd

from src.utils.paths import (
    FEATURES_DIR,
    DIAGNOSTICS_DIR,
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


ORIGIN_COLUMN = (
    "origin_copper_price_usd_per_ton"
)

HORIZONS = [
    1,
    3,
    6,
    12,
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


def main():
    print("=" * 80)
    print("NAIVE MULTI-HORIZON BACKTEST")
    print("=" * 80)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    prediction_rows = []
    metric_rows = []

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        if target_column not in df.columns:
            raise ValueError(
                f"Missing target: {target_column}"
            )

        subset = df[
            [
                "date",
                target_date_column,
                ORIGIN_COLUMN,
                target_column,
            ]
        ].dropna().copy()

        subset[
            "prediction"
        ] = subset[
            ORIGIN_COLUMN
        ]

        for _, row in subset.iterrows():
            prediction_rows.append(
                {
                    "model": "naive",
                    "horizon": horizon,
                    "origin_date": row[
                        "date"
                    ],
                    "target_date": row[
                        target_date_column
                    ],
                    "origin_price": row[
                        ORIGIN_COLUMN
                    ],
                    "actual": row[
                        target_column
                    ],
                    "prediction": row[
                        "prediction"
                    ],
                    "error": (
                        row[
                            "prediction"
                        ]
                        - row[
                            target_column
                        ]
                    ),
                }
            )

        actual = (
            subset[
                target_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        predicted = (
            subset[
                "prediction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        origin = (
            subset[
                ORIGIN_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        metric_rows.append(
            {
                "model": "naive",
                "horizon": horizon,
                "observations": len(
                    subset
                ),
                "mae": mae(
                    actual,
                    predicted,
                ),
                "rmse": rmse(
                    actual,
                    predicted,
                ),
                "mape_pct": mape(
                    actual,
                    predicted,
                ),
                "smape_pct": smape(
                    actual,
                    predicted,
                ),
                "bias": bias(
                    actual,
                    predicted,
                ),
                "directional_accuracy_pct": (
                    directional_accuracy(
                        origin,
                        actual,
                        predicted,
                    )
                ),
            }
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    metrics = pd.DataFrame(
        metric_rows
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
