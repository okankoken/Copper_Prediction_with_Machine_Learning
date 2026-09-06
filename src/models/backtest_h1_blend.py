import numpy as np
import pandas as pd

from src.utils.paths import DIAGNOSTICS_DIR


NAIVE_FILE = (
    DIAGNOSTICS_DIR
    / "naive_backtest_predictions.csv"
)

LINEAR_FILE = (
    DIAGNOSTICS_DIR
    / "linear_models_ridge_elasticnet_backtest_predictions.csv"
)

OUTPUT_PREDICTIONS = (
    DIAGNOSTICS_DIR
    / "h1_blend_backtest_predictions.csv"
)

OUTPUT_METRICS = (
    DIAGNOSTICS_DIR
    / "h1_blend_backtest_metrics.csv"
)


ELASTICNET_WINDOW_YEARS = 8

ELASTICNET_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def mape(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = np.abs(
        actual
    ) > 1e-12

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


def mae(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


def rmse(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    predicted
                    - actual
                ) ** 2
            )
        )
    )


def bias(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            predicted
            - actual
        )
    )


def smape(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    mask = denominator > 1e-12

    return float(
        np.mean(
            2.0
            * np.abs(
                predicted[mask]
                - actual[mask]
            )
            / denominator[mask]
        )
        * 100.0
    )


def directional_accuracy(
    origin_price,
    actual,
    predicted,
):
    origin_price = np.asarray(
        origin_price,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    actual_direction = np.sign(
        actual
        - origin_price
    )

    predicted_direction = np.sign(
        predicted
        - origin_price
    )

    return float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100.0
    )


def main():
    print("=" * 100)
    print("H1 NAIVE + ELASTICNET 8Y BLEND BACKTEST")
    print("=" * 100)

    if not NAIVE_FILE.exists():
        raise FileNotFoundError(
            f"Naive prediction file not found: "
            f"{NAIVE_FILE}"
        )

    if not LINEAR_FILE.exists():
        raise FileNotFoundError(
            f"Linear prediction file not found: "
            f"{LINEAR_FILE}"
        )

    naive = pd.read_csv(
        NAIVE_FILE,
        parse_dates=[
            "origin_date",
            "target_date",
        ],
    )

    linear = pd.read_csv(
        LINEAR_FILE,
        parse_dates=[
            "origin_date",
            "target_date",
        ],
    )

    naive = naive[
        naive[
            "horizon"
        ].astype(int)
        == 1
    ].copy()

    elasticnet = linear[
        (
            linear[
                "horizon"
            ].astype(int)
            == 1
        )
        & (
            linear[
                "model"
            ].astype(str)
            == "elasticnet"
        )
        & (
            linear[
                "training_window_years"
            ].astype(int)
            == ELASTICNET_WINDOW_YEARS
        )
    ].copy()

    if naive.empty:
        raise RuntimeError(
            "No H1 naive predictions found"
        )

    if elasticnet.empty:
        raise RuntimeError(
            "No H1 ElasticNet 8Y predictions found"
        )

    naive = naive.rename(
        columns={
            "prediction":
                "naive_prediction",
        }
    )

    elasticnet = elasticnet.rename(
        columns={
            "prediction":
                "elasticnet_prediction",
        }
    )

    merge_columns = [
        "origin_date",
        "target_date",
        "horizon",
    ]

    merged = naive[
        merge_columns
        + [
            "origin_price",
            "actual",
            "naive_prediction",
        ]
    ].merge(
        elasticnet[
            merge_columns
            + [
                "elasticnet_prediction",
            ]
        ],
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged.sort_values(
            "origin_date"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "[INFO] Common H1 origins:",
        len(
            merged
        ),
    )

    if len(
        merged
    ) < 10:
        raise RuntimeError(
            "Too few common H1 origins"
        )

    prediction_rows = []
    metric_rows = []

    for weight in ELASTICNET_WEIGHTS:
        current = merged.copy()

        current[
            "elasticnet_weight"
        ] = weight

        current[
            "naive_weight"
        ] = (
            1.0
            - weight
        )

        current[
            "blend_prediction"
        ] = (
            weight
            * current[
                "elasticnet_prediction"
            ]
            + (
                1.0
                - weight
            )
            * current[
                "naive_prediction"
            ]
        )

        prediction_rows.append(
            current
        )

        actual = current[
            "actual"
        ].to_numpy(
            dtype=float
        )

        predicted = current[
            "blend_prediction"
        ].to_numpy(
            dtype=float
        )

        origin_price = current[
            "origin_price"
        ].to_numpy(
            dtype=float
        )

        metric_rows.append(
            {
                "elasticnet_weight":
                    weight,
                "naive_weight":
                    1.0
                    - weight,
                "observations":
                    len(
                        current
                    ),
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
                        origin_price,
                        actual,
                        predicted,
                    ),
            }
        )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics[
        "rank_mape"
    ] = (
        metrics[
            "mape_pct"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    metrics[
        "rank_rmse"
    ] = (
        metrics[
            "rmse"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    metrics = (
        metrics.sort_values(
            [
                "mape_pct",
                "rmse",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    print()
    print("=" * 100)
    print("H1 BLEND RESULTS")
    print("=" * 100)
    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    best = metrics.iloc[0]

    print()
    print("=" * 100)
    print("BEST H1 BLEND")
    print("=" * 100)
    print()

    print(
        f"ElasticNet weight : "
        f"{best['elasticnet_weight']:.2f}"
    )

    print(
        f"Naive weight      : "
        f"{best['naive_weight']:.2f}"
    )

    print(
        f"MAPE              : "
        f"{best['mape_pct']:.4f}%"
    )

    print(
        f"RMSE              : "
        f"{best['rmse']:.2f}"
    )

    print(
        f"Bias              : "
        f"{best['bias']:.2f}"
    )

    print(
        f"Directional Acc.  : "
        f"{best['directional_accuracy_pct']:.2f}%"
    )

    print()
    print(
        "[OK] Predictions:",
        OUTPUT_PREDICTIONS,
    )

    print(
        "[OK] Metrics:",
        OUTPUT_METRICS,
    )


if __name__ == "__main__":
    main()
