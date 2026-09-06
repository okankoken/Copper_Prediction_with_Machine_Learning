import numpy as np
import pandas as pd

from src.utils.paths import DIAGNOSTICS_DIR


INPUT_FILE = (
    DIAGNOSTICS_DIR
    / "ensemble_expert_predictions.csv"
)

OUTPUT_PREDICTIONS = (
    DIAGNOSTICS_DIR
    / "ensemble_topn_backtest_predictions.csv"
)

OUTPUT_METRICS = (
    DIAGNOSTICS_DIR
    / "ensemble_topn_backtest_metrics.csv"
)

OUTPUT_SELECTIONS = (
    DIAGNOSTICS_DIR
    / "ensemble_topn_selections.csv"
)


HORIZONS = [
    1,
    3,
    6,
    12,
]

TOP_N_VALUES = [
    3,
    5,
]

MIN_HISTORY = 3

EPSILON = 1e-8


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


def get_prior_performance(
    df,
    horizon,
    origin_date,
):
    history = df[
        (
            df["horizon"]
            == horizon
        )
        & (
            df["origin_date"]
            < origin_date
        )
    ].copy()

    if history.empty:
        return pd.DataFrame()

    history[
        "absolute_error"
    ] = np.abs(
        history["prediction"]
        - history["actual"]
    )

    summary = (
        history.groupby(
            "expert"
        )
        .agg(
            prior_mae=(
                "absolute_error",
                "mean",
            ),
            prior_observations=(
                "absolute_error",
                "size",
            ),
        )
        .reset_index()
    )

    summary = summary[
        summary[
            "prior_observations"
        ]
        >= MIN_HISTORY
    ].copy()

    return summary.sort_values(
        "prior_mae",
        ascending=True,
    )


def make_prediction(
    current,
    prior_performance,
    top_n,
    weighted,
):
    available = (
        current[
            [
                "expert",
                "prediction",
            ]
        ]
        .merge(
            prior_performance,
            on="expert",
            how="inner",
        )
        .sort_values(
            "prior_mae",
            ascending=True,
        )
        .head(top_n)
        .copy()
    )

    if available.empty:
        return (
            np.nan,
            available,
        )

    if weighted:
        available[
            "raw_weight"
        ] = (
            1.0
            / (
                available[
                    "prior_mae"
                ]
                + EPSILON
            )
        )

        available[
            "weight"
        ] = (
            available[
                "raw_weight"
            ]
            / available[
                "raw_weight"
            ].sum()
        )

        prediction = float(
            np.sum(
                available[
                    "prediction"
                ]
                * available[
                    "weight"
                ]
            )
        )

    else:
        available[
            "weight"
        ] = (
            1.0
            / len(
                available
            )
        )

        prediction = float(
            available[
                "prediction"
            ].mean()
        )

    return (
        prediction,
        available,
    )


def main():
    print("=" * 80)
    print("TOP-N ENSEMBLE WALK-FORWARD BACKTEST")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "origin_date",
            "target_date",
        ],
    )

    df = (
        df.sort_values(
            [
                "horizon",
                "origin_date",
                "expert",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    prediction_rows = []
    selection_rows = []

    for horizon in HORIZONS:
        horizon_data = df[
            df["horizon"]
            == horizon
        ].copy()

        origins = (
            horizon_data[
                "origin_date"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        print()
        print(
            f"[INFO] H{horizon}: "
            f"{len(origins)} origins"
        )

        for origin_date in origins:
            current = horizon_data[
                horizon_data[
                    "origin_date"
                ]
                == origin_date
            ].copy()

            if current.empty:
                continue

            prior = get_prior_performance(
                df=df,
                horizon=horizon,
                origin_date=origin_date,
            )

            # The first three origins do not yet
            # have enough historical observations.
            if prior.empty:
                continue

            actual = float(
                current[
                    "actual"
                ].iloc[0]
            )

            origin_price = float(
                current[
                    "origin_price"
                ].iloc[0]
            )

            target_date = (
                current[
                    "target_date"
                ].iloc[0]
            )

            for top_n in TOP_N_VALUES:
                for weighted in [
                    False,
                    True,
                ]:
                    method = (
                        "weighted"
                        if weighted
                        else "simple"
                    )

                    model_name = (
                        f"top{top_n}_"
                        f"{method}_ensemble"
                    )

                    (
                        prediction,
                        selected,
                    ) = make_prediction(
                        current=current,
                        prior_performance=prior,
                        top_n=top_n,
                        weighted=weighted,
                    )

                    if np.isnan(
                        prediction
                    ):
                        continue

                    prediction_rows.append(
                        {
                            "model":
                                model_name,
                            "horizon":
                                horizon,
                            "origin_date":
                                origin_date,
                            "target_date":
                                target_date,
                            "origin_price":
                                origin_price,
                            "actual":
                                actual,
                            "prediction":
                                prediction,
                            "selected_expert_count":
                                len(
                                    selected
                                ),
                        }
                    )

                    for _, row in (
                        selected.iterrows()
                    ):
                        selection_rows.append(
                            {
                                "model":
                                    model_name,
                                "horizon":
                                    horizon,
                                "origin_date":
                                    origin_date,
                                "expert":
                                    row[
                                        "expert"
                                    ],
                                "prior_mae":
                                    row[
                                        "prior_mae"
                                    ],
                                "prior_observations":
                                    row[
                                        "prior_observations"
                                    ],
                                "weight":
                                    row[
                                        "weight"
                                    ],
                            }
                        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise RuntimeError(
            "No Top-N ensemble predictions produced"
        )

    predictions[
        "error"
    ] = (
        predictions[
            "prediction"
        ]
        - predictions[
            "actual"
        ]
    )

    metric_rows = []

    for (
        model_name,
        horizon,
    ), group in predictions.groupby(
        [
            "model",
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
                    model_name,
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

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    pd.DataFrame(
        selection_rows
    ).to_csv(
        OUTPUT_SELECTIONS,
        index=False,
    )

    print()
    print("=" * 80)
    print("TOP-N ENSEMBLE RESULTS")
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
        OUTPUT_PREDICTIONS,
    )

    print(
        "[OK] Metrics:",
        OUTPUT_METRICS,
    )

    print(
        "[OK] Selections:",
        OUTPUT_SELECTIONS,
    )


if __name__ == "__main__":
    main()
