import numpy as np
import pandas as pd

from src.utils.paths import DIAGNOSTICS_DIR


HORIZONS = list(range(1, 13))

ANCHORS = [1, 3, 6, 12]

DIRECT_WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


PREDICTION_FILES = [
    DIAGNOSTICS_DIR
    / "naive_backtest_predictions.csv",

    DIAGNOSTICS_DIR
    / "arima_backtest_predictions.csv",

    DIAGNOSTICS_DIR
    / "sarimax_backtest_predictions.csv",

    DIAGNOSTICS_DIR
    / "linear_models_ridge_elasticnet_backtest_predictions.csv",
]


WINNER_FILE = (
    DIAGNOSTICS_DIR
    / "model_horizon_winners_h1_h12.csv"
)


OUTPUT_PREDICTIONS = (
    DIAGNOSTICS_DIR
    / "reconciled_curve_backtest_predictions.csv"
)

OUTPUT_METRICS = (
    DIAGNOSTICS_DIR
    / "reconciled_curve_backtest_metrics.csv"
)

OUTPUT_SEGMENTS = (
    DIAGNOSTICS_DIR
    / "reconciled_curve_segment_comparison.csv"
)


def build_expert_name(row):
    model = str(
        row["model"]
    )

    if (
        "training_window_years"
        in row.index
        and pd.notna(
            row["training_window_years"]
        )
    ):
        window = int(
            float(
                row["training_window_years"]
            )
        )

        return (
            f"{model}_{window}y"
        )

    return model


def load_predictions():
    frames = []

    for path in PREDICTION_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {path}"
            )

        df = pd.read_csv(
            path
        )

        df["origin_date"] = pd.to_datetime(
            df["origin_date"]
        )

        df["target_date"] = pd.to_datetime(
            df["target_date"]
        )

        if (
            "training_window_years"
            not in df.columns
        ):
            df[
                "training_window_years"
            ] = np.nan

        df["expert"] = df.apply(
            build_expert_name,
            axis=1,
        )

        frames.append(
            df[
                [
                    "expert",
                    "horizon",
                    "origin_date",
                    "target_date",
                    "origin_price",
                    "actual",
                    "prediction",
                ]
            ].copy()
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_winners():
    if not WINNER_FILE.exists():
        raise FileNotFoundError(
            f"Winner file not found: {WINNER_FILE}"
        )

    winners = pd.read_csv(
        WINNER_FILE
    )

    winners["horizon"] = (
        winners["horizon"]
        .astype(int)
    )

    return winners[
        [
            "horizon",
            "expert",
        ]
    ].copy()


def build_winner_predictions(
    predictions,
    winners,
):
    rows = []

    for _, winner in winners.iterrows():
        horizon = int(
            winner["horizon"]
        )

        expert = str(
            winner["expert"]
        )

        current = predictions[
            (
                predictions["horizon"]
                == horizon
            )
            & (
                predictions["expert"]
                == expert
            )
        ].copy()

        if current.empty:
            raise RuntimeError(
                f"No predictions found for "
                f"H{horizon} / {expert}"
            )

        rows.append(
            current
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def get_anchor_pair(
    horizon,
):
    if horizon == 2:
        return 1, 3

    if horizon in {
        4,
        5,
    }:
        return 3, 6

    if horizon in {
        7,
        8,
        9,
        10,
        11,
    }:
        return 6, 12

    return None


def log_interpolate(
    left_price,
    right_price,
    left_horizon,
    right_horizon,
    horizon,
):
    if (
        left_price <= 0
        or right_price <= 0
    ):
        raise ValueError(
            "Prices must be positive for "
            "log interpolation"
        )

    fraction = (
        horizon
        - left_horizon
    ) / (
        right_horizon
        - left_horizon
    )

    log_price = (
        np.log(
            left_price
        )
        + fraction
        * (
            np.log(
                right_price
            )
            - np.log(
                left_price
            )
        )
    )

    return float(
        np.exp(
            log_price
        )
    )


def mape(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

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


def rmse(
    actual,
    predicted,
):
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


def bias(
    actual,
    predicted,
):
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


def directional_accuracy(
    origin,
    actual,
    predicted,
):
    origin = np.asarray(
        origin,
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

    return float(
        np.mean(
            np.sign(
                actual
                - origin
            )
            == np.sign(
                predicted
                - origin
            )
        )
        * 100.0
    )


def monthly_return_error(
    left_actual,
    right_actual,
    left_prediction,
    right_prediction,
):
    actual_return = (
        np.log(
            right_actual
            / left_actual
        )
        * 100.0
    )

    predicted_return = (
        np.log(
            right_prediction
            / left_prediction
        )
        * 100.0
    )

    return abs(
        predicted_return
        - actual_return
    )


def main():
    print("=" * 100)
    print("BACKTEST RECONCILED H1-H12 FORECAST CURVE")
    print("=" * 100)

    predictions = (
        load_predictions()
    )

    winners = (
        load_winners()
    )

    winner_predictions = (
        build_winner_predictions(
            predictions,
            winners,
        )
    )

    lookup = {}

    for _, row in winner_predictions.iterrows():
        key = (
            pd.Timestamp(
                row["origin_date"]
            ),
            int(
                row["horizon"]
            ),
        )

        lookup[key] = row

    output_rows = []

    for horizon in HORIZONS:
        current = winner_predictions[
            winner_predictions[
                "horizon"
            ]
            == horizon
        ].copy()

        if horizon in ANCHORS:
            for _, row in current.iterrows():
                for direct_weight in DIRECT_WEIGHTS:
                    output_rows.append(
                        {
                            "origin_date":
                                row[
                                    "origin_date"
                                ],
                            "target_date":
                                row[
                                    "target_date"
                                ],
                            "horizon":
                                horizon,
                            "expert":
                                row[
                                    "expert"
                                ],
                            "direct_weight":
                                direct_weight,
                            "origin_price":
                                row[
                                    "origin_price"
                                ],
                            "actual":
                                row[
                                    "actual"
                                ],
                            "direct_prediction":
                                row[
                                    "prediction"
                                ],
                            "anchor_prediction":
                                row[
                                    "prediction"
                                ],
                            "reconciled_prediction":
                                row[
                                    "prediction"
                                ],
                            "anchor_left":
                                horizon,
                            "anchor_right":
                                horizon,
                        }
                    )

            continue

        anchor_pair = get_anchor_pair(
            horizon
        )

        if anchor_pair is None:
            continue

        left_horizon, right_horizon = (
            anchor_pair
        )

        for _, row in current.iterrows():
            origin_date = pd.Timestamp(
                row["origin_date"]
            )

            left_key = (
                origin_date,
                left_horizon,
            )

            right_key = (
                origin_date,
                right_horizon,
            )

            if (
                left_key not in lookup
                or right_key not in lookup
            ):
                continue

            left_row = lookup[
                left_key
            ]

            right_row = lookup[
                right_key
            ]

            anchor_prediction = (
                log_interpolate(
                    left_price=float(
                        left_row[
                            "prediction"
                        ]
                    ),
                    right_price=float(
                        right_row[
                            "prediction"
                        ]
                    ),
                    left_horizon=
                        left_horizon,
                    right_horizon=
                        right_horizon,
                    horizon=horizon,
                )
            )

            direct_prediction = float(
                row[
                    "prediction"
                ]
            )

            for direct_weight in (
                DIRECT_WEIGHTS
            ):
                reconciled_prediction = (
                    direct_weight
                    * direct_prediction
                    + (
                        1.0
                        - direct_weight
                    )
                    * anchor_prediction
                )

                output_rows.append(
                    {
                        "origin_date":
                            row[
                                "origin_date"
                            ],
                        "target_date":
                            row[
                                "target_date"
                            ],
                        "horizon":
                            horizon,
                        "expert":
                            row[
                                "expert"
                            ],
                        "direct_weight":
                            direct_weight,
                        "origin_price":
                            row[
                                "origin_price"
                            ],
                        "actual":
                            row[
                                "actual"
                            ],
                        "direct_prediction":
                            direct_prediction,
                        "anchor_prediction":
                            anchor_prediction,
                        "reconciled_prediction":
                            reconciled_prediction,
                        "anchor_left":
                            left_horizon,
                        "anchor_right":
                            right_horizon,
                    }
                )

    result = pd.DataFrame(
        output_rows
    )

    result = (
        result.sort_values(
            [
                "direct_weight",
                "origin_date",
                "horizon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    metric_rows = []

    for (
        horizon,
        direct_weight,
    ), group in result.groupby(
        [
            "horizon",
            "direct_weight",
        ]
    ):
        actual = group[
            "actual"
        ].to_numpy(
            dtype=float
        )

        predicted = group[
            "reconciled_prediction"
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
                "horizon":
                    int(
                        horizon
                    ),
                "direct_weight":
                    float(
                        direct_weight
                    ),
                "observations":
                    len(
                        group
                    ),
                "mape_pct":
                    mape(
                        actual,
                        predicted,
                    ),
                "rmse":
                    rmse(
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
    )

    metrics = (
        metrics.sort_values(
            [
                "horizon",
                "mape_pct",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    segment_definitions = {
        "short_h1_h3":
            [
                1,
                2,
                3,
            ],
        "medium_h3_h6":
            [
                3,
                4,
                5,
                6,
            ],
        "long_h6_h12":
            [
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            ],
    }

    segment_rows = []

    for (
        segment_name,
        segment_horizons,
    ) in segment_definitions.items():
        for direct_weight in DIRECT_WEIGHTS:
            current = result[
                (
                    result[
                        "direct_weight"
                    ]
                    == direct_weight
                )
                & (
                    result[
                        "horizon"
                    ]
                    .isin(
                        segment_horizons
                    )
                )
            ].copy()

            origin_counts = (
                current
                .groupby(
                    "origin_date"
                )[
                    "horizon"
                ]
                .nunique()
            )

            complete_origins = (
                origin_counts[
                    origin_counts
                    == len(
                        segment_horizons
                    )
                ]
                .index
            )

            current = current[
                current[
                    "origin_date"
                ]
                .isin(
                    complete_origins
                )
            ].copy()

            if current.empty:
                continue

            actual = current[
                "actual"
            ].to_numpy(
                dtype=float
            )

            predicted = current[
                "reconciled_prediction"
            ].to_numpy(
                dtype=float
            )

            return_errors = []
            max_jumps = []

            for origin_date, curve in (
                current.groupby(
                    "origin_date"
                )
            ):
                curve = (
                    curve.sort_values(
                        "horizon"
                    )
                )

                actual_values = (
                    curve[
                        "actual"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                prediction_values = (
                    curve[
                        "reconciled_prediction"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                for index in range(
                    1,
                    len(curve),
                ):
                    return_errors.append(
                        monthly_return_error(
                            left_actual=
                                actual_values[
                                    index - 1
                                ],
                            right_actual=
                                actual_values[
                                    index
                                ],
                            left_prediction=
                                prediction_values[
                                    index - 1
                                ],
                            right_prediction=
                                prediction_values[
                                    index
                                ],
                        )
                    )

                predicted_returns = (
                    np.diff(
                        np.log(
                            prediction_values
                        )
                    )
                    * 100.0
                )

                if len(
                    predicted_returns
                ) > 0:
                    max_jumps.append(
                        float(
                            np.max(
                                np.abs(
                                    predicted_returns
                                )
                            )
                        )
                    )

            segment_rows.append(
                {
                    "segment":
                        segment_name,
                    "direct_weight":
                        direct_weight,
                    "complete_origins":
                        len(
                            complete_origins
                        ),
                    "observations":
                        len(
                            current
                        ),
                    "curve_mape_pct":
                        mape(
                            actual,
                            predicted,
                        ),
                    "curve_rmse":
                        rmse(
                            actual,
                            predicted,
                        ),
                    "monthly_return_mae_pct":
                        float(
                            np.mean(
                                return_errors
                            )
                        )
                        if return_errors
                        else np.nan,
                    "mean_max_monthly_jump_pct":
                        float(
                            np.mean(
                                max_jumps
                            )
                        )
                        if max_jumps
                        else np.nan,
                }
            )

    segment_results = (
        pd.DataFrame(
            segment_rows
        )
        .sort_values(
            [
                "segment",
                "direct_weight",
            ]
        )
    )

    segment_results.to_csv(
        OUTPUT_SEGMENTS,
        index=False,
    )

    print()
    print("=" * 100)
    print("BEST RECONCILIATION WEIGHT BY HORIZON")
    print("=" * 100)
    print()

    best_by_horizon = (
        metrics
        .sort_values(
            [
                "horizon",
                "mape_pct",
                "rmse",
            ]
        )
        .groupby(
            "horizon",
            as_index=False,
        )
        .first()
    )

    print(
        best_by_horizon.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("SEGMENT TRADE-OFF")
    print("=" * 100)
    print()

    print(
        segment_results.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print()
    print(
        "direct_weight=1.00 -> original independent horizon forecast"
    )

    print(
        "direct_weight=0.00 -> fully anchor-reconciled forecast"
    )

    print(
        "Values between 0 and 1 blend direct signal with the anchor curve."
    )

    print()
    print(
        "[OK] Predictions:",
        OUTPUT_PREDICTIONS,
    )

    print(
        "[OK] Horizon metrics:",
        OUTPUT_METRICS,
    )

    print(
        "[OK] Segment comparison:",
        OUTPUT_SEGMENTS,
    )


if __name__ == "__main__":
    main()
