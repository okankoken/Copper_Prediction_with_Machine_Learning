from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import DIAGNOSTICS_DIR


BACKTEST_START = pd.Timestamp("2025-01-31")

HORIZONS = [
    1,
    3,
    6,
    12,
]

MIN_HISTORY_FOR_WEIGHTING = 3

EPSILON = 1e-8


MODEL_FILES = {
    "naive": (
        DIAGNOSTICS_DIR
        / "naive_backtest_predictions.csv"
    ),
    "arima": (
        DIAGNOSTICS_DIR
        / "arima_backtest_predictions.csv"
    ),
    "sarimax": (
        DIAGNOSTICS_DIR
        / "sarimax_backtest_predictions.csv"
    ),
    "lightgbm_level": (
        DIAGNOSTICS_DIR
        / "lightgbm_backtest_predictions.csv"
    ),
    "linear_models": (
        DIAGNOSTICS_DIR
        / "linear_models_ridge_elasticnet_backtest_predictions.csv"
    ),
    "xgboost_return": (
        DIAGNOSTICS_DIR
        / "xgboost_return_backtest_predictions.csv"
    ),
    "lightgbm_return": (
        DIAGNOSTICS_DIR
        / "lightgbm_return_backtest_predictions.csv"
    ),
}


OUTPUT_PREDICTIONS = (
    DIAGNOSTICS_DIR
    / "ensemble_backtest_predictions.csv"
)

OUTPUT_METRICS = (
    DIAGNOSTICS_DIR
    / "ensemble_backtest_metrics.csv"
)

OUTPUT_WEIGHTS = (
    DIAGNOSTICS_DIR
    / "ensemble_dynamic_weights.csv"
)

OUTPUT_EXPERTS = (
    DIAGNOSTICS_DIR
    / "ensemble_expert_predictions.csv"
)


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


def standardize_model_name(
    value,
    source_name,
):
    if pd.isna(value):
        return source_name

    value = str(value).strip()

    if not value:
        return source_name

    return value


def build_expert_name(
    model_name,
    training_window,
):
    if pd.isna(training_window):
        return model_name

    try:
        window = int(
            float(
                training_window
            )
        )

        return (
            f"{model_name}_{window}y"
        )

    except Exception:
        return (
            f"{model_name}_"
            f"{training_window}"
        )


def load_prediction_file(
    source_name,
    path,
):
    path = Path(path)

    if not path.exists():
        print(
            f"[WARN] Missing file: {path}"
        )
        return pd.DataFrame()

    df = pd.read_csv(
        path
    )

    required = {
        "horizon",
        "origin_date",
        "actual",
        "prediction",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        print(
            f"[WARN] Skipping {path.name}. "
            f"Missing columns: "
            f"{sorted(missing)}"
        )
        return pd.DataFrame()

    df["origin_date"] = pd.to_datetime(
        df["origin_date"]
    )

    if "target_date" in df.columns:
        df["target_date"] = pd.to_datetime(
            df["target_date"]
        )
    else:
        df["target_date"] = pd.NaT

    if "origin_price" not in df.columns:
        print(
            f"[WARN] {path.name} has no "
            "origin_price column. "
            "Direction metrics may be unavailable."
        )

        df["origin_price"] = np.nan

    if "model" not in df.columns:
        df["model"] = source_name

    df["model"] = df["model"].apply(
        lambda value:
        standardize_model_name(
            value,
            source_name,
        )
    )

    if (
        "training_window_years"
        not in df.columns
    ):
        df[
            "training_window_years"
        ] = np.nan

    df["expert"] = [
        build_expert_name(
            model_name,
            window,
        )
        for model_name, window
        in zip(
            df["model"],
            df[
                "training_window_years"
            ],
        )
    ]

    df = df[
        df["origin_date"]
        >= BACKTEST_START
    ].copy()

    df = df[
        df["horizon"].isin(
            HORIZONS
        )
    ].copy()

    output_columns = [
        "expert",
        "model",
        "training_window_years",
        "horizon",
        "origin_date",
        "target_date",
        "origin_price",
        "actual",
        "prediction",
    ]

    df = df[
        output_columns
    ].copy()

    df["absolute_error"] = np.abs(
        df["prediction"]
        - df["actual"]
    )

    print(
        f"[OK] Loaded {path.name}: "
        f"{len(df)} rows, "
        f"{df['expert'].nunique()} experts"
    )

    return df


def load_all_predictions():
    frames = []

    for (
        source_name,
        path,
    ) in MODEL_FILES.items():
        current = (
            load_prediction_file(
                source_name,
                path,
            )
        )

        if not current.empty:
            frames.append(
                current
            )

    if not frames:
        raise RuntimeError(
            "No model prediction files "
            "could be loaded"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_mask = (
        combined.duplicated(
            subset=[
                "expert",
                "horizon",
                "origin_date",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():
        duplicates = combined[
            duplicate_mask
        ].sort_values(
            [
                "expert",
                "horizon",
                "origin_date",
            ]
        )

        print()
        print(
            "[WARN] Duplicate expert predictions "
            "found. Averaging duplicates."
        )

        combined = (
            combined.groupby(
                [
                    "expert",
                    "model",
                    "training_window_years",
                    "horizon",
                    "origin_date",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                target_date=(
                    "target_date",
                    "first",
                ),
                origin_price=(
                    "origin_price",
                    "first",
                ),
                actual=(
                    "actual",
                    "first",
                ),
                prediction=(
                    "prediction",
                    "mean",
                ),
            )
        )

        combined[
            "absolute_error"
        ] = np.abs(
            combined[
                "prediction"
            ]
            - combined[
                "actual"
            ]
        )

    return (
        combined.sort_values(
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


def get_prior_mae(
    all_predictions,
    expert,
    horizon,
    current_origin,
):
    history = all_predictions[
        (
            all_predictions[
                "expert"
            ]
            == expert
        )
        & (
            all_predictions[
                "horizon"
            ]
            == horizon
        )
        & (
            all_predictions[
                "origin_date"
            ]
            < current_origin
        )
    ]

    if (
        len(history)
        < MIN_HISTORY_FOR_WEIGHTING
    ):
        return np.nan

    return float(
        history[
            "absolute_error"
        ].mean()
    )


def calculate_dynamic_weights(
    current_predictions,
    all_predictions,
    horizon,
    origin_date,
):
    weight_data = []

    for _, row in (
        current_predictions.iterrows()
    ):
        prior_mae = get_prior_mae(
            all_predictions=all_predictions,
            expert=row["expert"],
            horizon=horizon,
            current_origin=origin_date,
        )

        if np.isnan(
            prior_mae
        ):
            raw_weight = np.nan
        else:
            raw_weight = (
                1.0
                / (
                    prior_mae
                    + EPSILON
                )
            )

        weight_data.append(
            {
                "expert":
                    row["expert"],
                "prior_mae":
                    prior_mae,
                "raw_weight":
                    raw_weight,
            }
        )

    weights = pd.DataFrame(
        weight_data
    )

    valid_history = (
        weights[
            "raw_weight"
        ]
        .notna()
    )

    if not valid_history.any():
        weights[
            "normalized_weight"
        ] = (
            1.0
            / len(weights)
        )

        weights[
            "weight_method"
        ] = "equal_warmup"

        return weights

    # Experts without enough prior history
    # are excluded from weighted prediction
    # once historical performance exists.
    weights.loc[
        ~valid_history,
        "raw_weight",
    ] = 0.0

    total_weight = (
        weights[
            "raw_weight"
        ].sum()
    )

    if total_weight <= 0:
        weights[
            "normalized_weight"
        ] = (
            1.0
            / len(weights)
        )

        weights[
            "weight_method"
        ] = "equal_fallback"

        return weights

    weights[
        "normalized_weight"
    ] = (
        weights[
            "raw_weight"
        ]
        / total_weight
    )

    weights[
        "weight_method"
    ] = "inverse_prior_mae"

    return weights


def main():
    print("=" * 80)
    print("ENSEMBLE WALK-FORWARD BACKTEST")
    print("=" * 80)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_predictions = (
        load_all_predictions()
    )

    all_predictions.to_csv(
        OUTPUT_EXPERTS,
        index=False,
    )

    print()
    print(
        "[INFO] Total experts:",
        all_predictions[
            "expert"
        ].nunique(),
    )

    print(
        "[INFO] Total expert predictions:",
        len(
            all_predictions
        ),
    )

    ensemble_rows = []
    weight_rows = []

    for horizon in HORIZONS:
        horizon_data = (
            all_predictions[
                all_predictions[
                    "horizon"
                ]
                == horizon
            ]
            .copy()
        )

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

            actual_values = (
                current[
                    "actual"
                ]
                .dropna()
                .unique()
            )

            if (
                len(actual_values)
                != 1
            ):
                raise RuntimeError(
                    "Models disagree on actual value "
                    f"for H{horizon} "
                    f"at {origin_date}"
                )

            actual = float(
                actual_values[0]
            )

            origin_prices = (
                current[
                    "origin_price"
                ]
                .dropna()
                .unique()
            )

            if len(
                origin_prices
            ) == 0:
                origin_price = np.nan
            else:
                origin_price = float(
                    origin_prices[0]
                )

            target_dates = (
                current[
                    "target_date"
                ]
                .dropna()
                .unique()
            )

            if len(
                target_dates
            ) == 0:
                target_date = pd.NaT
            else:
                target_date = (
                    target_dates[0]
                )

            simple_prediction = float(
                current[
                    "prediction"
                ].mean()
            )

            ensemble_rows.append(
                {
                    "model":
                        "simple_ensemble",
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
                        simple_prediction,
                    "expert_count":
                        len(current),
                }
            )

            weights = (
                calculate_dynamic_weights(
                    current_predictions=current,
                    all_predictions=all_predictions,
                    horizon=horizon,
                    origin_date=origin_date,
                )
            )

            current_weighted = (
                current[
                    [
                        "expert",
                        "prediction",
                    ]
                ]
                .merge(
                    weights,
                    on="expert",
                    how="left",
                )
            )

            weighted_prediction = float(
                np.sum(
                    current_weighted[
                        "prediction"
                    ]
                    * current_weighted[
                        "normalized_weight"
                    ]
                )
            )

            ensemble_rows.append(
                {
                    "model":
                        "weighted_ensemble",
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
                        weighted_prediction,
                    "expert_count":
                        len(current),
                }
            )

            for _, weight_row in (
                current_weighted.iterrows()
            ):
                weight_rows.append(
                    {
                        "horizon":
                            horizon,
                        "origin_date":
                            origin_date,
                        "expert":
                            weight_row[
                                "expert"
                            ],
                        "prior_mae":
                            weight_row[
                                "prior_mae"
                            ],
                        "weight":
                            weight_row[
                                "normalized_weight"
                            ],
                        "weight_method":
                            weight_row[
                                "weight_method"
                            ],
                    }
                )

    ensemble = pd.DataFrame(
        ensemble_rows
    )

    ensemble[
        "error"
    ] = (
        ensemble[
            "prediction"
        ]
        - ensemble[
            "actual"
        ]
    )

    metric_rows = []

    for (
        model_name,
        horizon,
    ), group in ensemble.groupby(
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

        if np.isnan(
            origin
        ).any():
            direction = np.nan
        else:
            direction = (
                directional_accuracy(
                    origin,
                    actual,
                    predicted,
                )
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
                    direction,
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

    ensemble.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    pd.DataFrame(
        weight_rows
    ).to_csv(
        OUTPUT_WEIGHTS,
        index=False,
    )

    print()
    print("=" * 80)
    print("ENSEMBLE RESULTS")
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
        "[OK] Dynamic weights:",
        OUTPUT_WEIGHTS,
    )

    print(
        "[OK] Expert predictions:",
        OUTPUT_EXPERTS,
    )


if __name__ == "__main__":
    main()
