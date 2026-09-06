import numpy as np
import pandas as pd

from src.utils.paths import DIAGNOSTICS_DIR


HORIZONS = list(
    range(1, 13)
)


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


OUTPUT_FILE = (
    DIAGNOSTICS_DIR
    / "model_common_period_comparison_h1_h12.csv"
)

WINNER_FILE = (
    DIAGNOSTICS_DIR
    / "model_horizon_winners_h1_h12.csv"
)


def build_name(
    row,
):
    model = str(
        row["model"]
    )

    if (
        "training_window_years"
        in row.index
        and pd.notna(
            row[
                "training_window_years"
            ]
        )
    ):
        window = int(
            float(
                row[
                    "training_window_years"
                ]
            )
        )

        return (
            f"{model}_{window}y"
        )

    return model


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
    return float(
        np.mean(
            np.sign(
                actual - origin
            )
            == np.sign(
                predicted - origin
            )
        )
        * 100.0
    )


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

        df[
            "origin_date"
        ] = pd.to_datetime(
            df[
                "origin_date"
            ]
        )

        if (
            "training_window_years"
            not in df.columns
        ):
            df[
                "training_window_years"
            ] = np.nan

        df[
            "expert"
        ] = df.apply(
            build_name,
            axis=1,
        )

        frames.append(
            df[
                [
                    "expert",
                    "model",
                    "training_window_years",
                    "horizon",
                    "origin_date",
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


def main():
    print("=" * 100)
    print("H1-H12 COMMON-PERIOD MODEL COMPARISON")
    print("=" * 100)

    all_predictions = (
        load_predictions()
    )

    result_rows = []

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

        experts = sorted(
            horizon_data[
                "expert"
            ]
            .unique()
        )

        origin_sets = []

        for expert in experts:
            origins = set(
                horizon_data.loc[
                    horizon_data[
                        "expert"
                    ]
                    == expert,
                    "origin_date",
                ]
            )

            origin_sets.append(
                origins
            )

        if not origin_sets:
            continue

        common_origins = sorted(
            set.intersection(
                *origin_sets
            )
        )

        print(
            f"[INFO] H{horizon:02d}: "
            f"{len(common_origins)} "
            "common origins"
        )

        for expert in experts:
            current = horizon_data[
                (
                    horizon_data[
                        "expert"
                    ]
                    == expert
                )
                & (
                    horizon_data[
                        "origin_date"
                    ]
                    .isin(
                        common_origins
                    )
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
                "prediction"
            ].to_numpy(
                dtype=float
            )

            origin = current[
                "origin_price"
            ].to_numpy(
                dtype=float
            )

            result_rows.append(
                {
                    "horizon":
                        horizon,
                    "expert":
                        expert,
                    "observations":
                        len(current),
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

    result = (
        pd.DataFrame(
            result_rows
        )
        .sort_values(
            [
                "horizon",
                "mape_pct",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    winners = (
        result
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

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    winners.to_csv(
        WINNER_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("H1-H12 WINNERS")
    print("=" * 100)
    print()

    print(
        winners[
            [
                "horizon",
                "expert",
                "observations",
                "mae",
                "rmse",
                "mape_pct",
                "smape_pct",
                "bias",
                "directional_accuracy_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("TOP 5 PER HORIZON")
    print("=" * 100)

    for horizon in HORIZONS:
        print()
        print(
            f"H{horizon:02d}"
        )
        print("-" * 100)

        current = (
            result[
                result[
                    "horizon"
                ]
                == horizon
            ]
            .head(5)
        )

        print(
            current.to_string(
                index=False
            )
        )

    print()
    print(
        "[OK] Full comparison:",
        OUTPUT_FILE,
    )

    print(
        "[OK] Horizon winners:",
        WINNER_FILE,
    )


if __name__ == "__main__":
    main()
