import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.utils.paths import (
    DIAGNOSTICS_DIR,
    FEATURES_DIR,
    MONTHLY_DIR,
)


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

ARCHIVE_DIR = (
    PREDICTIONS_DIR
    / "archive"
)

LATEST_FORECAST_FILE = (
    PREDICTIONS_DIR
    / "copper_production_forecast_latest.csv"
)

LATEST_MODEL_DETAIL_FILE = (
    PREDICTIONS_DIR
    / "copper_production_model_details_latest.csv"
)

LATEST_RUN_MANIFEST_FILE = (
    PREDICTIONS_DIR
    / "production_run_manifest_latest.json"
)

RECONCILED_METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "reconciled_curve_backtest_metrics.csv"
)

RECONCILED_SEGMENTS_FILE = (
    DIAGNOSTICS_DIR
    / "reconciled_curve_segment_comparison.csv"
)

H1_BLEND_METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "h1_blend_backtest_metrics.csv"
)


# =====================================================================
# MLFLOW
# =====================================================================

EXPERIMENT_NAME = (
    "Copper_Prediction_with_Machine_Learning"
)

DEFAULT_TRACKING_URI = (
    "http://127.0.0.1:5000"
)


# =====================================================================
# PRODUCTION POLICY
# =====================================================================

PRODUCTION_DIRECT_WEIGHTS = {
    1: 1.00,
    2: 0.50,
    3: 1.00,
    4: 0.00,
    5: 0.00,
    6: 1.00,
    7: 0.00,
    8: 0.00,
    9: 0.00,
    10: 0.00,
    11: 0.00,
    12: 1.00,
}

ANCHOR_HORIZONS = {
    1,
    3,
    6,
    12,
}


# =====================================================================
# FILE HELPERS
# =====================================================================

def sha256_file(
    file_path,
):
    hash_object = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file_object:
        for chunk in iter(
            lambda: file_object.read(
                1024 * 1024
            ),
            b"",
        ):
            hash_object.update(
                chunk
            )

    return hash_object.hexdigest()


def get_file_metadata(
    file_path,
):
    if file_path is None:
        return None

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        return None

    return {
        "path":
            str(
                file_path
            ),
        "file_name":
            file_path.name,
        "size_bytes":
            int(
                file_path.stat().st_size
            ),
        "sha256":
            sha256_file(
                file_path
            ),
    }


def parse_snapshot_timestamp(
    file_path,
):
    timestamp_text = (
        Path(
            file_path
        )
        .stem
        .rsplit(
            "_",
            1,
        )[-1]
    )

    try:
        return pd.to_datetime(
            timestamp_text,
            format="%Y%m%dT%H%M%S",
        )

    except ValueError:
        return pd.NaT


def find_latest_snapshot(
    directory,
    base_name,
    origin_tag,
    not_after,
):
    directory = Path(
        directory
    )

    pattern = (
        f"{base_name}_"
        f"{origin_tag}_"
        "*.csv"
    )

    candidates = []

    for file_path in directory.glob(
        pattern
    ):
        timestamp = (
            parse_snapshot_timestamp(
                file_path
            )
        )

        if pd.isna(
            timestamp
        ):
            continue

        if timestamp <= not_after:
            candidates.append(
                (
                    timestamp,
                    file_path,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[-1][1]


# =====================================================================
# METRIC HELPERS
# =====================================================================

def log_metric_if_valid(
    key,
    value,
):
    if value is None:
        return

    if pd.isna(
        value
    ):
        return

    mlflow.log_metric(
        key,
        float(
            value
        ),
    )


def load_h1_production_metrics():
    if not H1_BLEND_METRICS_FILE.exists():
        return None

    metrics = pd.read_csv(
        H1_BLEND_METRICS_FILE
    )

    selected = metrics[
        np.isclose(
            metrics[
                "elasticnet_weight"
            ].astype(float),
            0.50,
        )
        & np.isclose(
            metrics[
                "naive_weight"
            ].astype(float),
            0.50,
        )
    ].copy()

    if selected.empty:
        return None

    return selected.iloc[0]


def load_reconciled_metrics():
    if not RECONCILED_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Reconciled metrics not found: "
            f"{RECONCILED_METRICS_FILE}"
        )

    metrics = pd.read_csv(
        RECONCILED_METRICS_FILE
    )

    metrics[
        "horizon"
    ] = (
        metrics[
            "horizon"
        ]
        .astype(int)
    )

    return metrics


def get_production_backtest_metrics(
    horizon,
    forecast_row,
    reconciled_metrics,
    h1_metrics,
):
    # H1 uses the separately backtested Naive + ElasticNet blend.
    if (
        horizon == 1
        and h1_metrics is not None
    ):
        return {
            "mape_pct":
                float(
                    h1_metrics[
                        "mape_pct"
                    ]
                ),
            "rmse":
                float(
                    h1_metrics[
                        "rmse"
                    ]
                ),
            "bias":
                float(
                    h1_metrics[
                        "bias"
                    ]
                ),
            "directional_accuracy_pct":
                float(
                    h1_metrics[
                        "directional_accuracy_pct"
                    ]
                ),
            "observations":
                int(
                    h1_metrics[
                        "observations"
                    ]
                ),
            "source":
                "h1_blend_backtest",
        }

    direct_weight = (
        PRODUCTION_DIRECT_WEIGHTS[
            horizon
        ]
    )

    selected = reconciled_metrics[
        (
            reconciled_metrics[
                "horizon"
            ]
            == horizon
        )
        & np.isclose(
            reconciled_metrics[
                "direct_weight"
            ].astype(float),
            direct_weight,
        )
    ].copy()

    if not selected.empty:
        row = selected.iloc[0]

        return {
            "mape_pct":
                float(
                    row[
                        "mape_pct"
                    ]
                ),
            "rmse":
                float(
                    row[
                        "rmse"
                    ]
                ),
            "bias":
                float(
                    row[
                        "bias"
                    ]
                ),
            "directional_accuracy_pct":
                float(
                    row[
                        "directional_accuracy_pct"
                    ]
                ),
            "observations":
                int(
                    row[
                        "observations"
                    ]
                ),
            "source":
                "reconciled_curve_backtest",
        }

    # Safe fallback to original winner metrics.
    return {
        "mape_pct":
            float(
                forecast_row[
                    "backtest_mape_pct"
                ]
            ),
        "rmse":
            float(
                forecast_row[
                    "backtest_rmse"
                ]
            ),
        "bias":
            float(
                forecast_row[
                    "backtest_bias"
                ]
            ),
        "directional_accuracy_pct":
            float(
                forecast_row[
                    "backtest_directional_accuracy_pct"
                ]
            ),
        "observations":
            int(
                forecast_row[
                    "backtest_observations"
                ]
            ),
        "source":
            "raw_winner_fallback",
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(
        "=" * 100
    )

    print(
        "LOG FINAL COPPER PRODUCTION RUN TO MLFLOW"
    )

    print(
        "=" * 100
    )

    if not LATEST_FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast not found: "
            f"{LATEST_FORECAST_FILE}"
        )

    if not LATEST_MODEL_DETAIL_FILE.exists():
        raise FileNotFoundError(
            f"Model details not found: "
            f"{LATEST_MODEL_DETAIL_FILE}"
        )

    forecast = pd.read_csv(
        LATEST_FORECAST_FILE,
        parse_dates=[
            "generated_at",
            "forecast_origin",
            "forecast_date",
        ],
    )

    model_details = pd.read_csv(
        LATEST_MODEL_DETAIL_FILE,
        parse_dates=[
            "generated_at",
            "forecast_origin",
            "forecast_date",
        ],
    )

    forecast = (
        forecast
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        forecast
    ) != 12:
        raise RuntimeError(
            f"Expected 12 horizons, got "
            f"{len(forecast)}"
        )

    forecast_origin = pd.Timestamp(
        forecast[
            "forecast_origin"
        ].iloc[0]
    )

    generated_at = pd.Timestamp(
        forecast[
            "generated_at"
        ].iloc[0]
    )

    origin_tag = (
        forecast_origin.strftime(
            "%Y-%m-%d"
        )
    )

    run_timestamp = (
        generated_at.strftime(
            "%Y%m%dT%H%M%S"
        )
    )

    policy_version = str(
        forecast[
            "production_policy_version"
        ].iloc[0]
    )

    origin_price = float(
        forecast[
            "origin_price_usd_per_ton"
        ].iloc[0]
    )

    h1_metrics = (
        load_h1_production_metrics()
    )

    reconciled_metrics = (
        load_reconciled_metrics()
    )

    # =================================================================
    # INPUT SNAPSHOTS
    # =================================================================

    master_snapshot = (
        find_latest_snapshot(
            MONTHLY_DIR
            / "archive",
            "copper_monthly_master",
            origin_tag,
            generated_at,
        )
    )

    model_features_snapshot = (
        find_latest_snapshot(
            FEATURES_DIR
            / "archive",
            "copper_model_features",
            origin_tag,
            generated_at,
        )
    )

    multi_horizon_snapshot = (
        find_latest_snapshot(
            FEATURES_DIR
            / "archive",
            "copper_multi_horizon_features",
            origin_tag,
            generated_at,
        )
    )

    feature_manifest_snapshot = (
        find_latest_snapshot(
            FEATURES_DIR
            / "archive",
            "copper_model_feature_manifest",
            origin_tag,
            generated_at,
        )
    )

    input_snapshots = {
        "monthly_master":
            get_file_metadata(
                master_snapshot
            ),
        "model_features":
            get_file_metadata(
                model_features_snapshot
            ),
        "multi_horizon_features":
            get_file_metadata(
                multi_horizon_snapshot
            ),
        "feature_manifest":
            get_file_metadata(
                feature_manifest_snapshot
            ),
    }

    # =================================================================
    # PRODUCTION METRICS TABLE
    # =================================================================

    production_metric_rows = []

    for _, row in forecast.iterrows():
        horizon = int(
            row[
                "horizon"
            ]
        )

        production_metrics = (
            get_production_backtest_metrics(
                horizon=
                    horizon,
                forecast_row=
                    row,
                reconciled_metrics=
                    reconciled_metrics,
                h1_metrics=
                    h1_metrics,
            )
        )

        production_metric_rows.append(
            {
                "horizon":
                    horizon,
                "production_expert":
                    row[
                        "production_expert"
                    ],
                "forecast_date":
                    row[
                        "forecast_date"
                    ],
                "final_forecast_usd_per_ton":
                    float(
                        row[
                            "final_forecast_price_usd_per_ton"
                        ]
                    ),
                "raw_forecast_usd_per_ton":
                    float(
                        row[
                            "raw_forecast_price_usd_per_ton"
                        ]
                    ),
                "production_backtest_mape_pct":
                    production_metrics[
                        "mape_pct"
                    ],
                "production_backtest_rmse":
                    production_metrics[
                        "rmse"
                    ],
                "production_backtest_bias":
                    production_metrics[
                        "bias"
                    ],
                "production_directional_accuracy_pct":
                    production_metrics[
                        "directional_accuracy_pct"
                    ],
                "production_backtest_observations":
                    production_metrics[
                        "observations"
                    ],
                "production_metric_source":
                    production_metrics[
                        "source"
                    ],
                "raw_winner_backtest_mape_pct":
                    float(
                        row[
                            "backtest_mape_pct"
                        ]
                    ),
                "raw_winner_backtest_rmse":
                    float(
                        row[
                            "backtest_rmse"
                        ]
                    ),
                "mom_change_pct":
                    float(
                        row[
                            "month_over_month_change_pct"
                        ]
                    ),
                "change_pct_from_origin":
                    float(
                        row[
                            "change_pct_from_origin"
                        ]
                    ),
                "mom_direction":
                    row[
                        "mom_direction"
                    ],
                "direction_vs_origin":
                    row[
                        "direction_vs_origin"
                    ],
                "confidence_flag":
                    row[
                        "confidence_flag"
                    ],
            }
        )

    production_metrics_df = pd.DataFrame(
        production_metric_rows
    )

    # =================================================================
    # MANIFEST
    # =================================================================

    manifest = {
        "project":
            "Copper_Prediction_with_Machine_Learning",
        "forecast_origin":
            origin_tag,
        "generated_at":
            generated_at.isoformat(),
        "production_policy_version":
            policy_version,
        "origin_price_usd_per_ton":
            origin_price,
        "input_snapshots":
            input_snapshots,
    }

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_manifest_file = (
        ARCHIVE_DIR
        / (
            "production_run_manifest_"
            f"{origin_tag}_"
            f"{run_timestamp}.json"
        )
    )

    with open(
        archive_manifest_file,
        "w",
        encoding="utf-8",
    ) as file_object:
        json.dump(
            manifest,
            file_object,
            indent=2,
            ensure_ascii=True,
        )

    with open(
        LATEST_RUN_MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as file_object:
        json.dump(
            manifest,
            file_object,
            indent=2,
            ensure_ascii=True,
        )

    production_metrics_file = (
        PREDICTIONS_DIR
        / "copper_production_metrics_latest.csv"
    )

    production_metrics_archive_file = (
        ARCHIVE_DIR
        / (
            "copper_production_metrics_"
            f"{origin_tag}_"
            f"{run_timestamp}.csv"
        )
    )

    production_metrics_df.to_csv(
        production_metrics_file,
        index=False,
    )

    production_metrics_df.to_csv(
        production_metrics_archive_file,
        index=False,
    )

    # =================================================================
    # MLFLOW CONNECTION
    # =================================================================

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    run_name = (
        f"production_"
        f"{origin_tag}_"
        f"{run_timestamp}"
    )

    print()

    print(
        "[INFO] Tracking URI:",
        tracking_uri,
    )

    print(
        "[INFO] Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "[INFO] Run:",
        run_name,
    )

    # =================================================================
    # MLFLOW RUN
    # =================================================================

    with mlflow.start_run(
        run_name=run_name
    ) as run:
        mlflow.set_tags(
            {
                "project":
                    "Copper_Prediction_with_Machine_Learning",
                "run_type":
                    "monthly_production_forecast",
                "forecast_origin":
                    origin_tag,
                "production_policy_version":
                    policy_version,
                "frequency":
                    "monthly",
                "status":
                    "production",
            }
        )

        mlflow.log_params(
            {
                "forecast_origin":
                    origin_tag,
                "production_policy_version":
                    policy_version,
                "origin_price_usd_per_ton":
                    round(
                        origin_price,
                        4,
                    ),
                "forecast_horizons":
                    "H1-H12",
                "anchor_horizons":
                    "H1,H3,H6,H12",
                "h1_policy":
                    "50pct_naive_50pct_elasticnet_8y",
                "h2_direct_weight":
                    0.50,
                "h4_h5_direct_weight":
                    0.00,
                "h7_h11_direct_weight":
                    0.00,
            }
        )

        # =============================================================
        # HORIZON METRICS
        # =============================================================

        for _, row in (
            production_metrics_df.iterrows()
        ):
            horizon = int(
                row[
                    "horizon"
                ]
            )

            prefix = (
                f"h{horizon:02d}"
            )

            log_metric_if_valid(
                f"{prefix}_final_forecast_usd_per_ton",
                row[
                    "final_forecast_usd_per_ton"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_production_backtest_mape_pct",
                row[
                    "production_backtest_mape_pct"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_production_backtest_rmse",
                row[
                    "production_backtest_rmse"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_production_backtest_bias",
                row[
                    "production_backtest_bias"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_production_directional_accuracy_pct",
                row[
                    "production_directional_accuracy_pct"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_raw_winner_backtest_mape_pct",
                row[
                    "raw_winner_backtest_mape_pct"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_mom_change_pct",
                row[
                    "mom_change_pct"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_change_pct_from_origin",
                row[
                    "change_pct_from_origin"
                ],
            )

        # =============================================================
        # EXECUTIVE / SUMMARY METRICS
        # =============================================================

        anchor_df = (
            production_metrics_df[
                production_metrics_df[
                    "horizon"
                ]
                .isin(
                    ANCHOR_HORIZONS
                )
            ]
        )

        log_metric_if_valid(
            "production_mean_mape_pct",
            production_metrics_df[
                "production_backtest_mape_pct"
            ].mean(),
        )

        log_metric_if_valid(
            "anchor_mean_mape_pct",
            anchor_df[
                "production_backtest_mape_pct"
            ].mean(),
        )

        log_metric_if_valid(
            "production_mean_directional_accuracy_pct",
            production_metrics_df[
                "production_directional_accuracy_pct"
            ].mean(),
        )

        log_metric_if_valid(
            "max_abs_forecast_monthly_change_pct",
            production_metrics_df[
                "mom_change_pct"
            ]
            .abs()
            .max(),
        )

        log_metric_if_valid(
            "forecast_min_price_usd_per_ton",
            production_metrics_df[
                "final_forecast_usd_per_ton"
            ].min(),
        )

        log_metric_if_valid(
            "forecast_max_price_usd_per_ton",
            production_metrics_df[
                "final_forecast_usd_per_ton"
            ].max(),
        )

        min_row = (
            production_metrics_df
            .sort_values(
                "final_forecast_usd_per_ton"
            )
            .iloc[0]
        )

        mlflow.log_param(
            "forecast_min_price_horizon",
            f"H{int(min_row['horizon'])}",
        )

        # =============================================================
        # IMPORTANT ANCHOR METRICS
        # =============================================================

        for horizon in [
            1,
            3,
            6,
            12,
        ]:
            row = (
                production_metrics_df[
                    production_metrics_df[
                        "horizon"
                    ]
                    == horizon
                ]
                .iloc[0]
            )

            prefix = (
                f"anchor_h{horizon:02d}"
            )

            log_metric_if_valid(
                f"{prefix}_forecast",
                row[
                    "final_forecast_usd_per_ton"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_mape_pct",
                row[
                    "production_backtest_mape_pct"
                ],
            )

            log_metric_if_valid(
                f"{prefix}_directional_accuracy_pct",
                row[
                    "production_directional_accuracy_pct"
                ],
            )

        # =============================================================
        # ARTIFACTS
        # =============================================================

        mlflow.log_artifact(
            str(
                LATEST_FORECAST_FILE
            ),
            artifact_path="production",
        )

        mlflow.log_artifact(
            str(
                LATEST_MODEL_DETAIL_FILE
            ),
            artifact_path="production",
        )

        mlflow.log_artifact(
            str(
                production_metrics_file
            ),
            artifact_path="monitoring",
        )

        mlflow.log_artifact(
            str(
                archive_manifest_file
            ),
            artifact_path="reproducibility",
        )

        if RECONCILED_SEGMENTS_FILE.exists():
            mlflow.log_artifact(
                str(
                    RECONCILED_SEGMENTS_FILE
                ),
                artifact_path="validation",
            )

        if H1_BLEND_METRICS_FILE.exists():
            mlflow.log_artifact(
                str(
                    H1_BLEND_METRICS_FILE
                ),
                artifact_path="validation",
            )

        print()

        print(
            "[OK] MLflow run ID:",
            run.info.run_id,
        )

    # =================================================================
    # TERMINAL SUMMARY
    # =================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "PRODUCTION MODEL MONITORING SUMMARY"
    )

    print(
        "=" * 100
    )

    print()

    display_columns = [
        "horizon",
        "production_expert",
        "final_forecast_usd_per_ton",
        "production_backtest_mape_pct",
        "production_directional_accuracy_pct",
        "mom_change_pct",
        "change_pct_from_origin",
        "confidence_flag",
    ]

    print(
        production_metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "[INFO] Input snapshots:"
    )

    for name, metadata in (
        input_snapshots.items()
    ):
        if metadata is None:
            print(
                f"       {name}: NOT FOUND"
            )
        else:
            print(
                f"       {name}: "
                f"{metadata['file_name']}"
            )

    print()

    print(
        "[OK] Production metrics:",
        production_metrics_file,
    )

    print(
        "[OK] Archived metrics:",
        production_metrics_archive_file,
    )

    print(
        "[OK] Run manifest:",
        archive_manifest_file,
    )


if __name__ == "__main__":
    main()
