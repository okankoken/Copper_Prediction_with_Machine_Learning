import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.features.time_aware_feature_selector import select_features
from src.utils.paths import (
    DIAGNOSTICS_DIR,
    FEATURES_DIR,
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

INPUT_FILE = (
    FEATURES_DIR
    / "copper_multi_horizon_features.csv"
)

WINNER_FILE = (
    DIAGNOSTICS_DIR
    / "model_horizon_winners_h1_h12.csv"
)

H1_BLEND_METRICS_FILE = (
    DIAGNOSTICS_DIR
    / "h1_blend_backtest_metrics.csv"
)

LATEST_FORECAST_FILE = (
    PREDICTIONS_DIR
    / "copper_production_forecast_latest.csv"
)

LATEST_MODEL_DETAIL_FILE = (
    PREDICTIONS_DIR
    / "copper_production_model_details_latest.csv"
)


# =====================================================================
# PRODUCTION POLICY
# =====================================================================

PRODUCTION_POLICY_VERSION = (
    "v1_reconciled_h1_blend"
)

HORIZONS = list(
    range(1, 13)
)

ANCHORS = {
    1,
    3,
    6,
    12,
}

ORIGIN_PRICE_COLUMN = (
    "origin_copper_price_usd_per_ton"
)


# ---------------------------------------------------------------------
# H1 policy
#
# Backtested:
#
# 50% Naive
# 50% ElasticNet 8Y
#
# H1 blend backtest:
# MAPE ~3.3304%
# RMSE ~451.98
# Directional Accuracy ~63.16%
# ---------------------------------------------------------------------

H1_SIGNAL_MODEL = (
    "elasticnet"
)

H1_SIGNAL_WINDOW_YEARS = (
    8
)

H1_ELASTICNET_WEIGHT = (
    0.50
)

H1_NAIVE_WEIGHT = (
    0.50
)


# ---------------------------------------------------------------------
# Curve reconciliation policy
#
# H1, H3, H6, H12:
# anchor horizons
#
# H2:
# 50% direct + 50% H1-H3 anchor path
#
# H4-H5:
# fully reconciled between H3-H6
#
# H7-H11:
# fully reconciled between H6-H12
# ---------------------------------------------------------------------

DIRECT_WEIGHT_BY_HORIZON = {
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


# =====================================================================
# MODEL SEARCH GRIDS
# =====================================================================

RIDGE_ALPHA_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

ELASTICNET_ALPHA_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]

ELASTICNET_L1_RATIO_GRID = [
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
]

SARIMAX_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (0, 1, 2),
]


# =====================================================================
# HELPERS
# =====================================================================

def parse_expert_name(
    expert,
):
    expert = str(
        expert
    ).strip()

    if expert == "naive":
        return (
            "naive",
            None,
        )

    if expert.endswith(
        "y"
    ):
        base, window = expert.rsplit(
            "_",
            1,
        )

        if window[:-1].isdigit():
            return (
                base,
                int(
                    window[:-1]
                ),
            )

    raise ValueError(
        f"Unsupported expert name: {expert}"
    )


def remove_other_targets(
    frame,
    current_target,
):
    drop_columns = []

    for horizon in HORIZONS:
        target_column = (
            f"target_h{horizon}"
        )

        target_date_column = (
            f"target_date_h{horizon}"
        )

        if (
            target_column
            != current_target
            and target_column
            in frame.columns
        ):
            drop_columns.append(
                target_column
            )

        if (
            target_date_column
            in frame.columns
        ):
            drop_columns.append(
                target_date_column
            )

    return frame.drop(
        columns=drop_columns,
        errors="ignore",
    )


def prepare_training(
    df,
    forecast_origin,
    horizon,
    window_years,
):
    target_column = (
        f"target_h{horizon}"
    )

    target_date_column = (
        f"target_date_h{horizon}"
    )

    window_start = (
        forecast_origin
        - pd.DateOffset(
            years=window_years
        )
    )

    training = df[
        (
            df["date"]
            >= window_start
        )
        & (
            df["date"]
            < forecast_origin
        )
        & (
            df[
                target_date_column
            ]
            <= forecast_origin
        )
        & (
            df[
                target_column
            ]
            .notna()
        )
    ].copy()

    return training


def get_cv_splits(
    training_rows,
):
    if training_rows >= 90:
        return 5

    if training_rows >= 60:
        return 4

    return 3


# =====================================================================
# LINEAR MODELS
# =====================================================================

def build_ridge_search(
    training_rows,
):
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha":
                RIDGE_ALPHA_GRID,
        },
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(
            n_splits=get_cv_splits(
                training_rows
            )
        ),
        n_jobs=-1,
        refit=True,
    )


def build_elasticnet_search(
    training_rows,
):
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                ElasticNet(
                    max_iter=50000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha":
                ELASTICNET_ALPHA_GRID,
            "model__l1_ratio":
                ELASTICNET_L1_RATIO_GRID,
        },
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(
            n_splits=get_cv_splits(
                training_rows
            )
        ),
        n_jobs=-1,
        refit=True,
    )


def select_linear_features(
    training,
    target_column,
):
    selector_data = (
        remove_other_targets(
            training,
            target_column,
        )
    )

    (
        selected_features,
        ranking,
        decisions,
    ) = select_features(
        selector_data,
        target_column=target_column,
    )

    return [
        feature
        for feature
        in selected_features
        if (
            feature
            in training.columns
            and not feature.startswith(
                "target_"
            )
        )
    ]


def predict_linear(
    df,
    production_row,
    forecast_origin,
    horizon,
    model_name,
    window_years,
):
    target_column = (
        f"target_h{horizon}"
    )

    training = prepare_training(
        df=df,
        forecast_origin=forecast_origin,
        horizon=horizon,
        window_years=window_years,
    )

    if len(
        training
    ) < 36:
        raise RuntimeError(
            f"Insufficient training rows for "
            f"{model_name} H{horizon}: "
            f"{len(training)}"
        )

    selected_features = (
        select_linear_features(
            training,
            target_column,
        )
    )

    if not selected_features:
        raise RuntimeError(
            f"No features selected for "
            f"{model_name} H{horizon}"
        )

    X_train = (
        training[
            selected_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(
            float
        )
    )

    y_train = (
        training[
            target_column
        ]
        .astype(
            float
        )
    )

    X_production = (
        production_row[
            selected_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(
            float
        )
    )

    if model_name == "ridge":
        search = (
            build_ridge_search(
                len(
                    training
                )
            )
        )

    elif model_name == "elasticnet":
        search = (
            build_elasticnet_search(
                len(
                    training
                )
            )
        )

    else:
        raise ValueError(
            f"Unsupported linear model: "
            f"{model_name}"
        )

    search.fit(
        X_train,
        y_train,
    )

    prediction = float(
        search.predict(
            X_production
        )[0]
    )

    details = {
        "training_rows":
            len(
                training
            ),
        "selected_feature_count":
            len(
                selected_features
            ),
        "selected_features":
            "|".join(
                selected_features
            ),
        "best_params":
            str(
                search.best_params_
            ),
        "sarimax_order":
            None,
        "aic":
            None,
    }

    return (
        prediction,
        details,
    )


# =====================================================================
# SARIMAX
# =====================================================================

def get_max_exog(
    window_years,
):
    if window_years == 5:
        return 3

    if window_years == 8:
        return 4

    return 5


def select_sarimax_features(
    training,
    target_column,
    max_exog,
):
    selector_data = (
        remove_other_targets(
            training,
            target_column,
        )
    )

    (
        selected_features,
        ranking,
        decisions,
    ) = select_features(
        selector_data,
        target_column=target_column,
    )

    selected_features = [
        feature
        for feature
        in selected_features
        if (
            feature
            in training.columns
            and not feature.startswith(
                "target_"
            )
            and feature
            != ORIGIN_PRICE_COLUMN
        )
    ]

    return selected_features[
        :max_exog
    ]


def impute_sarimax_exog(
    X_train,
    X_forecast,
):
    X_train = (
        X_train
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(
            float
        )
    )

    X_forecast = (
        X_forecast
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype(
            float
        )
    )

    medians = (
        X_train
        .median(
            axis=0
        )
    )

    valid_columns = (
        medians[
            medians.notna()
        ]
        .index
        .tolist()
    )

    if not valid_columns:
        raise RuntimeError(
            "No valid SARIMAX exogenous columns"
        )

    X_train = (
        X_train[
            valid_columns
        ]
    )

    X_forecast = (
        X_forecast[
            valid_columns
        ]
    )

    medians = (
        medians[
            valid_columns
        ]
    )

    X_train = (
        X_train.fillna(
            medians
        )
    )

    X_forecast = (
        X_forecast.fillna(
            medians
        )
    )

    return (
        X_train,
        X_forecast,
        valid_columns,
    )


def fit_best_sarimax(
    y_train,
    X_train,
):
    best_result = None
    best_order = None
    best_aic = np.inf

    y_array = np.asarray(
        y_train,
        dtype=float,
    )

    X_array = np.asarray(
        X_train,
        dtype=float,
    )

    for order in SARIMAX_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                model = SARIMAX(
                    endog=y_array,
                    exog=X_array,
                    order=order,
                    seasonal_order=(
                        0,
                        0,
                        0,
                        0,
                    ),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                result = model.fit(
                    disp=False,
                    maxiter=200,
                )

            if (
                np.isfinite(
                    result.aic
                )
                and result.aic
                < best_aic
            ):
                best_result = (
                    result
                )

                best_order = (
                    order
                )

                best_aic = float(
                    result.aic
                )

        except Exception:
            continue

    if best_result is None:
        raise RuntimeError(
            "No SARIMAX candidate could be fitted"
        )

    return (
        best_result,
        best_order,
        best_aic,
    )


def predict_sarimax(
    df,
    forecast_origin,
    horizon,
    window_years,
):
    target_column = (
        f"target_h{horizon}"
    )

    training = prepare_training(
        df=df,
        forecast_origin=forecast_origin,
        horizon=horizon,
        window_years=window_years,
    )

    if len(
        training
    ) < 36:
        raise RuntimeError(
            f"Insufficient SARIMAX training rows "
            f"for H{horizon}: "
            f"{len(training)}"
        )

    selected_exog = (
        select_sarimax_features(
            training=training,
            target_column=target_column,
            max_exog=get_max_exog(
                window_years
            ),
        )
    )

    if not selected_exog:
        raise RuntimeError(
            f"No SARIMAX features selected "
            f"for H{horizon}"
        )

    last_training_origin = (
        training[
            "date"
        ]
        .max()
    )

    forecast_path = df[
        (
            df[
                "date"
            ]
            > last_training_origin
        )
        & (
            df[
                "date"
            ]
            <= forecast_origin
        )
    ].copy()

    if (
        len(
            forecast_path
        )
        != horizon
    ):
        raise RuntimeError(
            f"Unexpected SARIMAX exog path "
            f"for H{horizon}. "
            f"Expected={horizon}, "
            f"actual={len(forecast_path)}"
        )

    y_train = (
        training[
            target_column
        ]
        .astype(
            float
        )
    )

    X_train = (
        training[
            selected_exog
        ]
        .copy()
    )

    X_forecast = (
        forecast_path[
            selected_exog
        ]
        .copy()
    )

    (
        X_train,
        X_forecast,
        selected_exog,
    ) = impute_sarimax_exog(
        X_train,
        X_forecast,
    )

    (
        model_result,
        best_order,
        best_aic,
    ) = fit_best_sarimax(
        y_train,
        X_train,
    )

    forecast = (
        model_result
        .get_forecast(
            steps=horizon,
            exog=np.asarray(
                X_forecast,
                dtype=float,
            ),
        )
        .predicted_mean
    )

    prediction = float(
        np.asarray(
            forecast
        )[-1]
    )

    details = {
        "training_rows":
            len(
                training
            ),
        "selected_feature_count":
            len(
                selected_exog
            ),
        "selected_features":
            "|".join(
                selected_exog
            ),
        "best_params":
            None,
        "sarimax_order":
            str(
                best_order
            ),
        "aic":
            best_aic,
    }

    return (
        prediction,
        details,
    )


# =====================================================================
# RECONCILIATION
# =====================================================================

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
            "Anchor prices must be positive"
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


def get_anchor_pair(
    horizon,
):
    if horizon == 2:
        return (
            1,
            3,
        )

    if horizon in {
        4,
        5,
    }:
        return (
            3,
            6,
        )

    if horizon in {
        7,
        8,
        9,
        10,
        11,
    }:
        return (
            6,
            12,
        )

    return None


def apply_reconciliation(
    raw_forecast_df,
):
    raw_lookup = {
        int(
            row[
                "horizon"
            ]
        ): float(
            row[
                "raw_forecast_price_usd_per_ton"
            ]
        )
        for _, row
        in raw_forecast_df.iterrows()
    }

    rows = []

    for _, row in (
        raw_forecast_df.iterrows()
    ):
        horizon = int(
            row[
                "horizon"
            ]
        )

        raw_prediction = float(
            row[
                "raw_forecast_price_usd_per_ton"
            ]
        )

        direct_weight = (
            DIRECT_WEIGHT_BY_HORIZON[
                horizon
            ]
        )

        if horizon in ANCHORS:
            anchor_prediction = (
                raw_prediction
            )

            final_prediction = (
                raw_prediction
            )

            anchor_left = (
                horizon
            )

            anchor_right = (
                horizon
            )

        else:
            anchor_pair = (
                get_anchor_pair(
                    horizon
                )
            )

            if anchor_pair is None:
                raise RuntimeError(
                    f"No anchor pair for H{horizon}"
                )

            (
                anchor_left,
                anchor_right,
            ) = anchor_pair

            anchor_prediction = (
                log_interpolate(
                    left_price=
                        raw_lookup[
                            anchor_left
                        ],
                    right_price=
                        raw_lookup[
                            anchor_right
                        ],
                    left_horizon=
                        anchor_left,
                    right_horizon=
                        anchor_right,
                    horizon=
                        horizon,
                )
            )

            final_prediction = (
                direct_weight
                * raw_prediction
                + (
                    1.0
                    - direct_weight
                )
                * anchor_prediction
            )

        updated = (
            row.to_dict()
        )

        updated[
            "anchor_forecast_price_usd_per_ton"
        ] = (
            anchor_prediction
        )

        updated[
            "direct_weight"
        ] = (
            direct_weight
        )

        updated[
            "anchor_left_horizon"
        ] = (
            anchor_left
        )

        updated[
            "anchor_right_horizon"
        ] = (
            anchor_right
        )

        updated[
            "final_forecast_price_usd_per_ton"
        ] = (
            final_prediction
        )

        rows.append(
            updated
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# FINAL CURVE METRICS
# =====================================================================

def add_final_curve_metrics(
    forecast_df,
    origin_price,
):
    forecast_df = (
        forecast_df
        .sort_values(
            "horizon"
        )
        .reset_index(
            drop=True
        )
    )

    forecast_df[
        "change_usd_from_origin"
    ] = (
        forecast_df[
            "final_forecast_price_usd_per_ton"
        ]
        - origin_price
    )

    forecast_df[
        "change_pct_from_origin"
    ] = (
        (
            forecast_df[
                "final_forecast_price_usd_per_ton"
            ]
            / origin_price
        )
        - 1.0
    ) * 100.0

    month_over_month = []

    previous_price = (
        origin_price
    )

    for _, row in (
        forecast_df.iterrows()
    ):
        current_price = float(
            row[
                "final_forecast_price_usd_per_ton"
            ]
        )

        monthly_change = (
            (
                current_price
                / previous_price
            )
            - 1.0
        ) * 100.0

        month_over_month.append(
            monthly_change
        )

        previous_price = (
            current_price
        )

    forecast_df[
        "month_over_month_change_pct"
    ] = (
        month_over_month
    )

    mom_directions = []

    for value in (
        forecast_df[
            "month_over_month_change_pct"
        ]
    ):
        if value > 0.25:
            mom_directions.append(
                "UP"
            )

        elif value < -0.25:
            mom_directions.append(
                "DOWN"
            )

        else:
            mom_directions.append(
                "FLAT"
            )

    forecast_df[
        "mom_direction"
    ] = (
        mom_directions
    )

    origin_directions = []

    for value in (
        forecast_df[
            "change_pct_from_origin"
        ]
    ):
        if value > 0.25:
            origin_directions.append(
                "UP"
            )

        elif value < -0.25:
            origin_directions.append(
                "DOWN"
            )

        else:
            origin_directions.append(
                "FLAT"
            )

    forecast_df[
        "direction_vs_origin"
    ] = (
        origin_directions
    )

    return forecast_df


# =====================================================================
# H1 BACKTEST METRICS
# =====================================================================

def load_h1_blend_metrics():
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
            H1_ELASTICNET_WEIGHT,
        )
        & np.isclose(
            metrics[
                "naive_weight"
            ].astype(float),
            H1_NAIVE_WEIGHT,
        )
    ].copy()

    if selected.empty:
        return None

    return selected.iloc[0]


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(
        "=" * 100
    )

    print(
        "COPPER H1-H12 FINAL RECONCILED PRODUCTION FORECAST"
    )

    print(
        "=" * 100
    )

    start_time = (
        time.time()
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    if not WINNER_FILE.exists():
        raise FileNotFoundError(
            f"Winner file not found: "
            f"{WINNER_FILE}"
        )

    parse_date_columns = [
        "date",
    ] + [
        f"target_date_h{horizon}"
        for horizon
        in HORIZONS
    ]

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=parse_date_columns,
    )

    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    winners = pd.read_csv(
        WINNER_FILE
    )

    production_row = (
        df.iloc[
            [-1]
        ]
        .copy()
    )

    forecast_origin = (
        production_row[
            "date"
        ].iloc[0]
    )

    origin_price = float(
        production_row[
            ORIGIN_PRICE_COLUMN
        ].iloc[0]
    )

    generated_at = (
        pd.Timestamp.now()
    )

    run_timestamp = (
        generated_at.strftime(
            "%Y%m%dT%H%M%S"
        )
    )

    origin_tag = (
        forecast_origin.strftime(
            "%Y-%m-%d"
        )
    )

    archive_forecast_file = (
        ARCHIVE_DIR
        / (
            "copper_production_forecast_"
            f"{origin_tag}_"
            f"{run_timestamp}.csv"
        )
    )

    archive_model_detail_file = (
        ARCHIVE_DIR
        / (
            "copper_production_model_details_"
            f"{origin_tag}_"
            f"{run_timestamp}.csv"
        )
    )

    if archive_forecast_file.exists():
        raise FileExistsError(
            f"Archive forecast already exists: "
            f"{archive_forecast_file}"
        )

    if archive_model_detail_file.exists():
        raise FileExistsError(
            f"Archive model detail already exists: "
            f"{archive_model_detail_file}"
        )

    h1_blend_metrics = (
        load_h1_blend_metrics()
    )

    print()

    print(
        "[INFO] Production policy:",
        PRODUCTION_POLICY_VERSION,
    )

    print(
        "[INFO] Production origin:",
        forecast_origin.strftime(
            "%Y-%m-%d"
        ),
    )

    print(
        "[INFO] Origin copper price:",
        f"{origin_price:,.2f}",
    )

    print(
        "[INFO] Generated at:",
        generated_at,
    )

    print()

    raw_rows = []

    detail_rows = []

    for horizon in HORIZONS:
        winner = (
            winners[
                winners[
                    "horizon"
                ]
                .astype(
                    int
                )
                == horizon
            ]
            .iloc[0]
        )

        expert = str(
            winner[
                "expert"
            ]
        )

        (
            model_name,
            window_years,
        ) = parse_expert_name(
            expert
        )

        target_date = (
            production_row[
                f"target_date_h{horizon}"
            ]
            .iloc[0]
        )

        print(
            f"[INFO] H{horizon:02d} "
            f"| target={target_date.date()} "
            f"| winner={expert}"
        )

        # -------------------------------------------------------------
        # H1 production blend
        # -------------------------------------------------------------

        if horizon == 1:
            naive_prediction = (
                origin_price
            )

            (
                elasticnet_prediction,
                elasticnet_details,
            ) = predict_linear(
                df=df,
                production_row=
                    production_row,
                forecast_origin=
                    forecast_origin,
                horizon=1,
                model_name=
                    H1_SIGNAL_MODEL,
                window_years=
                    H1_SIGNAL_WINDOW_YEARS,
            )

            prediction = (
                H1_NAIVE_WEIGHT
                * naive_prediction
                + H1_ELASTICNET_WEIGHT
                * elasticnet_prediction
            )

            production_expert = (
                "h1_blend_naive_elasticnet_8y"
            )

            details = {
                "training_rows":
                    elasticnet_details[
                        "training_rows"
                    ],
                "selected_feature_count":
                    elasticnet_details[
                        "selected_feature_count"
                    ],
                "selected_features":
                    elasticnet_details[
                        "selected_features"
                    ],
                "best_params":
                    elasticnet_details[
                        "best_params"
                    ],
                "sarimax_order":
                    None,
                "aic":
                    None,
                "naive_prediction":
                    naive_prediction,
                "signal_prediction":
                    elasticnet_prediction,
                "signal_model":
                    "elasticnet_8y",
            }

            if h1_blend_metrics is not None:
                backtest_observations = int(
                    h1_blend_metrics[
                        "observations"
                    ]
                )

                backtest_mape = float(
                    h1_blend_metrics[
                        "mape_pct"
                    ]
                )

                backtest_rmse = float(
                    h1_blend_metrics[
                        "rmse"
                    ]
                )

                backtest_da = float(
                    h1_blend_metrics[
                        "directional_accuracy_pct"
                    ]
                )

                backtest_bias = float(
                    h1_blend_metrics[
                        "bias"
                    ]
                )

            else:
                backtest_observations = int(
                    winner[
                        "observations"
                    ]
                )

                backtest_mape = float(
                    winner[
                        "mape_pct"
                    ]
                )

                backtest_rmse = float(
                    winner[
                        "rmse"
                    ]
                )

                backtest_da = float(
                    winner[
                        "directional_accuracy_pct"
                    ]
                )

                backtest_bias = float(
                    winner[
                        "bias"
                    ]
                )

        # -------------------------------------------------------------
        # Other linear models
        # -------------------------------------------------------------

        elif model_name in {
            "ridge",
            "elasticnet",
        }:
            (
                prediction,
                details,
            ) = predict_linear(
                df=df,
                production_row=
                    production_row,
                forecast_origin=
                    forecast_origin,
                horizon=
                    horizon,
                model_name=
                    model_name,
                window_years=
                    window_years,
            )

            details[
                "naive_prediction"
            ] = None

            details[
                "signal_prediction"
            ] = None

            details[
                "signal_model"
            ] = None

            production_expert = (
                expert
            )

            backtest_observations = int(
                winner[
                    "observations"
                ]
            )

            backtest_mape = float(
                winner[
                    "mape_pct"
                ]
            )

            backtest_rmse = float(
                winner[
                    "rmse"
                ]
            )

            backtest_da = float(
                winner[
                    "directional_accuracy_pct"
                ]
            )

            backtest_bias = float(
                winner[
                    "bias"
                ]
            )

        # -------------------------------------------------------------
        # SARIMAX
        # -------------------------------------------------------------

        elif model_name == "sarimax":
            (
                prediction,
                details,
            ) = predict_sarimax(
                df=df,
                forecast_origin=
                    forecast_origin,
                horizon=
                    horizon,
                window_years=
                    window_years,
            )

            details[
                "naive_prediction"
            ] = None

            details[
                "signal_prediction"
            ] = None

            details[
                "signal_model"
            ] = None

            production_expert = (
                expert
            )

            backtest_observations = int(
                winner[
                    "observations"
                ]
            )

            backtest_mape = float(
                winner[
                    "mape_pct"
                ]
            )

            backtest_rmse = float(
                winner[
                    "rmse"
                ]
            )

            backtest_da = float(
                winner[
                    "directional_accuracy_pct"
                ]
            )

            backtest_bias = float(
                winner[
                    "bias"
                ]
            )

        # -------------------------------------------------------------
        # Naive fallback
        # -------------------------------------------------------------

        elif model_name == "naive":
            prediction = (
                origin_price
            )

            production_expert = (
                expert
            )

            details = {
                "training_rows":
                    None,
                "selected_feature_count":
                    0,
                "selected_features":
                    None,
                "best_params":
                    None,
                "sarimax_order":
                    None,
                "aic":
                    None,
                "naive_prediction":
                    origin_price,
                "signal_prediction":
                    None,
                "signal_model":
                    None,
            }

            backtest_observations = int(
                winner[
                    "observations"
                ]
            )

            backtest_mape = float(
                winner[
                    "mape_pct"
                ]
            )

            backtest_rmse = float(
                winner[
                    "rmse"
                ]
            )

            backtest_da = float(
                winner[
                    "directional_accuracy_pct"
                ]
            )

            backtest_bias = float(
                winner[
                    "bias"
                ]
            )

        else:
            raise ValueError(
                f"Unsupported production model: "
                f"{model_name}"
            )

        confidence_flag = (
            "LOW"
            if backtest_observations < 10
            else "NORMAL"
        )

        raw_rows.append(
            {
                "generated_at":
                    generated_at,
                "production_policy_version":
                    PRODUCTION_POLICY_VERSION,
                "forecast_origin":
                    forecast_origin,
                "horizon":
                    horizon,
                "forecast_date":
                    target_date,
                "selected_backtest_winner":
                    expert,
                "production_expert":
                    production_expert,
                "model":
                    model_name,
                "training_window_years":
                    window_years,
                "origin_price_usd_per_ton":
                    origin_price,
                "raw_forecast_price_usd_per_ton":
                    prediction,
                "backtest_observations":
                    backtest_observations,
                "backtest_mape_pct":
                    backtest_mape,
                "backtest_rmse":
                    backtest_rmse,
                "backtest_bias":
                    backtest_bias,
                "backtest_directional_accuracy_pct":
                    backtest_da,
                "confidence_flag":
                    confidence_flag,
            }
        )

        detail_rows.append(
            {
                "generated_at":
                    generated_at,
                "production_policy_version":
                    PRODUCTION_POLICY_VERSION,
                "forecast_origin":
                    forecast_origin,
                "horizon":
                    horizon,
                "forecast_date":
                    target_date,
                "selected_backtest_winner":
                    expert,
                "production_expert":
                    production_expert,
                "model":
                    model_name,
                "training_window_years":
                    window_years,
                "training_rows":
                    details[
                        "training_rows"
                    ],
                "selected_feature_count":
                    details[
                        "selected_feature_count"
                    ],
                "selected_features":
                    details[
                        "selected_features"
                    ],
                "best_params":
                    details[
                        "best_params"
                    ],
                "sarimax_order":
                    details[
                        "sarimax_order"
                    ],
                "aic":
                    details[
                        "aic"
                    ],
                "h1_naive_prediction":
                    details.get(
                        "naive_prediction"
                    ),
                "h1_signal_prediction":
                    details.get(
                        "signal_prediction"
                    ),
                "h1_signal_model":
                    details.get(
                        "signal_model"
                    ),
                "h1_naive_weight":
                    (
                        H1_NAIVE_WEIGHT
                        if horizon == 1
                        else None
                    ),
                "h1_elasticnet_weight":
                    (
                        H1_ELASTICNET_WEIGHT
                        if horizon == 1
                        else None
                    ),
            }
        )

        print(
            f"       Raw forecast: "
            f"{prediction:,.2f} USD/t"
        )

        if horizon == 1:
            print(
                f"       H1 Naive: "
                f"{details['naive_prediction']:,.2f} USD/t"
            )

            print(
                f"       H1 ElasticNet 8Y: "
                f"{details['signal_prediction']:,.2f} USD/t"
            )

            print(
                f"       H1 Blend: "
                f"{prediction:,.2f} USD/t "
                f"(50% Naive + 50% ElasticNet)"
            )

    # =================================================================
    # RECONCILE RAW FORECASTS
    # =================================================================

    raw_forecast_df = pd.DataFrame(
        raw_rows
    )

    forecast_df = (
        apply_reconciliation(
            raw_forecast_df
        )
    )

    forecast_df = (
        add_final_curve_metrics(
            forecast_df,
            origin_price,
        )
    )

    details_df = pd.DataFrame(
        detail_rows
    )

    # =================================================================
    # SAVE IMMUTABLE ARCHIVE
    # =================================================================

    forecast_df.to_csv(
        archive_forecast_file,
        index=False,
    )

    details_df.to_csv(
        archive_model_detail_file,
        index=False,
    )

    # =================================================================
    # SAVE CURRENT OPERATIONAL VERSION
    #
    # These two files intentionally overwrite because Airflow,
    # Streamlit and downstream systems need one current version.
    #
    # Historical versions are preserved in archive/.
    # =================================================================

    forecast_df.to_csv(
        LATEST_FORECAST_FILE,
        index=False,
    )

    details_df.to_csv(
        LATEST_MODEL_DETAIL_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    # =================================================================
    # DISPLAY
    # =================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL RECONCILED PRODUCTION FORECAST"
    )

    print(
        "=" * 100
    )

    print()

    display_columns = [
        "horizon",
        "forecast_date",
        "production_expert",
        "raw_forecast_price_usd_per_ton",
        "anchor_forecast_price_usd_per_ton",
        "direct_weight",
        "final_forecast_price_usd_per_ton",
        "month_over_month_change_pct",
        "mom_direction",
        "change_pct_from_origin",
        "direction_vs_origin",
        "backtest_mape_pct",
        "confidence_flag",
    ]

    print(
        forecast_df[
            display_columns
        ].to_string(
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
        "[OK] Archive forecast:",
        archive_forecast_file,
    )

    print(
        "[OK] Latest forecast:",
        LATEST_FORECAST_FILE,
    )

    print(
        "[OK] Archive model details:",
        archive_model_detail_file,
    )

    print(
        "[OK] Latest model details:",
        LATEST_MODEL_DETAIL_FILE,
    )


if __name__ == "__main__":
    main()
