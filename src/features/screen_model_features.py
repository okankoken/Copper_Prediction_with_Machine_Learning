import hashlib

import numpy as np
import pandas as pd

from src.utils.paths import (
    DIAGNOSTICS_DIR,
    FEATURES_DIR,
)


FEATURE_FILE = (
    FEATURES_DIR
    / "copper_model_features.csv"
)

MANIFEST_FILE = (
    FEATURES_DIR
    / "copper_model_feature_manifest.csv"
)

SCREENING_FILE = (
    DIAGNOSTICS_DIR
    / "model_feature_screening.csv"
)

DUPLICATE_FILE = (
    DIAGNOSTICS_DIR
    / "model_feature_duplicate_pairs.csv"
)

SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "model_feature_screening_summary.csv"
)


TARGET_COLUMN = "cash_settlement_usd_per_ton"

MIN_GLOBAL_OBSERVATIONS = 36
HIGH_MISSING_RATIO = 0.50


def load_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {MANIFEST_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    return df, manifest


def normalize_numeric_series(series):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return numeric


def series_hash(series):
    values = normalize_numeric_series(
        series
    )

    normalized = (
        values
        .round(12)
        .fillna(
            -9.999999999999999e307
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()


def build_manifest_lookup(
    manifest,
):
    if "feature" not in manifest.columns:
        raise ValueError(
            "Manifest does not contain feature column"
        )

    return (
        manifest
        .drop_duplicates(
            subset=[
                "feature"
            ]
        )
        .set_index(
            "feature"
        )
    )


def build_feature_screening(
    df,
    manifest_lookup,
):
    rows = []

    for column in df.columns:
        if column in {
            "date",
            TARGET_COLUMN,
        }:
            continue

        series = normalize_numeric_series(
            df[column]
        )

        non_null_count = int(
            series.notna().sum()
        )

        missing_count = int(
            series.isna().sum()
        )

        missing_ratio = float(
            series.isna().mean()
        )

        unique_count = int(
            series.dropna().nunique()
        )

        variance = (
            float(
                series.var()
            )
            if non_null_count > 1
            else np.nan
        )

        first_valid_month = (
            df.loc[
                series.notna(),
                "date",
            ].min()
            if non_null_count > 0
            else pd.NaT
        )

        last_valid_month = (
            df.loc[
                series.notna(),
                "date",
            ].max()
            if non_null_count > 0
            else pd.NaT
        )

        if column in manifest_lookup.index:
            metadata = (
                manifest_lookup
                .loc[
                    column
                ]
            )

            raw_feature = metadata.get(
                "raw_feature",
                None,
            )

            category = metadata.get(
                "category",
                None,
            )

            source_frequency = metadata.get(
                "source_frequency",
                None,
            )

            transform = metadata.get(
                "transform",
                None,
            )

            lag_months = metadata.get(
                "lag_months",
                None,
            )

            leakage_safe = metadata.get(
                "leakage_safe",
                None,
            )

        else:
            raw_feature = None
            category = None
            source_frequency = None
            transform = None
            lag_months = None
            leakage_safe = None

        is_all_null = (
            non_null_count == 0
        )

        is_constant = (
            unique_count <= 1
            and non_null_count > 0
        )

        low_observation_count = (
            non_null_count
            < MIN_GLOBAL_OBSERVATIONS
        )

        high_missing = (
            missing_ratio
            > HIGH_MISSING_RATIO
        )

        if is_all_null:
            structural_status = "FAIL"

        elif is_constant:
            structural_status = "FAIL"

        elif low_observation_count:
            structural_status = "WARNING"

        elif high_missing:
            structural_status = "WARNING"

        else:
            structural_status = "PASS"

        rows.append(
            {
                "feature": column,
                "raw_feature": raw_feature,
                "category": category,
                "source_frequency": source_frequency,
                "transform": transform,
                "lag_months": lag_months,
                "leakage_safe": leakage_safe,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "unique_count": unique_count,
                "variance": variance,
                "first_valid_month": first_valid_month,
                "last_valid_month": last_valid_month,
                "is_all_null": is_all_null,
                "is_constant": is_constant,
                "low_observation_count": low_observation_count,
                "high_missing_ratio": high_missing,
                "structural_status": structural_status,
            }
        )

    return pd.DataFrame(
        rows
    )


def find_exact_duplicate_features(
    df,
):
    feature_columns = [
        column
        for column in df.columns
        if column not in {
            "date",
            TARGET_COLUMN,
        }
    ]

    hash_groups = {}

    for column in feature_columns:
        column_hash = series_hash(
            df[column]
        )

        hash_groups.setdefault(
            column_hash,
            []
        ).append(
            column
        )

    rows = []

    duplicate_group_id = 0

    for columns in hash_groups.values():
        if len(columns) <= 1:
            continue

        reference = columns[
            0
        ]

        reference_series = (
            normalize_numeric_series(
                df[
                    reference
                ]
            )
        )

        confirmed = [
            reference
        ]

        for candidate in columns[
            1:
        ]:
            candidate_series = (
                normalize_numeric_series(
                    df[
                        candidate
                    ]
                )
            )

            same = (
                reference_series.equals(
                    candidate_series
                )
            )

            if same:
                confirmed.append(
                    candidate
                )

        if len(confirmed) <= 1:
            continue

        duplicate_group_id += 1

        for duplicate in confirmed[
            1:
        ]:
            rows.append(
                {
                    "duplicate_group": duplicate_group_id,
                    "reference_feature": reference,
                    "duplicate_feature": duplicate,
                }
            )

    return pd.DataFrame(
        rows
    )


def build_summary(
    screening,
    duplicates,
):
    status_counts = (
        screening[
            "structural_status"
        ]
        .value_counts()
        .to_dict()
    )

    rows = [
        {
            "metric": "total_model_features",
            "value": len(
                screening
            ),
        },
        {
            "metric": "pass_features",
            "value": int(
                status_counts.get(
                    "PASS",
                    0,
                )
            ),
        },
        {
            "metric": "warning_features",
            "value": int(
                status_counts.get(
                    "WARNING",
                    0,
                )
            ),
        },
        {
            "metric": "fail_features",
            "value": int(
                status_counts.get(
                    "FAIL",
                    0,
                )
            ),
        },
        {
            "metric": "all_null_features",
            "value": int(
                screening[
                    "is_all_null"
                ].sum()
            ),
        },
        {
            "metric": "constant_features",
            "value": int(
                screening[
                    "is_constant"
                ].sum()
            ),
        },
        {
            "metric": "high_missing_features",
            "value": int(
                screening[
                    "high_missing_ratio"
                ].sum()
            ),
        },
        {
            "metric": "low_observation_features",
            "value": int(
                screening[
                    "low_observation_count"
                ].sum()
            ),
        },
        {
            "metric": "exact_duplicate_pairs",
            "value": len(
                duplicates
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def main():
    print(
        "=" * 80
    )

    print(
        "MODEL FEATURE STRUCTURAL SCREENING"
    )

    print(
        "=" * 80
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, manifest = load_data()

    manifest_lookup = (
        build_manifest_lookup(
            manifest
        )
    )

    screening = (
        build_feature_screening(
            df,
            manifest_lookup,
        )
    )

    duplicates = (
        find_exact_duplicate_features(
            df
        )
    )

    summary = (
        build_summary(
            screening,
            duplicates,
        )
    )

    screening = screening.sort_values(
        [
            "structural_status",
            "missing_ratio",
            "non_null_count",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    screening.to_csv(
        SCREENING_FILE,
        index=False,
    )

    duplicates.to_csv(
        DUPLICATE_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "FEATURE STATUS BY CATEGORY"
    )

    category_status = (
        screening
        .groupby(
            [
                "category",
                "structural_status",
            ],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        category_status.to_string()
    )

    print()
    print(
        "TOP 30 FEATURES BY MISSING RATIO"
    )

    print(
        screening[
            [
                "feature",
                "category",
                "transform",
                "non_null_count",
                "missing_ratio",
                "structural_status",
            ]
        ]
        .sort_values(
            "missing_ratio",
            ascending=False,
        )
        .head(30)
        .to_string(
            index=False
        )
    )

    print()

    if duplicates.empty:
        print(
            "[INFO] Exact duplicate engineered features: 0"
        )
    else:
        print(
            "[INFO] Exact duplicate engineered features:",
            len(
                duplicates
            ),
        )

        print()

        print(
            duplicates
            .head(30)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "[OK] Screening report:",
        SCREENING_FILE,
    )

    print(
        "[OK] Duplicate report:",
        DUPLICATE_FILE,
    )

    print(
        "[OK] Summary:",
        SUMMARY_FILE,
    )

    print()
    print(
        "[IMPORTANT] No feature was removed."
    )

    print(
        "[IMPORTANT] Final feature selection must happen inside each training fold."
    )


if __name__ == "__main__":
    main()
