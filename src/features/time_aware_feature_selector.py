import numpy as np
import pandas as pd


TARGET_COLUMN = "cash_settlement_usd_per_ton"

MIN_PAIRED_OBSERVATIONS = 24
MAX_MISSING_RATIO = 0.50
MAX_REDUNDANCY_CORRELATION = 0.95
ABSOLUTE_MAX_FEATURES = 20


def _to_numeric(series):
    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def _get_dynamic_max_features(
    training_rows,
):
    # Keep the feature count conservative relative to sample size.
    # Around one selected feature per five training observations.
    dynamic_limit = max(
        5,
        training_rows // 5,
    )

    return min(
        ABSOLUTE_MAX_FEATURES,
        dynamic_limit,
    )


def _safe_spearman(
    feature,
    target,
):
    pair = pd.concat(
        [
            feature.rename("feature"),
            target.rename("target"),
        ],
        axis=1,
    ).dropna()

    if len(pair) < MIN_PAIRED_OBSERVATIONS:
        return np.nan, len(pair)

    if (
        pair["feature"].nunique() <= 1
        or pair["target"].nunique() <= 1
    ):
        return np.nan, len(pair)

    correlation = pair[
        "feature"
    ].corr(
        pair["target"],
        method="spearman",
    )

    return correlation, len(pair)


def _feature_pair_correlation(
    left,
    right,
):
    pair = pd.concat(
        [
            left.rename("left"),
            right.rename("right"),
        ],
        axis=1,
    ).dropna()

    if len(pair) < MIN_PAIRED_OBSERVATIONS:
        return np.nan

    if (
        pair["left"].nunique() <= 1
        or pair["right"].nunique() <= 1
    ):
        return np.nan

    return pair[
        "left"
    ].corr(
        pair["right"],
        method="spearman",
    )


def select_features(
    training_df,
    target_column=TARGET_COLUMN,
):
    if target_column not in training_df.columns:
        raise ValueError(
            f"Target not found: {target_column}"
        )

    working = training_df.copy()

    target = _to_numeric(
        working[
            target_column
        ]
    )

    candidate_columns = [
        column
        for column in working.columns
        if column not in {
            "date",
            target_column,
        }
    ]

    ranking_rows = []

    for column in candidate_columns:
        series = _to_numeric(
            working[
                column
            ]
        )

        non_null_count = int(
            series.notna().sum()
        )

        missing_ratio = float(
            series.isna().mean()
        )

        unique_count = int(
            series.dropna().nunique()
        )

        if non_null_count == 0:
            continue

        if unique_count <= 1:
            continue

        if missing_ratio > MAX_MISSING_RATIO:
            continue

        correlation, paired_count = (
            _safe_spearman(
                series,
                target,
            )
        )

        if pd.isna(
            correlation
        ):
            continue

        ranking_rows.append(
            {
                "feature": column,
                "non_null_count": non_null_count,
                "missing_ratio": missing_ratio,
                "paired_observations": paired_count,
                "spearman_correlation": correlation,
                "abs_spearman_correlation": abs(
                    correlation
                ),
            }
        )

    ranking = pd.DataFrame(
        ranking_rows
    )

    if ranking.empty:
        return [], ranking, pd.DataFrame()

    ranking = ranking.sort_values(
        [
            "abs_spearman_correlation",
            "paired_observations",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    max_features = (
        _get_dynamic_max_features(
            len(
                working
            )
        )
    )

    selected = []
    decision_rows = []

    for _, row in ranking.iterrows():
        candidate = row[
            "feature"
        ]

        redundant_with = None
        highest_redundancy = np.nan

        for selected_feature in selected:
            correlation = (
                _feature_pair_correlation(
                    _to_numeric(
                        working[
                            candidate
                        ]
                    ),
                    _to_numeric(
                        working[
                            selected_feature
                        ]
                    ),
                )
            )

            if pd.isna(
                correlation
            ):
                continue

            if (
                pd.isna(
                    highest_redundancy
                )
                or abs(
                    correlation
                )
                > highest_redundancy
            ):
                highest_redundancy = abs(
                    correlation
                )

            if (
                abs(
                    correlation
                )
                >= MAX_REDUNDANCY_CORRELATION
            ):
                redundant_with = (
                    selected_feature
                )
                break

        if redundant_with is not None:
            decision_rows.append(
                {
                    "feature": candidate,
                    "decision": "REJECT_REDUNDANT",
                    "redundant_with": redundant_with,
                    "max_abs_feature_correlation": (
                        highest_redundancy
                    ),
                    "target_abs_spearman": row[
                        "abs_spearman_correlation"
                    ],
                }
            )

            continue

        selected.append(
            candidate
        )

        decision_rows.append(
            {
                "feature": candidate,
                "decision": "SELECT",
                "redundant_with": None,
                "max_abs_feature_correlation": (
                    highest_redundancy
                ),
                "target_abs_spearman": row[
                    "abs_spearman_correlation"
                ],
            }
        )

        if len(
            selected
        ) >= max_features:
            break

    decisions = pd.DataFrame(
        decision_rows
    )

    return (
        selected,
        ranking,
        decisions,
    )
