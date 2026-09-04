# -*- coding: utf-8 -*-

"""
ICSG Copper annual data quality checks.
"""

from pathlib import Path

import pandas as pd
from src.utils.paths import MINING_RAW_DIR, QUALITY_DIR




INPUT_FILE = (
    MINING_RAW_DIR
    / "icsg_copper_annual.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "icsg_data_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "icsg_data_quality_anomalies.csv"
)


def add_result(
    results,
    check_name,
    status,
    value,
    message,
):
    """
    Quality sonucu ekler.
    """

    results.append(
        {
            "check_name": check_name,
            "status": status,
            "value": value,
            "message": message,
        }
    )


def main():

    print(
        "[INFO] Starting ICSG data quality checks..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    results = []
    anomalies = []

    # =========================
    # BASIC CHECKS
    # =========================

    add_result(
        results,
        "row_count",
        "INFO",
        len(df),
        "Total ICSG annual rows",
    )

    min_year = int(
        df["observation_year"].min()
    )

    max_year = int(
        df["observation_year"].max()
    )

    add_result(
        results,
        "first_observation_year",
        "INFO",
        min_year,
        "First observation year",
    )

    add_result(
        results,
        "last_observation_year",
        "INFO",
        max_year,
        "Last observation year",
    )

    # =========================
    # DUPLICATE YEARS
    # =========================

    duplicate_mask = (
        df["observation_year"]
        .duplicated(
            keep=False
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count == 0:
        status = "PASS"
    else:
        status = "FAIL"

        rows = df[
            duplicate_mask
        ].copy()

        rows["issue"] = (
            "duplicate_observation_year"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "duplicate_observation_years",
        status,
        duplicate_count,
        "Observation year must be unique",
    )

    # =========================
    # MISSING YEARS
    # =========================

    expected_years = set(
        range(
            min_year,
            max_year + 1,
        )
    )

    actual_years = set(
        df["observation_year"]
        .astype(int)
        .tolist()
    )

    missing_years = sorted(
        expected_years
        - actual_years
    )

    status = (
        "PASS"
        if not missing_years
        else "FAIL"
    )

    add_result(
        results,
        "missing_observation_years",
        status,
        len(missing_years),
        (
            "Missing years: "
            + (
                ", ".join(
                    map(
                        str,
                        missing_years,
                    )
                )
                if missing_years
                else "None"
            )
        ),
    )

    # =========================
    # NULL CHECKS
    # =========================

    columns = [
        "icsg_mine_production_ton",
        "icsg_refined_production_ton",
        "icsg_refined_usage_ton",
        "icsg_refined_balance_ton",
    ]

    for column in columns:

        null_mask = (
            df[column].isna()
        )

        null_count = int(
            null_mask.sum()
        )

        if null_count == 0:
            status = "PASS"
        else:
            status = "WARNING"

            rows = df[
                null_mask
            ].copy()

            rows["issue"] = (
                f"null_{column}"
            )

            anomalies.append(
                rows
            )

        add_result(
            results,
            f"null_{column}",
            status,
            null_count,
            f"Null count for {column}",
        )

    # =========================
    # NON-POSITIVE CHECKS
    # =========================

    positive_columns = [
        "icsg_mine_production_ton",
        "icsg_refined_production_ton",
        "icsg_refined_usage_ton",
    ]

    for column in positive_columns:

        invalid_mask = (
            df[column].notna()
            & (
                df[column] <= 0
            )
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        if invalid_count == 0:
            status = "PASS"
        else:
            status = "FAIL"

            rows = df[
                invalid_mask
            ].copy()

            rows["issue"] = (
                f"nonpositive_{column}"
            )

            anomalies.append(
                rows
            )

        add_result(
            results,
            f"nonpositive_{column}",
            status,
            invalid_count,
            (
                f"Non-positive values for "
                f"{column}"
            ),
        )

    # Balance negatif olabilir
    balance_null_count = int(
        df[
            "icsg_refined_balance_ton"
        ].isna().sum()
    )

    add_result(
        results,
        "balance_sign_check",
        "PASS",
        balance_null_count,
        (
            "Negative balance values are valid "
            "and represent market deficit"
        ),
    )

    # =========================
    # FORECAST LOGIC
    # =========================

    forecast_rows = df[
        df["is_forecast"] == True
    ]

    forecast_count = len(
        forecast_rows
    )

    add_result(
        results,
        "forecast_rows",
        "INFO",
        forecast_count,
        "Number of forecast observations",
    )

    # Forecast observation year should not be
    # earlier than report year
    invalid_forecast_mask = (
        (df["is_forecast"] == True)
        & (
            df["observation_year"]
            < df["report_year"]
        )
    )

    invalid_forecast_count = int(
        invalid_forecast_mask.sum()
    )

    if invalid_forecast_count == 0:
        status = "PASS"
    else:
        status = "FAIL"

        rows = df[
            invalid_forecast_mask
        ].copy()

        rows["issue"] = (
            "invalid_forecast_year_logic"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "forecast_year_logic",
        status,
        invalid_forecast_count,
        (
            "Forecast observation year must "
            "not be earlier than report year"
        ),
    )

    # =========================
    # LARGE YEARLY CHANGES
    # =========================

    df = df.sort_values(
        "observation_year"
    ).copy()

    change_columns = [
        "icsg_mine_production_ton",
        "icsg_refined_production_ton",
        "icsg_refined_usage_ton",
    ]

    for column in change_columns:

        pct_change = (
            df[column]
            .pct_change(
                fill_method=None
            )
            .abs()
        )

        large_mask = (
            pct_change > 0.20
        )

        large_count = int(
            large_mask.sum()
        )

        if large_count == 0:
            status = "PASS"
        else:
            status = "WARNING"

            rows = df[
                large_mask
            ].copy()

            rows[
                "absolute_pct_change"
            ] = pct_change[
                large_mask
            ]

            rows["issue"] = (
                f"large_change_{column}"
            )

            anomalies.append(
                rows
            )

        add_result(
            results,
            f"large_change_{column}",
            status,
            large_count,
            (
                f"Annual absolute change above "
                f"20 percent for {column}"
            ),
        )

    # =========================
    # SAVE RESULTS
    # =========================

    summary_df = pd.DataFrame(
        results
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    if anomalies:

        anomaly_df = pd.concat(
            anomalies,
            ignore_index=True,
        )

    else:

        anomaly_df = pd.DataFrame()

    anomaly_df.to_csv(
        ANOMALY_FILE,
        index=False,
    )

    fail_count = int(
        summary_df[
            "status"
        ].eq(
            "FAIL"
        ).sum()
    )

    warning_count = int(
        summary_df[
            "status"
        ].eq(
            "WARNING"
        ).sum()
    )

    if fail_count > 0:
        overall_status = "FAIL"
    elif warning_count > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    print(
        "\n[INFO] ICSG Data Quality Summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        f"[RESULT] FAIL={fail_count} "
        f"WARNING={warning_count}"
    )

    print(
        f"[RESULT] Overall Status: "
        f"{overall_status}"
    )

    print(
        f"\n[OK] Saved summary: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"[OK] Saved anomalies: "
        f"{ANOMALY_FILE}"
    )


if __name__ == "__main__":
    main()