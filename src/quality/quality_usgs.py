# -*- coding: utf-8 -*-

"""
USGS Copper annual data quality checks.
"""

from pathlib import Path

import pandas as pd
from src.utils.paths import MINING_RAW_DIR, QUALITY_DIR




INPUT_FILE = (
    MINING_RAW_DIR
    / "usgs_copper_annual.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "usgs_data_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "usgs_data_quality_anomalies.csv"
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
        "[INFO] Starting USGS data quality checks..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    results = []
    anomalies = []

    # =========================
    # BASIC CHECKS
    # =========================

    row_count = len(df)

    add_result(
        results,
        "row_count",
        "INFO",
        row_count,
        "Total USGS annual rows",
    )

    min_year = df[
        "observation_year"
    ].min()

    max_year = df[
        "observation_year"
    ].max()

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

    duplicate_mask = df[
        "observation_year"
    ].duplicated(
        keep=False
    )

    duplicate_count = (
        duplicate_mask.sum()
    )

    if duplicate_count == 0:

        status = "PASS"

    else:

        status = "FAIL"

        duplicate_rows = df[
            duplicate_mask
        ].copy()

        duplicate_rows[
            "issue"
        ] = "duplicate_observation_year"

        anomalies.append(
            duplicate_rows
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
            int(min_year),
            int(max_year) + 1,
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

    if len(missing_years) == 0:

        status = "PASS"

    else:

        status = "FAIL"

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

    key_columns = [
        "world_copper_mine_production_ton",
        "world_copper_refinery_production_ton",
        "world_copper_reserves_ton",
    ]

    for column in key_columns:

        null_count = (
            df[column]
            .isna()
            .sum()
        )

        if null_count == 0:

            status = "PASS"

        else:

            # USGS annual data can naturally contain
            # missing refinery or reserve values
            status = "WARNING"

            null_rows = df[
                df[column].isna()
            ].copy()

            null_rows[
                "issue"
            ] = f"null_{column}"

            anomalies.append(
                null_rows
            )

        add_result(
            results,
            f"null_{column}",
            status,
            int(null_count),
            f"Null count for {column}",
        )

    # =========================
    # NON-POSITIVE VALUES
    # =========================

    for column in key_columns:

        invalid_mask = (
            df[column].notna()
            & (
                df[column] <= 0
            )
        )

        invalid_count = (
            invalid_mask.sum()
        )

        if invalid_count == 0:

            status = "PASS"

        else:

            status = "FAIL"

            invalid_rows = df[
                invalid_mask
            ].copy()

            invalid_rows[
                "issue"
            ] = f"nonpositive_{column}"

            anomalies.append(
                invalid_rows
            )

        add_result(
            results,
            f"nonpositive_{column}",
            status,
            int(invalid_count),
            (
                f"Non-positive values for "
                f"{column}"
            ),
        )

    # =========================
    # REPORT YEAR LOGIC
    # =========================

    invalid_report_mask = (
        df["report_year"]
        < df["observation_year"]
    )

    invalid_report_count = (
        invalid_report_mask.sum()
    )

    if invalid_report_count == 0:

        status = "PASS"

    else:

        status = "FAIL"

        invalid_rows = df[
            invalid_report_mask
        ].copy()

        invalid_rows[
            "issue"
        ] = "invalid_report_year"

        anomalies.append(
            invalid_rows
        )

    add_result(
        results,
        "report_year_logic",
        status,
        int(invalid_report_count),
        (
            "Report year must not be earlier "
            "than observation year"
        ),
    )

    # =========================
    # ESTIMATE CHECK
    # =========================

    estimate_count = (
        df["is_estimate"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if estimate_count <= 1:

        status = "PASS"

    else:

        status = "WARNING"

    add_result(
        results,
        "estimate_rows",
        status,
        int(estimate_count),
        (
            "Latest observation may be estimate; "
            "multiple estimate rows should be reviewed"
        ),
    )

    # =========================
    # LARGE YEARLY CHANGES
    # =========================

    df = df.sort_values(
        "observation_year"
    ).copy()

    change_columns = [
        "world_copper_mine_production_ton",
        "world_copper_refinery_production_ton",
        "world_copper_reserves_ton",
    ]

    for column in change_columns:

        pct_change = (
            df[column]
            .pct_change(
                fill_method=None
            )
            .abs()
        )

        # Annual supply variables:
        # >20 percent change is worth reviewing
        large_change_mask = (
            pct_change > 0.20
        )

        large_change_count = (
            large_change_mask.sum()
        )

        if large_change_count == 0:

            status = "PASS"

        else:

            status = "WARNING"

            warning_rows = df[
                large_change_mask
            ].copy()

            warning_rows[
                "issue"
            ] = (
                f"large_change_{column}"
            )

            warning_rows[
                "absolute_pct_change"
            ] = (
                pct_change[
                    large_change_mask
                ]
            )

            anomalies.append(
                warning_rows
            )

        add_result(
            results,
            f"large_change_{column}",
            status,
            int(large_change_count),
            (
                f"Annual absolute change above "
                f"20 percent for {column}"
            ),
        )

    # =========================
    # SAVE
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

    fail_count = (
        summary_df[
            "status"
        ]
        .eq("FAIL")
        .sum()
    )

    warning_count = (
        summary_df[
            "status"
        ]
        .eq("WARNING")
        .sum()
    )

    if fail_count > 0:

        overall_status = "FAIL"

    elif warning_count > 0:

        overall_status = "WARNING"

    else:

        overall_status = "PASS"

    print(
        "\n[INFO] USGS Data Quality Summary:"
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