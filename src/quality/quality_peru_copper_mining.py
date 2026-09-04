from pathlib import Path

import pandas as pd
from src.utils.paths import MINING_RAW_DIR, QUALITY_DIR




INPUT_FILE = (
    MINING_RAW_DIR
    / "peru_copper_mining_annual.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "peru_copper_mining_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "peru_copper_mining_quality_anomalies.csv"
)


def add_result(
    results,
    check_name,
    status,
    value,
    message,
):
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
        "[INFO] Starting Peru copper mining "
        "data quality checks..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    results = []
    anomalies = []

    # Basic checks
    add_result(
        results,
        "row_count",
        "INFO",
        len(df),
        "Total annual rows",
    )

    min_year = int(
        df["year"].min()
    )

    max_year = int(
        df["year"].max()
    )

    add_result(
        results,
        "first_year",
        "INFO",
        min_year,
        "First observation year",
    )

    add_result(
        results,
        "last_year",
        "INFO",
        max_year,
        "Last observation year",
    )

    # Duplicate years
    duplicate_mask = (
        df["year"]
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
            "duplicate_year"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "duplicate_years",
        status,
        duplicate_count,
        "Year must be unique",
    )

    # Missing years
    expected_years = set(
        range(
            min_year,
            max_year + 1,
        )
    )

    actual_years = set(
        df["year"]
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
        "missing_years",
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

    # Production null
    production_null = int(
        df[
            "peru_copper_production_ton"
        ]
        .isna()
        .sum()
    )

    add_result(
        results,
        "null_production",
        (
            "PASS"
            if production_null == 0
            else "FAIL"
        ),
        production_null,
        "Copper production null count",
    )

    # Production positive
    production_invalid_mask = (
        df[
            "peru_copper_production_ton"
        ].notna()
        & (
            df[
                "peru_copper_production_ton"
            ] <= 0
        )
    )

    production_invalid = int(
        production_invalid_mask.sum()
    )

    if production_invalid > 0:

        rows = df[
            production_invalid_mask
        ].copy()

        rows["issue"] = (
            "nonpositive_production"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "production_positive",
        (
            "PASS"
            if production_invalid == 0
            else "FAIL"
        ),
        production_invalid,
        "Copper production must be positive",
    )

    # Employment coverage
    employment_count = int(
        df[
            "peru_mining_employment"
        ]
        .notna()
        .sum()
    )

    add_result(
        results,
        "employment_observations",
        "INFO",
        employment_count,
        "Available complete employment years",
    )

    employment_null = int(
        df[
            "peru_mining_employment"
        ]
        .isna()
        .sum()
    )

    add_result(
        results,
        "null_employment",
        "WARNING",
        employment_null,
        (
            "Employment starts in 2020 "
            "in current MINEM source"
        ),
    )

    # Employment positive
    employment_invalid_mask = (
        df[
            "peru_mining_employment"
        ].notna()
        & (
            df[
                "peru_mining_employment"
            ] <= 0
        )
    )

    employment_invalid = int(
        employment_invalid_mask.sum()
    )

    if employment_invalid > 0:

        rows = df[
            employment_invalid_mask
        ].copy()

        rows["issue"] = (
            "nonpositive_employment"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "employment_positive",
        (
            "PASS"
            if employment_invalid == 0
            else "FAIL"
        ),
        employment_invalid,
        "Mining employment must be positive",
    )

    # Production YoY consistency
    expected_yoy = (
        df[
            "peru_copper_production_ton"
        ]
        .pct_change(
            fill_method=None
        )
        * 100
    )

    yoy_diff = (
        expected_yoy
        - df[
            "peru_copper_production_yoy_pct"
        ]
    ).abs()

    yoy_invalid_mask = (
        yoy_diff.notna()
        & (
            yoy_diff > 0.01
        )
    )

    yoy_invalid = int(
        yoy_invalid_mask.sum()
    )

    if yoy_invalid > 0:

        rows = df[
            yoy_invalid_mask
        ].copy()

        rows[
            "production_yoy_difference"
        ] = yoy_diff[
            yoy_invalid_mask
        ]

        rows["issue"] = (
            "production_yoy_mismatch"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "production_yoy_consistency",
        (
            "PASS"
            if yoy_invalid == 0
            else "FAIL"
        ),
        yoy_invalid,
        "Stored production YoY must match production",
    )

    # Large production movements
    production_large_mask = (
        df[
            "peru_copper_production_yoy_pct"
        ].notna()
        & (
            df[
                "peru_copper_production_yoy_pct"
            ].abs()
            > 30
        )
    )

    production_large = int(
        production_large_mask.sum()
    )

    if production_large > 0:

        rows = df[
            production_large_mask
        ].copy()

        rows["issue"] = (
            "large_production_yoy"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "large_production_yoy",
        (
            "WARNING"
            if production_large > 0
            else "PASS"
        ),
        production_large,
        (
            "Absolute annual production change "
            "above 30 percent"
        ),
    )

    # Employment YoY consistency
    employment_expected_yoy = (
        df[
            "peru_mining_employment"
        ]
        .pct_change(
            fill_method=None
        )
        * 100
    )

    employment_yoy_diff = (
        employment_expected_yoy
        - df[
            "peru_mining_employment_yoy_pct"
        ]
    ).abs()

    employment_yoy_invalid_mask = (
        employment_yoy_diff.notna()
        & (
            employment_yoy_diff
            > 0.01
        )
    )

    employment_yoy_invalid = int(
        employment_yoy_invalid_mask.sum()
    )

    if employment_yoy_invalid > 0:

        rows = df[
            employment_yoy_invalid_mask
        ].copy()

        rows["issue"] = (
            "employment_yoy_mismatch"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "employment_yoy_consistency",
        (
            "PASS"
            if employment_yoy_invalid == 0
            else "FAIL"
        ),
        employment_yoy_invalid,
        "Stored employment YoY must match employment",
    )

    # Save
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

        anomaly_df = (
            pd.DataFrame()
        )

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
        "\n[INFO] Peru Copper Mining "
        "Data Quality Summary:"
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