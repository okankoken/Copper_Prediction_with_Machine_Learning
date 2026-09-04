from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.paths import MINING_RAW_DIR, QUALITY_DIR




INPUT_FILE = (
    MINING_RAW_DIR
    / "chile_cochilco_copper_cost_annual.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "chile_cochilco_data_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "chile_cochilco_data_quality_anomalies.csv"
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
        "[INFO] Starting Chile COCHILCO "
        "data quality checks..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    results = []
    anomalies = []

    # Basic information
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

    # Duplicate year control
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

    # Null checks
    null_columns = [
        "chile_copper_production_ton",
        "chile_mining_wage_index",
        "chile_mining_wage_yoy_pct",
        "chile_copper_fuel_consumption_tj",
        "chile_copper_electricity_consumption_tj",
        "chile_copper_total_energy_consumption_tj",
        "chile_open_pit_fuel_mj_per_ton",
        "chile_concentrator_electricity_mj_per_ton",
        "chile_sulfuric_acid_consumption_ton",
        "chile_sulfuric_acid_production_ton",
        "chile_sulfuric_acid_import_ton",
        "chile_sulfuric_acid_export_ton",
        "chile_average_copper_ore_grade_pct",
    ]

    for column in null_columns:

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

    # Positive value checks
    positive_columns = [
        "chile_copper_production_ton",
        "chile_mining_wage_index",
        "chile_copper_fuel_consumption_tj",
        "chile_copper_electricity_consumption_tj",
        "chile_copper_total_energy_consumption_tj",
        "chile_open_pit_fuel_mj_per_ton",
        "chile_concentrator_electricity_mj_per_ton",
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

    # Energy consistency
    energy_sum = (
        df[
            "chile_copper_fuel_consumption_tj"
        ]
        + df[
            "chile_copper_electricity_consumption_tj"
        ]
    )

    energy_diff = (
        energy_sum
        - df[
            "chile_copper_total_energy_consumption_tj"
        ]
    ).abs()

    energy_tolerance = 1.0

    energy_invalid_mask = (
        energy_diff
        > energy_tolerance
    )

    energy_invalid_count = int(
        energy_invalid_mask.sum()
    )

    if energy_invalid_count == 0:
        status = "PASS"
    else:
        status = "FAIL"

        rows = df[
            energy_invalid_mask
        ].copy()

        rows[
            "energy_difference_tj"
        ] = energy_diff[
            energy_invalid_mask
        ]

        rows["issue"] = (
            "energy_balance_mismatch"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "energy_balance_check",
        status,
        energy_invalid_count,
        (
            "Fuel + electricity should equal "
            "total energy within tolerance"
        ),
    )

    # Sulfuric acid consistency
    acid_expected = (
        df[
            "chile_sulfuric_acid_production_ton"
        ]
        + df[
            "chile_sulfuric_acid_import_ton"
        ]
        - df[
            "chile_sulfuric_acid_export_ton"
        ]
    )

    acid_diff = (
        acid_expected
        - df[
            "chile_sulfuric_acid_consumption_ton"
        ]
    ).abs()

    acid_available = (
        df[
            [
                "chile_sulfuric_acid_consumption_ton",
                "chile_sulfuric_acid_production_ton",
                "chile_sulfuric_acid_import_ton",
                "chile_sulfuric_acid_export_ton",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    acid_tolerance_ton = 5000

    acid_invalid_mask = (
        acid_available
        & (
            acid_diff
            > acid_tolerance_ton
        )
    )

    acid_invalid_count = int(
        acid_invalid_mask.sum()
    )

    if acid_invalid_count == 0:
        status = "PASS"
    else:
        status = "WARNING"

        rows = df[
            acid_invalid_mask
        ].copy()

        rows[
            "acid_difference_ton"
        ] = acid_diff[
            acid_invalid_mask
        ]

        rows["issue"] = (
            "sulfuric_acid_balance_mismatch"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "sulfuric_acid_balance_check",
        status,
        acid_invalid_count,
        (
            "Production + imports - exports "
            "should approximate consumption"
        ),
    )

    # Ore grade range
    ore_grade_mask = (
        df[
            "chile_average_copper_ore_grade_pct"
        ].notna()
        & (
            (
                df[
                    "chile_average_copper_ore_grade_pct"
                ]
                <= 0
            )
            |
            (
                df[
                    "chile_average_copper_ore_grade_pct"
                ]
                > 5
            )
        )
    )

    ore_grade_count = int(
        ore_grade_mask.sum()
    )

    if ore_grade_count == 0:
        status = "PASS"
    else:
        status = "FAIL"

        rows = df[
            ore_grade_mask
        ].copy()

        rows["issue"] = (
            "invalid_ore_grade"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "ore_grade_range_check",
        status,
        ore_grade_count,
        (
            "Copper ore grade must be "
            "greater than 0 and below 5 percent"
        ),
    )

    # Wage YoY range check
    wage_yoy_mask = (
        df[
            "chile_mining_wage_yoy_pct"
        ].notna()
        & (
            df[
                "chile_mining_wage_yoy_pct"
            ].abs()
            > 30
        )
    )

    wage_yoy_count = int(
        wage_yoy_mask.sum()
    )

    if wage_yoy_count == 0:
        status = "PASS"
    else:
        status = "WARNING"

        rows = df[
            wage_yoy_mask
        ].copy()

        rows["issue"] = (
            "large_wage_yoy_change"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "wage_yoy_change_check",
        status,
        wage_yoy_count,
        (
            "Absolute mining wage YoY change "
            "above 30 percent"
        ),
    )

    # Large annual changes
    change_columns = [
        "chile_copper_production_ton",
        "chile_copper_fuel_consumption_tj",
        "chile_copper_electricity_consumption_tj",
        "chile_open_pit_fuel_mj_per_ton",
        "chile_concentrator_electricity_mj_per_ton",
    ]

    df = df.sort_values(
        "year"
    ).copy()

    for column in change_columns:

        pct_change = (
            df[column]
            .pct_change(
                fill_method=None
            )
            .abs()
        )

        large_mask = (
            pct_change
            > 0.30
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
                f"30 percent for {column}"
            ),
        )

    # Save outputs
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
        "\n[INFO] Chile COCHILCO "
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
