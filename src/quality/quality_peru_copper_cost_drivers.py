from pathlib import Path

import pandas as pd
from src.utils.paths import MINING_RAW_DIR, QUALITY_DIR




INPUT_FILE = (
    MINING_RAW_DIR
    / "peru_copper_cost_drivers_monthly.csv"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



SUMMARY_FILE = (
    QUALITY_DIR
    / "peru_copper_cost_drivers_quality_summary.csv"
)

ANOMALY_FILE = (
    QUALITY_DIR
    / "peru_copper_cost_drivers_quality_anomalies.csv"
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
        "[INFO] Starting Peru copper cost drivers "
        "data quality checks..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    results = []
    anomalies = []

    df["month_dt"] = pd.to_datetime(
        df["month"],
        format="%Y-%m",
        errors="coerce",
    )

    # Basic
    add_result(
        results,
        "row_count",
        "INFO",
        len(df),
        "Total monthly rows",
    )

    invalid_month = int(
        df["month_dt"]
        .isna()
        .sum()
    )

    add_result(
        results,
        "invalid_month_format",
        (
            "PASS"
            if invalid_month == 0
            else "FAIL"
        ),
        invalid_month,
        "Month must use YYYY-MM format",
    )

    valid_df = df[
        df["month_dt"].notna()
    ].copy()

    first_month = (
        valid_df[
            "month_dt"
        ].min()
        .strftime("%Y-%m")
    )

    last_month = (
        valid_df[
            "month_dt"
        ].max()
        .strftime("%Y-%m")
    )

    add_result(
        results,
        "first_month",
        "INFO",
        first_month,
        "First observation month",
    )

    add_result(
        results,
        "last_month",
        "INFO",
        last_month,
        "Last observation month",
    )

    # Duplicate months
    duplicate_mask = (
        df["month"]
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
            "duplicate_month"
        )

        anomalies.append(
            rows
        )

    add_result(
        results,
        "duplicate_months",
        status,
        duplicate_count,
        "Month must be unique",
    )

    # Missing months
    expected_months = pd.date_range(
        valid_df[
            "month_dt"
        ].min(),
        valid_df[
            "month_dt"
        ].max(),
        freq="MS",
    )

    actual_months = set(
        valid_df[
            "month_dt"
        ].tolist()
    )

    missing_months = [
        x.strftime("%Y-%m")
        for x in expected_months
        if x not in actual_months
    ]

    add_result(
        results,
        "missing_months",
        (
            "PASS"
            if not missing_months
            else "FAIL"
        ),
        len(missing_months),
        (
            "Missing months: "
            + (
                ", ".join(
                    missing_months[:12]
                )
                if missing_months
                else "None"
            )
        ),
    )

    # Null checks
    columns = [
        "peru_usd_pen",
        "peru_diesel_price_index",
        (
            "peru_industrial_"
            "electricity_tariff_index"
        ),
        "peru_formal_private_income_pen",
    ]

    for column in columns:

        null_mask = (
            df[column]
            .isna()
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

    # Positive checks
    for column in columns:

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

    # Monthly jump checks
    jump_thresholds = {
        "peru_usd_pen": 0.15,
        "peru_diesel_price_index": 0.25,
        (
            "peru_industrial_"
            "electricity_tariff_index"
        ): 0.20,
        "peru_formal_private_income_pen": 0.40,
    }

    df = df.sort_values(
        "month_dt"
    ).copy()

    for column, threshold in (
        jump_thresholds.items()
    ):

        pct_change = (
            df[column]
            .pct_change(
                fill_method=None
            )
            .abs()
        )

        jump_mask = (
            pct_change
            > threshold
        )

        jump_count = int(
            jump_mask.sum()
        )

        if jump_count == 0:
            status = "PASS"

        else:
            status = "WARNING"

            rows = df[
                jump_mask
            ].copy()

            rows[
                "absolute_pct_change"
            ] = pct_change[
                jump_mask
            ]

            rows["issue"] = (
                f"large_monthly_change_{column}"
            )

            anomalies.append(
                rows
            )

        add_result(
            results,
            f"large_monthly_change_{column}",
            status,
            jump_count,
            (
                "Monthly absolute change above "
                f"{threshold * 100:.0f} percent "
                f"for {column}"
            ),
        )

    # Latest row completeness
    latest_row = (
        df.sort_values(
            "month_dt"
        )
        .iloc[-1]
    )

    latest_missing = int(
        latest_row[
            columns
        ]
        .isna()
        .sum()
    )

    add_result(
        results,
        "latest_month_missing_values",
        (
            "PASS"
            if latest_missing == 0
            else "WARNING"
        ),
        latest_missing,
        (
            "Missing values in latest "
            "available month"
        ),
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

        anomaly_df = pd.DataFrame()

    if "month_dt" in anomaly_df.columns:
        anomaly_df = anomaly_df.drop(
            columns=["month_dt"]
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
        "\n[INFO] Peru Copper Cost Drivers "
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
