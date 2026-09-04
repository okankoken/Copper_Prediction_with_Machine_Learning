from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import (
    DIAGNOSTICS_DIR,
    RAW_DIR,
)


FILE_OUTPUT = (
    DIAGNOSTICS_DIR
    / "raw_data_catalog.csv"
)

COLUMN_OUTPUT = (
    DIAGNOSTICS_DIR
    / "raw_column_catalog.csv"
)

TEXT_OUTPUT = (
    DIAGNOSTICS_DIR
    / "raw_data_profile.txt"
)


DATE_CANDIDATES = [
    "date",
    "month",
    "period",
    "time",
    "datetime",
    "Date",
    "Month",
    "Period",
    "DATE",
    "MONTH",
    "PERIOD",
]


def detect_date_column(df):
    for candidate in DATE_CANDIDATES:
        if candidate in df.columns:
            return candidate

    return None


def parse_date_series(series):
    parsed = pd.to_datetime(
        series,
        errors="coerce",
    )

    valid_ratio = parsed.notna().mean()

    if valid_ratio < 0.50:
        return None

    return parsed


def infer_frequency(parsed_dates):
    values = (
        parsed_dates
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(values) < 3:
        return "unknown"

    diffs = (
        values
        .diff()
        .dropna()
        .dt.days
    )

    if len(diffs) == 0:
        return "unknown"

    median_days = float(
        diffs.median()
    )

    if median_days <= 2:
        return "daily"

    if median_days <= 10:
        return "weekly"

    if median_days <= 20:
        return "semi_monthly"

    if median_days <= 45:
        return "monthly"

    if median_days <= 120:
        return "quarterly"

    if median_days <= 220:
        return "semi_annual"

    if median_days <= 400:
        return "annual"

    return "irregular"


def get_category(path):
    relative = path.relative_to(
        RAW_DIR
    )

    if len(relative.parts) > 1:
        return relative.parts[0]

    return "uncategorized"


def safe_value(value):
    if pd.isna(value):
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    return str(value)


def profile_file(path):
    print(
        "[SCAN]",
        path.relative_to(RAW_DIR),
    )

    try:
        df = pd.read_csv(
            path
        )

    except Exception as exc:
        return (
            {
                "category": get_category(path),
                "file": str(
                    path.relative_to(RAW_DIR)
                ),
                "rows": None,
                "columns": None,
                "date_column": None,
                "start_date": None,
                "end_date": None,
                "inferred_frequency": None,
                "duplicate_dates": None,
                "status": "READ_ERROR",
                "error": str(exc),
            },
            [],
        )

    date_column = detect_date_column(
        df
    )

    parsed_dates = None
    start_date = None
    end_date = None
    inferred_frequency = None
    duplicate_dates = None

    if date_column is not None:
        parsed_dates = parse_date_series(
            df[date_column]
        )

        if parsed_dates is not None:
            valid_dates = parsed_dates.dropna()

            if len(valid_dates) > 0:
                start_date = (
                    valid_dates.min()
                    .date()
                    .isoformat()
                )

                end_date = (
                    valid_dates.max()
                    .date()
                    .isoformat()
                )

                inferred_frequency = (
                    infer_frequency(
                        valid_dates
                    )
                )

                duplicate_dates = int(
                    valid_dates.duplicated().sum()
                )

    file_result = {
        "category": get_category(path),
        "file": str(
            path.relative_to(RAW_DIR)
        ),
        "rows": len(df),
        "columns": len(df.columns),
        "date_column": date_column,
        "start_date": start_date,
        "end_date": end_date,
        "inferred_frequency": inferred_frequency,
        "duplicate_dates": duplicate_dates,
        "status": "OK",
        "error": None,
    }

    column_results = []

    for column in df.columns:
        series = df[column]

        non_null = int(
            series.notna().sum()
        )

        missing = int(
            series.isna().sum()
        )

        missing_percent = (
            missing / len(df) * 100
            if len(df) > 0
            else 0.0
        )

        unique_values = int(
            series.nunique(
                dropna=True
            )
        )

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_ratio = (
            numeric.notna().mean()
            if len(series) > 0
            else 0
        )

        numeric_min = None
        numeric_max = None
        numeric_mean = None

        if numeric_ratio >= 0.80:
            numeric_valid = (
                numeric.dropna()
            )

            if len(numeric_valid) > 0:
                numeric_min = float(
                    numeric_valid.min()
                )

                numeric_max = float(
                    numeric_valid.max()
                )

                numeric_mean = float(
                    numeric_valid.mean()
                )

        valid_values = (
            series
            .dropna()
        )

        first_value = None
        last_value = None

        if len(valid_values) > 0:
            first_value = safe_value(
                valid_values.iloc[0]
            )

            last_value = safe_value(
                valid_values.iloc[-1]
            )

        column_results.append(
            {
                "category": get_category(path),
                "file": str(
                    path.relative_to(RAW_DIR)
                ),
                "column": column,
                "dtype": str(series.dtype),
                "is_date_column": (
                    column == date_column
                ),
                "non_null": non_null,
                "missing": missing,
                "missing_percent": round(
                    missing_percent,
                    4,
                ),
                "unique_values": unique_values,
                "numeric_ratio": round(
                    float(numeric_ratio),
                    4,
                ),
                "min": numeric_min,
                "max": numeric_max,
                "mean": numeric_mean,
                "first_value": first_value,
                "last_value": last_value,
            }
        )

    return (
        file_result,
        column_results,
    )


def build_text_report(
    file_df,
    column_df,
):
    lines = []

    lines.append(
        "=" * 120
    )
    lines.append(
        "RAW DATA PROFILE"
    )
    lines.append(
        "=" * 120
    )
    lines.append("")

    lines.append(
        f"Files: {len(file_df)}"
    )

    lines.append(
        f"Columns: {len(column_df)}"
    )

    lines.append("")

    for _, file_row in file_df.iterrows():
        lines.append(
            "-" * 120
        )

        lines.append(
            f"FILE: {file_row['file']}"
        )

        lines.append(
            f"CATEGORY: {file_row['category']}"
        )

        lines.append(
            f"ROWS: {file_row['rows']}"
        )

        lines.append(
            f"COLUMNS: {file_row['columns']}"
        )

        lines.append(
            f"DATE COLUMN: {file_row['date_column']}"
        )

        lines.append(
            f"DATE RANGE: {file_row['start_date']} -> {file_row['end_date']}"
        )

        lines.append(
            f"FREQUENCY: {file_row['inferred_frequency']}"
        )

        lines.append(
            f"DUPLICATE DATES: {file_row['duplicate_dates']}"
        )

        lines.append("")

        current_columns = (
            column_df[
                column_df["file"]
                == file_row["file"]
            ]
        )

        for _, row in current_columns.iterrows():
            lines.append(
                (
                    f"  {row['column']} | "
                    f"dtype={row['dtype']} | "
                    f"missing={row['missing']} "
                    f"({row['missing_percent']:.2f}%) | "
                    f"unique={row['unique_values']} | "
                    f"min={row['min']} | "
                    f"max={row['max']}"
                )
            )

        lines.append("")

    return "\n".join(
        lines
    )


def main():
    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DIR.rglob("*.csv")
    )

    if not files:
        raise RuntimeError(
            f"No CSV files found under {RAW_DIR}"
        )

    print(
        "[INFO] CSV files found:",
        len(files),
    )

    print()

    file_results = []
    column_results = []

    for path in files:
        (
            file_result,
            file_columns,
        ) = profile_file(
            path
        )

        file_results.append(
            file_result
        )

        column_results.extend(
            file_columns
        )

    file_df = pd.DataFrame(
        file_results
    )

    column_df = pd.DataFrame(
        column_results
    )

    file_df = file_df.sort_values(
        [
            "category",
            "file",
        ]
    )

    column_df = column_df.sort_values(
        [
            "category",
            "file",
            "column",
        ]
    )

    file_df.to_csv(
        FILE_OUTPUT,
        index=False,
    )

    column_df.to_csv(
        COLUMN_OUTPUT,
        index=False,
    )

    text_report = build_text_report(
        file_df,
        column_df,
    )

    TEXT_OUTPUT.write_text(
        text_report,
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("RAW DATA CATALOG")
    print("=" * 120)

    display_columns = [
        "category",
        "file",
        "rows",
        "columns",
        "date_column",
        "start_date",
        "end_date",
        "inferred_frequency",
        "duplicate_dates",
        "status",
    ]

    print(
        file_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("HIGH MISSING COLUMNS")
    print("=" * 120)

    high_missing = (
        column_df[
            column_df[
                "missing_percent"
            ] > 10
        ]
        .sort_values(
            "missing_percent",
            ascending=False,
        )
    )

    if len(high_missing) == 0:
        print(
            "None"
        )

    else:
        print(
            high_missing[
                [
                    "file",
                    "column",
                    "missing",
                    "missing_percent",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("[SAVED]")
    print(FILE_OUTPUT)
    print(COLUMN_OUTPUT)
    print(TEXT_OUTPUT)


if __name__ == "__main__":
    main()
