from pathlib import Path

import pandas as pd
from src.utils.paths import SHIPPING_RAW_DIR
SOURCE_FILE = (
    SHIPPING_RAW_DIR
    / "Baltic Dry Index Geçmiş Verileri.csv"
)
OUTPUT_FILE = (
    SHIPPING_RAW_DIR
    / "baltic_dry_index_daily.csv"
)
def parse_investing_number(series):
    """
    Convert Investing.com Turkish formatted numbers.

    Examples:
    4.668,00 -> 4668.00
    969,00   -> 969.00
    """

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            ".",
            "",
            regex=False,
        )
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def load_source_data():
    """
    Load the manually downloaded Investing.com BDI file.
    """

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            "BDI source file not found: "
            f"{SOURCE_FILE}"
        )

    print(
        "[PASS] Source file exists:",
        SOURCE_FILE,
    )

    df = pd.read_csv(
        SOURCE_FILE,
        encoding="utf-8-sig",
    )

    print(
        "[INFO] Source rows:",
        len(df),
    )

    print(
        "[INFO] Source columns:"
    )

    for column in df.columns:
        print(
            column
        )

    return df


def validate_source_columns(df):
    """
    Validate required Investing.com source columns.
    """

    required_columns = [
        "Tarih",
        "Şimdi",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required source columns: "
            f"{missing_columns}"
        )

    print(
        "[PASS] Required source columns exist"
    )


def standardize_data(df):
    """
    Standardize BDI source data.
    """

    result = pd.DataFrame()

    result["date"] = pd.to_datetime(
        df["Tarih"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    result["baltic_dry_index"] = (
        parse_investing_number(
            df["Şimdi"]
        )
    )

    invalid_dates = int(
        result["date"]
        .isna()
        .sum()
    )

    invalid_values = int(
        result["baltic_dry_index"]
        .isna()
        .sum()
    )

    print(
        "[INFO] Invalid dates:",
        invalid_dates,
    )

    print(
        "[INFO] Invalid BDI values:",
        invalid_values,
    )

    result = (
        result
        .dropna(
            subset=[
                "date",
                "baltic_dry_index",
            ]
        )
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    return result


def validate_standardized_data(df):
    """
    Validate standardized BDI data before saving.
    """

    if df.empty:
        raise ValueError(
            "Standardized BDI dataset is empty"
        )

    if (
        df[
            "baltic_dry_index"
        ]
        .le(0)
        .any()
    ):
        raise ValueError(
            "BDI contains non-positive values"
        )

    if (
        df["date"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "BDI contains duplicate dates"
        )

    if not (
        df["date"]
        .is_monotonic_increasing
    ):
        raise ValueError(
            "BDI dates are not sorted"
        )

    print(
        "[PASS] Standardized BDI data is valid"
    )


def save_output(df):
    """
    Save standardized BDI dataset.
    """

    output_df = df.copy()

    output_df["date"] = (
        output_df["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "[OK] Saved:",
        OUTPUT_FILE,
    )

    return output_df


def delete_source_file():
    """
    Delete the manually downloaded source file
    after successful ingestion.
    """

    if not SOURCE_FILE.exists():
        return

    SOURCE_FILE.unlink()

    print(
        "[OK] Deleted source file:",
        SOURCE_FILE,
    )


def main():

    print(
        "=" * 100
    )

    print(
        "BALTIC DRY INDEX INGESTION"
    )

    print(
        "=" * 100
    )

    df = load_source_data()

    validate_source_columns(
        df
    )

    df = standardize_data(
        df
    )

    validate_standardized_data(
        df
    )

    output_df = save_output(
        df
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "[INFO] Rows:",
        len(output_df),
    )

    print(
        "[INFO] Range:",
        output_df["date"].min(),
        "->",
        output_df["date"].max(),
    )

    print(
        "[INFO] Minimum BDI:",
        output_df[
            "baltic_dry_index"
        ].min(),
    )

    print(
        "[INFO] Maximum BDI:",
        output_df[
            "baltic_dry_index"
        ].max(),
    )

    print(
        "\n[INFO] First 10 rows:"
    )

    print(
        output_df
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\n[INFO] Last 10 rows:"
    )

    print(
        output_df
        .tail(10)
        .to_string(
            index=False
        )
    )

    # Delete source only after successful processing and save
    delete_source_file()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "[PASS] BDI ingestion completed"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()