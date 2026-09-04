from pathlib import Path
from io import BytesIO

import pandas as pd
import requests
from src.utils.paths import RISK_RAW_DIR
OUTPUT_FILE = (
    RISK_RAW_DIR
    / "geopolitical_risk_monthly.csv"
)
SOURCE_URL = (
    "https://www.matteoiacoviello.com/"
    "gpr_files/data_gpr_export.xls"
)


SOURCE_COLUMNS = {
    "month": "date",
    "GPR": "geopolitical_risk_index",
    "GPRT": "geopolitical_threats_index",
    "GPRA": "geopolitical_acts_index",
}


START_DATE = pd.Timestamp(
    "1985-01-01"
)


def download_source():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        SOURCE_URL,
        headers=headers,
        timeout=120,
    )

    print(
        "[INFO] HTTP:",
        response.status_code,
    )

    response.raise_for_status()

    return response.content


def load_data(content):
    df = pd.read_excel(
        BytesIO(content),
        sheet_name="Sheet1",
        usecols=list(
            SOURCE_COLUMNS.keys()
        ),
    )

    df = df.rename(
        columns=SOURCE_COLUMNS
    )

    return df


def clean_data(df):
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df[
        df["date"].notna()
    ].copy()

    df = df[
        df["date"] >= START_DATE
    ].copy()

    numeric_columns = [
        "geopolitical_risk_index",
        "geopolitical_threats_index",
        "geopolitical_acts_index",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df[
            [
                "date",
                "geopolitical_risk_index",
                "geopolitical_threats_index",
                "geopolitical_acts_index",
            ]
        ]
        .drop_duplicates(
            subset=["date"]
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def validate_data(df):
    if df.empty:
        raise RuntimeError(
            "GPR dataset is empty"
        )

    if df["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates detected"
        )

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError(
            "Dates are not sorted"
        )

    required_columns = [
        "geopolitical_risk_index",
        "geopolitical_threats_index",
        "geopolitical_acts_index",
    ]

    for column in required_columns:
        if df[column].notna().sum() == 0:
            raise RuntimeError(
                f"No valid values found for {column}"
            )


def main():
    print(
        "[INFO] Source: Caldara-Iacoviello GPR"
    )

    content = download_source()

    df = load_data(
        content
    )

    df = clean_data(
        df
    )

    validate_data(
        df
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        "[OK] Saved:",
        OUTPUT_FILE,
    )

    print(
        "[INFO] Rows:",
        len(df),
    )

    print(
        "[INFO] Date range:",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    print(
        "[INFO] Columns:",
        list(df.columns),
    )

    print()

    print(
        df.tail(
            12
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
