from pathlib import Path
from io import BytesIO

import pandas as pd
import requests


BASE_DIR = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "economic_policy_uncertainty_monthly.csv"
)


SOURCE_URL = (
    "https://www.policyuncertainty.com/"
    "media/Global_Policy_Uncertainty_Data.xlsx"
)


OUTPUT_COLUMNS = [
    "date",
    "global_economic_policy_uncertainty_index",
    "global_economic_policy_uncertainty_ppp_index",
]


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
        usecols=[
            "Year",
            "Month",
            "GEPU_current",
            "GEPU_ppp",
        ],
    )

    return df


def clean_data(df):
    df["date"] = pd.to_datetime(
        {
            "year": pd.to_numeric(
                df["Year"],
                errors="coerce",
            ),
            "month": pd.to_numeric(
                df["Month"],
                errors="coerce",
            ),
            "day": 1,
        },
        errors="coerce",
    )

    df = df.rename(
        columns={
            "GEPU_current": (
                "global_economic_policy_uncertainty_index"
            ),
            "GEPU_ppp": (
                "global_economic_policy_uncertainty_ppp_index"
            ),
        }
    )

    df = df[
        OUTPUT_COLUMNS
    ].copy()

    numeric_columns = [
        "global_economic_policy_uncertainty_index",
        "global_economic_policy_uncertainty_ppp_index",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[
        df["date"].notna()
    ].copy()

    df = (
        df
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
            "Global EPU dataset is empty"
        )

    if df["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates detected"
        )

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError(
            "Dates are not sorted ascending"
        )

    for column in OUTPUT_COLUMNS[1:]:
        valid_count = df[
            column
        ].notna().sum()

        if valid_count == 0:
            raise RuntimeError(
                f"No valid values found for {column}"
            )


def main():
    print(
        "[INFO] Source: Global Economic Policy Uncertainty"
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
