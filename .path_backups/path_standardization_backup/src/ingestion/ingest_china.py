from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

BASE_API_URL = (
    "https://chinadata.live/api/v2/data"
)


DATASETS = {
    "refined_copper": {
        "slug": "china-refined-copper-production",
        "output_file": "china_refined_copper_monthly.csv",
    },
    "industrial_output": {
        "slug": "china-industrial-output",
        "output_file": "china_industrial_output_monthly.csv",
    },
    "ppi": {
        "slug": "china-ppi",
        "output_file": "china_ppi_monthly.csv",
    },
    
    "electricity_generation": {
        "slug": "china-electricity-generation",
        "output_file": "china_electricity_generation_monthly.csv",
    },
    
    "fixed_asset_investment": {
        "slug": "china-fai",
        "output_file": "china_fixed_asset_investment_monthly.csv",
    },
    "real_estate_investment": {
        "slug": "china-real-estate-investment-yoy",
        "output_file": "china_real_estate_investment_monthly.csv",
    },
}

def prepare_fixed_asset_investment(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value": "china_fixed_asset_investment_yoy_pct",
        }
    )

    df["month"] = df["month"].astype(str)

    df[
        "china_fixed_asset_investment_yoy_pct"
    ] = pd.to_numeric(
        df[
            "china_fixed_asset_investment_yoy_pct"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2007-01"
    ].copy()

    df["unit"] = (
        "% YoY cumulative comparable"
    )

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_fixed_asset_investment_yoy_pct",
            "unit",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )


def prepare_real_estate_investment(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value": "china_real_estate_investment_yoy_pct",
        }
    )

    df["month"] = df["month"].astype(str)

    df[
        "china_real_estate_investment_yoy_pct"
    ] = pd.to_numeric(
        df[
            "china_real_estate_investment_yoy_pct"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2007-01"
    ].copy()

    df["unit"] = (
        "% YoY cumulative"
    )

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_real_estate_investment_yoy_pct",
            "unit",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )

def prepare_electricity_generation(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value": "china_electricity_generation_100m_kwh",
        }
    )

    df["month"] = (
        df["month"]
        .astype(str)
    )

    df[
        "china_electricity_generation_100m_kwh"
    ] = pd.to_numeric(
        df[
            "china_electricity_generation_100m_kwh"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2007-01"
    ].copy()

    df["unit"] = (
        "100M kWh current period"
    )

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_electricity_generation_100m_kwh",
            "unit",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )

def fetch_dataset(slug):
    url = f"{BASE_API_URL}/{slug}"

    print(
        f"\n[INFO] Fetching: {slug}"
    )

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get("success"):
        raise ValueError(
            f"API returned success=false for {slug}"
        )

    dataset = payload.get(
        "data",
        {}
    )

    observations = dataset.get(
        "data",
        []
    )

    if not observations:
        raise ValueError(
            f"No observations returned for {slug}"
        )

    print(
        "[INFO] Title:",
        dataset.get("title"),
    )

    print(
        "[INFO] Source:",
        dataset.get("source"),
    )

    print(
        "[INFO] Unit:",
        dataset.get("unit"),
    )

    print(
        "[INFO] Frequency:",
        dataset.get("frequency"),
    )

    print(
        "[INFO] Observations:",
        len(observations),
    )

    return dataset


def prepare_refined_copper(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value": "source_value_10000_ton",
        }
    )

    df["month"] = (
        df["month"]
        .astype(str)
    )

    df[
        "source_value_10000_ton"
    ] = pd.to_numeric(
        df[
            "source_value_10000_ton"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2005-01"
    ].copy()

    df[
        "china_refined_copper_production_ton"
    ] = (
        df[
            "source_value_10000_ton"
        ]
        * 10000
    )

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_refined_copper_production_ton",
            "source_value_10000_ton",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )


def prepare_industrial_output(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value":
                "china_industrial_value_added_yoy_pct",
        }
    )

    df["month"] = (
        df["month"]
        .astype(str)
    )

    df[
        "china_industrial_value_added_yoy_pct"
    ] = pd.to_numeric(
        df[
            "china_industrial_value_added_yoy_pct"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2007-01"
    ].copy()

    df["unit"] = (
        "% YoY real comparable prices"
    )

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_industrial_value_added_yoy_pct",
            "unit",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )


def prepare_ppi(dataset):
    df = pd.DataFrame(
        dataset["data"]
    )

    df = df.rename(
        columns={
            "date": "month",
            "value": "china_ppi_yoy_pct",
        }
    )

    df["month"] = (
        df["month"]
        .astype(str)
    )

    df[
        "china_ppi_yoy_pct"
    ] = pd.to_numeric(
        df[
            "china_ppi_yoy_pct"
        ],
        errors="coerce",
    )

    df = df[
        df["month"] >= "2005-01"
    ].copy()

    df["unit"] = "% YoY"

    df["source"] = (
        "National Bureau of Statistics "
        "of China via China Data Portal"
    )

    df = df[
        [
            "month",
            "china_ppi_yoy_pct",
            "unit",
            "source",
        ]
    ]

    return clean_monthly_dataframe(
        df
    )


def clean_monthly_dataframe(df):
    df = (
        df
        .sort_values("month")
        .drop_duplicates(
            subset=["month"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def save_dataframe(
    df,
    output_file,
):
    output_path = (
        RAW_DIR
        / output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n[OK] Saved:"
    )

    print(
        output_path
    )

    print(
        "[INFO] Shape:",
        df.shape,
    )

    print(
        "[INFO] Period:",
        df["month"].min(),
        "->",
        df["month"].max(),
    )

    print(
        "\n[INFO] Latest rows:"
    )

    print(
        df.tail(5).to_string(
            index=False
        )
    )


def main():
    print(
        "=" * 80
    )

    print(
        "CHINA DATA INGESTION"
    )

    print(
        "=" * 80
    )

    refined_dataset = fetch_dataset(
        DATASETS[
            "refined_copper"
        ]["slug"]
    )

    refined_df = (
        prepare_refined_copper(
            refined_dataset
        )
    )

    save_dataframe(
        refined_df,
        DATASETS[
            "refined_copper"
        ]["output_file"],
    )

    industrial_dataset = fetch_dataset(
        DATASETS[
            "industrial_output"
        ]["slug"]
    )

    industrial_df = (
        prepare_industrial_output(
            industrial_dataset
        )
    )

    save_dataframe(
        industrial_df,
        DATASETS[
            "industrial_output"
        ]["output_file"],
    )

    ppi_dataset = fetch_dataset(
        DATASETS[
            "ppi"
        ]["slug"]
    )

    electricity_dataset = fetch_dataset(
        DATASETS[
            "electricity_generation"
        ]["slug"]
    )
    
    electricity_df = (
        prepare_electricity_generation(
            electricity_dataset
        )
    )
    
    fixed_asset_dataset = fetch_dataset(
        DATASETS[
            "fixed_asset_investment"
        ]["slug"]
    )
    
    fixed_asset_df = (
        prepare_fixed_asset_investment(
            fixed_asset_dataset
        )
    )
    
    save_dataframe(
        fixed_asset_df,
        DATASETS[
            "fixed_asset_investment"
        ]["output_file"],
    )
    
    
    real_estate_dataset = fetch_dataset(
        DATASETS[
            "real_estate_investment"
        ]["slug"]
    )
    
    real_estate_df = (
        prepare_real_estate_investment(
            real_estate_dataset
        )
    )
    
    save_dataframe(
        real_estate_df,
        DATASETS[
            "real_estate_investment"
        ]["output_file"],
    )
        
    
    save_dataframe(
        electricity_df,
        DATASETS[
            "electricity_generation"
        ]["output_file"],
    )

    ppi_df = prepare_ppi(
        ppi_dataset
    )

    save_dataframe(
        ppi_df,
        DATASETS[
            "ppi"
        ]["output_file"],
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "[OK] China ingestion completed."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
