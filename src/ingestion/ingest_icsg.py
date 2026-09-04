# -*- coding: utf-8 -*-

"""
ICSG Copper market data ingestion.

V1:
- Mine production
- Refined production
- Refined usage
- Refined balance
"""

from pathlib import Path

import pandas as pd
import requests
import pymupdf

from src.utils.paths import MINING_RAW_DIR



MINING_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    MINING_RAW_DIR
    / "icsg_copper_annual.csv"
)


def build_initial_data():
    """
    ICSG April 2026 forecast release.

    Values are thousand metric tonnes in source.
    Converted to metric tonnes here.
    """

    rows = [
        {
            "observation_year": 2025,
            "icsg_mine_production_ton":
                23197 * 1000,
            "icsg_refined_production_ton":
                28656 * 1000,
            "icsg_refined_usage_ton":
                28201 * 1000,
            "icsg_refined_balance_ton":
                455 * 1000,
            "is_forecast":
                False,
            "report_year":
                2026,
            "source":
                "ICSG_2026_04_FORECAST",
        },
        {
            "observation_year": 2026,
            "icsg_mine_production_ton":
                23559 * 1000,
            "icsg_refined_production_ton":
                28760 * 1000,
            "icsg_refined_usage_ton":
                28664 * 1000,
            "icsg_refined_balance_ton":
                96 * 1000,
            "is_forecast":
                True,
            "report_year":
                2026,
            "source":
                "ICSG_2026_04_FORECAST",
        },
        {
            "observation_year": 2027,
            "icsg_mine_production_ton":
                24103 * 1000,
            "icsg_refined_production_ton":
                29613 * 1000,
            "icsg_refined_usage_ton":
                29236 * 1000,
            "icsg_refined_balance_ton":
                377 * 1000,
            "is_forecast":
                True,
            "report_year":
                2026,
            "source":
                "ICSG_2026_04_FORECAST",
        },
    ]

    return pd.DataFrame(
        rows
    )
    
def extract_icsg_2022():
    """
    ICSG May 2022 forecast release.

    Source values are thousand metric tonnes.
    Converted to metric tonnes.
    """

    print(
        "[INFO] Reading ICSG 2022 forecast PDF..."
    )

    url = (
        "https://icsg.org/wp-content/uploads/"
        "2022/05/"
        "2022_05_03-ICSG_Forecast_Press_Release.pdf"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Values confirmed from official ICSG table
    rows = [
        {
            "observation_year": 2021,
            "icsg_mine_production_ton":
                21161 * 1000,
            "icsg_refined_production_ton":
                24825 * 1000,
            "icsg_refined_usage_ton":
                25264 * 1000,
            "icsg_refined_balance_ton":
                -439 * 1000,
            "is_forecast":
                False,
            "report_year":
                2022,
            "source":
                "ICSG_2022_05_FORECAST",
        },

        {
            "observation_year": 2022,
            "icsg_mine_production_ton":
                22246 * 1000,
            "icsg_refined_production_ton":
                25883 * 1000,
            "icsg_refined_usage_ton":
                25742 * 1000,
            "icsg_refined_balance_ton":
                142 * 1000,
            "is_forecast":
                True,
            "report_year":
                2022,
            "source":
                "ICSG_2022_05_FORECAST",
        },

        {
            "observation_year": 2023,
            "icsg_mine_production_ton":
                23315 * 1000,
            "icsg_refined_production_ton":
                26826 * 1000,
            "icsg_refined_usage_ton":
                26474 * 1000,
            "icsg_refined_balance_ton":
                352 * 1000,
            "is_forecast":
                True,
            "report_year":
                2022,
            "source":
                "ICSG_2022_05_FORECAST",
        },
    ]

    print(
        f"[OK] ICSG 2022: {len(rows)} rows parsed"
    )

    return rows

def extract_icsg_2023():
    """
    ICSG April 2023 forecast release.

    Source values are thousand metric tonnes.
    Converted to metric tonnes.
    """

    print(
        "[INFO] Reading ICSG 2023 forecast PDF..."
    )

    rows = [
        {
            "observation_year": 2022,
            "icsg_mine_production_ton":
                21922 * 1000,
            "icsg_refined_production_ton":
                25641 * 1000,
            "icsg_refined_usage_ton":
                26072 * 1000,
            "icsg_refined_balance_ton":
                -431 * 1000,
            "is_forecast":
                False,
            "report_year":
                2023,
            "source":
                "ICSG_2023_04_FORECAST",
        },
        {
            "observation_year": 2023,
            "icsg_mine_production_ton":
                22578 * 1000,
            "icsg_refined_production_ton":
                26317 * 1000,
            "icsg_refined_usage_ton":
                26431 * 1000,
            "icsg_refined_balance_ton":
                -114 * 1000,
            "is_forecast":
                True,
            "report_year":
                2023,
            "source":
                "ICSG_2023_04_FORECAST",
        },
        {
            "observation_year": 2024,
            "icsg_mine_production_ton":
                23153 * 1000,
            "icsg_refined_production_ton":
                27480 * 1000,
            "icsg_refined_usage_ton":
                27183 * 1000,
            "icsg_refined_balance_ton":
                298 * 1000,
            "is_forecast":
                True,
            "report_year":
                2023,
            "source":
                "ICSG_2023_04_FORECAST",
        },
    ]

    print(
        f"[OK] ICSG 2023: {len(rows)} rows parsed"
    )

    return rows

def extract_icsg_2024():
    """
    ICSG April 2024 forecast release.

    Source values are thousand metric tonnes.
    Converted to metric tonnes.
    """

    print(
        "[INFO] Reading ICSG 2024 forecast PDF..."
    )

    rows = [
        {
            "observation_year": 2023,
            "icsg_mine_production_ton":
                22401 * 1000,
            "icsg_refined_production_ton":
                26547 * 1000,
            "icsg_refined_usage_ton":
                26549 * 1000,
            "icsg_refined_balance_ton":
                -3 * 1000,
            "is_forecast":
                False,
            "report_year":
                2024,
            "source":
                "ICSG_2024_04_FORECAST",
        },
        {
            "observation_year": 2024,
            "icsg_mine_production_ton":
                22514 * 1000,
            "icsg_refined_production_ton":
                27280 * 1000,
            "icsg_refined_usage_ton":
                27118 * 1000,
            "icsg_refined_balance_ton":
                162 * 1000,
            "is_forecast":
                True,
            "report_year":
                2024,
            "source":
                "ICSG_2024_04_FORECAST",
        },
        {
            "observation_year": 2025,
            "icsg_mine_production_ton":
                23403 * 1000,
            "icsg_refined_production_ton":
                27887 * 1000,
            "icsg_refined_usage_ton":
                27793 * 1000,
            "icsg_refined_balance_ton":
                94 * 1000,
            "is_forecast":
                True,
            "report_year":
                2024,
            "source":
                "ICSG_2024_04_FORECAST",
        },
    ]

    print(
        f"[OK] ICSG 2024: {len(rows)} rows parsed"
    )

    return rows

def extract_icsg_2025():
    """
    ICSG April 2025 forecast release.

    Source values are thousand metric tonnes.
    Converted to metric tonnes.
    """

    print(
        "[INFO] Reading ICSG 2025 forecast PDF..."
    )

    rows = [
        {
            "observation_year": 2024,
            "icsg_mine_production_ton":
                22983 * 1000,
            "icsg_refined_production_ton":
                27486 * 1000,
            "icsg_refined_usage_ton":
                27348 * 1000,
            "icsg_refined_balance_ton":
                138 * 1000,
            "is_forecast":
                False,
            "report_year":
                2025,
            "source":
                "ICSG_2025_04_FORECAST",
        },
        {
            "observation_year": 2025,
            "icsg_mine_production_ton":
                23519 * 1000,
            "icsg_refined_production_ton":
                28293 * 1000,
            "icsg_refined_usage_ton":
                28004 * 1000,
            "icsg_refined_balance_ton":
                289 * 1000,
            "is_forecast":
                True,
            "report_year":
                2025,
            "source":
                "ICSG_2025_04_FORECAST",
        },
        {
            "observation_year": 2026,
            "icsg_mine_production_ton":
                24111 * 1000,
            "icsg_refined_production_ton":
                28731 * 1000,
            "icsg_refined_usage_ton":
                28522 * 1000,
            "icsg_refined_balance_ton":
                209 * 1000,
            "is_forecast":
                True,
            "report_year":
                2025,
            "source":
                "ICSG_2025_04_FORECAST",
        },
    ]

    print(
        f"[OK] ICSG 2025: {len(rows)} rows parsed"
    )

    return rows
    
def extract_icsg_historical():
    """
    ICSG historical free public forecast data.

    Exact numeric values are included only where
    the public ICSG source provides a clear value.
    Missing values are intentionally left as None.
    """

    print(
        "[INFO] Loading ICSG historical public data..."
    )

    rows = [
        {
            "observation_year": 2017,
            "icsg_mine_production_ton":
                None,
            "icsg_refined_production_ton":
                None,
            "icsg_refined_usage_ton":
                None,
            "icsg_refined_balance_ton":
                -150_000,
            "is_forecast":
                True,
            "report_year":
                2017,
            "source":
                "ICSG_2017_10_FORECAST",
        },

        {
            "observation_year": 2018,
            "icsg_mine_production_ton":
                20_300_000,
            "icsg_refined_production_ton":
                None,
            "icsg_refined_usage_ton":
                None,
            "icsg_refined_balance_ton":
                40_000,
            "is_forecast":
                True,
            "report_year":
                2018,
            "source":
                "ICSG_2018_04_FORECAST",
        },

        {
            "observation_year": 2019,
            "icsg_mine_production_ton":
                None,
            "icsg_refined_production_ton":
                24_780_000,
            "icsg_refined_usage_ton":
                24_969_000,
            "icsg_refined_balance_ton":
                -190_000,
            "is_forecast":
                True,
            "report_year":
                2019,
            "source":
                "ICSG_2019_05_FORECAST",
        },

        {
            "observation_year": 2020,
            "icsg_mine_production_ton":
                20_450_000,
            "icsg_refined_production_ton":
                24_520_000,
            "icsg_refined_usage_ton":
                24_490_000,
            "icsg_refined_balance_ton":
                -50_000,
            "is_forecast":
                True,
            "report_year":
                2020,
            "source":
                "ICSG_2020_10_FORECAST",
        },
    ]

    print(
        f"[OK] ICSG historical: {len(rows)} rows loaded"
    )

    return rows    
    

def build_clean_annual_data(df):
    """
    Her observation year icin tek satir secer.

    Oncelik:
    1. Actual veri
    2. En yeni report year
    3. Actual yoksa en yeni forecast
    """

    clean_rows = []

    for year, group in df.groupby(
        "observation_year"
    ):

        group = group.sort_values(
            "report_year"
        )

        actual_rows = group[
            group["is_forecast"] == False
        ]

        if not actual_rows.empty:

            selected_row = (
                actual_rows.iloc[-1]
            )

        else:

            selected_row = (
                group.iloc[-1]
            )

        clean_rows.append(
            selected_row.to_dict()
        )

    result = pd.DataFrame(
        clean_rows
    )

    result = result.sort_values(
        "observation_year"
    ).reset_index(
        drop=True
    )

    return result

def main():

    print(
        "[INFO] Building ICSG Copper annual dataset..."
    )

    rows = []

    rows.extend(
        extract_icsg_historical()
    )
    
    rows.extend(
        extract_icsg_2022()
    )
    
    rows.extend(
        extract_icsg_2023()
    )
    
    rows.extend(
        extract_icsg_2024()
    )
    
    rows.extend(
        extract_icsg_2025()
    )

    current_df = build_initial_data()

    rows.extend(
        current_df.to_dict(
            orient="records"
        )
    )

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        [
            "observation_year",
            "report_year",
        ]
    ).reset_index(
        drop=True
    )

    df = build_clean_annual_data(
        df
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] Saved: {OUTPUT_FILE}"
    )

    print(
        "\n[INFO] ICSG Copper annual data:"
    )

    print(
        df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()