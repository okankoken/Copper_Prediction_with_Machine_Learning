from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from src.utils.paths import MARKET_RAW_DIR
PALLADIUM_OUTPUT_FILE = (
    MARKET_RAW_DIR
    / "palladium_daily.csv"
)
BIST_PALLADIUM_URL = (
    "https://borsaistanbul.com/"
    "metal-fiyatlari.php"
)
OUTPUT_FILE = (
    MARKET_RAW_DIR
    / "lme_other_metals_daily.csv"
)
BASE_URL = (
    "https://www.westmetall.com/en/"
    "markdaten.php"
)


METALS = {
    "aluminum": "LME_Al_cash",
    "nickel": "LME_Ni_cash",
    "zinc": "LME_Zn_cash",
    "lead": "LME_Pb_cash",
    "tin": "LME_Sn_cash",
}


START_YEAR = 2008


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    )
}


def clean_numeric(
    series,
):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.strip(),
        errors="coerce",
    )


def fetch_metal_year(
    metal_name,
    field,
    year,
):
    print(
        f"[INFO] Fetching {metal_name}: "
        f"{year}"
    )

    params = {
        "action": "table",
        "field": field,
        "year": year,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(
            response.text
        )
    )

    if not tables:
        raise ValueError(
            f"No table found for "
            f"{metal_name} {year}"
        )

    df = tables[0].copy()

    if df.shape[1] != 4:
        raise ValueError(
            f"Unexpected column count for "
            f"{metal_name} {year}: "
            f"{df.shape[1]}"
        )

    df.columns = [
        "date",
        f"{metal_name}_cash_usd_per_ton",
        f"{metal_name}_3_month_usd_per_ton",
        f"{metal_name}_stock_ton",
    ]

    # Remove repeated header rows.
    df = df[
        df["date"].astype(str).str.lower()
        != "date"
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d. %B %Y",
        errors="coerce",
    )

    numeric_columns = [
        f"{metal_name}_cash_usd_per_ton",
        f"{metal_name}_3_month_usd_per_ton",
        f"{metal_name}_stock_ton",
    ]

    for column in numeric_columns:
        df[column] = clean_numeric(
            df[column]
        )

    df = df.dropna(
        subset=[
            "date",
        ]
    )

    df = (
        df
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

    print(
        f"[OK] {metal_name} {year}: "
        f"{len(df)} rows"
    )

    return df

def fetch_palladium():
    """
    Fetch Palladium prices from Borsa Istanbul.

    The selected observations are:
    - Palladium
    - USD
    - OZ
    - Metal price reference
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BORSA ISTANBUL PALLADIUM INGESTION"
    )

    print(
        "=" * 100
    )

    params = {
        "op": "fetchMetalFiyatlari",
        "startDate": "2005-01-01",
        "endDate": (
            pd.Timestamp.today()
            .strftime("%Y-%m-%d")
        ),
        "priceType": "PD",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        ),
        "Referer": (
            "https://borsaistanbul.com/en/data/"
            "precious-metals-and-diamond-market"
        ),
    }

    print(
        "[INFO] Fetching Palladium..."
    )

    response = requests.get(
        BIST_PALLADIUM_URL,
        params=params,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    data = payload.get(
        "data",
        []
    )

    rows = []

    for item in data:

        if item.get("priceRef") != "MTL":
            continue

        if item.get("priceType") != "PD":
            continue

        if item.get("priceCurrency") != "USD":
            continue

        if item.get("priceWeight") != "OZ":
            continue

        rows.append(
            {
                "date": item.get(
                    "priceDate"
                ),
                "palladium_usd_per_troy_ounce": (
                    item.get(
                        "priceValue"
                    )
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No Palladium USD/OZ data found"
        )

    df = pd.DataFrame(
        rows
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df[
        "palladium_usd_per_troy_ounce"
    ] = pd.to_numeric(
        df[
            "palladium_usd_per_troy_ounce"
        ],
        errors="coerce",
    )

    df = (
        df
        .dropna()
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

    PALLADIUM_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PALLADIUM_OUTPUT_FILE,
        index=False,
    )

    print(
        "[OK] Palladium observations:",
        len(df),
    )

    print(
        "[OK] Palladium range:",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    print(
        "[OK] Saved:",
        PALLADIUM_OUTPUT_FILE,
    )

    print(
        "\n[INFO] Last 10 rows:"
    )

    print(
        df.tail(10).to_string(
            index=False
        )
    )

    return df

def fetch_metal(
    metal_name,
    field,
):
    current_year = (
        pd.Timestamp.today().year
    )

    frames = []

    for year in range(
        START_YEAR,
        current_year + 1,
    ):
        try:
            df_year = fetch_metal_year(
                metal_name=metal_name,
                field=field,
                year=year,
            )

            if not df_year.empty:
                frames.append(
                    df_year
                )

        except Exception as exc:
            print(
                f"[WARNING] {metal_name} "
                f"{year} failed: {exc}"
            )

    if not frames:
        raise ValueError(
            f"No valid data found for "
            f"{metal_name}"
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
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

    print(
        f"[OK] {metal_name}: "
        f"{len(result)} daily rows"
    )

    print(
        f"[OK] {metal_name} range: "
        f"{result['date'].min().date()} -> "
        f"{result['date'].max().date()}"
    )

    return result


def build_metals_dataset():
    print(
        "=" * 100
    )

    print(
        "WESTMETALL LME OTHER METALS INGESTION"
    )

    print(
        "=" * 100
    )

    metal_frames = []

    for metal_name, field in METALS.items():
        metal_df = fetch_metal(
            metal_name=metal_name,
            field=field,
        )

        metal_frames.append(
            metal_df
        )

    result = metal_frames[0].copy()

    for frame in metal_frames[1:]:
        result = result.merge(
            frame,
            on="date",
            how="outer",
        )

    result = (
        result
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"[OK] Saved: {OUTPUT_FILE}"
    )

    print(
        f"[OK] Rows: {len(result)}"
    )

    print(
        "[OK] First date:",
        result["date"].min().date(),
    )

    print(
        "[OK] Last date:",
        result["date"].max().date(),
    )

    print(
        "\n[INFO] Columns:"
    )

    for column in result.columns:
        print(
            column
        )

    print(
        "\n[INFO] Last 10 rows:"
    )

    print(
        result.tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    build_metals_dataset()
    fetch_palladium()