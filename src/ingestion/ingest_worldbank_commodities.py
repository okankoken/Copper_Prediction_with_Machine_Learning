from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from src.utils.paths import MARKET_RAW_DIR
OUTPUT_FILE = (
    MARKET_RAW_DIR
    / "worldbank_commodities_monthly.csv"
)
URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-"
    "0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    )
}


COLUMN_MAP = {
    "Gold": "gold_usd_per_troy_ounce",
    "Silver": "silver_usd_per_troy_ounce",
    "Platinum": "platinum_usd_per_troy_ounce",
    "Iron ore, cfr spot": "iron_ore_usd_per_dmtu",
    "Crude oil, WTI": "wti_usd_per_barrel",
    "Coal, Australian": "coal_australia_usd_per_ton",
}


def get_last_full_month():
    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_full_month_end.to_period("M")


def download_workbook():
    print(
        "[INFO] Downloading World Bank "
        "commodity data..."
    )

    response = requests.get(
        URL,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    print(
        "[OK] World Bank workbook downloaded"
    )

    return response.content


def load_monthly_prices(
    content,
):
    # Row 4 contains commodity names.
    df = pd.read_excel(
        BytesIO(content),
        sheet_name="Monthly Prices",
        header=4,
    )

    first_column = df.columns[0]

    df = df.rename(
        columns={
            first_column: "date"
        }
    )

    required_columns = [
        "date",
        *COLUMN_MAP.keys(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing World Bank columns: "
            f"{missing_columns}"
        )

    df = df[
        required_columns
    ].copy()

    df = df.rename(
        columns=COLUMN_MAP
    )

    return df


def clean_data(
    df,
    start_period="2005-01",
):
    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
    )

    valid_date = (
        df["date"]
        .str.match(
            r"^\d{4}M\d{2}$",
            na=False,
        )
    )

    df = df[
        valid_date
    ].copy()

    df["date"] = pd.PeriodIndex(
        df["date"]
        .str.replace(
            "M",
            "-",
            regex=False,
        ),
        freq="M",
    )

    numeric_columns = [
        column
        for column in df.columns
        if column != "date"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df = df[
        (df["date"] >= start_month)
        & (df["date"] <= last_full_month)
    ].copy()

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

    return df


def print_summary(
    df,
):
    print(
        "\n"
        + "=" * 100
    )

    print(
        "WORLD BANK COMMODITY SUMMARY"
    )

    print(
        "=" * 100
    )

    for column in df.columns:

        if column == "date":
            continue

        valid = df[
            column
        ].notna()

        print(
            f"\n{column}"
        )

        print(
            "  observations:",
            int(valid.sum()),
        )

        if valid.any():

            print(
                "  range:",
                df.loc[
                    valid,
                    "date",
                ].min(),
                "->",
                df.loc[
                    valid,
                    "date",
                ].max(),
            )

            print(
                "  min:",
                df.loc[
                    valid,
                    column,
                ].min(),
            )

            print(
                "  max:",
                df.loc[
                    valid,
                    column,
                ].max(),
            )


def main():

    print(
        "=" * 100
    )

    print(
        "WORLD BANK COMMODITY INGESTION"
    )

    print(
        "=" * 100
    )

    content = (
        download_workbook()
    )

    df = load_monthly_prices(
        content
    )

    df = clean_data(
        df,
        start_period="2005-01",
    )

    if df.empty:
        raise ValueError(
            "World Bank commodity dataset is empty"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] Saved: {OUTPUT_FILE}"
    )

    print(
        "[OK] Rows:",
        len(df),
    )

    print(
        "[OK] Range:",
        df["date"].min(),
        "->",
        df["date"].max(),
    )

    print_summary(
        df
    )

    print(
        "\n[INFO] Last 12 rows:"
    )

    print(
        df.tail(12).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
