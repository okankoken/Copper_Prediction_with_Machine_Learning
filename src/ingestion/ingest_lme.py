# -*- coding: utf-8 -*-

"""
Copper data ingestion module.

V1 scope:
- Fetch daily LME Copper data from Westmetall
- Save raw daily data under data/raw
- Keep source data unmodified except basic parsing and type conversion

Code comments use ASCII-only Turkish.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================
# PROJECT PATHS
# =========================

from src.utils.paths import MARKET_RAW_DIR


MARKET_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LME_OUTPUT_FILE = (
    MARKET_RAW_DIR
    / "lme_copper_daily.csv"
)


# =========================
# WESTMETALL SETTINGS
# =========================

WESTMETALL_BASE_URL = (
    "https://www.westmetall.com/en/markdaten.php"
    "?action=table&field=LME_Cu_cash"
)

START_YEAR = 2008

MIN_DATE = pd.Timestamp("2008-01-01")

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "Chrome/151 Safari/537.36"
    ),
    "Accept-Language": "en,en-US;q=0.9",
}


# =========================
# DATE PARSING
# =========================

DATE_RE = re.compile(
    r"^\s*(\d{1,2})\.\s+([A-Za-z]+)\s+(\d{4})$"
)


def parse_date_en(value: str) -> pd.Timestamp:
    """
    Ingilizce tarih metnini pandas Timestamp'e cevirir.

    Ornek:
    08. August 2026
    """

    value = str(value).strip()

    match = DATE_RE.match(value)

    if not match:
        return pd.NaT

    day, month_name, year = match.groups()

    try:
        parsed = datetime.strptime(
            f"{int(day)} {month_name} {int(year)}",
            "%d %B %Y",
        )

        return pd.Timestamp(parsed)

    except ValueError:
        return pd.NaT


# =========================
# HTML TABLE FINDER
# =========================

def find_year_tables(html: bytes):
    """
    Westmetall sayfasindaki yil bazli tablolarini bulur.
    """

    soup = BeautifulSoup(html, "html.parser")

    anchors = {}

    for anchor in soup.find_all("a"):
        anchor_id = anchor.get("id") or ""

        if anchor_id.startswith("y") and anchor_id[1:].isdigit():
            anchors[int(anchor_id[1:])] = anchor

    # Ilk yontem calismazsa h2 basliklarindan yil ara
    if not anchors:
        for h2 in soup.find_all("h2"):
            text = h2.get_text() or ""

            years = re.findall(
                r"(20\d{2}|19\d{2})",
                text,
            )

            if years:
                anchors[int(years[0])] = h2

    for year in sorted(anchors.keys()):
        node = anchors[year]

        if node.name == "h2":
            heading = node
        else:
            heading = node.find_next("h2")

        if heading:
            table = heading.find_next("table")
        else:
            table = node.find_next("table")

        if table:
            yield year, str(table)


# =========================
# LME INGESTION
# =========================

def fetch_lme_copper() -> pd.DataFrame:
    """
    Westmetall'dan 2008'den bugune kadar
    gunluk LME Copper verisini ceker.

    Beklenen kolonlar:
    - date
    - cash_settlement_usd_per_ton
    - copper_3_month_usd_per_ton
    - copper_stock_ton
    """

    current_year = pd.Timestamp.today().year

    print(
        f"[INFO] Fetching Westmetall LME Copper data "
        f"from {START_YEAR} to {current_year}..."
    )

    parts = []

    for year in range(START_YEAR, current_year + 1):

        year_url = (
            f"{WESTMETALL_BASE_URL}&year={year}"
        )

        print(
            f"[INFO] Fetching year {year}..."
        )

        try:
            response = requests.get(
                year_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            print(
                f"[WARN] Request failed for {year}: {exc}"
            )
            continue

        try:
            tables = pd.read_html(
                io.StringIO(response.text),
                decimal=".",
                thousands=",",
            )

        except Exception as exc:
            print(
                f"[WARN] Could not read HTML tables for {year}: {exc}"
            )
            continue

        year_df = None

        for table_df in tables:

            normalized_columns = []

            for column in table_df.columns:

                clean_column = (
                    str(column)
                    .lower()
                    .strip()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                normalized_columns.append(
                    clean_column
                )

            table_df.columns = normalized_columns

            rename_map = {}

            for column in table_df.columns:

                if column.startswith("date"):
                    rename_map[column] = "date"

                elif "cash" in column:
                    rename_map[column] = (
                        "cash_settlement_usd_per_ton"
                    )

                elif (
                    "3_month" in column
                    or "3month" in column
                ):
                    rename_map[column] = (
                        "copper_3_month_usd_per_ton"
                    )

                elif "stock" in column:
                    rename_map[column] = (
                        "copper_stock_ton"
                    )

            table_df = table_df.rename(
                columns=rename_map
            )

            required_columns = {
                "date",
                "cash_settlement_usd_per_ton",
                "copper_3_month_usd_per_ton",
                "copper_stock_ton",
            }

            if required_columns.issubset(
                table_df.columns
            ):
                year_df = table_df.copy()
                break

        if year_df is None:
            print(
                f"[WARN] LME Copper table not found for {year}"
            )
            continue

        year_df = year_df[
            [
                "date",
                "cash_settlement_usd_per_ton",
                "copper_3_month_usd_per_ton",
                "copper_stock_ton",
            ]
        ].copy()

        year_df["date"] = (
            year_df["date"]
            .astype(str)
            .map(parse_date_en)
        )

        numeric_columns = [
            "cash_settlement_usd_per_ton",
            "copper_3_month_usd_per_ton",
            "copper_stock_ton",
        ]

        for column in numeric_columns:

            year_df[column] = pd.to_numeric(
                year_df[column],
                errors="coerce",
            )

        year_df = year_df.dropna(
            subset=["date"]
        )

        # Yanlis yil sayfasi gelirse filtrele
        year_df = year_df[
            year_df["date"].dt.year == year
        ].copy()

        if year_df.empty:
            print(
                f"[WARN] No valid rows found for {year}"
            )
            continue

        parts.append(year_df)

        print(
            f"[OK] {year}: "
            f"{len(year_df)} rows parsed"
        )

        time.sleep(SLEEP_SECONDS)

    if not parts:
        raise RuntimeError(
            "No LME Copper data could be downloaded."
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    result = result[
        result["date"] >= MIN_DATE
    ].copy()

    return result


# =========================
# SAVE RAW DATA
# =========================

def save_lme_raw(df: pd.DataFrame) -> None:
    """
    Gunluk LME verisini raw katmanina kaydeder.
    """

    df.to_csv(
        LME_OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] Saved raw LME data: "
        f"{LME_OUTPUT_FILE}"
    )

    print(
        f"[OK] Rows: {len(df)}"
    )

    print(
        f"[OK] First date: "
        f"{df['date'].min()}"
    )

    print(
        f"[OK] Last date: "
        f"{df['date'].max()}"
    )


# =========================
# MAIN
# =========================

def main():

    lme_df = fetch_lme_copper()

    save_lme_raw(lme_df)

    print("\n[INFO] Last 5 rows:")

    print(
        lme_df.tail(5).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
