# -*- coding: utf-8 -*-

"""
USGS Copper annual ingestion.

Source:
USGS Mineral Commodity Summaries Data Releases.

V1 scope:
- MCS2022
- MCS2023
- MCS2024
- MCS2025

Outputs:
- World copper mine production
- World copper refinery production
- World copper reserves
- Report year
- Estimate flag
- Source release

Ayni observation year farkli raporlarda tekrar gelebilir.
Bu bilincli bir tercihtir; revision history korunur.
"""

from pathlib import Path
import re

import pandas as pd
import requests
from io import BytesIO
import zipfile
import pymupdf


# =========================
# PROJECT PATHS
# =========================

from src.utils.paths import MINING_RAW_DIR


MINING_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    MINING_RAW_DIR
    / "usgs_copper_annual.csv"
)


# =========================
# USGS RELEASES
# =========================

USGS_RELEASES = {
    2022: "61ead310d34e8b818ad9f3a5",
    2023: "63d1a36ed34e06fef150068f",
    2024: "65b7d77ed34e36a39045b4b2",
    2025: "6797fba5d34ea8c18376e15d",
}

MCS2025_ZIP_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "677eaf95d34e760b392c4970"
    "?f=__disk__70%2Fcf%2F36%2F70cf3695ad9405884df4a4758e4b609013e3fb1e"
)


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# =========================
# HELPERS
# =========================

def normalize_columns(df):
    """
    Kolon isimlerini standart hale getirir.
    """

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    return df


def find_world_csv_url(item_id):
    """
    ScienceBase item metadata icinden world CSV dosyasini bulur.
    """

    metadata_url = (
        f"https://www.sciencebase.gov/catalog/item/"
        f"{item_id}?format=json"
    )

    response = requests.get(
        metadata_url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    files = data.get(
        "files",
        []
    )

    for file_info in files:

        name = str(
            file_info.get("name", "")
        ).lower()

        url = file_info.get(
            "url"
        )

        if (
            "world" in name
            and name.endswith(".csv")
            and url
        ):
            return url

    return None


def read_world_csv(
    report_year,
    item_id,
):
    """
    Belirli bir MCS release icindeki world CSV dosyasini okur.
    MCS2025 ZIP formatinda oldugu icin ayri ele alinir.
    """

    print(
        f"[INFO] Reading MCS{report_year}..."
    )

    if report_year == 2025:

        response = requests.get(
            MCS2025_ZIP_URL,
            headers=HEADERS,
            timeout=60,
        )

        response.raise_for_status()

        with zipfile.ZipFile(
            BytesIO(response.content)
        ) as z:

            csv_name = None

            for name in z.namelist():

                if (
                    "world_data.csv"
                    in name.lower()
                ):
                    csv_name = name
                    break

            if not csv_name:
                raise RuntimeError(
                    "MCS2025 CSV not found inside ZIP."
                )

            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        print(
            "[OK] MCS2025 ZIP CSV found."
        )

    else:

        url = find_world_csv_url(
            item_id
        )

        if not url:
            raise RuntimeError(
                f"MCS{report_year} world CSV not found."
            )

        print(
            f"[OK] MCS{report_year} world CSV found."
        )

        df = pd.read_csv(
            url
        )

    df = normalize_columns(
        df
    )

    return df


def find_column(
    columns,
    patterns,
):
    """
    Kolon isimleri farkli release'lerde degisebildigi icin
    regex ile uygun kolonu bulur.
    """

    for column in columns:

        for pattern in patterns:

            if re.search(
                pattern,
                column,
                flags=re.I,
            ):
                return column

    return None


def extract_year_from_column(
    column_name,
):
    """
    Kolon isminden 4 haneli yili cikarir.
    """

    match = re.search(
        r"(20\d{2})",
        str(column_name),
    )

    if match:
        return int(
            match.group(1)
        )

    return None


def safe_numeric(value):
    """
    Degeri guvenli sekilde numeric'e cevirir.
    """

    return pd.to_numeric(
        value,
        errors="coerce",
    )


# =========================
# EXTRACTION
# =========================

def extract_world_rows(
    df,
    report_year,
):
    """
    World total mine/refinery/reserve verilerini cikarir.
    """

    columns = list(
        df.columns
    )

    country_col = find_column(
        columns,
        [
            r"^country$",
        ],
    )

    type_col = find_column(
        columns,
        [
            r"^type$",
        ],
    )

    if not country_col:
        raise RuntimeError(
            f"MCS{report_year}: country column not found."
        )

    if not type_col:
        raise RuntimeError(
            f"MCS{report_year}: type column not found."
        )

    # Dosyada commodity kolonu varsa sadece copper verisini tut
    commodity_col = find_column(
        columns,
        [
            r"^commodity$",
        ],
    )

    if commodity_col:

        df = df[
            df[commodity_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("copper")
        ].copy()

        if df.empty:
            raise RuntimeError(
                f"MCS{report_year}: copper rows not found."
            )


    world = df[
        df[country_col]
        .astype(str)
        .str.contains(
            "world total",
            case=False,
            na=False,
        )
    ].copy()

    if world.empty:
        raise RuntimeError(
            f"MCS{report_year}: world total rows not found."
        )

    mine_rows = world[
        world[type_col]
        .astype(str)
        .str.contains(
            "mine production",
            case=False,
            na=False,
        )
    ].copy()

    refinery_rows = world[
        world[type_col]
        .astype(str)
        .str.contains(
            "refinery production",
            case=False,
            na=False,
        )
    ].copy()

    if mine_rows.empty:
        raise RuntimeError(
            f"MCS{report_year}: mine production row not found."
        )

    mine_row = mine_rows.iloc[0]

    refinery_row = (
        refinery_rows.iloc[0]
        if not refinery_rows.empty
        else None
    )

    # -------------------------
    # Production columns
    # -------------------------

    production_columns = []

    for column in columns:

        if (
            "prod" in column.lower()
            and re.search(
                r"20\d{2}",
                column
            )
        ):
            production_columns.append(
                column
            )

    rows = []

    for production_column in production_columns:

        observation_year = (
            extract_year_from_column(
                production_column
            )
        )

        if observation_year is None:
            continue

        mine_value = safe_numeric(
            mine_row.get(
                production_column
            )
        )

        refinery_value = None

        if refinery_row is not None:
            refinery_value = safe_numeric(
                refinery_row.get(
                    production_column
                )
            )

        # USGS world tablolarinda production genelde kt
        if pd.notna(mine_value):
            mine_value = (
                float(mine_value)
                * 1000
            )

        if pd.notna(refinery_value):
            refinery_value = (
                float(refinery_value)
                * 1000
            )

        column_lower = (
            production_column
            .lower()
        )

        is_estimate = any(
            token in column_lower
            for token in [
                "est",
                "estimate",
            ]
        )

        rows.append(
            {
                "observation_year":
                    observation_year,

                "world_copper_mine_production_ton":
                    mine_value,

                "world_copper_refinery_production_ton":
                    refinery_value,

                "world_copper_reserves_ton":
                    None,

                "is_estimate":
                    is_estimate,

                "report_year":
                    report_year,

                "source":
                    f"USGS_MCS{report_year}",
            }
        )

    # -------------------------
    # Reserve column
    # -------------------------

    reserve_column = find_column(
        columns,
        [
            r"reserve.*20\d{2}",
            r"reserves_kt",
            r"^reserves$",
        ],
    )

    if reserve_column:

        reserve_year = (
            extract_year_from_column(
                reserve_column
            )
        )

        if reserve_year is None:
            reserve_year = (
                report_year - 1
            )

        reserve_value = safe_numeric(
            mine_row.get(
                reserve_column
            )
        )

        if pd.notna(reserve_value):

            reserve_value = (
                float(reserve_value)
                * 1000
            )

            matching_row = None

            for row in rows:

                if (
                    row["observation_year"]
                    == reserve_year
                ):
                    matching_row = row
                    break

            if matching_row is not None:

                matching_row[
                    "world_copper_reserves_ton"
                ] = reserve_value

            else:

                rows.append(
                    {
                        "observation_year":
                            reserve_year,

                        "world_copper_mine_production_ton":
                            None,

                        "world_copper_refinery_production_ton":
                            None,

                        "world_copper_reserves_ton":
                            reserve_value,

                        "is_estimate":
                            False,

                        "report_year":
                            report_year,

                        "source":
                            f"USGS_MCS{report_year}",
                    }
                )

    return rows

def extract_mcs2021_pdf_rows():
    """
    USGS MCS2021 Copper PDF tablosundan
    dunya bakir verilerini cikarir.

    PDF tablosu:
    - 2019 actual
    - 2020 estimate
    - 2020 reserves
    """

    print(
        "[INFO] Reading MCS2021 Copper PDF..."
    )

    url = (
        "https://pubs.usgs.gov/"
        "periodicals/mcs2021/mcs2021.pdf"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Copper World Production tablosu PDF page 57
    page = doc[56]

    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    world_index = None

    for i, line in enumerate(lines):

        if (
            "world total" in line.lower()
        ):
            world_index = i
            break

    if world_index is None:
        raise RuntimeError(
            "MCS2021 world total row not found."
        )

    # World total satirindan sonraki degerler:
    # Mine 2019
    # Mine 2020 estimate
    # Refinery 2019
    # Refinery 2020 estimate
    # Reserves 2020

    values = []

    for line in lines[
        world_index + 1:
        world_index + 10
    ]:

        cleaned = (
            line
            .replace(",", "")
            .strip()
        )

        try:
            value = float(cleaned)
            values.append(value)
        except ValueError:
            continue

        if len(values) == 5:
            break

    if len(values) < 5:
        raise RuntimeError(
            "MCS2021 world values could not be parsed."
        )

    mine_2019 = values[0] * 1000
    mine_2020 = values[1] * 1000

    refinery_2019 = values[2] * 1000
    refinery_2020 = values[3] * 1000

    reserves_2020 = values[4] * 1000

    rows = [
        {
            "observation_year": 2019,

            "world_copper_mine_production_ton":
                mine_2019,

            "world_copper_refinery_production_ton":
                refinery_2019,

            "world_copper_reserves_ton":
                None,

            "is_estimate":
                False,

            "report_year":
                2021,

            "source":
                "USGS_MCS2021",
        },

        {
            "observation_year": 2020,

            "world_copper_mine_production_ton":
                mine_2020,

            "world_copper_refinery_production_ton":
                refinery_2020,

            "world_copper_reserves_ton":
                reserves_2020,

            "is_estimate":
                True,

            "report_year":
                2021,

            "source":
                "USGS_MCS2021",
        },
    ]

    print(
        "[OK] MCS2021: "
        f"{len(rows)} observation rows parsed"
    )

    return rows

def extract_mcs2018_pdf_rows():
    """
    USGS MCS2018 Copper PDF verilerini cikarir.

    PDF:
    - 2016 actual mine production
    - 2017 estimate mine production
    - 2017 reserves

    Global refinery production tabloda verilmedigi icin
    refinery bos birakilir.
    """

    print(
        "[INFO] Reading MCS2018 Copper PDF..."
    )

    url = (
        "https://apps.usgs.gov/"
        "minerals-information-archives/"
        "mcs/mcs2018.pdf"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Copper world table PDF page 57
    page = doc[56]

    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    world_index = None

    for i, line in enumerate(lines):

        if "world total" in line.lower():
            world_index = i
            break

    if world_index is None:
        raise RuntimeError(
            "MCS2018 world total row not found."
        )

    values = []

    for line in lines[
        world_index + 1:
        world_index + 8
    ]:

        cleaned = (
            line
            .replace(",", "")
            .strip()
        )

        try:
            value = float(cleaned)
            values.append(value)
        except ValueError:
            continue

        if len(values) == 3:
            break

    if len(values) < 3:
        raise RuntimeError(
            "MCS2018 world values could not be parsed."
        )

    mine_2016 = values[0] * 1000
    mine_2017 = values[1] * 1000
    reserves_2017 = values[2] * 1000

    rows = [
        {
            "observation_year": 2016,
            "world_copper_mine_production_ton":
                mine_2016,
            "world_copper_refinery_production_ton":
                None,
            "world_copper_reserves_ton":
                None,
            "is_estimate":
                False,
            "report_year":
                2018,
            "source":
                "USGS_MCS2018",
        },
        {
            "observation_year": 2017,
            "world_copper_mine_production_ton":
                mine_2017,
            "world_copper_refinery_production_ton":
                None,
            "world_copper_reserves_ton":
                reserves_2017,
            "is_estimate":
                True,
            "report_year":
                2018,
            "source":
                "USGS_MCS2018",
        },
    ]

    print(
        "[OK] MCS2018: "
        f"{len(rows)} observation rows parsed"
    )

    return rows


def extract_mcs2019_pdf_rows():
    """
    USGS MCS2019 Copper PDF verilerini cikarir.

    PDF:
    - 2017 actual mine production
    - 2018 estimate mine production
    - 2018 reserves

    Global refinery production sayisal olarak verilmedigi icin
    bu release icin refinery bos birakilir.
    """

    print(
        "[INFO] Reading MCS2019 Copper PDF..."
    )

    url = (
        "https://apps.usgs.gov/"
        "minerals-information-archives/"
        "mcs/mcs2019.pdf"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Copper world table PDF page 57
    page = doc[56]

    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    world_index = None

    for i, line in enumerate(lines):

        if "world total" in line.lower():
            world_index = i
            break

    if world_index is None:
        raise RuntimeError(
            "MCS2019 world total row not found."
        )

    values = []

    for line in lines[
        world_index + 1:
        world_index + 8
    ]:

        cleaned = (
            line
            .replace(",", "")
            .strip()
        )

        try:
            value = float(cleaned)
            values.append(value)
        except ValueError:
            continue

        if len(values) == 3:
            break

    if len(values) < 3:
        raise RuntimeError(
            "MCS2019 world values could not be parsed."
        )

    mine_2017 = values[0] * 1000
    mine_2018 = values[1] * 1000
    reserves_2018 = values[2] * 1000

    rows = [
        {
            "observation_year": 2017,
            "world_copper_mine_production_ton":
                mine_2017,
            "world_copper_refinery_production_ton":
                None,
            "world_copper_reserves_ton":
                None,
            "is_estimate":
                False,
            "report_year":
                2019,
            "source":
                "USGS_MCS2019",
        },
        {
            "observation_year": 2018,
            "world_copper_mine_production_ton":
                mine_2018,
            "world_copper_refinery_production_ton":
                None,
            "world_copper_reserves_ton":
                reserves_2018,
            "is_estimate":
                True,
            "report_year":
                2019,
            "source":
                "USGS_MCS2019",
        },
    ]

    print(
        "[OK] MCS2019: "
        f"{len(rows)} observation rows parsed"
    )

    return rows


    
def extract_mcs2020_pdf_rows():
    """
    USGS MCS2020 Copper PDF verilerini cikarir.

    PDF:
    - 2018 actual
    - 2019 estimate
    - 2019 reserves
    """

    print(
        "[INFO] Reading MCS2020 Copper PDF..."
    )

    url = (
        "https://pubs.usgs.gov/"
        "periodicals/mcs2020/mcs2020.pdf"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Copper world table PDF page 57
    page = doc[56]

    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    world_index = None

    for i, line in enumerate(lines):

        if "world total" in line.lower():
            world_index = i
            break

    if world_index is None:
        raise RuntimeError(
            "MCS2020 world total row not found."
        )

    values = []

    for line in lines[
        world_index + 1:
        world_index + 8
    ]:

        cleaned = (
            line
            .replace(",", "")
            .strip()
        )

        try:
            value = float(cleaned)
            values.append(value)
        except ValueError:
            continue

        if len(values) == 3:
            break

    if len(values) < 3:
        raise RuntimeError(
            "MCS2020 world values could not be parsed."
        )

    mine_2018 = values[0] * 1000
    mine_2019 = values[1] * 1000
    reserves_2019 = values[2] * 1000

    # Refinery values are stated in the Copper text
    refinery_2018 = 24_400_000
    refinery_2019 = 25_000_000

    rows = [
        {
            "observation_year": 2018,
            "world_copper_mine_production_ton":
                mine_2018,
            "world_copper_refinery_production_ton":
                refinery_2018,
            "world_copper_reserves_ton":
                None,
            "is_estimate":
                False,
            "report_year":
                2020,
            "source":
                "USGS_MCS2020",
        },
        {
            "observation_year": 2019,
            "world_copper_mine_production_ton":
                mine_2019,
            "world_copper_refinery_production_ton":
                refinery_2019,
            "world_copper_reserves_ton":
                reserves_2019,
            "is_estimate":
                True,
            "report_year":
                2020,
            "source":
                "USGS_MCS2020",
        },
    ]

    print(
        "[OK] MCS2020: "
        f"{len(rows)} observation rows parsed"
    )

    return rows    

def extract_mcs2026_pdf_rows():
    """
    USGS MCS2026 Copper PDF verilerini cikarir.

    PDF:
    - 2024 actual
    - 2025 estimate
    - 2025 reserves
    """

    print(
        "[INFO] Reading MCS2026 Copper PDF..."
    )

    url = (
        "https://pubs.usgs.gov/"
        "periodicals/mcs2026/mcs2026.pdf"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    doc = pymupdf.open(
        stream=response.content,
        filetype="pdf",
    )

    # Copper world table PDF page 77
    page = doc[76]

    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    world_index = None

    for i, line in enumerate(lines):

        if "world total" in line.lower():
            world_index = i
            break

    if world_index is None:
        raise RuntimeError(
            "MCS2026 world total row not found."
        )

    values = []

    for line in lines[
        world_index + 1:
        world_index + 10
    ]:

        cleaned = (
            line
            .replace(",", "")
            .strip()
        )

        try:
            value = float(cleaned)
            values.append(value)
        except ValueError:
            continue

        if len(values) == 5:
            break

    if len(values) < 5:
        raise RuntimeError(
            "MCS2026 world values could not be parsed."
        )

    mine_2024 = values[0] * 1000
    mine_2025 = values[1] * 1000

    refinery_2024 = values[2] * 1000
    refinery_2025 = values[3] * 1000

    reserves_2025 = values[4] * 1000

    rows = [
        {
            "observation_year": 2024,
            "world_copper_mine_production_ton":
                mine_2024,
            "world_copper_refinery_production_ton":
                refinery_2024,
            "world_copper_reserves_ton":
                None,
            "is_estimate":
                False,
            "report_year":
                2026,
            "source":
                "USGS_MCS2026",
        },
        {
            "observation_year": 2025,
            "world_copper_mine_production_ton":
                mine_2025,
            "world_copper_refinery_production_ton":
                refinery_2025,
            "world_copper_reserves_ton":
                reserves_2025,
            "is_estimate":
                True,
            "report_year":
                2026,
            "source":
                "USGS_MCS2026",
        },
    ]

    print(
        "[OK] MCS2026: "
        f"{len(rows)} observation rows parsed"
    )

    return rows


# =========================
# MAIN INGESTION
# =========================

def fetch_all_usgs_data():
    """
    Tum tanimli USGS release'lerini dolasir.
    """

    all_rows = []
    
    try:

        rows_2018 = (
            extract_mcs2018_pdf_rows()
        )

        all_rows.extend(
            rows_2018
        )

    except Exception as exc:

        print(
            f"[WARN] MCS2018 failed: {exc}"
        )
    
    try:

        rows_2019 = (
            extract_mcs2019_pdf_rows()
        )

        all_rows.extend(
            rows_2019
        )

    except Exception as exc:

        print(
            f"[WARN] MCS2019 failed: {exc}"
        )
    
    try:

        rows_2020 = (
            extract_mcs2020_pdf_rows()
        )

        all_rows.extend(
            rows_2020
        )

    except Exception as exc:

        print(
            f"[WARN] MCS2020 failed: {exc}"
        )
    
        # MCS2021 eski PDF formatinda
    try:

        rows_2021 = (
            extract_mcs2021_pdf_rows()
        )

        all_rows.extend(
            rows_2021
        )
        
    except Exception as exc:

        print(
            f"[WARN] MCS2021 failed: {exc}"
        )
        
    try:

        rows_2026 = (
            extract_mcs2026_pdf_rows()
        )

        all_rows.extend(
            rows_2026
        )

    except Exception as exc:

        print(
            f"[WARN] MCS2026 failed: {exc}"
        )

    for report_year, item_id in USGS_RELEASES.items():

        try:

            df = read_world_csv(
                report_year=report_year,
                item_id=item_id,
            )

            rows = extract_world_rows(
                df=df,
                report_year=report_year,
            )

            all_rows.extend(
                rows
            )

            print(
                f"[OK] MCS{report_year}: "
                f"{len(rows)} observation rows parsed"
            )

        except Exception as exc:

            print(
                f"[WARN] MCS{report_year} failed: {exc}"
            )

    if not all_rows:
        raise RuntimeError(
            "No USGS Copper data could be parsed."
        )

    result = pd.DataFrame(
        all_rows
    )

    result = result.sort_values(
        [
            "observation_year",
            "report_year",
        ]
    ).reset_index(
        drop=True
    )

    return result

def build_clean_annual_data(df):
    """
    Her observation year icin tek satir uretir.

    Once gercek production verisi tercih edilir.
    Gercek veri yoksa en guncel tahmin kullanilir.
    Reserve icin ayni yildaki en guncel mevcut deger kullanilir.
    """

    clean_rows = []

    for year, group in df.groupby(
        "observation_year"
    ):

        group = group.sort_values(
            "report_year"
        )

        # Gercek production verisini tercih et
        actual_rows = group[
            group["is_estimate"] == False
        ]

        if not actual_rows.empty:

            production_row = (
                actual_rows.iloc[-1]
            )

        else:

            production_row = (
                group.iloc[-1]
            )

        # Reserve icin mevcut en guncel degeri bul
        reserve_rows = group[
            group[
                "world_copper_reserves_ton"
            ].notna()
        ]

        if not reserve_rows.empty:

            reserve_row = (
                reserve_rows.iloc[-1]
            )

            reserve_value = (
                reserve_row[
                    "world_copper_reserves_ton"
                ]
            )

        else:

            reserve_value = None

        clean_rows.append(
            {
                "observation_year":
                    year,

                "world_copper_mine_production_ton":
                    production_row[
                        "world_copper_mine_production_ton"
                    ],

                "world_copper_refinery_production_ton":
                    production_row[
                        "world_copper_refinery_production_ton"
                    ],

                "world_copper_reserves_ton":
                    reserve_value,

                "is_estimate":
                    production_row[
                        "is_estimate"
                    ],

                "report_year":
                    production_row[
                        "report_year"
                    ],

                "source":
                    production_row[
                        "source"
                    ],
            }
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


# =========================
# SAVE
# =========================

def save_data(df):
    """
    Modelde kullanilacak tekil USGS annual datasetini kaydeder.
    """

    clean_df = build_clean_annual_data(
        df
    )

    clean_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\n[OK] Saved: {OUTPUT_FILE}"
    )

    print(
        f"[OK] Rows: {len(clean_df)}"
    )

    print(
        "\n[INFO] Clean USGS Copper annual data:"
    )

    print(
        clean_df.to_string(
            index=False
        )
    )


# =========================
# MAIN
# =========================

def main():

    print(
        "[INFO] Starting USGS Copper ingestion..."
    )

    df = fetch_all_usgs_data()

    save_data(
        df
    )


if __name__ == "__main__":
    main()