from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from src.utils.paths import COCHILCO_SOURCE_DIR, MINING_RAW_DIR
SOURCE_FILE = (
    COCHILCO_SOURCE_DIR
    / "chile_cochilco_yearbook_2005_2024.xlsx"
)
OUTPUT_FILE = (
    MINING_RAW_DIR
    / "chile_cochilco_copper_cost_annual.csv"
)
def clean_number(value):
    """
    Sayisal degeri temizler.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return float(value)

    text = str(value).strip()

    invalid_values = {
        "",
        "-",
        "*",
        "(*)",
        "n.d.",
        "n.d",
        "ND",
        "N/D",
    }

    if text in invalid_values:
        return np.nan

    text = text.replace(",", ".")

    try:
        return float(text)

    except ValueError:
        return np.nan


def normalize_text(text):
    """
    Metni ASCII uyumlu hale getirir
    ve fazla bosluklari temizler.
    """

    text = str(text)

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = text.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )

    # Birden fazla boslugu tek bosluga indir
    text = " ".join(
        text.split()
    )

    return text.lower().strip()


def extract_year_series(
    sheet_name,
    row_keyword,
):
    """
    Sheet icinden yil bazli hedef satiri cikarir.
    """

    df = pd.read_excel(
        SOURCE_FILE,
        sheet_name=sheet_name,
        header=None,
    )

    target_row = None

    normalized_keyword = normalize_text(
        row_keyword
    )

    for i in range(len(df)):

        row_text = normalize_text(
            " | ".join(
                df.iloc[i]
                .astype(str)
                .tolist()
            )
        )

        if normalized_keyword in row_text:
            target_row = i
            break

    if target_row is None:
        raise ValueError(
            f"Row not found: {row_keyword} "
            f"in {sheet_name}"
        )

    # Hedef satirin ustunde yil satirini ara
    year_row = None

    search_start = target_row - 1
    search_end = max(
        -1,
        target_row - 12,
    )

    for i in range(
        search_start,
        search_end,
        -1,
    ):

        numeric_years = []

        for value in df.iloc[i]:

            try:
                year = int(
                    float(value)
                )

                if 1900 <= year <= 2100:
                    numeric_years.append(
                        year
                    )

            except (
                ValueError,
                TypeError,
            ):
                continue

        if len(numeric_years) >= 2:
            year_row = i
            break

    if year_row is None:
        raise ValueError(
            f"Year row not found in "
            f"{sheet_name} for "
            f"{row_keyword}"
        )

    result = {}

    for col in range(
        df.shape[1]
    ):

        try:
            year = int(
                float(
                    df.iloc[
                        year_row,
                        col,
                    ]
                )
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

        if 2000 <= year <= 2100:

            value = clean_number(
                df.iloc[
                    target_row,
                    col,
                ]
            )

            result[year] = value

    return result


def merge_series(
    *series_dicts,
):
    """
    Birden fazla yil serisini birlestirir.
    """

    merged = {}

    for series in series_dicts:
        merged.update(
            series
        )

    return merged


def extract_copper_production():
    """
    Sili toplam bakir uretimini cikarir.

    CODELCO Total + Other Producers.

    Kaynak birimi:
    kMT copper content.

    Cikti birimi:
    metric ton.
    """

    result = {}

    sheets = [
        "Tabla 2.1",
        "Tabla 2.2",
    ]

    for sheet in sheets:

        codelco = extract_year_series(
            sheet,
            "CODELCO-CHILE Total",
        )

        others = extract_year_series(
            sheet,
            "OTROS PRODUCTORES",
        )

        years = (
            set(codelco.keys())
            | set(others.keys())
        )

        for year in years:

            codelco_value = (
                codelco.get(
                    year,
                    np.nan,
                )
            )

            other_value = (
                others.get(
                    year,
                    np.nan,
                )
            )

            if (
                pd.notna(
                    codelco_value
                )
                and pd.notna(
                    other_value
                )
            ):

                result[year] = (
                    codelco_value
                    + other_value
                ) * 1000

            else:
                result[year] = (
                    np.nan
                )

    return result


def extract_mining_wage():
    """
    Mining nominal wage index cikarir.

    Baz yil degisimleri nedeniyle ham index
    ile birlikte YoY degisim de hesaplanir.
    """

    wage_1 = extract_year_series(
        "Tabla 33.1",
        "Mineria / Mining",
    )

    wage_2 = extract_year_series(
        "Tabla 33.2",
        "Mineria / Mining",
    )

    wage = merge_series(
        wage_1,
        wage_2,
    )

    wage_yoy = {}

    for year in sorted(
        wage.keys()
    ):

        current = wage.get(
            year,
            np.nan,
        )

        previous = wage.get(
            year - 1,
            np.nan,
        )

        # Baz yil degisen gecislerde
        # YoY hesaplanmaz.
        if year in {
            2016,
            2023,
        }:
            wage_yoy[year] = (
                np.nan
            )

        elif (
            pd.notna(current)
            and pd.notna(previous)
            and previous != 0
        ):

            wage_yoy[year] = (
                (
                    current
                    / previous
                )
                - 1
            ) * 100

        else:
            wage_yoy[year] = (
                np.nan
            )

    return (
        wage,
        wage_yoy,
    )


def extract_energy():
    """
    Copper mining enerji verilerini cikarir.

    Birim:
    TJ.
    """

    fuel = merge_series(
        extract_year_series(
            "Tabla 36.1",
            "Combustibles / Fuels",
        ),
        extract_year_series(
            "Tabla 36.2",
            "Combustibles / Fuels",
        ),
    )

    electricity = merge_series(
        extract_year_series(
            "Tabla 36.1",
            "Electricidad / Electricity",
        ),
        extract_year_series(
            "Tabla 36.2",
            "Electricidad / Electricity",
        ),
    )

    total_energy = merge_series(
        extract_year_series(
            "Tabla 36.1",
            "Total Nacional / Domestic Total",
        ),
        extract_year_series(
            "Tabla 36.2",
            "Total Nacional / Domestic Total",
        ),
    )

    return (
        fuel,
        electricity,
        total_energy,
    )


def extract_open_pit_fuel_intensity():
    """
    Open pit fuel intensity cikarir.

    Birim:
    MJ/MT copper content.
    """

    series_1 = extract_year_series(
        "Tabla 37.1",
        "Mina Rajo / Open Pit",
    )

    series_2 = extract_year_series(
        "Tabla 37.2",
        "Mina Rajo / Open Pit",
    )

    return merge_series(
        series_1,
        series_2,
    )


def extract_concentrator_electricity_intensity():
    """
    Concentrator electricity intensity cikarir.

    Birim:
    MJ/MT copper content.
    """

    result = {}

    for sheet_name in [
        "Tabla 39.1",
        "Tabla 39.2",
    ]:

        df = pd.read_excel(
            SOURCE_FILE,
            sheet_name=sheet_name,
            header=None,
        )

        target_row = None

        # Sadece ilk kolondaki proses adini ara.
        for i in range(len(df)):

            first_cell = normalize_text(
                df.iloc[i, 0]
            )

            if (
                "concentradora"
                in first_cell
                and
                "concentrating plant"
                in first_cell
            ):
                target_row = i
                break

        if target_row is None:
            raise ValueError(
                f"Concentrator row not found "
                f"in {sheet_name}"
            )

        # Hedef satirin ustundeki yil satirini bul.
        year_row = None

        for i in range(
            target_row - 1,
            max(-1, target_row - 12),
            -1,
        ):

            years_found = 0

            for value in df.iloc[i]:

                try:
                    year = int(float(value))

                    if 2000 <= year <= 2100:
                        years_found += 1

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

            if years_found >= 5:
                year_row = i
                break

        if year_row is None:
            raise ValueError(
                f"Year row not found "
                f"in {sheet_name}"
            )

        for col in range(
            df.shape[1]
        ):

            try:
                year = int(
                    float(
                        df.iloc[
                            year_row,
                            col,
                        ]
                    )
                )

            except (
                ValueError,
                TypeError,
            ):
                continue

            if 2005 <= year <= 2024:

                value = clean_number(
                    df.iloc[
                        target_row,
                        col,
                    ]
                )

                result[year] = value

    return result


def extract_sulfuric_acid():
    """
    Sulfuric acid market verilerini cikarir.

    Kaynak birimi:
    kMT.

    Cikti birimi:
    metric ton.
    """

    consumption = (
        extract_year_series(
            "Tabla 41",
            (
                "CONSUMO APARENTE / "
                "APPARENT CONSUMPTION"
            ),
        )
    )

    production = (
        extract_year_series(
            "Tabla 41",
            (
                "PRODUCCION TOTAL / "
                "TOTAL PRODUCTION"
            ),
        )
    )

    imports = (
        extract_year_series(
            "Tabla 41",
            (
                "EMBARQUES DE IMPORTACION / "
                "IMPORTS"
            ),
        )
    )

    exports = (
        extract_year_series(
            "Tabla 41",
            (
                "EMBARQUES DE EXPORTACION / "
                "EXPORTS"
            ),
        )
    )

    series_list = [
        consumption,
        production,
        imports,
        exports,
    ]

    for series in series_list:

        for year in series:

            if pd.notna(
                series[year]
            ):
                series[year] = (
                    series[year]
                    * 1000
                )

    return (
        consumption,
        production,
        imports,
        exports,
    )


def extract_ore_grade():
    """
    Chile average copper ore grade cikarir.

    Birim:
    percent.
    """

    return extract_year_series(
        "Tabla 50",
        (
            "Promedio Chile / "
            "Chile Average"
        ),
    )


def main():
    """
    Ana ingestion fonksiyonu.
    """

    print(
        "[INFO] Building Chile COCHILCO "
        "copper cost dataset..."
    )

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: "
            f"{SOURCE_FILE}"
        )

    production = (
        extract_copper_production()
    )

    (
        wage,
        wage_yoy,
    ) = extract_mining_wage()

    (
        fuel,
        electricity,
        total_energy,
    ) = extract_energy()

    open_pit_fuel = (
        extract_open_pit_fuel_intensity()
    )

    concentrator_electricity = (
        extract_concentrator_electricity_intensity()
    )

    (
        acid_consumption,
        acid_production,
        acid_import,
        acid_export,
    ) = extract_sulfuric_acid()

    ore_grade = (
        extract_ore_grade()
    )

    rows = []

    for year in range(
        2005,
        2025,
    ):

        row = {
            "year":
                year,

            "chile_copper_production_ton":
                production.get(
                    year,
                    np.nan,
                ),

            "chile_mining_wage_index":
                wage.get(
                    year,
                    np.nan,
                ),

            "chile_mining_wage_yoy_pct":
                wage_yoy.get(
                    year,
                    np.nan,
                ),

            "chile_copper_fuel_consumption_tj":
                fuel.get(
                    year,
                    np.nan,
                ),

            "chile_copper_electricity_consumption_tj":
                electricity.get(
                    year,
                    np.nan,
                ),

            "chile_copper_total_energy_consumption_tj":
                total_energy.get(
                    year,
                    np.nan,
                ),

            "chile_open_pit_fuel_mj_per_ton":
                open_pit_fuel.get(
                    year,
                    np.nan,
                ),

            "chile_concentrator_electricity_mj_per_ton":
                concentrator_electricity.get(
                    year,
                    np.nan,
                ),

            "chile_sulfuric_acid_consumption_ton":
                acid_consumption.get(
                    year,
                    np.nan,
                ),

            "chile_sulfuric_acid_production_ton":
                acid_production.get(
                    year,
                    np.nan,
                ),

            "chile_sulfuric_acid_import_ton":
                acid_import.get(
                    year,
                    np.nan,
                ),

            "chile_sulfuric_acid_export_ton":
                acid_export.get(
                    year,
                    np.nan,
                ),

            "chile_average_copper_ore_grade_pct":
                ore_grade.get(
                    year,
                    np.nan,
                ),

            "source":
                "COCHILCO_YEARBOOK_2005_2024",
        }

        rows.append(
            row
        )

    output = pd.DataFrame(
        rows
    )

    output = output.sort_values(
        "year"
    ).reset_index(
        drop=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] Saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        "\n[INFO] Chile COCHILCO data:"
    )

    print(
        output.to_string(
            index=False
        )
    )

    print(
        "\n[INFO] Non-null counts:"
    )

    print(
        output.notna()
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()