from io import StringIO
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

EMPLOYMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "peru"
    / "peru_mining_employment_2020_2026.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "peru_copper_mining_annual.csv"
)

BCRP_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/anuales/resultados/"
    "PM05146AA/html/2005/2025/"
)


def normalize_text(text):
    """
    Metni ASCII uyumlu hale getirir.
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

    text = " ".join(
        text.split()
    )

    return text.lower().strip()


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

    if text in {
        "",
        "-",
        "nan",
        "None",
    }:
        return np.nan

    # Peru formatinda ondalik virgul olabilir.
    text = text.replace(
        ".",
        ""
    )

    text = text.replace(
        ",",
        "."
    )

    try:
        return float(text)

    except ValueError:
        return np.nan


def extract_bcrp_copper_production():
    """
    BCRP Peru copper production serisini ceker.

    Series:
    PM05146AA

    Kaynak birimi:
    thousand tonnes

    Cikti birimi:
    metric ton
    """

    print(
        "[INFO] Reading Peru copper production "
        "from BCRP..."
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        BCRP_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(
            response.text
        ),
        decimal=",",
        thousands=".",
    )

    if not tables:
        raise ValueError(
            "No tables found on BCRP page"
        )

    production_df = None

    for table in tables:

        if table.shape[1] < 2:
            continue

        valid_year_count = 0

        for value in table.iloc[:, 0]:

            try:
                year = int(
                    float(value)
                )

                if 2005 <= year <= 2025:
                    valid_year_count += 1

            except (
                ValueError,
                TypeError,
            ):
                continue

        if valid_year_count >= 10:

            production_df = (
                table.copy()
            )

            break

    if production_df is None:

        raise ValueError(
            "BCRP production table "
            "could not be identified"
        )

    result = {}

    for _, row in production_df.iterrows():

        try:
            year = int(
                float(
                    row.iloc[0]
                )
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

        if not (
            2005 <= year <= 2025
        ):
            continue

        raw_value = (
            row.iloc[1]
        )

        if pd.isna(
            raw_value
        ):
            continue

        # read_html degeri zaten numeric
        # okuyabilir.
        if isinstance(
            raw_value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            value = float(
                raw_value
            )

        else:

            text = str(
                raw_value
            ).strip()

            # BCRP decimal separator
            # virgul olabilir.
            text = text.replace(
                " ",
                ""
            )

            text = text.replace(
                ",",
                "."
            )

            try:
                value = float(
                    text
                )

            except ValueError:
                continue

        # BCRP serisi thousand tonnes.
        result[year] = (
            value * 1000
        )

    if not result:

        raise ValueError(
            "No Peru copper production "
            "values parsed"
        )

    print(
        f"[OK] BCRP production: "
        f"{len(result)} years parsed"
    )

    print(
        "[INFO] First production:",
        min(result),
        result[min(result)],
    )

    print(
        "[INFO] Last production:",
        max(result),
        result[max(result)],
    )

    return result

def extract_mining_employment():
    """
    MINEM departman bazli mining employment
    dosyasindan Peru toplam aylik istihdamini
    hesaplar.

    Daha sonra tamamlanmis yillar icin
    aylik toplamlarin yillik ortalamasi alinir.
    """

    print(
        "[INFO] Reading Peru mining employment "
        "from MINEM..."
    )

    if not EMPLOYMENT_FILE.exists():

        raise FileNotFoundError(
            f"Employment file not found: "
            f"{EMPLOYMENT_FILE}"
        )

    df = pd.read_excel(
        EMPLOYMENT_FILE,
        sheet_name="Empleo Minero",
        header=None,
    )

    # Dosyada yil basliklari 4. satirda,
    # ay basliklari 5. satirda bulunuyor.
    year_row = 3
    month_row = 4

    months = {
        "ene.",
        "feb.",
        "mar.",
        "abr.",
        "may.",
        "jun.",
        "jul.",
        "ago.",
        "set.",
        "oct.",
        "nov.",
        "dic.",
    }

    current_year = None

    monthly_columns = []

    for col in range(
        1,
        df.shape[1]
    ):

        year_value = (
            df.iloc[
                year_row,
                col,
            ]
        )

        if pd.notna(
            year_value
        ):

            try:
                possible_year = int(
                    float(
                        year_value
                    )
                )

                if (
                    2020
                    <= possible_year
                    <= 2026
                ):
                    current_year = (
                        possible_year
                    )

            except (
                ValueError,
                TypeError,
            ):
                pass

        month_value = normalize_text(
            df.iloc[
                month_row,
                col,
            ]
        )

        if (
            current_year is not None
            and month_value in months
        ):

            monthly_columns.append(
                {
                    "column":
                        col,
                    "year":
                        current_year,
                    "month":
                        month_value,
                }
            )

    if not monthly_columns:

        raise ValueError(
            "No monthly employment "
            "columns identified"
        )

    monthly_totals = []

    # Veri satirlari basliklardan sonra basliyor.
    for column_info in monthly_columns:

        col = column_info[
            "column"
        ]

        values = []

        for row_idx in range(
            5,
            len(df),
        ):

            department = normalize_text(
                df.iloc[
                    row_idx,
                    0,
                ]
            )

            if not department:
                continue

            # Toplam veya aciklama satirlarini
            # tekrar toplama.
            if (
                "total" in department
                or "fuente" in department
                or "source" in department
            ):
                continue

            value = clean_number(
                df.iloc[
                    row_idx,
                    col,
                ]
            )

            if pd.notna(value):
                values.append(
                    value
                )

        if values:

            monthly_totals.append(
                {
                    "year":
                        column_info[
                            "year"
                        ],
                    "month":
                        column_info[
                            "month"
                        ],
                    "employment":
                        sum(values),
                }
            )

    monthly_df = pd.DataFrame(
        monthly_totals
    )

    # Sadece tamamlanmis yillar.
    monthly_df = monthly_df[
        monthly_df["year"]
        <= 2025
    ].copy()

    month_counts = (
        monthly_df
        .groupby("year")["month"]
        .nunique()
    )

    complete_years = (
        month_counts[
            month_counts == 12
        ]
        .index
        .tolist()
    )

    monthly_df = monthly_df[
        monthly_df["year"]
        .isin(
            complete_years
        )
    ].copy()

    annual = (
        monthly_df
        .groupby(
            "year"
        )["employment"]
        .mean()
    )

    result = (
        annual
        .round()
        .astype(int)
        .to_dict()
    )

    print(
        f"[OK] MINEM employment: "
        f"{len(result)} complete years parsed"
    )

    print(
        "[INFO] Employment years:",
        sorted(
            result.keys()
        ),
    )

    return result


def calculate_yoy(series):
    """
    Yillik yuzde degisimi hesaplar.
    """

    result = {}

    for year in sorted(
        series.keys()
    ):

        current = series.get(
            year,
            np.nan,
        )

        previous = series.get(
            year - 1,
            np.nan,
        )

        if (
            pd.notna(current)
            and pd.notna(previous)
            and previous != 0
        ):

            result[year] = (
                (
                    current
                    / previous
                )
                - 1
            ) * 100

        else:

            result[year] = (
                np.nan
            )

    return result


def main():
    """
    Peru copper mining annual dataset.
    """

    print(
        "[INFO] Building Peru copper mining "
        "annual dataset..."
    )

    production = (
        extract_bcrp_copper_production()
    )

    employment = (
        extract_mining_employment()
    )

    production_yoy = (
        calculate_yoy(
            production
        )
    )

    employment_yoy = (
        calculate_yoy(
            employment
        )
    )

    years = range(
        2005,
        2026,
    )

    rows = []

    for year in years:

        rows.append(
            {
                "year":
                    year,

                "peru_copper_production_ton":
                    production.get(
                        year,
                        np.nan,
                    ),

                "peru_copper_production_yoy_pct":
                    production_yoy.get(
                        year,
                        np.nan,
                    ),

                "peru_mining_employment":
                    employment.get(
                        year,
                        np.nan,
                    ),

                "peru_mining_employment_yoy_pct":
                    employment_yoy.get(
                        year,
                        np.nan,
                    ),

                "production_source":
                    "BCRP_MINEM_PM05146AA",

                "employment_source":
                    (
                        "MINEM_EMPLEO_MINERO"
                        if year in employment
                        else np.nan
                    ),
            }
        )

    output = pd.DataFrame(
        rows
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
        f"\n[OK] Saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        "\n[INFO] Peru copper mining data:"
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
