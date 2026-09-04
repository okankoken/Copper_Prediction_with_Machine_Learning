from io import StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "peru_copper_cost_drivers_monthly.csv"
)

BCRP_USD_PEN_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/mensuales/resultados/"
    "PN01246PM/html/2005-01/2026-07/"
)

BCRP_DIESEL_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/mensuales/resultados/"
    "PN01441PM/html/2010-01/2026-07/"
)

BCRP_ELECTRICITY_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/mensuales/resultados/"
    "PN01445PM/html/2010-01/2026-07/"
)

BCRP_LABOR_PROXY_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/mensuales/resultados/"
    "PN37696PM/html/2015-01/2026-06/"
)

def extract_usd_pen():
    """
    Peru nominal exchange rate.

    Series:
    PN01246PM

    Unit:
    PEN per USD

    Frequency:
    monthly average
    """

    print(
        "[INFO] Reading USD/PEN from BCRP..."
    )

    response = requests.get(
        BCRP_USD_PEN_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(
            response.text
        )
    )

    if not tables:
        raise ValueError(
            "No BCRP USD/PEN table found"
        )

    result_table = None

    month_prefixes = (
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic",
    )

    for table in tables:

        if table.shape[1] < 2:
            continue

        valid_dates = 0

        for value in table.iloc[:, 0]:

            text = str(
                value
            ).strip()

            if (
                text.startswith(
                    month_prefixes
                )
                and len(text) >= 5
            ):
                valid_dates += 1

        if valid_dates >= 12:

            result_table = (
                table.copy()
            )

            break

    if result_table is None:

        raise ValueError(
            "USD/PEN table could not "
            "be identified"
        )

    rows = []

    month_map = {
        "Ene": 1,
        "Feb": 2,
        "Mar": 3,
        "Abr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dic": 12,
    }

    for _, row in result_table.iterrows():

        date_text = str(
            row.iloc[0]
        ).strip()

        if len(date_text) < 5:
            continue

        month_text = (
            date_text[:3]
        )

        if month_text not in month_map:
            continue

        try:
            year_2digit = int(
                date_text[3:5]
            )

        except ValueError:
            continue

        if year_2digit <= 30:
            year = (
                2000
                + year_2digit
            )
        else:
            year = (
                1900
                + year_2digit
            )

        if not (
            2005 <= year <= 2026
        ):
            continue

        raw_value = (
            row.iloc[1]
        )

        if pd.isna(
            raw_value
        ):
            continue

        if isinstance(
            raw_value,
            (
                int,
                float,
            ),
        ):

            value = float(
                raw_value
            )

        else:

            text = str(
                raw_value
            ).strip()

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

        month_number = (
            month_map[
                month_text
            ]
        )

        month = (
            f"{year}-"
            f"{month_number:02d}"
        )

        rows.append(
            {
                "month":
                    month,
                "peru_usd_pen":
                    value,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise ValueError(
            "No USD/PEN observations parsed"
        )

    result = (
        result
        .drop_duplicates(
            subset=["month"],
            keep="last",
        )
        .sort_values(
            "month"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"[OK] USD/PEN: "
        f"{len(result)} months parsed"
    )

    print(
        "[INFO] First:",
        result.iloc[0].to_dict(),
    )

    print(
        "[INFO] Last:",
        result.iloc[-1].to_dict(),
    )

    return result

def extract_bcrp_monthly_series(
    url,
    output_column,
    start_year,
    end_year,
):
    """
    BCRP aylik serisini ceker.

    Tarih formati:
    Ene05, Feb05, Mar05...
    """

    print(
        f"[INFO] Reading {output_column} "
        f"from BCRP..."
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30,
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
            f"{output_column}"
        )

    month_prefixes = (
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic",
    )

    result_table = None

    for table in tables:

        if table.shape[1] < 2:
            continue

        valid_dates = 0

        for value in table.iloc[:, 0]:

            text = str(
                value
            ).strip()

            if (
                text.startswith(
                    month_prefixes
                )
                and len(text) >= 5
            ):
                valid_dates += 1

        if valid_dates >= 12:

            result_table = (
                table.copy()
            )

            break

    if result_table is None:

        raise ValueError(
            f"Table could not be identified "
            f"for {output_column}"
        )

    month_map = {
        "Ene": 1,
        "Feb": 2,
        "Mar": 3,
        "Abr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dic": 12,
    }

    rows = []

    for _, row in result_table.iterrows():

        date_text = str(
            row.iloc[0]
        ).strip()

        if len(date_text) < 5:
            continue

        month_text = (
            date_text[:3]
        )

        if month_text not in month_map:
            continue

        try:
            year_2digit = int(
                date_text[3:5]
            )

        except ValueError:
            continue

        if year_2digit <= 30:
            year = (
                2000
                + year_2digit
            )
        else:
            year = (
                1900
                + year_2digit
            )

        if not (
            start_year
            <= year
            <= end_year
        ):
            continue

        raw_value = (
            row.iloc[1]
        )

        if pd.isna(
            raw_value
        ):
            continue

        if isinstance(
            raw_value,
            (
                int,
                float,
            ),
        ):

            value = float(
                raw_value
            )

        else:

            text = str(
                raw_value
            ).strip()

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

        month_number = (
            month_map[
                month_text
            ]
        )

        month = (
            f"{year}-"
            f"{month_number:02d}"
        )

        rows.append(
            {
                "month":
                    month,
                output_column:
                    value,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise ValueError(
            f"No observations parsed "
            f"for {output_column}"
        )

    result = (
        result
        .drop_duplicates(
            subset=["month"],
            keep="last",
        )
        .sort_values(
            "month"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"[OK] {output_column}: "
        f"{len(result)} months parsed"
    )

    return result


def main():

    print(
        "[INFO] Building Peru copper "
        "cost drivers dataset..."
    )

    usd_pen = (
        extract_usd_pen()
    )

    diesel = (
        extract_bcrp_monthly_series(
            BCRP_DIESEL_URL,
            "peru_diesel_price_index",
            2010,
            2026,
        )
    )

    electricity = (
        extract_bcrp_monthly_series(
            BCRP_ELECTRICITY_URL,
            (
                "peru_industrial_"
                "electricity_tariff_index"
            ),
            2010,
            2026,
        )
    )

    labor_proxy = (
        extract_bcrp_monthly_series(
            BCRP_LABOR_PROXY_URL,
            "peru_formal_private_income_pen",
            2015,
            2026,
        )
    )

    output = (
        usd_pen
        .merge(
            diesel,
            on="month",
            how="left",
        )
        .merge(
            electricity,
            on="month",
            how="left",
        )
        .merge(
            labor_proxy,
            on="month",
            how="left",
        )
    )

    output = output.sort_values(
        "month"
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
        f"\n[OK] Saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        "\n[INFO] Peru cost drivers:"
    )

    print(
        output.tail(
            24
        ).to_string(
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