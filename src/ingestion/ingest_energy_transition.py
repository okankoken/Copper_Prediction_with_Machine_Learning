from io import StringIO
from itertools import product
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.utils.paths import ENERGY_TRANSITION_RAW_DIR
OUTPUT_PATH = (
    ENERGY_TRANSITION_RAW_DIR
    / "energy_transition_annual.csv"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


IEA_SALES_URL = (
    "https://www.iea.org/data-and-statistics/"
    "charts/electric-car-sales-by-region-2015-2025"
)

IEA_STOCK_URL = (
    "https://www.iea.org/data-and-statistics/"
    "charts/electric-car-stock-by-region-2015-2025"
)

IRENA_URL = (
    "https://pxweb.irena.org/api/v1/en/"
    "IRENASTAT/"
    "Power Capacity and Generation/"
    "Region_ELECCAP_2026_H1_v-PX 1.px"
)


def fetch_iea_chart(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    chart = soup.find(
        attrs={
            "data-behavior": "chart"
        }
    )

    if chart is None:
        raise RuntimeError(
            f"IEA chart block not found: {url}"
        )

    raw_csv = chart.get(
        "data-chart-csv",
        ""
    )

    if not raw_csv.strip():
        raise RuntimeError(
            f"IEA inline CSV is empty: {url}"
        )

    return pd.read_csv(
        StringIO(raw_csv),
        sep=";",
    )


def ingest_iea_sales():
    df = fetch_iea_chart(
        IEA_SALES_URL
    )

    df = df.rename(
        columns={
            "Unnamed: 0": "year",
            "United States": "usa_ev_sales_units",
            "Europe": "europe_ev_sales_units",
            "China": "china_ev_sales_units",
            "Rest of World": "rest_of_world_ev_sales_units",
            "Share": "global_ev_sales_share_percent",
        }
    )

    unit_columns = [
        "usa_ev_sales_units",
        "europe_ev_sales_units",
        "china_ev_sales_units",
        "rest_of_world_ev_sales_units",
    ]

    for column in unit_columns:
        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            * 1_000_000
        )

        df[column] = df[column].round().astype(
            "Int64"
        )

    df[
        "global_ev_sales_units"
    ] = df[unit_columns].sum(
        axis=1
    ).astype(
        "Int64"
    )

    df[
        "global_ev_sales_share_percent"
    ] = pd.to_numeric(
        df["global_ev_sales_share_percent"],
        errors="coerce",
    )

    return df[
        [
            "year",
            "global_ev_sales_units",
            "china_ev_sales_units",
            "europe_ev_sales_units",
            "usa_ev_sales_units",
            "rest_of_world_ev_sales_units",
            "global_ev_sales_share_percent",
        ]
    ]


def ingest_iea_stock():
    df = fetch_iea_chart(
        IEA_STOCK_URL
    )

    df = df.rename(
        columns={
            "Unnamed: 0": "year",
            "United States": "usa_ev_stock_units",
            "Europe": "europe_ev_stock_units",
            "China": "china_ev_stock_units",
            "Rest of world": "rest_of_world_ev_stock_units",
        }
    )

    unit_columns = [
        "usa_ev_stock_units",
        "europe_ev_stock_units",
        "china_ev_stock_units",
        "rest_of_world_ev_stock_units",
    ]

    for column in unit_columns:
        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            * 1_000_000
        )

        df[column] = df[column].round().astype(
            "Int64"
        )

    df[
        "global_ev_stock_units"
    ] = df[unit_columns].sum(
        axis=1
    ).astype(
        "Int64"
    )

    return df[
        [
            "year",
            "global_ev_stock_units",
            "china_ev_stock_units",
            "europe_ev_stock_units",
            "usa_ev_stock_units",
            "rest_of_world_ev_stock_units",
        ]
    ]


def jsonstat2_to_dataframe(data):
    dimension_ids = data["id"]

    labels = {}

    for dimension_id in dimension_ids:
        dimension = data[
            "dimension"
        ][
            dimension_id
        ]

        index_map = dimension[
            "category"
        ][
            "index"
        ]

        label_map = dimension[
            "category"
        ][
            "label"
        ]

        ordered_codes = sorted(
            index_map,
            key=index_map.get,
        )

        labels[
            dimension_id
        ] = [
            label_map[code]
            for code in ordered_codes
        ]

    rows = []

    for coordinates, value in zip(
        product(
            *[
                labels[dimension_id]
                for dimension_id
                in dimension_ids
            ]
        ),
        data["value"],
    ):
        row = dict(
            zip(
                dimension_ids,
                coordinates,
            )
        )

        row["value"] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def ingest_irena_capacity():
    # Yeni IRENA tablosunda kapasite ve uretim ayni tabloda bulunuyor.
    url = (
        "https://pxweb.irena.org/api/v1/en/IRENASTAT/"
        "Power%20Capacity%20and%20Generation/"
        "Region_ELECSTAT_2026_H2_PX.px"
    )

    headers = {
        **HEADERS,
        "Accept": "application/json",
    }

    # Kodlari metadata etiketlerinden bul.
    metadata_response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )
    metadata_response.raise_for_status()
    metadata = metadata_response.json()

    variables = {
        variable["code"]: variable
        for variable in metadata["variables"]
    }

    def code_for_label(dimension, label):
        variable = variables[dimension]
        for code, text in zip(
            variable["values"],
            variable["valueTexts"],
        ):
            if text.strip().casefold() == label.casefold():
                return code
        raise ValueError(
            f"IRENA etiketi bulunamadi: {dimension} / {label}"
        )

    technology_columns = {
        "Total Renewable": "global_renewable_capacity_mw",
        "Solar energy": "global_solar_capacity_mw",
        "Wind energy": "global_wind_capacity_mw",
    }

    year_variable = variables["Year"]
    year_codes = [
        code
        for code, label in zip(
            year_variable["values"],
            year_variable["valueTexts"],
        )
        if label.isdigit() and int(label) >= 2015
    ]

    if not year_codes:
        raise ValueError("IRENA yil secimi bos.")

    selections = {
        "Region": [code_for_label("Region", "World")],
        "Technology": [
            code_for_label("Technology", label)
            for label in technology_columns
        ],
        "Data Type": [
            code_for_label(
                "Data Type",
                "Electrical Installed Capacity (MW)",
            )
        ],
        "Year": year_codes,
    }

    payload = {
        "query": [
            {
                "code": dimension,
                "selection": {
                    "filter": "item",
                    "values": codes,
                },
            }
            for dimension, codes in selections.items()
        ],
        "response": {"format": "json-stat2"},
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    df = jsonstat2_to_dataframe(response.json())
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if df.empty or df["value"].isna().any():
        raise ValueError("IRENA kapasite verisi bos veya eksik.")

    if df.duplicated(["Year", "Technology"]).any():
        raise ValueError("IRENA yil ve teknoloji eslesmeleri tekrar ediyor.")

    pivot = df.pivot(
        index="Year",
        columns="Technology",
        values="value",
    ).reset_index()

    pivot = pivot.rename(
        columns={"Year": "year", **technology_columns}
    )

    required_columns = ["year", *technology_columns.values()]
    missing = [
        column for column in required_columns
        if column not in pivot.columns
    ]
    if missing:
        raise ValueError(f"IRENA cikti kolonlari eksik: {missing}")

    pivot["year"] = pd.to_numeric(
        pivot["year"],
        errors="raise",
    ).astype("Int64")

    return (
        pivot[required_columns]
        .sort_values("year")
        .reset_index(drop=True)
    )



def main():
    print(
        "[INFO] Fetching IEA EV sales"
    )

    sales = ingest_iea_sales()

    print(
        "[INFO] Fetching IEA EV stock"
    )

    stock = ingest_iea_stock()

    print(
        "[INFO] Fetching IRENA capacity"
    )

    try:
        capacity = ingest_irena_capacity()

    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as exc:

        print(
            "[WARN] IRENA baglanti hatasi: "
            f"{type(exc).__name__}: {exc}"
        )

        if not OUTPUT_PATH.exists():
            raise RuntimeError(
                "IRENA erisilemiyor ve kayitli veri bulunamadi."
            ) from exc

        capacity_columns = [
            "global_renewable_capacity_mw",
            "global_solar_capacity_mw",
            "global_wind_capacity_mw",
        ]

        cached = pd.read_csv(OUTPUT_PATH)

        required = ["date", *capacity_columns]
        missing = [
            column
            for column in required
            if column not in cached.columns
        ]

        if missing:
            raise ValueError(
                f"Kayitli IRENA kolonlari eksik: {missing}"
            )

        dates = pd.to_datetime(
            cached["date"],
            errors="raise",
        )

        if dates.isna().any():
            raise ValueError("Kayitli IRENA tarihleri eksik.")

        capacity = cached[capacity_columns].copy()
        capacity.insert(0, "year", dates.dt.year)

        if capacity["year"].duplicated().any():
            raise ValueError("Kayitli IRENA yillari tekrar ediyor.")

        for column in capacity_columns:
            capacity[column] = pd.to_numeric(
                capacity[column],
                errors="raise",
            )

        if (
            capacity[capacity_columns].isna().any().any()
            or (capacity[capacity_columns] <= 0).any().any()
            or capacity[capacity_columns].isin(
                [float("inf"), float("-inf")]
            ).any().any()
        ):
            raise ValueError(
                "Kayitli IRENA kapasite degerleri gecersiz."
            )

        # Bu calisma icin beklenen tarihsel kapsam.
        expected_years = set(range(2015, 2026))
        missing_years = sorted(
            expected_years - set(capacity["year"])
        )

        if missing_years:
            raise ValueError(
                f"Kayitli IRENA yillari eksik: {missing_years}"
            )

        capacity = capacity.sort_values(
            "year"
        ).reset_index(drop=True)

        print(
            "[WARN] CANLI IRENA VERISI ALINAMADI. "
            "Onceki dosyadaki kapasite verileri kullaniliyor."
        )
        print(
            "[WARN] Kayitli IRENA yil araligi: "
            f"{capacity['year'].min()} - "
            f"{capacity['year'].max()}"
        )
        print(
            "[INFO] IEA satis ve stok verileri "
            "bu calismada yeniden cekildi."
        )


    final = (
        sales
        .merge(
            stock,
            on="year",
            how="outer",
        )
        .merge(
            capacity,
            on="year",
            how="outer",
        )
        .sort_values(
            "year"
        )
        .reset_index(
            drop=True
        )
    )

    final.insert(
        0,
        "date",
        pd.to_datetime(
            final["year"].astype(
                str
            )
            + "-12-31"
        ),
    )

    final = final.drop(
        columns=[
            "year"
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "[INFO] Saved:",
        OUTPUT_PATH
    )

    print(
        "[INFO] Rows:",
        len(final)
    )

    print(
        "[INFO] Columns:",
        len(final.columns)
    )

    print(
        "[INFO] Date range:",
        final["date"].min(),
        "->",
        final["date"].max(),
    )

    print()
    print(
        final.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
