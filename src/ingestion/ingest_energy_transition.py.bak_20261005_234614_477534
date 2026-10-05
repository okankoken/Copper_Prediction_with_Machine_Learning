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
    year_codes = [
        str(value)
        for value in range(
            15,
            26,
        )
    ]

    payload = {
        "query": [
            {
                "code": "Region",
                "selection": {
                    "filter": "item",
                    "values": [
                        "GLO"
                    ],
                },
            },
            {
                "code": "Technology",
                "selection": {
                    "filter": "item",
                    "values": [
                        "0",
                        "1",
                        "2",
                    ],
                },
            },
            {
                "code": "Grid connection",
                "selection": {
                    "filter": "item",
                    "values": [
                        "0",
                        "1",
                    ],
                },
            },
            {
                "code": "Year",
                "selection": {
                    "filter": "item",
                    "values": year_codes,
                },
            },
        ],
        "response": {
            "format": "json-stat2"
        },
    }

    response = requests.post(
        IRENA_URL,
        json=payload,
        headers={
            **HEADERS,
            "Accept": "application/json",
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    df = jsonstat2_to_dataframe(
        data
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    grouped = (
        df.groupby(
            [
                "Year",
                "Technology",
            ],
            as_index=False,
        )[
            "value"
        ]
        .sum(
            min_count=1
        )
    )

    pivot = grouped.pivot(
        index="Year",
        columns="Technology",
        values="value",
    ).reset_index()

    pivot = pivot.rename(
        columns={
            "Year": "year",
            "Total renewable energy":
                "global_renewable_capacity_mw",
            "Solar energy":
                "global_solar_capacity_mw",
            "Wind energy":
                "global_wind_capacity_mw",
        }
    )

    pivot["year"] = pd.to_numeric(
        pivot["year"],
        errors="coerce",
    ).astype(
        "Int64"
    )

    return pivot[
        [
            "year",
            "global_renewable_capacity_mw",
            "global_solar_capacity_mw",
            "global_wind_capacity_mw",
        ]
    ]


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

    capacity = ingest_irena_capacity()

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
