import os
from pathlib import Path

import pandas as pd
from evds import evdsAPI
from src.utils.paths import MACRO_RAW_DIR
OUTPUT_PATH = (
    MACRO_RAW_DIR
    / "turkey_monthly.csv"
)
START_DATE = "01-01-2015"
END_DATE = (
    pd.Timestamp.today()
    .strftime("%d-%m-%Y")
)


SERIES = {
    "turkey_cpi_index": "TP.TUKFIY2025.GENEL",
    "turkey_domestic_ppi_index": "TP.TUFE1YI.T1",
    "turkey_unemployment_rate_percent": "TP.TIG08",
    "turkey_real_sector_confidence_index": "TP.GY1.N2.MA",
    "turkey_consumer_confidence_index": "TP.TG2.Y01",
    "turkey_manufacturing_capacity_utilization_percent": "TP.KKO.MA",

    "turkey_ppi_metal_ores_index": "TP.TUFE1YI.T9",
    "turkey_ppi_nonferrous_metal_ores_index": "TP.TUFE1YI.T11",
    "turkey_ppi_basic_metals_index": "TP.TUFE1YI.T73",
    "turkey_ppi_nonferrous_metals_index": "TP.TUFE1YI.T77",
    "turkey_ppi_electrical_equipment_index": "TP.TUFE1YI.T93",
    "turkey_ppi_wire_cable_index": "TP.TUFE1YI.T96",
    "turkey_ppi_household_appliances_index": "TP.TUFE1YI.T98",
    "turkey_ppi_motor_vehicles_index": "TP.TUFE1YI.T105",

    "turkey_capacity_intermediate_goods_percent": "TP.KKO2.IS.INTM",
    "turkey_capacity_investment_goods_percent": "TP.KKO2.IS.INVE",
    "turkey_capacity_basic_metals_percent": "TP.KKO2.IS.24",
    "turkey_capacity_fabricated_metals_percent": "TP.KKO2.IS.25",
    "turkey_capacity_electrical_equipment_percent": "TP.KKO2.IS.27",
    "turkey_capacity_machinery_percent": "TP.KKO2.IS.28",
    "turkey_capacity_motor_vehicles_percent": "TP.KKO2.IS.29",

    "turkey_exports_usd": "TP.IHRISICREV4.TT",
    "turkey_imports_usd": "TP.ITHISICREV4.TT",
}


FX_SERIES = {
    "turkey_usd_try_monthly_average": "TP.DK.USD.A.YTL",
    "turkey_eur_try_monthly_average": "TP.DK.EUR.A.YTL",
}


def load_env(path=".env"):
    path = Path(path)

    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1,
        )

        os.environ.setdefault(
            key.strip(),
            value.strip(),
        )


def get_api():
    load_env()

    api_key = os.getenv(
        "TCMB_EVDS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "TCMB_EVDS_API_KEY not found in .env"
        )

    return evdsAPI(
        api_key
    )


def parse_monthly_date(series):
    return pd.to_datetime(
        series.astype(str),
        format="%Y-%m",
        errors="coerce",
    )


def fetch_monthly_series(
    api,
    output_name,
    series_code,
):
    df = api.get_data(
        [series_code],
        startdate=START_DATE,
        enddate=END_DATE,
    )

    if "Tarih" not in df.columns:
        raise RuntimeError(
            f"Date column not found for {series_code}"
        )

    value_columns = [
        column
        for column in df.columns
        if column != "Tarih"
    ]

    if len(value_columns) != 1:
        raise RuntimeError(
            f"Unexpected columns for {series_code}: "
            f"{value_columns}"
        )

    value_column = value_columns[0]

    output = pd.DataFrame()

    output["date"] = parse_monthly_date(
        df["Tarih"]
    )

    output[output_name] = pd.to_numeric(
        df[value_column],
        errors="coerce",
    )

    output = output.dropna(
        subset=[
            "date"
        ]
    )

    return output



def fetch_fx_monthly(api):
    print(
        "[INFO] Fetching monthly FX averages from EVDS"
    )

    df = api.get_data(
        list(FX_SERIES.values()),
        startdate=START_DATE,
        enddate=END_DATE,
        aggregation_types="avg",
        frequency=5,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "No FX data returned from EVDS"
        )

    if "Tarih" not in df.columns:
        raise RuntimeError(
            "Date column not found in FX response"
        )

    code_to_column = {
        "TP_DK_USD_A_YTL":
            "turkey_usd_try_monthly_average",
        "TP_DK_EUR_A_YTL":
            "turkey_eur_try_monthly_average",
    }

    output = pd.DataFrame()

    output["date"] = pd.to_datetime(
        df["Tarih"].astype(str),
        format="%Y-%m",
        errors="coerce",
    )

    for source_column, output_column in code_to_column.items():

        if source_column not in df.columns:
            raise RuntimeError(
                f"FX column not found: {source_column}"
            )

        output[output_column] = pd.to_numeric(
            df[source_column],
            errors="coerce",
        )

    output = (
        output
        .dropna(
            subset=[
                "date"
            ]
        )
        .drop_duplicates(
            subset=[
                "date"
            ],
            keep="last",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    return output

def main():
    api = get_api()

    print(
        "[INFO] Fetching FX"
    )

    final = fetch_fx_monthly(
        api
    )

    for output_name, series_code in SERIES.items():

        print(
            "[INFO] Fetching:",
            output_name,
            "|",
            series_code,
        )

        series_df = fetch_monthly_series(
            api,
            output_name,
            series_code,
        )

        final = final.merge(
            series_df,
            on="date",
            how="outer",
        )

    final = final.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    trade_columns = [
        "turkey_exports_usd",
        "turkey_imports_usd",
    ]

    for column in trade_columns:
        final[column] = (
            final[column]
            * 1000
        )

    final[
        "turkey_foreign_trade_balance_usd"
    ] = (
        final[
            "turkey_exports_usd"
        ]
        - final[
            "turkey_imports_usd"
        ]
    )

    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(
            day=1
        )
        - pd.Timedelta(
            days=1
        )
    )

    last_full_month = (
        last_full_month_end
        .to_period(
            "M"
        )
        .to_timestamp()
    )

    final = final[
        final["date"]
        <= last_full_month
    ].copy()

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
        final.tail(
            12
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
