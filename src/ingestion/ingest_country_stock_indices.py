from pathlib import Path
import time

import pandas as pd
import requests
from src.utils.paths import EQUITIES_RAW_DIR
OUTPUT_FILE = (
    EQUITIES_RAW_DIR
    / "country_stock_indices_monthly.csv"
)
START_DATE = "2015-01-01"


YAHOO_INDICES = {
    "usa_sp500": "^GSPC",
    "usa_dow_jones": "^DJI",
    "usa_nasdaq_composite": "^IXIC",
    "germany_dax": "^GDAXI",
    "japan_nikkei225": "^N225",
    "india_nifty50": "^NSEI",
    "south_korea_kospi": "^KS11",
    "indonesia_jakarta_composite": "^JKSE",
    "turkey_bist100": "XU100.IS",
    "uk_ftse100": "^FTSE",
    "hong_kong_hang_seng": "^HSI",
    "australia_asx200": "^AXJO",
    "canada_tsx_composite": "^GSPTSE",
}


INVESTING_FILES = {
    "chile_ipsa": (
        EQUITIES_RAW_DIR
        / "Chile_IPSA_Historical_Data.csv"
    ),
    "china_csi300": (
        EQUITIES_RAW_DIR
        / "China_CSI300_Historical_Data.csv"
    ),
    "poland_wig20": (
        EQUITIES_RAW_DIR
        / "Poland_WIG20_Historical_Data.csv"
    ),
}


BCRP_URL = (
    "https://estadisticas.bcrp.gob.pe/"
    "estadisticas/series/api/"
    "PN01142MM/json/2015-1/2030-12/ing"
)


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


FINAL_COLUMNS = [
    "date",
    "usa_sp500",
    "usa_dow_jones",
    "usa_nasdaq_composite",
    "china_csi300",
    "chile_ipsa",
    "peru_bvl_general",
    "germany_dax",
    "japan_nikkei225",
    "india_nifty50",
    "south_korea_kospi",
    "indonesia_jakarta_composite",
    "turkey_bist100",
    "uk_ftse100",
    "hong_kong_hang_seng",
    "australia_asx200",
    "canada_tsx_composite",
    "poland_wig20",
]


def to_unix(date_string):
    return int(
        pd.Timestamp(
            date_string,
            tz="UTC",
        ).timestamp()
    )


def fetch_yahoo_daily(
    column_name,
    ticker,
):
    today = (
        pd.Timestamp.today()
        .normalize()
        + pd.Timedelta(
            days=1
        )
    )

    encoded_ticker = requests.utils.quote(
        ticker,
        safe="",
    )

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{encoded_ticker}"
    )

    params = {
        "period1": to_unix(
            START_DATE
        ),
        "period2": int(
            today
            .tz_localize("UTC")
            .timestamp()
        ),
        "interval": "1d",
        "events": "history",
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    result = (
        payload
        .get("chart", {})
        .get("result")
    )

    if not result:
        raise RuntimeError(
            f"No Yahoo data for {column_name}"
        )

    item = result[0]

    timestamps = item.get(
        "timestamp",
        []
    )

    quote = (
        item
        .get("indicators", {})
        .get("quote", [{}])[0]
    )

    close_values = quote.get(
        "close",
        []
    )

    timezone_name = (
        item
        .get("meta", {})
        .get(
            "exchangeTimezoneName",
            "UTC",
        )
    )

    dates = (
        pd.to_datetime(
            timestamps,
            unit="s",
            utc=True,
        )
        .tz_convert(
            timezone_name
        )
        .tz_localize(
            None
        )
        .normalize()
    )

    df = pd.DataFrame(
        {
            "date": dates,
            column_name: close_values,
        }
    )

    df[column_name] = pd.to_numeric(
        df[column_name],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "date",
                column_name,
            ]
        )
        .drop_duplicates(
            subset=["date"],
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
        f"[OK] Yahoo {column_name}: "
        f"{len(df)} daily rows"
    )

    return df


def parse_investing_price(series):
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            ",",
            "",
            regex=False,
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def load_investing_daily(
    column_name,
    file_path,
):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing manual file: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8",
    )

    df["date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    df[column_name] = parse_investing_price(
        df["Price"]
    )

    df = (
        df[
            [
                "date",
                column_name,
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["date"],
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
        f"[OK] Investing {column_name}: "
        f"{len(df)} daily rows"
    )

    return df


def daily_to_monthly_mean(
    df,
    column_name,
):
    result = (
        df
        .set_index("date")[
            column_name
        ]
        .resample("MS")
        .mean()
        .reset_index()
    )

    return result


def fetch_peru_monthly():
    response = requests.get(
        BCRP_URL,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    rows = []

    for period in payload.get(
        "periods",
        []
    ):
        name = period.get(
            "name"
        )

        values = period.get(
            "values",
            []
        )

        if (
            not name
            or not values
        ):
            continue

        month_text, year_text = name.split(
            ".",
            maxsplit=1,
        )

        if month_text not in month_map:
            continue

        value = pd.to_numeric(
            values[0],
            errors="coerce",
        )

        rows.append(
            {
                "date": pd.Timestamp(
                    year=int(year_text),
                    month=month_map[
                        month_text
                    ],
                    day=1,
                ),
                "peru_bvl_general": value,
            }
        )

    df = pd.DataFrame(
        rows
    )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset=["date"],
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
        f"[OK] BCRP peru_bvl_general: "
        f"{len(df)} monthly rows"
    )

    return df


def merge_monthly_frames(
    frames,
):
    result = None

    for frame in frames:
        if result is None:
            result = frame.copy()
        else:
            result = result.merge(
                frame,
                on="date",
                how="outer",
                validate="one_to_one",
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

    return result


def remove_current_partial_month(
    df,
):
    today = pd.Timestamp.today().normalize()

    current_month_start = today.replace(
        day=1
    )

    return df[
        df["date"] < current_month_start
    ].copy()


def main():
    print(
        "=" * 100
    )

    print(
        "COUNTRY STOCK INDEX INGESTION"
    )

    print(
        "=" * 100
    )

    monthly_frames = []

    print()
    print(
        "[INFO] Yahoo indices"
    )

    for column_name, ticker in (
        YAHOO_INDICES.items()
    ):
        daily_df = fetch_yahoo_daily(
            column_name,
            ticker,
        )

        monthly_df = daily_to_monthly_mean(
            daily_df,
            column_name,
        )

        monthly_frames.append(
            monthly_df
        )

        time.sleep(
            0.4
        )

    print()
    print(
        "[INFO] Investing indices"
    )

    for column_name, file_path in (
        INVESTING_FILES.items()
    ):
        daily_df = load_investing_daily(
            column_name,
            file_path,
        )

        monthly_df = daily_to_monthly_mean(
            daily_df,
            column_name,
        )

        monthly_frames.append(
            monthly_df
        )

    print()
    print(
        "[INFO] Peru BVL"
    )

    peru_df = fetch_peru_monthly()

    monthly_frames.append(
        peru_df
    )

    final_df = merge_monthly_frames(
        monthly_frames
    )

    final_df = remove_current_partial_month(
        final_df
    )

    final_df = final_df[
        FINAL_COLUMNS
    ]

    final_df = (
        final_df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    EQUITIES_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL DATASET"
    )

    print(
        "=" * 100
    )

    print(
        "[OK] Saved:",
        OUTPUT_FILE,
    )

    print(
        "[INFO] Rows:",
        len(final_df),
    )

    print(
        "[INFO] Columns:",
        len(final_df.columns),
    )

    print(
        "[INFO] Date range:",
        final_df["date"].min().date(),
        "->",
        final_df["date"].max().date(),
    )

    print()

    print(
        "[INFO] Coverage:"
    )

    for column in FINAL_COLUMNS[1:]:
        valid = final_df[
            [
                "date",
                column,
            ]
        ].dropna()

        if valid.empty:
            print(
                f"[FAIL] {column:<32} "
                "no observations"
            )

            continue

        print(
            f"[OK] {column:<32} "
            f"months={len(valid):<4} "
            f"{valid['date'].min().date()} -> "
            f"{valid['date'].max().date()}"
        )

    print()

    print(
        final_df.tail(
            12
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()