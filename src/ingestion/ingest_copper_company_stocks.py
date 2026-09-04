import time

import pandas as pd
import requests

from src.utils.paths import (
    EQUITIES_RAW_DIR,
)


OUTPUT_FILE = (
    EQUITIES_RAW_DIR
    / "copper_company_stocks_monthly.csv"
)

START_DATE = "2015-01-01"


COMPANIES = {
    "usa_sp500_freeport_mcmoran": "FCX",
    "usa_sp500_southern_copper": "SCCO",
    "uk_ftse100_antofagasta": "ANTO.L",
    "uk_ftse100_glencore": "GLEN.L",
    "australia_asx200_bhp": "BHP.AX",
    "australia_asx200_rio_tinto": "RIO.AX",
    "germany_dax_aurubis": "NDA.DE",
    "poland_wig20_kghm": "KGH.WA",
    "hong_kong_hang_seng_zijin_mining": "2899.HK",
    "china_csi300_jiangxi_copper": "600362.SS",
    "canada_tsx_composite_teck_resources": "TECK-B.TO",
    "canada_tsx_composite_lundin_mining": "LUN.TO",
    "canada_tsx_composite_first_quantum_minerals": "FM.TO",
    "canada_tsx_composite_hudbay_minerals": "HBM.TO",
    "canada_tsx_composite_capstone_copper": "CS.TO",

    # Additional mining / critical-material companies
    "usa_nyse_mp_materials": "MP",
    "usa_nyse_hecla_mining": "HL",
}


HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


FINAL_COLUMNS = [
    "date",
    *COMPANIES.keys(),
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
        + pd.Timedelta(days=1)
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

    if not timestamps:
        raise RuntimeError(
            f"No timestamps for {column_name}"
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
        f"[OK] {column_name:<52} "
        f"ticker={ticker:<10} "
        f"daily_rows={len(df):<5} "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


def daily_to_monthly_mean(
    df,
    column_name,
):
    monthly = (
        df
        .set_index("date")[
            column_name
        ]
        .resample("MS")
        .mean()
        .reset_index()
    )

    return monthly


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

    return (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )


def remove_current_partial_month(
    df,
):
    today = pd.Timestamp.today().normalize()

    current_month_start = today.replace(
        day=1
    )

    return (
        df[
            df["date"] < current_month_start
        ]
        .copy()
    )


def main():
    print(
        "=" * 120
    )

    print(
        "COPPER COMPANY STOCK INGESTION"
    )

    print(
        "=" * 120
    )

    monthly_frames = []

    for column_name, ticker in COMPANIES.items():

        daily_df = fetch_yahoo_daily(
            column_name=column_name,
            ticker=ticker,
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
        .sort_values("date")
        .reset_index(drop=True)
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
        "=" * 120
    )

    print(
        "FINAL DATASET"
    )

    print(
        "=" * 120
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

        valid = (
            final_df[
                [
                    "date",
                    column,
                ]
            ]
            .dropna()
        )

        if valid.empty:

            print(
                f"[FAIL] {column:<52} "
                "no observations"
            )

            continue

        print(
            f"[OK] {column:<52} "
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

    print()
    print(
        "[DONE] Copper company stock ingestion completed"
    )


if __name__ == "__main__":
    main()
