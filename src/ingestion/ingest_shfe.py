from pathlib import Path
import time

import pandas as pd
import requests
from src.utils.paths import MARKET_RAW_DIR
OUTPUT_PATH = (
    MARKET_RAW_DIR
    / "shfe_copper_daily.csv"
)
START_DATE = pd.Timestamp(
    "2016-01-01"
)

BASE_URL = (
    "https://www.shfe.com.cn/data/tradedata/"
    "future/dailydata/kx{}.dat"
)

REPAIR_DATES = [
    "2020-03-24",
    "2023-05-29",
    "2026-07-20",
]


def get_existing_data():
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        OUTPUT_PATH
    )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    return df


def fetch_json(
    session,
    date,
    max_retries=3,
):
    date_text = (
        date.strftime("%Y%m%d")
    )

    url = BASE_URL.format(
        date_text
    )

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            response = session.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.json()

        except (
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            if attempt == max_retries:
                print(
                    f"[WARN] {date.date()} failed "
                    f"after {max_retries} attempts: {exc}"
                )

                return None

            time.sleep(
                attempt * 2
            )

        except Exception as exc:
            print(
                f"[WARN] {date.date()} failed: {exc}"
            )

            return None


def choose_main_copper_contract(
    payload,
    date,
):
    rows = payload.get(
        "o_curinstrument",
        []
    )

    if not rows:
        return None

    df = pd.DataFrame(
        rows
    )

    if "PRODUCTID" not in df.columns:
        return None

    df["PRODUCTID"] = (
        df["PRODUCTID"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["PRODUCTID"] == "cu_f"
    ].copy()

    if df.empty:
        return None

    df["DELIVERYMONTH"] = (
        df["DELIVERYMONTH"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["DELIVERYMONTH"]
        .str.match(
            r"^\d{4}$",
            na=False,
        )
    ].copy()

    if df.empty:
        return None

    df["VOLUME"] = pd.to_numeric(
        df["VOLUME"],
        errors="coerce",
    )

    df = df.sort_values(
        "VOLUME",
        ascending=False,
    )

    row = df.iloc[0]

    return {
        "date": date,
        "delivery_month": row.get(
            "DELIVERYMONTH"
        ),
        "open_cny_per_ton": row.get(
            "OPENPRICE"
        ),
        "high_cny_per_ton": row.get(
            "HIGHESTPRICE"
        ),
        "low_cny_per_ton": row.get(
            "LOWESTPRICE"
        ),
        "close_cny_per_ton": row.get(
            "CLOSEPRICE"
        ),
        "settlement_cny_per_ton": row.get(
            "SETTLEMENTPRICE"
        ),
        "pre_settlement_cny_per_ton": row.get(
            "PRESETTLEMENTPRICE"
        ),
        "volume_lots": row.get(
            "VOLUME"
        ),
        "open_interest_lots": row.get(
            "OPENINTEREST"
        ),
        "open_interest_change": row.get(
            "OPENINTERESTCHG"
        ),
        "turnover": row.get(
            "TURNOVER"
        ),
    }


def build_download_dates(
    existing_df,
):
    today = pd.Timestamp.today().normalize()

    dates = set()

    if existing_df.empty:
        start_date = START_DATE
    else:
        last_date = (
            existing_df["date"]
            .max()
            .normalize()
        )

        start_date = (
            last_date
            + pd.Timedelta(days=1)
        )

    if start_date <= today:
        for date in pd.date_range(
            start_date,
            today,
            freq="D",
        ):
            if date.weekday() < 5:
                dates.add(
                    date
                )

    existing_dates = set()

    if not existing_df.empty:
        existing_dates = set(
            existing_df[
                "date"
            ]
            .dt
            .normalize()
        )

    for repair_date in REPAIR_DATES:
        date = pd.Timestamp(
            repair_date
        )

        if date not in existing_dates:
            dates.add(
                date
            )

    return sorted(
        dates
    )


def main():
    existing_df = get_existing_data()

    print(
        "[INFO] Existing rows:",
        len(existing_df),
    )

    if not existing_df.empty:
        print(
            "[INFO] Existing period:",
            existing_df["date"].min(),
            "->",
            existing_df["date"].max(),
        )

    download_dates = build_download_dates(
        existing_df
    )

    print(
        "[INFO] Dates to check:",
        len(download_dates),
    )

    session = requests.Session()

    new_rows = []

    for index, date in enumerate(
        download_dates,
        start=1,
    ):
        payload = fetch_json(
            session,
            date,
        )

        if payload is None:
            continue

        row = choose_main_copper_contract(
            payload,
            date,
        )

        if row is not None:
            new_rows.append(
                row
            )

        if index % 100 == 0:
            print(
                f"[INFO] Checked {index} dates"
            )

    new_df = pd.DataFrame(
        new_rows
    )

    if existing_df.empty:
        combined_df = new_df
    elif new_df.empty:
        combined_df = existing_df
    else:
        combined_df = pd.concat(
            [
                existing_df,
                new_df,
            ],
            ignore_index=True,
        )

    combined_df["date"] = pd.to_datetime(
        combined_df["date"],
        errors="coerce",
    )

    combined_df = (
        combined_df
        .dropna(
            subset=["date"]
        )
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n[OK] Saved:",
        OUTPUT_PATH,
    )

    print(
        "[OK] Rows:",
        len(combined_df),
    )

    print(
        "[OK] First date:",
        combined_df["date"].min(),
    )

    print(
        "[OK] Last date:",
        combined_df["date"].max(),
    )

    if not new_df.empty:
        print(
            "[OK] New rows added:",
            len(new_df),
        )


if __name__ == "__main__":
    main()