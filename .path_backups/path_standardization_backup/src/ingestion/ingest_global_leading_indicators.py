from pathlib import Path
from io import StringIO

import pandas as pd
import requests


BASE_DIR = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

OUTPUT_FILE = (
    RAW_DIR
    / "global_leading_indicators_monthly.csv"
)


OECD_G20_CLI_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/"
    "G20.M.LI...AA...H"
    "?startPeriod=2015-01"
    "&dimensionAtObservation=AllDimensions"
)


HEADERS = {
    "Accept": "text/csv",
    "User-Agent": "Mozilla/5.0",
}


def fetch_g20_cli():
    response = requests.get(
        OECD_G20_CLI_URL,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    source_df = pd.read_csv(
        StringIO(
            response.text
        )
    )

    required_columns = [
        "TIME_PERIOD",
        "OBS_VALUE",
    ]

    missing = [
        column
        for column in required_columns
        if column not in source_df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing OECD columns: {missing}"
        )

    df = source_df[
        [
            "TIME_PERIOD",
            "OBS_VALUE",
        ]
    ].copy()

    df["date"] = pd.to_datetime(
        df["TIME_PERIOD"],
        format="%Y-%m",
        errors="coerce",
    )

    df[
        "g20_composite_leading_indicator"
    ] = pd.to_numeric(
        df["OBS_VALUE"],
        errors="coerce",
    )

    df = (
        df[
            [
                "date",
                "g20_composite_leading_indicator",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "[OK] G20 CLI:",
        len(df),
        "rows |",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    return df


def remove_future_dates(df):
    current_month_start = (
        pd.Timestamp.today()
        .normalize()
        .replace(day=1)
    )

    return (
        df[
            df["date"] <= current_month_start
        ]
        .copy()
    )


def main():
    print(
        "=" * 100
    )

    print(
        "GLOBAL LEADING INDICATORS INGESTION"
    )

    print(
        "=" * 100
    )

    g20_df = fetch_g20_cli()

    final_df = remove_future_dates(
        g20_df
    )

    RAW_DIR.mkdir(
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
        final_df.tail(
            12
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "[DONE] Global leading indicators ingestion completed"
    )


if __name__ == "__main__":
    main()
