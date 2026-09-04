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
    / "fred_macro_monthly.csv"
)

FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


FRED_SERIES = {
    # USA
    "fed_interest_rate": "FEDFUNDS",
    "us_cpi_index": "CPIAUCNS",
    "us_industrial_production": "INDPRO",
    "us_10y_treasury_yield": "GS10",
    "us_2y_yield": "GS2",
    "us_10y_2y_spread": "T10Y2YM",
    "us_housing_starts": "HOUST",
    "us_ppi": "PPIACO",
    "m2_money_supply": "M2SL",

    # Commodities
    "brent_oil_usd": "POILBREUSDM",
    "natural_gas_usd": "DHHNGSP",
    "copper_fred_usd_per_ton": "PCOPPUSDM",

    # FX
    "eur_usd": "DEXUSEU",
    "usd_cny": "DEXCHUS",
    "usd_jpy": "DEXJPUS",
    "usd_gbp": "DEXUSUK",
    "usd_inr": "DEXINUS",
    "usd_krw": "DEXKOUS",
    "dollar_index": "DTWEXBGS",

    # Risk
    "vix_index": "VIXCLS",

    # Euro Area / Germany
    "euro_area_hicp": "CP0000EZ19M086NEST",
    "germany_10y_bund_yield": "IRLTLT01DEM156N",

    # Japan
    "japan_10y_yield": "IRLTLT01JPM156N",
    "japan_manufacturing_growth": "JPNPRMNTO01GPSAM",

    # India
    "india_10y_yield": "INDIRLTLT01STM",
    "india_call_money_rate": "IRSTCI01INM156N",
    "india_industrial_production": "INDPRINTO01GYSAM",

    # South Korea
    "korea_10y_yield": "IRLTLT01KRM156N",
    "korea_exports": "XTEXVA01KRM667N",
    "korea_industrial_production": "KORPRINTO01GYSAM",
    "korea_call_money_rate": "IRSTCI01KRM156N",
    
    # Germany
    "germany_10y_bund_yield": "IRLTLT01DEM156N",
    "germany_exports": "XTEXVA01DEM667S",
    "germany_manufacturing_order_books": "BSOBLV02DEM460S",
    
    # Indonesia
    "indonesia_exports": "XTEXVA01IDM667S",
    "indonesia_call_money_rate": "IRSTCI01IDM156N",
    "usd_idr": "CCUSMA02IDM618N",
    "indonesia_reer": "CCRETT01IDM661N",
    
    
}


def get_fred_api_key():
    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        raise FileNotFoundError(
            f".env file not found: {env_file}"
        )

    for line in env_file.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if line.startswith("FRED_API_KEY="):
            api_key = line.split(
                "=",
                1,
            )[1].strip()

            if not api_key:
                raise ValueError(
                    "FRED_API_KEY is empty"
                )

            return api_key

    raise ValueError(
        "FRED_API_KEY not found in .env"
    )


def get_last_full_month():
    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_full_month_end


def fetch_fred_series(
    api_key,
    feature_name,
    series_id,
    start_date,
    end_date,
):
    print(
        f"[INFO] Fetching "
        f"{feature_name} "
        f"({series_id})..."
    )

    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": (
                start_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "observation_end": (
                end_date.strftime(
                    "%Y-%m-%d"
                )
            ),
        },
        timeout=60,
        headers={
            "User-Agent": (
                "Mozilla/5.0"
            )
        },
    )

    response.raise_for_status()

    data = response.json()

    observations = data.get(
        "observations",
        [],
    )

    if not observations:
        raise ValueError(
            f"No observations returned "
            f"for {feature_name} "
            f"({series_id})"
        )

    df = pd.DataFrame(
        observations
    )

    required_columns = {
        "date",
        "value",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for "
            f"{feature_name}: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "date",
            "value",
        ]
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df[feature_name] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.drop(
        columns=[
            "value",
        ]
    )

    df = df.dropna(
        subset=[
            "date",
        ]
    )

    df = df[
        (
            df["date"]
            >= start_date
        )
        &
        (
            df["date"]
            <= end_date
        )
    ].copy()

    if df.empty:
        raise ValueError(
            f"No valid dated observations "
            f"for {feature_name}"
        )

    df["date"] = (
        df["date"]
        .dt.to_period("M")
    )

    monthly = (
        df.groupby(
            "date",
            as_index=False,
        )[feature_name]
        .mean()
    )

    valid_count = (
        monthly[
            feature_name
        ]
        .notna()
        .sum()
    )

    print(
        f"[OK] {feature_name}: "
        f"{valid_count} observations"
    )

    return monthly


def build_fred_macro(
    start_date="2005-01-01",
):
    api_key = get_fred_api_key()

    start_date = pd.Timestamp(
        start_date
    )

    end_date = get_last_full_month()

    print(
        f"[INFO] FRED start date: "
        f"{start_date.date()}"
    )

    print(
        f"[INFO] FRED end date: "
        f"{end_date.date()}"
    )

    frames = []

    for (
        feature_name,
        series_id,
    ) in FRED_SERIES.items():

        try:
            frame = fetch_fred_series(
                api_key=api_key,
                feature_name=feature_name,
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
            )

            frames.append(
                frame
            )

        except Exception as exc:
            print(
                f"[FAIL] {feature_name} "
                f"({series_id}): "
                f"{repr(exc)}"
            )

            raise

    if not frames:
        raise ValueError(
            "No FRED series were loaded"
        )

    result = frames[0]

    for frame in frames[1:]:
        result = result.merge(
            frame,
            on="date",
            how="outer",
        )

    full_month_index = pd.period_range(
        start=start_date.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    calendar = pd.DataFrame(
        {
            "date": full_month_index
        }
    )

    result = calendar.merge(
        result,
        on="date",
        how="left",
    )

    result = result.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    result["date"] = (
        result["date"]
        .astype(str)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"[OK] Saved FRED data: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"[OK] Rows: "
        f"{len(result)}"
    )

    print(
        f"[OK] First month: "
        f"{result['date'].iloc[0]}"
    )

    print(
        f"[OK] Last month: "
        f"{result['date'].iloc[-1]}"
    )

    print(
        "\n[INFO] Last 5 rows:"
    )

    print(
        result.tail(
            5
        ).to_string(
            index=False
        )
    )

    return result


if __name__ == "__main__":
    build_fred_macro()