from io import BytesIO
from pathlib import Path

import os
import pandas as pd
import requests
from dotenv import load_dotenv
import re

PROJECT_ROOT = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "global_macro_monthly.csv"
)


EUROSTAT_BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
)

BOJ_API_URL = (
    "https://www.stat-search.boj.or.jp/"
    "api/v1/getDataCode"
)

JAPAN_CPI_URL = (
    "https://www.e-stat.go.jp/en/stat-search/"
    "file-download?fileKind=1&statInfId=000040482945"
)

METI_IIP_CURRENT_URL = (
    "https://www.meti.go.jp/english/statistics/"
    "tyo/iip/xls/b2020_gsm1e.xlsx"
)

METI_IIP_HISTORY_URL = (
    "https://www.meti.go.jp/english/statistics/"
    "tyo/iip/xls/b2020_sgs1e.xlsx"
)


EUROSTAT_SERIES = {
    "euro_area_ppi": {
        "dataset": "sts_inpp_m",
        "params": {
            "indic_bt": "PRC_PRR",
            "nace_r2": "B-D",
            "s_adj": "NSA",
            "unit": "I21",
            "geo": "EA20",
        },
    },
    "euro_area_copper_production_ppi": {
        "dataset": "sts_inpp_m",
        "params": {
            "indic_bt": "PRC_PRR",
            "nace_r2": "C2444",
            "s_adj": "NSA",
            "unit": "I21",
            "geo": "EA20",
        },
    },
    "euro_area_industrial_production": {
        "dataset": "sts_inpr_m",
        "params": {
            "indic_bt": "PRD",
            "nace_r2": "B-D",
            "s_adj": "SCA",
            "unit": "I21",
            "geo": "EA20",
        },
    },
    "euro_area_construction_output": {
        "dataset": "sts_copr_m",
        "params": {
            "indic_bt": "PRD",
            "nace_r2": "F",
            "s_adj": "SCA",
            "unit": "I21",
            "geo": "EA20",
        },
    },
}


BOJ_SERIES = {
    "boj_call_rate": {
        "db": "FM01",
        "code": "STRDCLUCON",
        "frequency": "daily",
    },
    "japan_ppi": {
        "db": "PR01",
        "code": "PRCG20_2200000000",
        "frequency": "monthly",
    },
}


def get_last_full_month():
    """
    Return the last fully completed calendar month.
    """

    today = pd.Timestamp.today().normalize()

    last_full_month_end = (
        today.replace(day=1)
        - pd.Timedelta(days=1)
    )

    return last_full_month_end.to_period("M")


def parse_eurostat_series(
    data,
    variable_name,
):
    """
    Parse a Eurostat JSON response into a monthly dataframe.
    """

    time_category = (
        data["dimension"]
        ["time"]
        ["category"]
    )

    time_index = time_category["index"]

    if not isinstance(
        time_index,
        dict,
    ):
        raise ValueError(
            "Unexpected Eurostat time index format"
        )

    periods = sorted(
        time_index,
        key=time_index.get,
    )

    values = data.get(
        "value",
        {},
    )

    rows = []

    for position, period in enumerate(
        periods
    ):
        value = values.get(
            str(position)
        )

        rows.append(
            {
                "date": period,
                variable_name: value,
            }
        )

    df = pd.DataFrame(
        rows
    )

    df["date"] = pd.PeriodIndex(
        df["date"],
        freq="M",
    )

    df[variable_name] = pd.to_numeric(
        df[variable_name],
        errors="coerce",
    )

    return df


def fetch_eurostat_series(
    variable_name,
    config,
    start_period="2005-01",
):
    """
    Fetch one monthly Eurostat series.
    """

    dataset = config["dataset"]

    url = (
        EUROSTAT_BASE_URL
        + dataset
    )

    params = config[
        "params"
    ].copy()

    params[
        "sinceTimePeriod"
    ] = start_period

    params["lang"] = "en"

    print(
        f"[INFO] Fetching "
        f"{variable_name} "
        f"from Eurostat..."
    )

    response = requests.get(
        url,
        params=params,
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    df = parse_eurostat_series(
        data,
        variable_name,
    )

    valid_count = int(
        df[
            variable_name
        ]
        .notna()
        .sum()
    )

    print(
        f"[OK] {variable_name}: "
        f"{valid_count} observations"
    )

    return df


def fetch_ecb_policy_rate(
    start_period="2005-01",
):
    """
    Fetch the ECB deposit facility rate and map it
    to monthly observations using the latest rate
    effective at each month end.
    """

    url = (
        "https://www.ecb.europa.eu/stats/"
        "policy_and_exchange_rates/"
        "key_ecb_interest_rates/"
        "html/index.en.html"
    )

    print(
        "[INFO] Fetching "
        "ecb_policy_rate "
        "from ECB..."
    )

    tables = pd.read_html(
        url
    )

    if not tables:
        raise ValueError(
            "ECB interest rate table not found"
        )

    df = tables[0].copy()

    # Use positions because ECB table headers
    # may contain duplicate column names.
    year_series = df.iloc[
        :,
        0,
    ]

    month_day_series = df.iloc[
        :,
        1,
    ]

    deposit_series = df.iloc[
        :,
        2,
    ]

    df_clean = pd.DataFrame(
        {
            "year": pd.to_numeric(
                year_series,
                errors="coerce",
            ),
            "month_day": (
                month_day_series
                .astype(str)
                .str.replace(
                    r"[^A-Za-z0-9. ]",
                    "",
                    regex=True,
                )
                .str.strip()
            ),
            "ecb_policy_rate": (
                pd.to_numeric(
                    deposit_series,
                    errors="coerce",
                )
            ),
        }
    )

    df_clean = df_clean[
        df_clean[
            "year"
        ].notna()
        & df_clean[
            "month_day"
        ].notna()
        & df_clean[
            "ecb_policy_rate"
        ].notna()
    ].copy()

    df_clean["year"] = (
        df_clean[
            "year"
        ]
        .astype(int)
        .astype(str)
    )

    df_clean[
        "effective_date"
    ] = pd.to_datetime(
        df_clean[
            "month_day"
        ]
        + " "
        + df_clean[
            "year"
        ],
        errors="coerce",
        dayfirst=True,
    )

    df_clean = df_clean[
        df_clean[
            "effective_date"
        ].notna()
    ].copy()

    df_clean = df_clean.sort_values(
        "effective_date"
    )

    start_date = (
        pd.Period(
            start_period,
            freq="M",
        )
        .to_timestamp(
            how="end"
        )
        .normalize()
    )

    last_full_month = (
        get_last_full_month()
    )

    end_date = (
        last_full_month
        .to_timestamp(
            how="end"
        )
        .normalize()
    )

    monthly_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="ME",
    )

    monthly = pd.DataFrame(
        {
            "month_end": (
                monthly_dates
            )
        }
    )

    changes = df_clean[
        [
            "effective_date",
            "ecb_policy_rate",
        ]
    ].copy()

    monthly = pd.merge_asof(
        monthly.sort_values(
            "month_end"
        ),
        changes.sort_values(
            "effective_date"
        ),
        left_on="month_end",
        right_on="effective_date",
        direction="backward",
    )

    monthly["date"] = (
        monthly[
            "month_end"
        ]
        .dt.to_period("M")
    )

    monthly = monthly[
        [
            "date",
            "ecb_policy_rate",
        ]
    ].copy()

    valid_count = int(
        monthly[
            "ecb_policy_rate"
        ]
        .notna()
        .sum()
    )

    print(
        "[OK] ecb_policy_rate: "
        f"{valid_count} observations"
    )

    return monthly


def fetch_boj_series(
    variable_name,
    config,
    start_period="2005-01",
):
    """
    Fetch daily or monthly series from the
    Bank of Japan API.
    """

    print(
        f"[INFO] Fetching "
        f"{variable_name} "
        f"from BOJ..."
    )

    response = requests.get(
        BOJ_API_URL,
        params={
            "format": "json",
            "lang": "en",
            "db": config["db"],
            "code": config["code"],
            "startDate": (
                start_period.replace(
                    "-",
                    "",
                )
            ),
        },
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    if data.get(
        "STATUS"
    ) != 200:
        raise ValueError(
            f"BOJ API error for "
            f"{variable_name}: "
            f"{data.get('MESSAGE')}"
        )

    resultset = data.get(
        "RESULTSET",
        [],
    )

    if not resultset:
        raise ValueError(
            f"No BOJ data for "
            f"{variable_name}"
        )

    values_block = (
        resultset[0]
        .get(
            "VALUES",
            {},
        )
    )

    dates = values_block.get(
        "SURVEY_DATES",
        [],
    )

    values = values_block.get(
        "VALUES",
        [],
    )

    if len(
        dates
    ) != len(
        values
    ):
        raise ValueError(
            f"BOJ date/value mismatch for "
            f"{variable_name}"
        )

    df = pd.DataFrame(
        {
            "raw_date": dates,
            variable_name: values,
        }
    )

    df[variable_name] = (
        pd.to_numeric(
            df[
                variable_name
            ],
            errors="coerce",
        )
    )

    if (
        config[
            "frequency"
        ]
        == "daily"
    ):

        df["raw_date"] = (
            pd.to_datetime(
                df[
                    "raw_date"
                ].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
        )

        df = df[
            df[
                "raw_date"
            ].notna()
        ].copy()

        df["date"] = (
            df[
                "raw_date"
            ]
            .dt.to_period("M")
        )

        # Monthly mean uses only actual
        # daily observations.
        df = (
            df.groupby(
                "date",
                as_index=False,
            )[
                variable_name
            ]
            .mean()
        )

    elif (
        config[
            "frequency"
        ]
        == "monthly"
    ):

        df["date"] = (
            pd.PeriodIndex(
                df[
                    "raw_date"
                ].astype(str),
                freq="M",
            )
        )

        df = df[
            [
                "date",
                variable_name,
            ]
        ].copy()

    else:
        raise ValueError(
            f"Unknown BOJ frequency: "
            f"{config['frequency']}"
        )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df = df[
        (
            df["date"]
            >= start_month
        )
        & (
            df["date"]
            <= last_full_month
        )
    ].copy()

    valid_count = int(
        df[
            variable_name
        ]
        .notna()
        .sum()
    )

    print(
        f"[OK] {variable_name}: "
        f"{valid_count} observations"
    )

    return df


def fetch_japan_cpi(
    start_period="2005-01",
):
    """
    Fetch Japan CPI from the official e-Stat CSV.
    """

    print(
        "[INFO] Fetching "
        "japan_cpi "
        "from e-Stat..."
    )

    response = requests.get(
        JAPAN_CPI_URL,
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    df = pd.read_csv(
        BytesIO(
            response.content
        ),
        encoding="cp932",
    )

    if df.shape[1] < 2:
        raise ValueError(
            "Unexpected Japan CPI file structure"
        )

    # The first column contains metadata
    # and YYYYMM period values.
    raw_date = (
        df.iloc[
            :,
            0,
        ]
        .astype(str)
        .str.strip()
    )

    # The second column contains
    # the All Items CPI.
    raw_cpi = df.iloc[
        :,
        1,
    ]

    valid_date_mask = (
        raw_date.str.fullmatch(
            r"\d{6}",
            na=False,
        )
    )

    df_clean = pd.DataFrame(
        {
            "raw_date": (
                raw_date[
                    valid_date_mask
                ]
            ),
            "japan_cpi": (
                raw_cpi[
                    valid_date_mask
                ]
            ),
        }
    ).copy()

    df_clean["date"] = (
        pd.PeriodIndex(
            df_clean[
                "raw_date"
            ],
            freq="M",
        )
    )

    df_clean[
        "japan_cpi"
    ] = pd.to_numeric(
        df_clean[
            "japan_cpi"
        ],
        errors="coerce",
    )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df_clean = df_clean[
        (
            df_clean["date"]
            >= start_month
        )
        & (
            df_clean["date"]
            <= last_full_month
        )
    ].copy()

    df_clean = df_clean[
        [
            "date",
            "japan_cpi",
        ]
    ].copy()

    valid_count = int(
        df_clean[
            "japan_cpi"
        ]
        .notna()
        .sum()
    )

    print(
        "[OK] japan_cpi: "
        f"{valid_count} observations"
    )

    return df_clean


def fetch_japan_industrial_production(
    start_period="2005-01",
):
    """
    Fetch Japan industrial production
    from official METI historical and
    current Excel files.
    """

    print(
        "[INFO] Fetching "
        "japan_industrial_production "
        "from METI..."
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        ),
        "Referer": (
            "https://www.meti.go.jp/english/"
            "statistics/tyo/iip/"
            "b2020_result-2.html"
        ),
    }

    # -------------------------------------------------
    # Historical connected series
    # -------------------------------------------------

    history_response = requests.get(
        METI_IIP_HISTORY_URL,
        headers=headers,
        timeout=60,
    )

    history_response.raise_for_status()

    history_raw = pd.read_excel(
        BytesIO(
            history_response.content
        ),
        sheet_name="Production",
        header=None,
    )

    # Locate Mining and manufacturing
    # using the official METI item code.
    history_item_row = (
        history_raw.iloc[
            1
        ]
        .astype(str)
    )

    history_matches = (
        history_item_row[
            history_item_row
            .str.strip()
            .eq(
                "1000000000"
            )
        ]
    )

    if history_matches.empty:
        raise ValueError(
            "METI historical industry "
            "item code not found"
        )

    history_value_col = (
        history_matches
        .index[0]
    )

    history_dates = (
        history_raw.iloc[
            4:,
            1,
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    history_values = pd.to_numeric(
        history_raw.iloc[
            4:,
            history_value_col,
        ],
        errors="coerce",
    )

    history = pd.DataFrame(
        {
            "raw_date": (
                history_dates
            ),
            "japan_industrial_production": (
                history_values.values
            ),
        }
    )

    history = history[
        history[
            "raw_date"
        ]
        .str.fullmatch(
            r"\d{6}",
            na=False,
        )
    ].copy()

    history["date"] = (
        pd.PeriodIndex(
            history[
                "raw_date"
            ],
            freq="M",
        )
    )

    history = history[
        [
            "date",
            "japan_industrial_production",
        ]
    ].copy()

    # -------------------------------------------------
    # Current METI series
    # -------------------------------------------------

    current_response = requests.get(
        METI_IIP_CURRENT_URL,
        headers=headers,
        timeout=60,
    )

    current_response.raise_for_status()

    current_raw = pd.read_excel(
        BytesIO(
            current_response.content
        ),
        sheet_name="Production",
        header=None,
    )

    item_codes = pd.to_numeric(
        current_raw.iloc[
            :,
            0,
        ],
        errors="coerce",
    )

    target_rows = current_raw[
        item_codes.eq(
            1000000000
        )
    ]

    if target_rows.empty:
        raise ValueError(
            "METI current industry "
            "item code not found"
        )

    target_row_index = (
        target_rows.index[0]
    )

    current_dates = (
        current_raw.iloc[
            2,
            3:,
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )

    current_values = pd.to_numeric(
        current_raw.iloc[
            target_row_index,
            3:,
        ],
        errors="coerce",
    )

    current = pd.DataFrame(
        {
            "raw_date": (
                current_dates.values
            ),
            "japan_industrial_production": (
                current_values.values
            ),
        }
    )

    current = current[
        current[
            "raw_date"
        ]
        .str.fullmatch(
            r"\d{6}",
            na=False,
        )
    ].copy()

    current["date"] = (
        pd.PeriodIndex(
            current[
                "raw_date"
            ],
            freq="M",
        )
    )

    current = current[
        [
            "date",
            "japan_industrial_production",
        ]
    ].copy()

    # Current observations have priority
    # when history and current files overlap.
    combined = pd.concat(
        [
            history,
            current,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset="date",
            keep="last",
        )
    )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    combined = combined[
        (
            combined["date"]
            >= start_month
        )
        & (
            combined["date"]
            <= last_full_month
        )
    ].copy()

    combined = (
        combined
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    valid_count = int(
        combined[
            "japan_industrial_production"
        ]
        .notna()
        .sum()
    )

    if valid_count == 0:
        raise ValueError(
            "No valid METI industrial "
            "production observations"
        )

    first_valid = combined.loc[
        combined[
            "japan_industrial_production"
        ].notna(),
        "date",
    ].min()

    last_valid = combined.loc[
        combined[
            "japan_industrial_production"
        ].notna(),
        "date",
    ].max()

    print(
        "[OK] "
        "japan_industrial_production: "
        f"{valid_count} observations"
    )

    print(
        "[OK] "
        "japan_industrial_production "
        f"range: {first_valid} -> "
        f"{last_valid}"
    )

    return combined


def fetch_bps_indonesia_inflation(
    start_period="2005-01",
):
    """
    Fetch Indonesia national monthly
    month-to-month inflation from BPS.
    """

    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "BPS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "BPS_API_KEY not found in .env"
        )

    print(
        "[INFO] Fetching "
        "indonesia_inflation_mom "
        "from BPS Indonesia..."
    )

    base_url = (
        "https://webapi.bps.go.id/"
        "v1/api/list/model/data"
    )

    # BPS variable:
    # Inflasi Bulanan (M-to-M)
    var_id = 1

    # National Indonesia aggregate
    vervar_id = 9999

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    start_year = (
        start_month.year
    )

    end_year = (
        last_full_month.year
    )

    rows = []

    # BPS period IDs follow this mapping:
    #
    # 2026 -> 126
    # 2025 -> 125
    # 2024 -> 124
    #
    # Therefore:
    # th_id = year - 1900
    #
    # The monthly request structure below has been
    # validated directly against the BPS API.
    for year in range(
        start_year,
        end_year + 1,
    ):
        th_id = (
            year
            - 1900
        )

        first_month = 1
        final_month = 12

        if year == start_year:
            first_month = (
                start_month.month
            )

        if year == end_year:
            final_month = (
                last_full_month.month
            )

        for month in range(
            first_month,
            final_month + 1,
        ):
            url = (
                f"{base_url}/"
                f"domain/0000/"
                f"var/{var_id}/"
                f"th/{th_id}/"
                f"turth/{month}/"
                f"vervar/{vervar_id}/"
                f"key/{api_key}"
            )

            try:
                response = requests.get(
                    url,
                    timeout=60,
                )

                response.raise_for_status()

                payload = (
                    response.json()
                )

            except (
                requests.RequestException
            ) as exc:
                print(
                    "[WARN] BPS request "
                    "failed for "
                    f"{year}-{month:02d}: "
                    f"{exc}"
                )

                continue

            if (
                payload.get(
                    "data-availability"
                )
                != "available"
            ):
                continue

            content = payload.get(
                "datacontent",
                {},
            )

            if not content:
                continue

            values = list(
                content.values()
            )

            if not values:
                continue

            value = values[0]

            rows.append(
                {
                    "date": (
                        f"{year}-"
                        f"{month:02d}"
                    ),
                    "indonesia_inflation_mom": (
                        value
                    ),
                }
            )

        print(
            "[INFO] BPS Indonesia "
            "inflation completed: "
            f"{year}"
        )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise ValueError(
            "No Indonesia inflation data "
            "returned from BPS"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m",
        errors="coerce",
    ).dt.to_period(
        "M"
    )

    df[
        "indonesia_inflation_mom"
    ] = pd.to_numeric(
        df[
            "indonesia_inflation_mom"
        ],
        errors="coerce",
    )

    df = df[
        df[
            "date"
        ].notna()
    ].copy()

    df = df[
        (
            df["date"]
            >= start_month
        )
        & (
            df["date"]
            <= last_full_month
        )
    ].copy()

    df = (
        df
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

    valid_count = int(
        df[
            "indonesia_inflation_mom"
        ]
        .notna()
        .sum()
    )

    if valid_count == 0:
        raise ValueError(
            "No valid Indonesia inflation "
            "observations returned from BPS"
        )

    first_valid = df.loc[
        df[
            "indonesia_inflation_mom"
        ].notna(),
        "date",
    ].min()

    last_valid = df.loc[
        df[
            "indonesia_inflation_mom"
        ].notna(),
        "date",
    ].max()

    print(
        "[OK] "
        "indonesia_inflation_mom: "
        f"{valid_count} observations"
    )

    print(
        "[OK] "
        "indonesia_inflation_mom "
        f"range: {first_valid} -> "
        f"{last_valid}"
    )

    return df

def fetch_drc_policy_rate(
    start_period="2005-01",
):
    """
    Fetch the DRC monthly policy rate
    from the Banque Centrale du Congo
    official statistics page.
    """

    print(
        "[INFO] Fetching "
        "drc_policy_rate "
        "from Banque Centrale du Congo..."
    )

    url = (
        "https://www.bcc.cd/statistiques/"
        "secteur-monetaire/taux-interet"
    )

    series_key = (
        "taux-interet--taux-directeur-bcc"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    html = response.text

    pattern = re.compile(
        r'\\"period\\":\\"(\d{4}-\d{2})\\"'
        r'.*?'
        + re.escape(
            '\\"'
            + series_key
            + '\\":'
        )
        + r'(-?\d+(?:\.\d+)?)',
        flags=re.DOTALL,
    )

    matches = pattern.findall(
        html
    )

    if not matches:
        raise ValueError(
            "No DRC policy rate data found"
        )

    rows = [
        {
            "date": period,
            "drc_policy_rate": float(value),
        }
        for period, value in matches
    ]

    df = pd.DataFrame(
        rows
    )

    df["date"] = pd.PeriodIndex(
        df["date"],
        freq="M",
    )

    df[
        "drc_policy_rate"
    ] = pd.to_numeric(
        df[
            "drc_policy_rate"
        ],
        errors="coerce",
    )

    df = (
        df
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df = df[
        (
            df["date"]
            >= start_month
        )
        & (
            df["date"]
            <= last_full_month
        )
    ].copy()

    valid_count = int(
        df[
            "drc_policy_rate"
        ]
        .notna()
        .sum()
    )

    if valid_count == 0:
        raise ValueError(
            "No valid DRC policy rate observations"
        )

    first_valid = df.loc[
        df[
            "drc_policy_rate"
        ].notna(),
        "date",
    ].min()

    last_valid = df.loc[
        df[
            "drc_policy_rate"
        ].notna(),
        "date",
    ].max()

    print(
        "[OK] drc_policy_rate: "
        f"{valid_count} observations"
    )

    print(
        "[OK] drc_policy_rate range: "
        f"{first_valid} -> "
        f"{last_valid}"
    )

    return df

def fetch_drc_fx(
    start_period="2005-01",
):
    """
    Fetch the official DRC CDF/USD exchange rate
    from Banque Centrale du Congo and convert
    daily observations to monthly averages.
    """

    print(
        "[INFO] Fetching usd_cdf "
        "from Banque Centrale du Congo..."
    )

    url = (
        "https://www.bcc.cd/statistiques/"
        "secteur-exterieur/marche-des-changes"
    )

    series_key = (
        "marche-des-changes--cours-indicatif-cdf-usd"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    html = response.text

    pattern = re.compile(
        r'\\"period\\":\\"([^"]+)\\"'
        r'.*?'
        + re.escape(
            '\\"'
            + series_key
            + '\\":'
        )
        + r'(-?\d+(?:\.\d+)?)',
        flags=re.DOTALL,
    )

    matches = pattern.findall(
        html
    )

    if not matches:
        raise ValueError(
            "No DRC FX data found"
        )

    df = pd.DataFrame(
        matches,
        columns=[
            "date",
            "usd_cdf",
        ],
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    df["usd_cdf"] = pd.to_numeric(
        df["usd_cdf"],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .sort_values(
            "date"
        )
    )

    df = (
        df
        .set_index(
            "date"
        )["usd_cdf"]
        .resample("MS")
        .mean()
        .reset_index()
    )

    df["date"] = (
        df["date"]
        .dt.to_period("M")
    )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df = df[
        (df["date"] >= start_month)
        & (df["date"] <= last_full_month)
    ].copy()

    valid = df[
        "usd_cdf"
    ].notna()

    print(
        "[OK] usd_cdf: "
        f"{valid.sum()} observations"
    )

    print(
        "[OK] usd_cdf range: "
        f"{df.loc[valid, 'date'].min()} -> "
        f"{df.loc[valid, 'date'].max()}"
    )

    return df


def fetch_drc_inflation(
    start_period="2005-01",
):
    """
    Fetch the official monthly DRC inflation rate
    from Banque Centrale du Congo.
    """

    print(
        "[INFO] Fetching drc_inflation_mom "
        "from Banque Centrale du Congo..."
    )

    url = (
        "https://www.bcc.cd/statistiques/"
        "secteur-reel/inflation"
    )

    series_key = (
        "inflation--inflation-mensuelle"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    html = response.text

    pattern = re.compile(
        r'\\"period\\":\\"([^"]+)\\"'
        r'.*?'
        + re.escape(
            '\\"'
            + series_key
            + '\\":'
        )
        + r'(-?\d+(?:\.\d+)?)',
        flags=re.DOTALL,
    )

    matches = pattern.findall(
        html
    )

    if not matches:
        raise ValueError(
            "No DRC inflation data found"
        )

    df = pd.DataFrame(
        matches,
        columns=[
            "date",
            "drc_inflation_mom",
        ],
    )

    df["date"] = (
        df["date"]
        .str.extract(
            r"(\d{4}-\d{2})",
            expand=False,
        )
    )

    df["date"] = pd.PeriodIndex(
        df["date"],
        freq="M",
    )

    df[
        "drc_inflation_mom"
    ] = pd.to_numeric(
        df[
            "drc_inflation_mom"
        ],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    start_month = pd.Period(
        start_period,
        freq="M",
    )

    last_full_month = (
        get_last_full_month()
    )

    df = df[
        (df["date"] >= start_month)
        & (df["date"] <= last_full_month)
    ].copy()

    valid = df[
        "drc_inflation_mom"
    ].notna()

    print(
        "[OK] drc_inflation_mom: "
        f"{valid.sum()} observations"
    )

    print(
        "[OK] drc_inflation_mom range: "
        f"{df.loc[valid, 'date'].min()} -> "
        f"{df.loc[valid, 'date'].max()}"
    )

    return df

def build_global_macro():
    """
    Fetch all official global macro datasets,
    merge them by monthly period, and save
    the consolidated raw dataset.
    """

    start_period = "2005-01"

    last_full_month = (
        get_last_full_month()
    )

    print(
        f"[INFO] Start period: "
        f"{start_period}"
    )

    print(
        f"[INFO] Last full month: "
        f"{last_full_month}"
    )

    # Every source-specific dataframe is first
    # collected here.
    #
    # The final merge is performed only after
    # all datasets have been fetched.
    frames = []

    # -------------------------------------------------
    # Euro Area - Eurostat
    # -------------------------------------------------

    for (
        variable_name,
        config,
    ) in EUROSTAT_SERIES.items():

        df = fetch_eurostat_series(
            variable_name,
            config,
            start_period=start_period,
        )

        frames.append(
            df
        )

    # -------------------------------------------------
    # Euro Area - ECB policy rate
    # -------------------------------------------------

    ecb_df = fetch_ecb_policy_rate(
        start_period=start_period,
    )

    frames.append(
        ecb_df
    )

    # -------------------------------------------------
    # Japan - BOJ call rate and PPI
    # -------------------------------------------------

    for (
        variable_name,
        config,
    ) in BOJ_SERIES.items():

        df = fetch_boj_series(
            variable_name,
            config,
            start_period=start_period,
        )

        frames.append(
            df
        )

    # -------------------------------------------------
    # Japan - CPI
    # -------------------------------------------------

    japan_cpi_df = fetch_japan_cpi(
        start_period=start_period,
    )

    frames.append(
        japan_cpi_df
    )

    # -------------------------------------------------
    # Japan - Industrial Production
    # -------------------------------------------------

    japan_ip_df = (
        fetch_japan_industrial_production(
            start_period=start_period,
        )
    )

    frames.append(
        japan_ip_df
    )

    # -------------------------------------------------
    # Indonesia - Monthly Inflation
    # -------------------------------------------------

    indonesia_df = (
        fetch_bps_indonesia_inflation(
            start_period=start_period,
        )
    )

    frames.append(
        indonesia_df
    )

    if not frames:
        raise ValueError(
            "No global macro datasets fetched"
        )

    # -------------------------------------------------
    # DRC - Policy Rate
    # -------------------------------------------------

    drc_policy_rate_df = (
        fetch_drc_policy_rate(
            start_period=start_period,
        )
    )

    frames.append(
        drc_policy_rate_df
    )

    drc_fx_df = fetch_drc_fx(
        start_period=start_period,
    )

    frames.append(
        drc_fx_df
    )

    drc_inflation_df = (
        fetch_drc_inflation(
            start_period=start_period,
        )
    )

    frames.append(
        drc_inflation_df
    )

    # -------------------------------------------------
    # Merge all monthly datasets
    # -------------------------------------------------

    # Start from the first dataframe.
    result = frames[
        0
    ].copy()

    # Outer merge is intentional.
    #
    # Official sources have different
    # publication delays. We keep those
    # missing observations instead of
    # silently filling them.
    for frame in frames[
        1:
    ]:
        result = result.merge(
            frame,
            on="date",
            how="outer",
        )

    # Never include the current incomplete month.
    result = result[
        result[
            "date"
        ]
        <= last_full_month
    ].copy()

    feature_cols = [
        col
        for col
        in result.columns
        if col != "date"
    ]

    # Drop only months where every feature
    # is missing.
    #
    # Do not drop a month just because one
    # source has a publication lag.
    result = result.dropna(
        subset=feature_cols,
        how="all",
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

    # Convert monthly Period values to YYYY-MM
    # before saving the CSV file.
    result["date"] = (
        result[
            "date"
        ]
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
        f"[OK] Saved: "
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
        "\n[INFO] Last 12 rows:"
    )

    print(
        result
        .tail(12)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    build_global_macro()