from pathlib import Path
import json
import requests
import pandas as pd
from src.utils.paths import SHIPPING_RAW_DIR
OUTPUT_FILE = (
    SHIPPING_RAW_DIR
    / "portwatch_shipping_activity_daily.csv"
)
API_URL = (
    "https://services9.arcgis.com/"
    "weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)


START_DATE = pd.Timestamp(
    "2019-01-01"
)


SUPPLY_PORTS = [
    "port728",   # Mejillones
    "port56",    # Antofagasta
    "port90",    # Ventanas
    "port88",    # Matarani
    "port1045",  # Callao
    "port1047",  # Ilo
    "port91",    # San Nicolas
]


CHINA_PORTS = [
    "port1188",  # Shanghai
    "port824",   # Ningbo
    "port1069",  # Qingdao Port
    "port1297",  # Tianjin Xin Gang
    "port339",   # Fangcheng
]


GLOBAL_STATISTICS = [
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls",
        "outStatisticFieldName": "global_portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls_container",
        "outStatisticFieldName": "global_container_portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls_dry_bulk",
        "outStatisticFieldName": "global_dry_bulk_portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls_cargo",
        "outStatisticFieldName": "global_cargo_portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "import",
        "outStatisticFieldName": "global_import",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "export",
        "outStatisticFieldName": "global_export",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "import_dry_bulk",
        "outStatisticFieldName": "global_dry_bulk_import",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "export_dry_bulk",
        "outStatisticFieldName": "global_dry_bulk_export",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "import_container",
        "outStatisticFieldName": "global_container_import",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "export_container",
        "outStatisticFieldName": "global_container_export",
    },
]


BASKET_STATISTICS = [
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls",
        "outStatisticFieldName": "portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "portcalls_dry_bulk",
        "outStatisticFieldName": "dry_bulk_portcalls",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "import_dry_bulk",
        "outStatisticFieldName": "dry_bulk_import",
    },
    {
        "statisticType": "sum",
        "onStatisticField": "export_dry_bulk",
        "outStatisticFieldName": "dry_bulk_export",
    },
]


def request_json(params):
    response = requests.get(
        API_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    payload = response.json()

    if "error" in payload:
        raise RuntimeError(
            payload["error"]
        )

    return payload


def get_latest_date():
    params = {
        "where": "1=1",
        "outFields": (
            "date,"
            "year,"
            "month,"
            "day"
        ),
        "returnGeometry": "false",
        "orderByFields": (
            "year DESC,"
            "month DESC,"
            "day DESC"
        ),
        "resultRecordCount": 1,
        "f": "json",
    }

    payload = request_json(
        params
    )

    features = payload.get(
        "features",
        []
    )

    if not features:
        raise RuntimeError(
            "No PortWatch records found"
        )

    attributes = features[0].get(
        "attributes",
        {}
    )

    year = attributes.get(
        "year"
    )

    month = attributes.get(
        "month"
    )

    day = attributes.get(
        "day"
    )

    if (
        year is None
        or month is None
        or day is None
    ):
        raise RuntimeError(
            "Latest PortWatch date fields are missing"
        )

    latest_date = pd.Timestamp(
        year=int(year),
        month=int(month),
        day=int(day),
    )

    return latest_date


def make_port_where(port_ids):
    values = ",".join(
        f"'{port_id}'"
        for port_id in port_ids
    )

    return (
        f"portid IN ({values})"
    )


def fetch_period(
    start_date,
    end_date,
    statistics,
    prefix=None,
    extra_where="1=1",
):
    where = (
        f"date >= DATE '{start_date:%Y-%m-%d}' "
        f"AND date <= DATE '{end_date:%Y-%m-%d}' "
        f"AND ({extra_where})"
    )

    params = {
        "where": where,
        "outStatistics": json.dumps(
            statistics
        ),
        "groupByFieldsForStatistics": "date",
        "orderByFields": "date ASC",
        "returnGeometry": "false",
        "f": "json",
    }

    payload = request_json(
        params
    )

    rows = [
        feature.get(
            "attributes",
            {}
        )
        for feature in payload.get(
            "features",
            []
        )
    ]

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if prefix is not None:
        rename_map = {
            column: f"{prefix}_{column}"
            for column in df.columns
            if column != "date"
        }

        df = df.rename(
            columns=rename_map
        )

    return df


def fetch_group_history(
    statistics,
    latest_date,
    prefix=None,
    extra_where="1=1",
):
    frames = []

    for year in range(
        START_DATE.year,
        latest_date.year + 1,
    ):
        year_start = pd.Timestamp(
            year=year,
            month=1,
            day=1,
        )

        year_end = pd.Timestamp(
            year=year,
            month=12,
            day=31,
        )

        period_start = max(
            START_DATE,
            year_start,
        )

        period_end = min(
            latest_date,
            year_end,
        )

        if period_start > period_end:
            continue

        print(
            "[INFO] Fetching:",
            prefix or "global",
            period_start.date(),
            "->",
            period_end.date(),
        )

        df = fetch_period(
            start_date=period_start,
            end_date=period_end,
            statistics=statistics,
            prefix=prefix,
            extra_where=extra_where,
        )

        if not df.empty:
            frames.append(
                df
            )

    if not frames:
        raise RuntimeError(
            f"No data returned for {prefix or 'global'}"
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
        .drop_duplicates(
            subset=["date"]
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    return result


def main():
    print(
        "[INFO] Source: IMF PortWatch"
    )

    latest_date = get_latest_date()

    print(
        "[INFO] Latest source date:",
        latest_date.date(),
    )

    print(
        "[INFO] Fetching global activity"
    )

    global_df = fetch_group_history(
        statistics=GLOBAL_STATISTICS,
        latest_date=latest_date,
    )

    print(
        "[INFO] Fetching copper supply port basket"
    )

    supply_df = fetch_group_history(
        statistics=BASKET_STATISTICS,
        latest_date=latest_date,
        prefix="copper_supply",
        extra_where=make_port_where(
            SUPPLY_PORTS
        ),
    )

    print(
        "[INFO] Fetching China copper-related port basket"
    )

    china_df = fetch_group_history(
        statistics=BASKET_STATISTICS,
        latest_date=latest_date,
        prefix="china_copper_related",
        extra_where=make_port_where(
            CHINA_PORTS
        ),
    )

    df = global_df.merge(
        supply_df,
        on="date",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        china_df,
        on="date",
        how="left",
        validate="one_to_one",
    )

    df = (
        df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    if df["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates detected"
        )

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError(
            "Dates are not sorted"
        )

    numeric_columns = [
        column
        for column in df.columns
        if column != "date"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
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
        len(df),
    )

    print(
        "[INFO] Columns:",
        len(df.columns),
    )

    print(
        "[INFO] Date range:",
        df["date"].min().date(),
        "->",
        df["date"].max().date(),
    )

    print()

    print(
        df.tail(
            10
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
