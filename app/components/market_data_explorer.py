from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawrect",
        "eraseshape",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "copper_market_chart",
        "scale": 2,
    },
}


CATEGORY_RULES: Dict[str, List[str]] = {
    "Copper & Metals": [
        "copper",
        "lme",
        "metal",
        "gold",
        "silver",
        "aluminum",
        "aluminium",
        "zinc",
        "nickel",
        "lead",
        "tin",
        "shfe",
    ],
    "Macro & Rates": [
        "fed",
        "rate",
        "yield",
        "treasury",
        "cpi",
        "inflation",
        "ppi",
        "m2",
        "money",
        "gdp",
        "gfcf",
        "pmi",
        "industrial",
        "cli",
        "unemployment",
    ],
    "China": [
        "china",
        "csi",
        "shanghai",
    ],
    "FX & Financial Markets": [
        "dxy",
        "dollar",
        "eurusd",
        "usd",
        "fx",
        "vix",
        "equity",
        "stock",
        "index",
        "ipsa",
        "wig20",
        "csi300",
    ],
    "Energy": [
        "oil",
        "brent",
        "wti",
        "gas",
        "energy",
        "electricity",
        "coal",
    ],
    "Shipping & Logistics": [
        "baltic",
        "bdi",
        "shipping",
        "freight",
        "port",
    ],
    "Risk & Policy": [
        "risk",
        "gpr",
        "geopolitical",
        "uncertainty",
        "policy",
        "epu",
    ],
    "Copper Supply": [
        "mine",
        "mining",
        "production",
        "reserve",
        "inventory",
        "stock",
        "icsg",
        "cochilco",
        "peru",
        "chile",
    ],
    "Demand & Energy Transition": [
        "ev",
        "vehicle",
        "renewable",
        "solar",
        "wind",
        "transition",
        "demand",
    ],
}


def infer_unit(column_name: str) -> str:
    name = column_name.lower()

    if (
        "usd_per_ton" in name
        or "usd_per_tonne" in name
        or "usd_ton" in name
    ):
        return "USD / ton"

    if (
        "pct" in name
        or "percent" in name
        or "percentage" in name
    ):
        return "%"

    if "yield" in name:
        return "%"

    if "rate" in name:
        return "% / rate"

    if "index" in name:
        return "Index"

    if (
        "ton" in name
        or "tonne" in name
    ):
        return "Ton"

    if (
        "count" in name
        or "sales" in name
        or "employment" in name
    ):
        return "Count"

    if "usd" in name:
        return "USD"

    return "Value"


def pretty_name(column_name: str) -> str:
    return (
        column_name
        .replace("_", " ")
        .strip()
        .title()
    )


def get_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "date",
        "month",
        "period",
        "timestamp",
    ]

    lowered = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    raise ValueError(
        "No date-like column found in monthly master."
    )


def get_numeric_columns(
    df: pd.DataFrame,
    date_column: str,
) -> List[str]:
    result = []

    for column in df.columns:
        if column == date_column:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.notna().sum() >= 3:
            result.append(
                column
            )

    return sorted(
        result
    )


def build_categories(
    numeric_columns: List[str],
) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "All Numeric Variables": numeric_columns,
    }

    for (
        category,
        keywords,
    ) in CATEGORY_RULES.items():

        matched = []

        for column in numeric_columns:
            lowered = column.lower()

            if any(
                keyword in lowered
                for keyword in keywords
            ):
                matched.append(
                    column
                )

        if matched:
            categories[
                category
            ] = sorted(
                set(matched)
            )

    uncategorized = [
        column
        for column in numeric_columns
        if not any(
            column in values
            for key, values
            in categories.items()
            if key != "All Numeric Variables"
        )
    ]

    if uncategorized:
        categories[
            "Other Variables"
        ] = uncategorized

    return categories


def calculate_change(
    series: pd.Series,
    periods: int,
) -> float:
    clean = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if len(clean) <= periods:
        return np.nan

    previous = clean.iloc[
        -(periods + 1)
    ]

    latest = clean.iloc[-1]

    if previous == 0:
        return np.nan

    return (
        (
            latest
            / previous
        )
        - 1
    ) * 100


def filter_period(
    df: pd.DataFrame,
    date_column: str,
    selection: str,
) -> pd.DataFrame:
    if selection == "All":
        return df.copy()

    years = {
        "1Y": 1,
        "3Y": 3,
        "5Y": 5,
        "10Y": 10,
    }[
        selection
    ]

    latest_date = df[
        date_column
    ].max()

    cutoff = (
        latest_date
        - pd.DateOffset(
            years=years
        )
    )

    return df.loc[
        df[
            date_column
        ] >= cutoff
    ].copy()


def render_market_data_explorer(
    master_df: pd.DataFrame,
) -> None:

    st.markdown(
        "## Market Data Explorer"
    )

    st.caption(
        "Explore production master data interactively. "
        "Select a category and variable, inspect exact values, "
        "zoom into periods, review statistics, and export data."
    )

    if master_df.empty:
        st.error(
            "Monthly master dataset is empty."
        )
        return

    df = master_df.copy()

    date_column = get_date_column(
        df
    )

    df[
        date_column
    ] = pd.to_datetime(
        df[
            date_column
        ],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                date_column
            ]
        )
        .sort_values(
            date_column
        )
        .reset_index(
            drop=True
        )
    )

    numeric_columns = get_numeric_columns(
        df,
        date_column,
    )

    categories = build_categories(
        numeric_columns
    )

    control1, control2, control3 = st.columns(
        [1.2, 2.1, 0.8]
    )

    with control1:
        category = st.selectbox(
            "Category",
            list(
                categories.keys()
            ),
        )

    with control2:
        variables = categories[
            category
        ]

        selected_variable = st.selectbox(
            "Variable",
            variables,
            format_func=pretty_name,
        )

    with control3:
        period = st.selectbox(
            "Period",
            [
                "1Y",
                "3Y",
                "5Y",
                "10Y",
                "All",
            ],
            index=2,
        )

    df[
        selected_variable
    ] = pd.to_numeric(
        df[
            selected_variable
        ],
        errors="coerce",
    )

    filtered_df = filter_period(
        df,
        date_column,
        period,
    )

    series = filtered_df[
        selected_variable
    ]

    clean_series = (
        series.dropna()
    )

    if clean_series.empty:
        st.warning(
            "Selected variable has no numeric observations "
            "for the selected period."
        )
        return

    latest_value = clean_series.iloc[-1]

    unit = infer_unit(
        selected_variable
    )

    one_month_change = calculate_change(
        series,
        1,
    )

    three_month_change = calculate_change(
        series,
        3,
    )

    twelve_month_change = calculate_change(
        series,
        12,
    )

    missing_pct = (
        series.isna().mean()
        * 100
    )

    k1, k2, k3, k4, k5, k6 = st.columns(
        6
    )

    k1.metric(
        "Latest",
        f"{latest_value:,.2f}",
        unit,
    )

    k2.metric(
        "1M Change",
        (
            f"{one_month_change:+.2f}%"
            if pd.notna(
                one_month_change
            )
            else "N/A"
        ),
    )

    k3.metric(
        "3M Change",
        (
            f"{three_month_change:+.2f}%"
            if pd.notna(
                three_month_change
            )
            else "N/A"
        ),
    )

    k4.metric(
        "12M Change",
        (
            f"{twelve_month_change:+.2f}%"
            if pd.notna(
                twelve_month_change
            )
            else "N/A"
        ),
    )

    k5.metric(
        "Observations",
        f"{clean_series.shape[0]:,}",
    )

    k6.metric(
        "Missing",
        f"{missing_pct:.1f}%",
    )

    st.divider()

    left, right = st.columns(
        [3.2, 1]
    )

    with left:
        st.markdown(
            f"### {pretty_name(selected_variable)}"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=filtered_df[
                    date_column
                ],
                y=filtered_df[
                    selected_variable
                ],
                mode="lines+markers",
                name=pretty_name(
                    selected_variable
                ),
                connectgaps=False,
                line=dict(
                    width=2.5,
                ),
                marker=dict(
                    size=6,
                ),
                customdata=np.column_stack(
                    [
                        filtered_df[
                            selected_variable
                        ],
                    ]
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b>"
                    "<br>"
                    "Value: %{y:,.4f}"
                    f"<br>Unit: {unit}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=560,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
            hovermode="x unified",
            dragmode="zoom",
            xaxis_title="Month",
            yaxis_title=unit,
            legend=dict(
                orientation="h",
                y=1.04,
            ),
        )

        fig.update_xaxes(
            showspikes=True,
            spikecolor="gray",
            spikethickness=1,
            spikedash="dot",
            spikesnap="cursor",
            rangeslider=dict(
                visible=True,
                thickness=0.08,
            ),
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        count=3,
                        label="3Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        count=5,
                        label="5Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        count=10,
                        label="10Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        label="ALL",
                        step="all",
                    ),
                ],
            ),
        )

        fig.update_yaxes(
            showspikes=True,
            spikecolor="gray",
            spikethickness=1,
            spikedash="dot",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        st.markdown(
            "### Statistics"
        )

        st.metric(
            "Mean",
            f"{clean_series.mean():,.2f}",
        )

        st.metric(
            "Median",
            f"{clean_series.median():,.2f}",
        )

        st.metric(
            "Minimum",
            f"{clean_series.min():,.2f}",
        )

        st.metric(
            "Maximum",
            f"{clean_series.max():,.2f}",
        )

        st.metric(
            "Std. Deviation",
            f"{clean_series.std():,.2f}",
        )

        latest_valid_index = (
            filtered_df[
                selected_variable
            ]
            .last_valid_index()
        )

        if latest_valid_index is not None:
            latest_date = filtered_df.loc[
                latest_valid_index,
                date_column,
            ]

            st.metric(
                "Latest Data Month",
                latest_date.strftime(
                    "%b %Y"
                ),
            )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Raw Data",
            "Descriptive Statistics",
            "Missing Data",
        ]
    )

    with tab1:
        raw_table = filtered_df[
            [
                date_column,
                selected_variable,
            ]
        ].copy()

        raw_table.columns = [
            "Date",
            pretty_name(
                selected_variable
            ),
        ]

        st.dataframe(
            raw_table,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        csv_data = raw_table.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            label="Download Selected Series",
            data=csv_data,
            file_name=(
                f"{selected_variable}.csv"
            ),
            mime="text/csv",
        )

    with tab2:
        stats = (
            clean_series
            .describe()
            .to_frame(
                name="Value"
            )
        )

        stats.loc[
            "missing_count"
        ] = series.isna().sum()

        stats.loc[
            "missing_pct"
        ] = missing_pct

        st.dataframe(
            stats,
            use_container_width=True,
        )

    with tab3:
        missing_table = pd.DataFrame(
            {
                "Metric": [
                    "Total rows",
                    "Valid observations",
                    "Missing observations",
                    "Missing %",
                    "First valid month",
                    "Last valid month",
                ],
                "Value": [
                    len(series),
                    series.notna().sum(),
                    series.isna().sum(),
                    f"{missing_pct:.2f}%",
                    (
                        filtered_df.loc[
                            series.first_valid_index(),
                            date_column,
                        ].strftime(
                            "%Y-%m-%d"
                        )
                        if series.first_valid_index()
                        is not None
                        else "N/A"
                    ),
                    (
                        filtered_df.loc[
                            series.last_valid_index(),
                            date_column,
                        ].strftime(
                            "%Y-%m-%d"
                        )
                        if series.last_valid_index()
                        is not None
                        else "N/A"
                    ),
                ],
            }
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True,
        )
