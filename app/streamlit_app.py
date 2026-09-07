from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "copper_production_forecast_latest.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "copper_production_metrics_latest.csv"
)

MODEL_DETAILS_FILE = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "copper_production_model_details_latest.csv"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly"
    / "copper_monthly_master.csv"
)


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="Copper Intelligence & Forecasting Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================

CUSTOM_CSS = """
<style>

html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif;
}

.block-container {
    padding-top: 3.1rem;
    padding-bottom: 3rem;
    max-width: 1850px;
}

[data-testid="stSidebar"] {
    min-width: 300px;
    max-width: 300px;
}

[data-testid="stSidebarContent"] {
    padding-top: 2rem;
}

.main-title {
    font-size: 2.55rem;
    font-weight: 850;
    letter-spacing: -0.035em;
    line-height: 1.25;
    margin-top: 0.25rem;
    margin-bottom: 0.15rem;
    padding-top: 0.2rem;
    padding-bottom: 0.1rem;
    overflow: visible;
}

.main-subtitle {
    font-size: 1rem;
    opacity: 0.66;
    margin-bottom: 1.6rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 760;
    margin-bottom: 0.75rem;
}

.kpi-card {
    border: 1px solid rgba(120, 120, 120, 0.20);
    border-radius: 18px;
    padding: 17px 19px 15px 19px;
    background:
        linear-gradient(
            145deg,
            rgba(120, 120, 120, 0.035),
            rgba(120, 120, 120, 0.010)
        );
    min-height: 125px;
}

.kpi-label {
    opacity: 0.68;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2rem;
    line-height: 1.05;
    font-weight: 820;
    letter-spacing: -0.035em;
}

.kpi-unit {
    font-size: 0.78rem;
    opacity: 0.58;
    margin-top: 7px;
}

.kpi-positive {
    color: #16a34a;
    font-size: 0.84rem;
    font-weight: 650;
    margin-top: 8px;
}

.kpi-negative {
    color: #dc2626;
    font-size: 0.84rem;
    font-weight: 650;
    margin-top: 8px;
}

.kpi-neutral {
    opacity: 0.72;
    font-size: 0.84rem;
    font-weight: 600;
    margin-top: 8px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(120, 120, 120, 0.18);
    border-radius: 16px;
    padding: 15px 17px;
    background: rgba(120, 120, 120, 0.025);
}

div[data-testid="stMetricValue"] {
    font-weight: 760;
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding-left: 18px;
    padding-right: 18px;
}

hr {
    margin-top: 1.4rem;
    margin-bottom: 1.4rem;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# =============================================================================
# PLOTLY CONFIG
# =============================================================================

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
        "filename": "copper_intelligence_chart",
        "scale": 2,
    },
}


# =============================================================================
# CATEGORY RULES
# =============================================================================

CATEGORY_RULES: Dict[str, List[str]] = {
    "Copper Price & Inventory": [
        "cash_settlement",
        "copper_stock",
        "copper",
        "lme",
    ],
    "Other Metals": [
        "aluminum",
        "aluminium",
        "lead",
        "nickel",
        "tin",
        "zinc",
        "gold",
        "silver",
        "metal",
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
        "fx",
        "vix",
        "equity",
        "stock",
        "index",
        "ipsa",
        "wig20",
        "csi300",
        "sp500",
        "tsx",
    ],
    "Energy": [
        "oil",
        "brent",
        "wti",
        "gas",
        "energy",
        "electricity",
        "coal",
        "fuel",
    ],
    "Shipping & Logistics": [
        "baltic",
        "bdi",
        "shipping",
        "freight",
        "port",
        "dry_bulk",
        "portcalls",
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
        "icsg",
        "cochilco",
        "peru",
        "chile",
        "ore_grade",
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


# =============================================================================
# DATA LOADERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def prepare_forecast(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()

    for column in [
        "forecast_date",
        "forecast_origin",
        "generated_at",
    ]:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            )

    if "horizon" in result.columns:
        result = result.sort_values(
            "horizon"
        )

    return result.reset_index(
        drop=True
    )


forecast_df = prepare_forecast(
    load_csv(
        FORECAST_FILE
    )
)

metrics_df = load_csv(
    METRICS_FILE
)

model_details_df = load_csv(
    MODEL_DETAILS_FILE
)

master_df = load_csv(
    MASTER_FILE
)


# =============================================================================
# HELPERS
# =============================================================================

def format_price(
    value: float,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def pretty_name(
    column_name: str,
) -> str:
    custom_names = {
        "cash_settlement_usd_per_ton":
            "LME Copper Cash Settlement",
        "copper_stock_ton":
            "LME Copper Stock",
        "china_refined_copper_production_ton":
            "China Refined Copper Production",
        "world_copper_mine_production_ton":
            "World Copper Mine Production",
        "world_copper_refinery_production_ton":
            "World Copper Refinery Production",
        "world_copper_reserves_ton":
            "World Copper Reserves",
        "chile_copper_production_ton":
            "Chile Copper Production",
        "peru_copper_production_ton":
            "Peru Copper Production",
    }

    if column_name in custom_names:
        return custom_names[
            column_name
        ]

    return (
        str(column_name)
        .replace("_", " ")
        .strip()
        .title()
    )


def infer_unit(
    column_name: str,
) -> str:
    name = str(
        column_name
    ).lower()

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

    if "tj" in name:
        return "TJ"

    if "usd" in name:
        return "USD"

    if (
        "count" in name
        or "employment" in name
        or "sales" in name
        or "portcalls" in name
    ):
        return "Count"

    return "Value"


def find_date_column(
    df: pd.DataFrame,
) -> str | None:
    candidates = [
        "date",
        "month",
        "period",
        "timestamp",
    ]

    lowered = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lowered:
            return lowered[
                candidate
            ]

    return None


def find_copper_target_column(
    df: pd.DataFrame,
) -> str | None:
    preferred = [
        "cash_settlement_usd_per_ton",
        "lme_cash_monthly_usd",
        "copper_usd_per_ton",
        "lme_cash_usd_per_ton",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    for column in df.columns:
        name = str(
            column
        ).lower()

        if (
            "cash" in name
            and "settlement" in name
            and "usd" in name
            and "ton" in name
        ):
            return column

    return None


def get_numeric_columns(
    df: pd.DataFrame,
    date_column: str | None,
) -> List[str]:
    columns = []

    for column in df.columns:
        if column == date_column:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric.notna().sum() >= 3:
            columns.append(
                column
            )

    return sorted(
        columns
    )


def build_categories(
    numeric_columns: List[str],
) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "All Numeric Variables":
            numeric_columns,
    }

    assigned = set()

    for category, keywords in CATEGORY_RULES.items():
        matches = []

        for column in numeric_columns:
            lowered = str(
                column
            ).lower()

            if any(
                keyword in lowered
                for keyword in keywords
            ):
                matches.append(
                    column
                )

                assigned.add(
                    column
                )

        if matches:
            categories[
                category
            ] = sorted(
                set(matches)
            )

    remaining = [
        column
        for column in numeric_columns
        if column not in assigned
    ]

    if remaining:
        categories[
            "Other Variables"
        ] = remaining

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
    period: str,
) -> pd.DataFrame:
    if period == "All":
        return df.copy()

    year_map = {
        "1Y": 1,
        "3Y": 3,
        "5Y": 5,
        "10Y": 10,
    }

    years = year_map[
        period
    ]

    latest = df[
        date_column
    ].max()

    cutoff = (
        latest
        - pd.DateOffset(
            years=years
        )
    )

    return df.loc[
        df[
            date_column
        ] >= cutoff
    ].copy()


def render_card(
    label: str,
    value: str,
    unit: str = "",
    delta: str | None = None,
) -> None:
    delta_html = ""

    if delta:
        try:
            numeric_delta = float(
                str(delta)
                .replace("%", "")
                .replace("+", "")
            )

            if numeric_delta > 0:
                css_class = (
                    "kpi-positive"
                )
            elif numeric_delta < 0:
                css_class = (
                    "kpi-negative"
                )
            else:
                css_class = (
                    "kpi-neutral"
                )

        except Exception:
            css_class = (
                "kpi-neutral"
            )

        delta_html = (
            f'<div class="{css_class}">'
            f'{delta}'
            f'</div>'
        )

    unit_html = (
        f'<div class="kpi-unit">'
        f'{unit}'
        f'</div>'
        if unit
        else ""
    )

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                {label}
            </div>
            <div class="kpi-value">
                {value}
            </div>
            {unit_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot(
    fig: go.Figure,
) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def configure_time_axis(
    fig: go.Figure,
    show_range_slider: bool = True,
) -> None:
    fig.update_xaxes(
        showspikes=True,
        spikethickness=1,
        spikedash="dot",
        spikesnap="cursor",
        rangeslider=dict(
            visible=show_range_slider,
            thickness=0.07,
        ),
        rangeselector=dict(
            x=0,
            y=1.03,
            xanchor="left",
            yanchor="bottom",
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
        spikethickness=1,
        spikedash="dot",
    )


def configure_top_legend(
    fig: go.Figure,
) -> None:
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=1,
            y=1.16,
            xanchor="right",
            yanchor="bottom",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        )
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown(
    "## Copper Intelligence"
)

st.sidebar.caption(
    "Production ML Forecasting Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Forecast Center",
        "Market Data Explorer",
    ],
)

st.sidebar.divider()

if not forecast_df.empty:

    origin = forecast_df[
        "forecast_origin"
    ].iloc[0]

    generated_at = forecast_df[
        "generated_at"
    ].iloc[0]

    st.sidebar.markdown(
        "### Production Status"
    )

    st.sidebar.success(
        "Latest forecast available"
    )

    if pd.notna(
        origin
    ):
        st.sidebar.caption(
            "Forecast origin"
        )

        st.sidebar.code(
            origin.strftime(
                "%Y-%m-%d"
            )
        )

    if pd.notna(
        generated_at
    ):
        st.sidebar.caption(
            "Generated at"
        )

        st.sidebar.code(
            generated_at.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    st.sidebar.caption(
        "Policy"
    )

    st.sidebar.code(
        str(
            forecast_df[
                "production_policy_version"
            ].iloc[0]
        )
    )

else:
    st.sidebar.error(
        "Forecast unavailable"
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">'
    'Copper Intelligence & Forecasting Center'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'LME Copper | Machine Learning Production Forecast | '
    'Market Intelligence | Data Analytics'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# GLOBAL FORECAST VALIDATION
# =============================================================================

if forecast_df.empty:
    st.error(
        "Production forecast file could not be loaded."
    )

    st.stop()


origin_price = float(
    forecast_df[
        "origin_price_usd_per_ton"
    ].iloc[0]
)


def horizon_row(
    horizon: int,
) -> pd.Series:
    result = forecast_df.loc[
        forecast_df[
            "horizon"
        ] == horizon
    ]

    if result.empty:
        raise ValueError(
            f"Horizon H{horizon} is missing."
        )

    return result.iloc[0]


h1 = horizon_row(
    1
)

h3 = horizon_row(
    3
)

h6 = horizon_row(
    6
)

h12 = horizon_row(
    12
)


# =============================================================================
# EXECUTIVE OVERVIEW
# =============================================================================

if page == "Executive Overview":

    cards = st.columns(
        5
    )

    with cards[0]:
        render_card(
            "Latest Copper",
            format_price(
                origin_price
            ),
            "USD / ton",
        )

    with cards[1]:
        render_card(
            "H1 Forecast",
            format_price(
                h1[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            "USD / ton",
            f"{h1['change_pct_from_origin']:+.2f}%",
        )

    with cards[2]:
        render_card(
            "H3 Forecast",
            format_price(
                h3[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            "USD / ton",
            f"{h3['change_pct_from_origin']:+.2f}%",
        )

    with cards[3]:
        render_card(
            "H6 Forecast",
            format_price(
                h6[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            "USD / ton",
            f"{h6['change_pct_from_origin']:+.2f}%",
        )

    with cards[4]:
        render_card(
            "H12 Forecast",
            format_price(
                h12[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            "USD / ton",
            f"{h12['change_pct_from_origin']:+.2f}%",
        )

    st.divider()

    left, right = st.columns(
        [3.15, 1]
    )

    with left:

        st.markdown(
            '<div class="section-title">'
            'Historical Copper + Production Forecast'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        date_column = find_date_column(
            master_df
        )

        target_column = (
            find_copper_target_column(
                master_df
            )
        )

        if (
            date_column
            and target_column
        ):

            historical = master_df[
                [
                    date_column,
                    target_column,
                ]
            ].copy()

            historical[
                date_column
            ] = pd.to_datetime(
                historical[
                    date_column
                ],
                errors="coerce",
            )

            historical[
                target_column
            ] = pd.to_numeric(
                historical[
                    target_column
                ],
                errors="coerce",
            )

            historical = (
                historical
                .dropna()
                .sort_values(
                    date_column
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=historical[
                        date_column
                    ],
                    y=historical[
                        target_column
                    ],
                    mode="lines",
                    name="Historical LME Copper",
                    line=dict(
                        width=2.2,
                    ),
                    hovertemplate=(
                        "<b>%{x|%b %Y}</b>"
                        "<br>"
                        "Actual: $%{y:,.2f}/t"
                        "<extra></extra>"
                    ),
                )
            )

        forecast_chart = forecast_df[
            [
                "forecast_date",
                "final_forecast_price_usd_per_ton",
                "horizon",
                "production_expert",
                "confidence_flag",
            ]
        ].copy()

        origin_point = pd.DataFrame(
            {
                "forecast_date": [
                    forecast_df[
                        "forecast_origin"
                    ].iloc[0]
                ],
                "final_forecast_price_usd_per_ton": [
                    origin_price
                ],
                "horizon": [
                    0
                ],
                "production_expert": [
                    "Forecast Origin"
                ],
                "confidence_flag": [
                    "ORIGIN"
                ],
            }
        )

        forecast_chart = pd.concat(
            [
                origin_point,
                forecast_chart,
            ],
            ignore_index=True,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_chart[
                    "forecast_date"
                ],
                y=forecast_chart[
                    "final_forecast_price_usd_per_ton"
                ],
                mode="lines+markers",
                name="Production Forecast",
                line=dict(
                    width=3,
                    dash="dash",
                ),
                marker=dict(
                    size=8,
                ),
                customdata=np.column_stack(
                    [
                        forecast_chart[
                            "horizon"
                        ],
                        forecast_chart[
                            "production_expert"
                        ],
                        forecast_chart[
                            "confidence_flag"
                        ],
                    ]
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b>"
                    "<br>"
                    "Price: $%{y:,.2f}/t"
                    "<br>"
                    "Horizon: H%{customdata[0]}"
                    "<br>"
                    "Expert: %{customdata[1]}"
                    "<br>"
                    "Confidence: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_vline(
            x=forecast_df[
                "forecast_origin"
            ].iloc[0],
            line_dash="dot",
            line_width=1.5,
        )

        fig.update_layout(
            height=650,
            margin=dict(
                l=20,
                r=20,
                t=115,
                b=20,
            ),
            hovermode="x unified",
            dragmode="zoom",
            xaxis_title="Month",
            yaxis_title="USD / ton",
        )

        configure_time_axis(
            fig,
            show_range_slider=True,
        )

        configure_top_legend(
            fig
        )

        render_plot(
            fig
        )

    with right:

        st.markdown(
            '<div class="section-title">'
            'Forecast Intelligence'
            '</div>',
            unsafe_allow_html=True,
        )

        max_row = forecast_df.loc[
            forecast_df[
                "final_forecast_price_usd_per_ton"
            ].idxmax()
        ]

        min_row = forecast_df.loc[
            forecast_df[
                "final_forecast_price_usd_per_ton"
            ].idxmin()
        ]

        st.metric(
            "Forecast Peak",
            format_price(
                max_row[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            f"H{int(max_row['horizon'])}",
        )

        st.metric(
            "Forecast Trough",
            format_price(
                min_row[
                    "final_forecast_price_usd_per_ton"
                ]
            ),
            f"H{int(min_row['horizon'])}",
        )

        st.metric(
            "H1 Backtest MAPE",
            f"{h1['backtest_mape_pct']:.2f}%",
        )

        st.metric(
            "H1 Direction Accuracy",
            (
                f"{h1['backtest_directional_accuracy_pct']:.1f}%"
            ),
        )

        low_horizons = (
            forecast_df.loc[
                forecast_df[
                    "confidence_flag"
                ]
                .astype(str)
                .str.upper()
                .eq("LOW"),
                "horizon",
            ]
            .astype(int)
            .tolist()
        )

        if low_horizons:
            text = ", ".join(
                f"H{horizon}"
                for horizon in low_horizons
            )

            st.warning(
                f"Low confidence: {text}"
            )

        else:
            st.success(
                "All horizons normal"
            )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Monthly Forecast Movement'
        '</div>',
        unsafe_allow_html=True,
    )

    movement_fig = go.Figure()

    movement_fig.add_trace(
        go.Bar(
            x=forecast_df[
                "forecast_date"
            ],
            y=forecast_df[
                "month_over_month_change_pct"
            ],
            customdata=np.column_stack(
                [
                    forecast_df[
                        "horizon"
                    ],
                    forecast_df[
                        "final_forecast_price_usd_per_ton"
                    ],
                    forecast_df[
                        "production_expert"
                    ],
                ]
            ),
            hovertemplate=(
                "<b>%{x|%b %Y}</b>"
                "<br>"
                "Horizon: H%{customdata[0]}"
                "<br>"
                "Forecast: $%{customdata[1]:,.2f}/t"
                "<br>"
                "MoM: %{y:+.2f}%"
                "<br>"
                "Expert: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    movement_fig.add_hline(
        y=0,
        line_width=1,
    )

    movement_fig.update_layout(
        height=380,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
        xaxis_title="Forecast Month",
        yaxis_title="Month-over-month change %",
        dragmode="zoom",
    )

    render_plot(
        movement_fig
    )


# =============================================================================
# FORECAST CENTER
# =============================================================================

elif page == "Forecast Center":

    st.markdown(
        '<div class="section-title">'
        'Forecast Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Explore the current H1-H12 production forecast, "
        "model diagnostics, errors, confidence, and raw forecast data."
    )

    top1, top2, top3, top4 = st.columns(
        4
    )

    top1.metric(
        "Forecast Origin",
        forecast_df[
            "forecast_origin"
        ].iloc[0].strftime(
            "%b %Y"
        ),
    )

    top2.metric(
        "Origin Price",
        f"${origin_price:,.2f}",
    )

    top3.metric(
        "Best MAPE",
        (
            f"{forecast_df['backtest_mape_pct'].min():.2f}%"
        ),
    )

    top4.metric(
        "Forecast Horizons",
        f"{len(forecast_df)}",
    )

    tab_curve, tab_table, tab_diag = st.tabs(
        [
            "Price Curve",
            "Forecast Table",
            "Model Diagnostics",
        ]
    )

    with tab_curve:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=forecast_df[
                    "forecast_date"
                ],
                y=forecast_df[
                    "final_forecast_price_usd_per_ton"
                ],
                mode="lines+markers",
                name="Final Forecast",
                line=dict(
                    width=3,
                ),
                marker=dict(
                    size=9,
                ),
                customdata=np.column_stack(
                    [
                        forecast_df[
                            "horizon"
                        ],
                        forecast_df[
                            "production_expert"
                        ],
                        forecast_df[
                            "month_over_month_change_pct"
                        ],
                        forecast_df[
                            "change_pct_from_origin"
                        ],
                        forecast_df[
                            "backtest_mape_pct"
                        ],
                        forecast_df[
                            "confidence_flag"
                        ],
                    ]
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b>"
                    "<br>"
                    "Horizon: H%{customdata[0]}"
                    "<br>"
                    "Forecast: $%{y:,.2f}/t"
                    "<br>"
                    "MoM: %{customdata[2]:+.2f}%"
                    "<br>"
                    "vs Origin: %{customdata[3]:+.2f}%"
                    "<br>"
                    "Model: %{customdata[1]}"
                    "<br>"
                    "MAPE: %{customdata[4]:.2f}%"
                    "<br>"
                    "Confidence: %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_hline(
            y=origin_price,
            line_dash="dot",
            annotation_text="Origin Price",
        )

        fig.update_layout(
            height=590,
            hovermode="x unified",
            dragmode="zoom",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20,
            ),
            xaxis_title="Forecast Month",
            yaxis_title="USD / ton",
        )

        configure_time_axis(
            fig,
            show_range_slider=True,
        )

        configure_top_legend(
            fig
        )

        render_plot(
            fig
        )

    with tab_table:

        forecast_table = forecast_df[
            [
                "horizon",
                "forecast_date",
                "origin_price_usd_per_ton",
                "raw_forecast_price_usd_per_ton",
                "anchor_forecast_price_usd_per_ton",
                "final_forecast_price_usd_per_ton",
                "month_over_month_change_pct",
                "change_pct_from_origin",
                "production_expert",
                "model",
                "training_window_years",
                "backtest_mape_pct",
                "backtest_rmse",
                "backtest_bias",
                "backtest_directional_accuracy_pct",
                "confidence_flag",
            ]
        ].copy()

        forecast_table.columns = [
            "Horizon",
            "Forecast Month",
            "Origin USD/t",
            "Raw Forecast USD/t",
            "Anchor Forecast USD/t",
            "Final Forecast USD/t",
            "MoM %",
            "vs Origin %",
            "Production Expert",
            "Model",
            "Training Window Years",
            "MAPE %",
            "RMSE",
            "Bias",
            "Directional Accuracy %",
            "Confidence",
        ]

        forecast_table[
            "Forecast Month"
        ] = forecast_table[
            "Forecast Month"
        ].dt.strftime(
            "%Y-%m-%d"
        )

        st.dataframe(
            forecast_table.style.format(
                {
                    "Origin USD/t":
                        "{:,.2f}",
                    "Raw Forecast USD/t":
                        "{:,.2f}",
                    "Anchor Forecast USD/t":
                        "{:,.2f}",
                    "Final Forecast USD/t":
                        "{:,.2f}",
                    "MoM %":
                        "{:.2f}",
                    "vs Origin %":
                        "{:.2f}",
                    "MAPE %":
                        "{:.2f}",
                    "RMSE":
                        "{:,.2f}",
                    "Bias":
                        "{:,.2f}",
                    "Directional Accuracy %":
                        "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        st.download_button(
            "Download Latest Forecast CSV",
            forecast_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                "copper_production_forecast_latest.csv"
            ),
            mime="text/csv",
        )

    with tab_diag:

        selected_horizon = st.select_slider(
            "Select Horizon",
            options=forecast_df[
                "horizon"
            ].astype(int).tolist(),
            value=1,
        )

        row = forecast_df.loc[
            forecast_df[
                "horizon"
            ] == selected_horizon
        ].iloc[0]

        d1, d2, d3, d4 = st.columns(
            4
        )

        d1.metric(
            "Model",
            str(
                row[
                    "production_expert"
                ]
            ),
        )

        d2.metric(
            "MAPE",
            f"{row['backtest_mape_pct']:.2f}%",
        )

        d3.metric(
            "RMSE",
            f"{row['backtest_rmse']:,.2f}",
        )

        d4.metric(
            "Directional Accuracy",
            (
                f"{row['backtest_directional_accuracy_pct']:.1f}%"
            ),
        )

        d5, d6, d7, d8 = st.columns(
            4
        )

        d5.metric(
            "Raw Forecast",
            (
                f"${row['raw_forecast_price_usd_per_ton']:,.2f}"
            ),
        )

        d6.metric(
            "Final Forecast",
            (
                f"${row['final_forecast_price_usd_per_ton']:,.2f}"
            ),
        )

        d7.metric(
            "Bias",
            f"{row['backtest_bias']:,.2f}",
        )

        d8.metric(
            "Confidence",
            str(
                row[
                    "confidence_flag"
                ]
            ),
        )

        left, right = st.columns(
            2
        )

        with left:

            mape_fig = go.Figure()

            mape_fig.add_trace(
                go.Bar(
                    x=forecast_df[
                        "horizon"
                    ],
                    y=forecast_df[
                        "backtest_mape_pct"
                    ],
                    hovertemplate=(
                        "H%{x}<br>"
                        "MAPE: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

            mape_fig.update_layout(
                title="MAPE by Horizon",
                height=390,
                xaxis_title="Horizon",
                yaxis_title="MAPE %",
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20,
                ),
            )

            render_plot(
                mape_fig
            )

        with right:

            accuracy_fig = (
                go.Figure()
            )

            accuracy_fig.add_trace(
                go.Bar(
                    x=forecast_df[
                        "horizon"
                    ],
                    y=forecast_df[
                        "backtest_directional_accuracy_pct"
                    ],
                    hovertemplate=(
                        "H%{x}<br>"
                        "Accuracy: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                )
            )

            accuracy_fig.update_layout(
                title=(
                    "Directional Accuracy by Horizon"
                ),
                height=390,
                xaxis_title="Horizon",
                yaxis_title=(
                    "Directional Accuracy %"
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20,
                ),
            )

            render_plot(
                accuracy_fig
            )


# =============================================================================
# MARKET DATA EXPLORER
# =============================================================================

elif page == "Market Data Explorer":

    st.markdown(
        '<div class="section-title">'
        'Market Data Explorer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Analyze any monthly market variable and optionally "
        "compare it directly with the LME copper price."
    )

    if master_df.empty:
        st.error(
            "Monthly master dataset is empty."
        )

        st.stop()

    explorer_df = master_df.copy()

    date_column = find_date_column(
        explorer_df
    )

    copper_column = (
        find_copper_target_column(
            explorer_df
        )
    )

    if date_column is None:
        st.error(
            "No date column could be detected."
        )

        st.stop()

    explorer_df[
        date_column
    ] = pd.to_datetime(
        explorer_df[
            date_column
        ],
        errors="coerce",
    )

    explorer_df = (
        explorer_df
        .dropna(
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

    numeric_columns = (
        get_numeric_columns(
            explorer_df,
            date_column,
        )
    )

    categories = build_categories(
        numeric_columns
    )

    controls = st.columns(
        [1.15, 2.0, 0.75, 1.0]
    )

    with controls[0]:

        category = st.selectbox(
            "Category",
            list(
                categories.keys()
            ),
        )

    with controls[1]:

        selected_variable = (
            st.selectbox(
                "Variable",
                categories[
                    category
                ],
                format_func=pretty_name,
            )
        )

    with controls[2]:

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

    with controls[3]:

        overlay_copper = (
            st.checkbox(
                "Show Copper Overlay",
                value=True,
                disabled=(
                    selected_variable
                    == copper_column
                ),
            )
        )

    comparison_mode = st.radio(
        "Chart Mode",
        [
            "Raw Values",
            "Indexed Comparison (Base = 100)",
        ],
        horizontal=True,
        disabled=(
            not overlay_copper
            or selected_variable
            == copper_column
        ),
    )

    explorer_df[
        selected_variable
    ] = pd.to_numeric(
        explorer_df[
            selected_variable
        ],
        errors="coerce",
    )

    if (
        copper_column
        and copper_column
        in explorer_df.columns
    ):
        explorer_df[
            copper_column
        ] = pd.to_numeric(
            explorer_df[
                copper_column
            ],
            errors="coerce",
        )

    filtered_df = filter_period(
        explorer_df,
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
            "Selected variable has no numeric data "
            "for the selected period."
        )

        st.stop()

    unit = infer_unit(
        selected_variable
    )

    latest_value = (
        clean_series.iloc[-1]
    )

    one_month_change = (
        calculate_change(
            series,
            1,
        )
    )

    three_month_change = (
        calculate_change(
            series,
            3,
        )
    )

    twelve_month_change = (
        calculate_change(
            series,
            12,
        )
    )

    missing_pct = (
        series.isna().mean()
        * 100
    )

    cards = st.columns(
        6
    )

    with cards[0]:
        render_card(
            "Latest",
            f"{latest_value:,.2f}",
            unit,
        )

    with cards[1]:
        render_card(
            "1M Change",
            (
                f"{one_month_change:+.2f}%"
                if pd.notna(
                    one_month_change
                )
                else "N/A"
            ),
        )

    with cards[2]:
        render_card(
            "3M Change",
            (
                f"{three_month_change:+.2f}%"
                if pd.notna(
                    three_month_change
                )
                else "N/A"
            ),
        )

    with cards[3]:
        render_card(
            "12M Change",
            (
                f"{twelve_month_change:+.2f}%"
                if pd.notna(
                    twelve_month_change
                )
                else "N/A"
            ),
        )

    with cards[4]:
        render_card(
            "Observations",
            f"{len(clean_series):,}",
        )

    with cards[5]:
        render_card(
            "Missing",
            f"{missing_pct:.1f}%",
        )

    st.divider()

    chart_col, stats_col = st.columns(
        [3.25, 1]
    )

    with chart_col:

        st.markdown(
            f"### {pretty_name(selected_variable)}"
        )

        st.caption(
            f"Primary unit: {unit}"
        )

        explorer_fig = (
            go.Figure()
        )

        use_overlay = (
            overlay_copper
            and copper_column
            and copper_column
            != selected_variable
        )

        if (
            use_overlay
            and comparison_mode
            == "Indexed Comparison (Base = 100)"
        ):

            comparison_df = (
                filtered_df[
                    [
                        date_column,
                        selected_variable,
                        copper_column,
                    ]
                ]
                .copy()
                .dropna()
            )

            if not comparison_df.empty:

                selected_base = (
                    comparison_df[
                        selected_variable
                    ].iloc[0]
                )

                copper_base = (
                    comparison_df[
                        copper_column
                    ].iloc[0]
                )

                if (
                    selected_base != 0
                    and copper_base != 0
                ):

                    comparison_df[
                        "selected_index"
                    ] = (
                        comparison_df[
                            selected_variable
                        ]
                        / selected_base
                        * 100
                    )

                    comparison_df[
                        "copper_index"
                    ] = (
                        comparison_df[
                            copper_column
                        ]
                        / copper_base
                        * 100
                    )

                    explorer_fig.add_trace(
                        go.Scatter(
                            x=comparison_df[
                                date_column
                            ],
                            y=comparison_df[
                                "selected_index"
                            ],
                            mode="lines+markers",
                            name=pretty_name(
                                selected_variable
                            ),
                            line=dict(
                                width=2.6,
                            ),
                            marker=dict(
                                size=6,
                            ),
                            hovertemplate=(
                                "<b>%{x|%b %Y}</b>"
                                "<br>"
                                "Index: %{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    explorer_fig.add_trace(
                        go.Scatter(
                            x=comparison_df[
                                date_column
                            ],
                            y=comparison_df[
                                "copper_index"
                            ],
                            mode="lines+markers",
                            name="LME Copper Price",
                            line=dict(
                                width=2.6,
                                dash="dash",
                            ),
                            marker=dict(
                                size=6,
                            ),
                            hovertemplate=(
                                "<b>%{x|%b %Y}</b>"
                                "<br>"
                                "Copper Index: %{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

            explorer_fig.update_layout(
                yaxis_title=(
                    "Index (Base = 100)"
                ),
            )

        else:

            explorer_fig.add_trace(
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
                        width=2.6,
                    ),
                    marker=dict(
                        size=6,
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

            if use_overlay:

                explorer_fig.add_trace(
                    go.Scatter(
                        x=filtered_df[
                            date_column
                        ],
                        y=filtered_df[
                            copper_column
                        ],
                        mode="lines",
                        name="LME Copper Price",
                        yaxis="y2",
                        line=dict(
                            width=2.5,
                            dash="dash",
                        ),
                        hovertemplate=(
                            "<b>%{x|%b %Y}</b>"
                            "<br>"
                            "Copper: $%{y:,.2f}/t"
                            "<extra></extra>"
                        ),
                    )
                )

                explorer_fig.update_layout(
                    yaxis=dict(
                        title=unit,
                    ),
                    yaxis2=dict(
                        title=(
                            "Copper USD / ton"
                        ),
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )

            else:

                explorer_fig.update_layout(
                    yaxis=dict(
                        title=unit,
                    ),
                )

        explorer_fig.update_layout(
            height=660,
            margin=dict(
                l=25,
                r=50,
                t=120,
                b=20,
            ),
            hovermode="x unified",
            dragmode="zoom",
            xaxis_title="Month",
        )

        configure_time_axis(
            explorer_fig,
            show_range_slider=True,
        )

        configure_top_legend(
            explorer_fig
        )

        render_plot(
            explorer_fig
        )

        if use_overlay:

            st.caption(
                "Tip: Click a legend item to hide/show a series. "
                "Double-click a legend item to isolate that series."
            )

    with stats_col:

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

        valid_mask = filtered_df[
            selected_variable
        ].notna()

        if valid_mask.any():

            latest_valid_date = (
                filtered_df.loc[
                    valid_mask,
                    date_column,
                ].iloc[-1]
            )

            st.metric(
                "Latest Data Month",
                latest_valid_date.strftime(
                    "%b %Y"
                ),
            )

        if use_overlay:

            aligned = filtered_df[
                [
                    selected_variable,
                    copper_column,
                ]
            ].dropna()

            if len(
                aligned
            ) >= 3:

                correlation = (
                    aligned[
                        selected_variable
                    ]
                    .corr(
                        aligned[
                            copper_column
                        ]
                    )
                )

                st.metric(
                    "Copper Correlation",
                    f"{correlation:+.3f}",
                )

    st.divider()

    raw_tab, stats_tab, missing_tab = (
        st.tabs(
            [
                "Raw Data",
                "Descriptive Statistics",
                "Missing Data",
            ]
        )
    )

    with raw_tab:

        raw_columns = [
            date_column,
            selected_variable,
        ]

        if (
            use_overlay
            and copper_column
            not in raw_columns
        ):
            raw_columns.append(
                copper_column
            )

        raw_table = (
            filtered_df[
                raw_columns
            ].copy()
        )

        rename_map = {
            date_column:
                "Date",
            selected_variable:
                pretty_name(
                    selected_variable
                ),
        }

        if use_overlay:
            rename_map[
                copper_column
            ] = (
                "LME Copper Price USD/t"
            )

        raw_table = (
            raw_table.rename(
                columns=rename_map
            )
        )

        st.dataframe(
            raw_table,
            use_container_width=True,
            hide_index=True,
            height=440,
        )

        st.download_button(
            "Download Selected Data",
            raw_table.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                f"{selected_variable}_analysis.csv"
            ),
            mime="text/csv",
        )

    with stats_tab:

        descriptive = (
            clean_series
            .describe()
            .to_frame(
                name="Value"
            )
        )

        descriptive.loc[
            "missing_count"
        ] = (
            series.isna().sum()
        )

        descriptive.loc[
            "missing_pct"
        ] = (
            missing_pct
        )

        descriptive.loc[
            "latest_value"
        ] = (
            latest_value
        )

        descriptive.loc[
            "1m_change_pct"
        ] = (
            one_month_change
        )

        descriptive.loc[
            "3m_change_pct"
        ] = (
            three_month_change
        )

        descriptive.loc[
            "12m_change_pct"
        ] = (
            twelve_month_change
        )

        st.dataframe(
            descriptive,
            use_container_width=True,
        )

    with missing_tab:

        valid_dates = filtered_df.loc[
            filtered_df[
                selected_variable
            ].notna(),
            date_column,
        ]

        first_valid = (
            valid_dates.iloc[0]
            .strftime(
                "%Y-%m-%d"
            )
            if not valid_dates.empty
            else "N/A"
        )

        last_valid = (
            valid_dates.iloc[-1]
            .strftime(
                "%Y-%m-%d"
            )
            if not valid_dates.empty
            else "N/A"
        )

        missing_table = pd.DataFrame(
            {
                "Metric": [
                    "Total rows",
                    "Valid observations",
                    "Missing observations",
                    "Missing %",
                    "First valid month",
                    "Last valid month",
                    "Unit",
                    "Source column",
                ],
                "Value": [
                    len(
                        series
                    ),
                    int(
                        series.notna().sum()
                    ),
                    int(
                        series.isna().sum()
                    ),
                    f"{missing_pct:.2f}%",
                    first_valid,
                    last_valid,
                    unit,
                    selected_variable,
                ],
            }
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True,
        )
