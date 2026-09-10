# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.market_data_explorer import (
    PLOTLY_CONFIG,
    TARGET_COLUMN,
    render_lead_lag_signal_lab,
    render_market_data_explorer,
)

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


st.set_page_config(
    page_title="Copper Intelligence & Forecasting Center",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>

:root {
    --copper: #f59e0b;
    --copper-soft: rgba(245, 158, 11, 0.18);
    --cyan: #22d3ee;
    --purple: #a78bfa;
    --green: #34d399;
    --red: #fb7185;
    --panel: rgba(15, 23, 42, 0.72);
    --border: rgba(148, 163, 184, 0.16);
}

html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(245, 158, 11, 0.11),
            transparent 26%
        ),
        radial-gradient(
            circle at 28% 15%,
            rgba(34, 211, 238, 0.07),
            transparent 30%
        ),
        #ffffff;
}

.block-container {
    padding-top: 3.2rem;
    padding-bottom: 3rem;
    max-width: 1850px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    min-width: 310px;
    max-width: 310px;

    background:
        linear-gradient(
            180deg,
            #eef4f8 0%,
            #f4f6f7 40%,
            #fff7eb 100%
        );

    border-right:
        1px solid rgba(
            245,
            158,
            11,
            0.25
        );
}

[data-testid="stSidebarContent"] {
    padding-top: 2rem;
    padding-bottom: 5.2rem;
}

/* Sidebar text */

[data-testid="stSidebar"] * {
    color: #263445;
}

/* Sidebar section separators */

[data-testid="stSidebar"] hr {
    border-color:
        rgba(
            71,
            85,
            105,
            0.15
        );
}

/* Sidebar radio container */

[data-testid="stSidebar"]
div[role="radiogroup"] {
    gap: 0.35rem;
}

/* Sidebar radio options */

[data-testid="stSidebar"]
div[role="radiogroup"] label {
    border-radius: 10px;
    padding: 7px 8px;
    transition:
        background 0.2s ease;
}

[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {
    background:
        rgba(
            245,
            158,
            11,
            0.10
        );
}

/* Sidebar code boxes */

[data-testid="stSidebar"]
[data-testid="stCode"] {
    background:
        rgba(
            255,
            255,
            255,
            0.70
        );

    border:
        1px solid
        rgba(
            148,
            163,
            184,
            0.20
        );

    border-radius: 10px;
}

/* Fixed disclaimer at the bottom of the sidebar */

.sidebar-disclaimer {
    position: fixed;
    left: 14px;
    bottom: 12px;
    width: 278px;
    z-index: 9999;

    padding: 10px 12px;

    border:
        1px solid
        rgba(
            148,
            163,
            184,
            0.18
        );

    border-radius: 10px;

    background:
        rgba(
            255,
            255,
            255,
            0.78
        );

    backdrop-filter: blur(8px);

    font-size: 0.70rem;
    line-height: 1.35;
    color: rgba(71, 85, 105, 0.72);

    box-shadow:
        0 6px 18px
        rgba(
            15,
            23,
            42,
            0.05
        );
}

/* Main title */

.main-title {
    font-size: 2.65rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    line-height: 1.22;
    margin-top: 0.3rem;
    margin-bottom: 0.18rem;

    background:
        linear-gradient(
            90deg,
            #f59e0b,
            #fbbf24,
            #0891b2
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    font-size: 1rem;
    opacity: 0.68;
    margin-bottom: 1.6rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 760;
    margin-bottom: 0.75rem;
}

.page-hero {
    border:
        1px solid
        rgba(
            148,
            163,
            184,
            0.16
        );

    border-radius: 24px;
    padding: 24px 28px;
    margin-bottom: 22px;

    box-shadow:
        0 18px 55px
        rgba(
            0,
            0,
            0,
            0.08
        );
}

.market-hero {
    background:
        linear-gradient(
            135deg,
            rgba(
                34,
                211,
                238,
                0.10
            ),
            rgba(
                255,
                255,
                255,
                0.90
            ) 48%,
            rgba(
                245,
                158,
                11,
                0.14
            )
        );
}

.signal-hero {
    background:
        linear-gradient(
            135deg,
            rgba(
                167,
                139,
                250,
                0.12
            ),
            rgba(
                255,
                255,
                255,
                0.90
            ) 50%,
            rgba(
                245,
                158,
                11,
                0.14
            )
        );
}


.hero-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    font-weight: 800;
    color: #d97706;
    margin-bottom: 7px;
}

.hero-title {
    font-size: 2rem;
    font-weight: 850;
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin-bottom: 8px;
}

.hero-copy {
    opacity: 0.70;
    max-width: 850px;
    line-height: 1.6;
}

/* KPI cards */

.kpi-card {
    border:
        1px solid
        rgba(
            148,
            163,
            184,
            0.22
        );

    border-radius: 18px;
    padding: 17px 19px 15px 19px;

    background:
        linear-gradient(
            145deg,
            rgba(
                241,
                245,
                249,
                0.96
            ),
            rgba(
                255,
                247,
                237,
                0.78
            )
        );

    min-height: 125px;

    box-shadow:
        0 10px 30px
        rgba(
            15,
            23,
            42,
            0.07
        );
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
    border:
        1px solid
        rgba(
            148,
            163,
            184,
            0.22
        );

    border-radius: 16px;
    padding: 15px 17px;

    background:
        linear-gradient(
            145deg,
            rgba(
                241,
                245,
                249,
                0.92
            ),
            rgba(
                255,
                247,
                237,
                0.72
            )
        );
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


@st.cache_data(show_spinner=False)
def load_csv(
    path: Path,
) -> pd.DataFrame:
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
                css_class = "kpi-positive"
            elif numeric_delta < 0:
                css_class = "kpi-negative"
            else:
                css_class = "kpi-neutral"

        except Exception:
            css_class = "kpi-neutral"

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
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {unit_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_price(
    value: float,
) -> str:
    if pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def configure_time_axis(
    fig: go.Figure,
    show_range_slider: bool = True,
) -> None:
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
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
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
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


forecast_df = prepare_forecast(
    load_csv(FORECAST_FILE)
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


st.sidebar.markdown(
    "## \U0001F7E0 Copper Intelligence"
)

st.sidebar.caption(
    "Production ML Forecasting Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "\U0001F3E0 Executive Overview",
        "\U0001F3AF Forecast Center",
        "\U0001F310 Market Data Explorer",
        "\U0001F9ED Lead-Lag Signals",
    ],
)

if not forecast_df.empty:
    origin = forecast_df[
        "forecast_origin"
    ].iloc[0]

    generated_at = forecast_df[
        "generated_at"
    ].iloc[0]

    st.sidebar.markdown(
        "### \U0001F7E2 Forecast Status"
    )

    st.sidebar.success(
        "Latest forecast available"
    )

    if pd.notna(origin):
        st.sidebar.caption(
            "Forecast origin"
        )

        st.sidebar.code(
            origin.strftime(
                "%Y-%m-%d"
            )
        )

    if pd.notna(generated_at):
        st.sidebar.caption(
            "Generated at"
        )

        st.sidebar.code(
            generated_at.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    st.sidebar.caption(
        "Model Strategy"
    )
    
    st.sidebar.code(
        str(
            forecast_df[
                "production_policy_version"
            ].iloc[0]
        )
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-disclaimer">
            &#9888;&#65039; Analytical use only. Not investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.sidebar.error(
        "Forecast unavailable"
    )


st.markdown(
    '<div class="main-title">'
    'Copper Intelligence & Forecasting Center'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'LME Copper | Machine Learning Production Forecast | '
    'Market Intelligence | Lead-Lag Analytics'
    '</div>',
    unsafe_allow_html=True,
)


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


h1 = horizon_row(1)
h3 = horizon_row(3)
h6 = horizon_row(6)
h12 = horizon_row(12)


if page == "\U0001F3E0 Executive Overview":
    cards = st.columns(5)

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
            "### Historical Copper + Production Forecast"
        )

        fig = go.Figure()

        if (
            not master_df.empty
            and "date" in master_df.columns
            and TARGET_COLUMN
            in master_df.columns
        ):
            historical = master_df[
                [
                    "date",
                    TARGET_COLUMN,
                ]
            ].copy()

            historical[
                "date"
            ] = pd.to_datetime(
                historical["date"],
                errors="coerce",
            )

            historical[
                TARGET_COLUMN
            ] = pd.to_numeric(
                historical[
                    TARGET_COLUMN
                ],
                errors="coerce",
            )

            historical = (
                historical
                .dropna()
                .sort_values("date")
            )

            fig.add_trace(
                go.Scatter(
                    x=historical["date"],
                    y=historical[
                        TARGET_COLUMN
                    ],
                    mode="lines",
                    name="Historical LME Copper",
                    line=dict(
                        width=2.5,
                        color="#22d3ee",
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
                "horizon": [0],
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
                    color="#f59e0b",
                ),
                marker=dict(
                    size=8,
                    color="#f59e0b",
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
            line_color="#fbbf24",
        )

        fig.update_layout(
            height=650,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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

        configure_top_legend(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        st.markdown(
            "### Forecast Intelligence"
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
                f"H{value}"
                for value in low_horizons
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
        "### Monthly Forecast Movement"
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
            marker=dict(
                color=forecast_df[
                    "month_over_month_change_pct"
                ],
                colorscale="RdYlGn",
                cmid=0,
            ),
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
        line_color="rgba(148,163,184,0.6)",
    )

    movement_fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=20,
            r=20,
            t=35,
            b=20,
        ),
        xaxis_title="Forecast Month",
        yaxis_title="Month-over-month change %",
    )

    movement_fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)",
    )

    movement_fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
    )

    st.plotly_chart(
        movement_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


elif page == "\U0001F3AF Forecast Center":
    st.markdown(
        """
        <div class="page-hero signal-hero">
            <div class="hero-kicker">PRODUCTION FORECAST</div>
            <div class="hero-title">Forecast Center</div>
            <div class="hero-copy">
                Inspect the H1-H12 production curve, final reconciled
                forecasts, model diagnostics and backtest quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
                    color="#f59e0b",
                ),
                marker=dict(
                    size=9,
                    color="#fbbf24",
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
            line_color="#22d3ee",
            annotation_text="Origin Price",
        )

        fig.update_layout(
            height=590,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
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
                    "Origin USD/t": "{:,.2f}",
                    "Raw Forecast USD/t": "{:,.2f}",
                    "Anchor Forecast USD/t": "{:,.2f}",
                    "Final Forecast USD/t": "{:,.2f}",
                    "MoM %": "{:.2f}",
                    "vs Origin %": "{:.2f}",
                    "MAPE %": "{:.2f}",
                    "RMSE": "{:,.2f}",
                    "Bias": "{:,.2f}",
                    "Directional Accuracy %": "{:.2f}",
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
            ).encode("utf-8"),
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

        d1, d2, d3, d4 = st.columns(4)

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

        d5, d6, d7, d8 = st.columns(4)

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


elif page == "\U0001F310 Market Data Explorer":
    render_market_data_explorer(
        master_df=master_df,
    )


elif page == "\U0001F9ED Lead-Lag Signals":
    render_lead_lag_signal_lab(
        master_df=master_df,
    )
