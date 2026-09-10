# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# CORE CONFIG
# =============================================================================

TARGET_COLUMN = "cash_settlement_usd_per_ton"

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
# CATEGORY DEFINITIONS
#
# Important:
# Emoji characters are stored as Unicode escapes. This avoids the "??" issue
# caused by terminals/editors that damage UTF-8 emoji during copy/paste.
# =============================================================================

CATEGORY_ORDER = [
    "Copper Price & LME",
    "Metal Inventories & Exchange Stocks",
    "Metal Prices",
    "Country Stock Indices",
    "Mining Company Stocks",
    "FX & Currencies",
    "China Economy & Copper Demand",
    "US Economy & Financial Conditions",
    "Europe Economy & Rates",
    "Latin America Mining & Economy",
    "Global Economy & Leading Indicators",
    "Energy & Power",
    "Shipping, Trade & Ports",
    "Copper Supply & Mining",
    "Demand & Energy Transition",
    "Risk & Uncertainty",
    "Macro & Economic Activity",
]


CATEGORY_ICON_MAP = {
    "All Indicators": "\U0001F50E",
    "Copper Price & LME": "\U0001F7E0",
    "Metal Inventories & Exchange Stocks": "\U0001F4E6",
    "Metal Prices": "\u2699\ufe0f",
    "Country Stock Indices": "\U0001F4C8",
    "Mining Company Stocks": "\u26CF\ufe0f",
    "FX & Currencies": "\U0001F4B1",
    "China Economy & Copper Demand": "\U0001F1E8\U0001F1F3",
    "US Economy & Financial Conditions": "\U0001F1FA\U0001F1F8",
    "Europe Economy & Rates": "\U0001F1EA\U0001F1FA",
    "Latin America Mining & Economy": "\U0001F30E",
    "Global Economy & Leading Indicators": "\U0001F310",
    "Energy & Power": "\u26A1",
    "Shipping, Trade & Ports": "\U0001F6A2",
    "Copper Supply & Mining": "\u26CF\ufe0f",
    "Demand & Energy Transition": "\U0001F50B",
    "Risk & Uncertainty": "\u26A0\ufe0f",
    "Macro & Economic Activity": "\U0001F3ED",
}


CATEGORY_DESCRIPTION = {
    "All Indicators":
        "Every numeric series in the monthly master dataset.",
    "Copper Price & LME":
        "LME copper cash price, copper futures and direct copper benchmark series.",
    "Metal Inventories & Exchange Stocks":
        "LME, SHFE and warehouse inventory measures for copper and other metals.",
    "Metal Prices":
        "Aluminum, zinc, nickel, lead, tin, gold, silver and other metal prices or ratios.",
    "Country Stock Indices":
        "Broad national equity indices such as ASX200, FTSE100, CSI300, IPSA, WIG20 and S&P 500.",
    "Mining Company Stocks":
        "Listed copper and mining companies such as Southern Copper, Jiangxi Copper and Capstone Copper.",
    "FX & Currencies":
        "US Dollar Index and currency pairs such as USD/JPY, USD/KRW, USD/INR, USD/GBP and EUR/USD.",
    "China Economy & Copper Demand":
        "China industrial production, PPI, investment, electricity, GDP and refined copper indicators.",
    "US Economy & Financial Conditions":
        "US rates, Treasury yields, CPI, money supply, industrial production, PMI and labor-cost indicators.",
    "Europe Economy & Rates":
        "ECB policy rates and European macroeconomic or financial indicators.",
    "Latin America Mining & Economy":
        "Chile and Peru economic, mining, labor, cost and local-market indicators.",
    "Global Economy & Leading Indicators":
        "World growth, GFCF, OECD and global leading indicators.",
    "Energy & Power":
        "Oil, Brent, WTI, natural gas, electricity, fuel and energy-cost indicators.",
    "Shipping, Trade & Ports":
        "Baltic Dry Index, freight, dry bulk, exports, imports and port-call indicators.",
    "Copper Supply & Mining":
        "Global mine/refinery production, reserves, ore grade, ICSG and other copper supply measures.",
    "Demand & Energy Transition":
        "EVs, renewables, batteries, grids and structural copper-demand indicators.",
    "Risk & Uncertainty":
        "VIX, geopolitical risk and economic or policy uncertainty measures.",
    "Macro & Economic Activity":
        "Remaining macro, production, employment, price and activity indicators.",
}


# =============================================================================
# FRIENDLY NAMES
# =============================================================================

CUSTOM_NAMES = {
    "cash_settlement_usd_per_ton": "LME Copper Cash Settlement",
    "copper_stock_ton": "LME Copper Stock",
    "copper_usd_per_ton": "Copper Price",
    "copper_futures": "Copper Futures",
    "gold_copper_ratio": "Gold / Copper Ratio",
    "fed_interest_rate": "Fed Interest Rate",
    "crude_oil_price_usd": "Crude Oil Price",
    "natural_gas_usd": "US Natural Gas",
    "m2_money_supply": "US M2 Money Supply",
    "us_10y_treasury_yield": "US 10Y Treasury Yield",
    "us_cpi_index": "US CPI",
    "us_inflation_rate_pct": "US Inflation Rate",
    "gold_usd_per_ounce": "Gold Price",
    "dollar_index": "US Dollar Index (DXY)",
    "eur_usd": "EUR / USD",
    "vix_index": "VIX Index",
    "baltic_dry_index": "Baltic Dry Index",
    "geopolitical_risk_index": "Geopolitical Risk Index",
    "china_industrial_production": "China Industrial Production",
    "china_gdp_growth_pct": "China GDP Growth",
    "china_refined_copper_production_ton": "China Refined Copper Production",
    "china_fixed_asset_investment_yoy_pct": "China Fixed Asset Investment YoY",
    "china_real_estate_investment_yoy_pct": "China Real Estate Investment YoY",
    "china_electricity_generation_100m_kwh": "China Electricity Generation",
    "world_copper_mine_production_ton": "World Copper Mine Production",
    "world_copper_refinery_production_ton": "World Copper Refinery Production",
    "world_copper_reserves_ton": "World Copper Reserves",
    "chile_copper_production_ton": "Chile Copper Production",
    "peru_copper_production_ton": "Peru Copper Production",
    "ev_sales_count": "EV Sales",
    "us_pmi": "US PMI",
    "composite_leading_indicator_cli_g20": "G20 Composite Leading Indicator",
    "chile_ipsa": "Chile IPSA",
    "china_csi300": "China CSI300",
    "poland_wig20": "Poland WIG20",
    "canada_tsx_composite_capstone_copper": "Capstone Copper",
    "china_csi300_jiangxi_copper": "Jiangxi Copper",
    "usa_sp500_southern_copper": "Southern Copper",
    "icsg_refined_usage_ton": "ICSG Refined Copper Usage",
    "icsg_mine_production_ton": "ICSG Mine Production",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def pretty_name(column_name: str) -> str:
    if column_name in CUSTOM_NAMES:
        return CUSTOM_NAMES[column_name]

    return (
        str(column_name)
        .replace("_", " ")
        .strip()
        .title()
    )


def category_icon(category_name: str) -> str:
    return CATEGORY_ICON_MAP.get(
        category_name,
        "\U0001F4C1",
    )


def category_label(category_name: str) -> str:
    return (
        f"{category_icon(category_name)} "
        f"{category_name}"
    )


def infer_unit(column_name: str) -> str:
    name = str(column_name).lower()

    if (
        "usd_per_ton" in name
        or "usd_per_tonne" in name
        or "usd_ton" in name
    ):
        return "USD / ton"

    if "usd_per_ounce" in name:
        return "USD / oz"

    if (
        "pct" in name
        or "percent" in name
        or "percentage" in name
        or "yoy" in name
    ):
        return "%"

    if (
        "yield" in name
        or "interest_rate" in name
        or "policy_rate" in name
        or "fedfund" in name
    ):
        return "%"

    if "100m_kwh" in name:
        return "100M kWh"

    if "_tj" in name or name.endswith("tj"):
        return "TJ"

    if (
        "ton" in name
        or "tonne" in name
    ):
        return "ton"

    if (
        "index" in name
        or "csi300" in name
        or "wig20" in name
        or "ipsa" in name
        or "sp500" in name
        or "tsx" in name
    ):
        return "Index"

    if (
        "count" in name
        or "sales" in name
        or "employment" in name
        or "portcall" in name
    ):
        return "Count"

    if "usd" in name:
        return "USD"

    return "Value"


def get_date_column(df: pd.DataFrame) -> str:
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
            return lowered[candidate]

    raise ValueError(
        "No date-like column found."
    )


def get_numeric_columns(
    df: pd.DataFrame,
    date_column: str,
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
            columns.append(column)

    return sorted(columns)


# =============================================================================
# SMART BUSINESS CATEGORY CLASSIFICATION
# =============================================================================

def _contains_any(
    value: str,
    keywords: List[str],
) -> bool:
    return any(
        keyword in value
        for keyword in keywords
    )


def _looks_like_currency_pair(
    name: str,
) -> bool:
    currency_codes = {
        "aud",
        "brl",
        "cad",
        "cdf",
        "chf",
        "clp",
        "cny",
        "cop",
        "eur",
        "gbp",
        "idr",
        "inr",
        "jpy",
        "krw",
        "mxn",
        "pen",
        "pln",
        "try",
        "usd",
        "zar",
    }

    normalized = (
        name
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )

    parts = [
        part
        for part in normalized.split("_")
        if part
    ]

    if len(parts) >= 2:
        for left, right in zip(
            parts,
            parts[1:],
        ):
            if (
                left in currency_codes
                and right in currency_codes
            ):
                return True

    return False


def _looks_like_country_index(
    name: str,
) -> bool:
    explicit_tokens = [
        "asx200",
        "ftse100",
        "ftse_100",
        "cac40",
        "cac_40",
        "dax40",
        "dax_40",
        "dax",
        "nikkei225",
        "nikkei_225",
        "hang_seng",
        "hsi",
        "kospi",
        "sensex",
        "nifty50",
        "nifty_50",
        "ibovespa",
        "bovespa",
        "bist100",
        "bist_100",
        "ipsa",
        "wig20",
        "csi300",
        "sp500",
        "s&p500",
        "s&p_500",
        "nasdaq",
        "dow_jones",
        "tsx_composite",
        "smi",
        "stoxx",
        "eurostoxx",
        "euro_stoxx",
    ]

    if _contains_any(
        name,
        explicit_tokens,
    ):
        return True

    country_tokens = [
        "australia",
        "austria",
        "belgium",
        "brazil",
        "canada",
        "chile",
        "china",
        "france",
        "germany",
        "hong_kong",
        "india",
        "indonesia",
        "italy",
        "japan",
        "korea",
        "mexico",
        "netherlands",
        "norway",
        "poland",
        "south_africa",
        "spain",
        "sweden",
        "switzerland",
        "turkey",
        "uk",
        "united_kingdom",
        "usa",
        "us",
    ]

    index_tokens = [
        "index",
        "composite",
        "equity_market",
        "stock_market",
    ]

    return (
        _contains_any(
            name,
            country_tokens,
        )
        and _contains_any(
            name,
            index_tokens,
        )
    )


def classify_column(
    column_name: str,
) -> str:
    name = str(
        column_name
    ).lower()

    if column_name == TARGET_COLUMN:
        return "Copper Price & LME"

    if _contains_any(
        name,
        [
            "cash_settlement",
            "copper_cash",
            "copper_futures",
            "copper_usd_per_ton",
        ],
    ):
        return "Copper Price & LME"

    if (
        _contains_any(
            name,
            [
                "inventory",
                "warehouse",
                "lme_stock",
                "shfe_stock",
                "metal_stock",
                "copper_stock",
                "exchange_stock",
            ],
        )
        or (
            "stock" in name
            and _contains_any(
                name,
                [
                    "copper",
                    "aluminum",
                    "aluminium",
                    "zinc",
                    "nickel",
                    "lead",
                    "tin",
                ],
            )
            and not _contains_any(
                name,
                [
                    "sp500",
                    "tsx",
                    "csi300",
                    "company",
                    "southern_copper",
                    "jiangxi_copper",
                    "capstone_copper",
                ],
            )
        )
    ):
        return "Metal Inventories & Exchange Stocks"

    if _contains_any(
        name,
        [
            "southern_copper",
            "jiangxi_copper",
            "capstone_copper",
            "freeport",
            "freeport_mcmoran",
            "rio_tinto",
            "glencore",
            "bhp",
            "antofagasta",
            "teck_resources",
            "mining_company",
        ],
    ):
        return "Mining Company Stocks"

    if _looks_like_country_index(
        name
    ):
        return "Country Stock Indices"

    if (
        _looks_like_currency_pair(
            name
        )
        or _contains_any(
            name,
            [
                "dollar_index",
                "dxy",
                "exchange_rate",
                "currency",
                "forex",
                "fx_rate",
                "fx_",
                "_fx",
            ],
        )
    ):
        return "FX & Currencies"

    if _contains_any(
        name,
        [
            "baltic",
            "bdi",
            "shipping",
            "freight",
            "dry_bulk",
            "portcall",
            "port_calls",
            "_export",
            "_import",
            "trade_",
            "_trade",
        ],
    ):
        return "Shipping, Trade & Ports"

    if _contains_any(
        name,
        [
            "crude_oil",
            "brent",
            "wti",
            "natural_gas",
            "electricity",
            "energy_",
            "_energy",
            "fuel",
            "diesel",
            "coal",
        ],
    ):
        return "Energy & Power"

    if _contains_any(
        name,
        [
            "vix",
            "geopolitical",
            "gpr",
            "uncertainty",
            "epu",
            "risk_index",
            "economic_policy_uncertainty",
        ],
    ):
        return "Risk & Uncertainty"

    if _contains_any(
        name,
        [
            "icsg",
            "cochilco",
            "mine_production",
            "mining_production",
            "refinery_production",
            "copper_production",
            "copper_reserve",
            "copper_reserves",
            "ore_grade",
            "smelter",
            "refinery",
            "treatment_charge",
            "tcrc",
            "mine_prod",
            "world_copper",
        ],
    ):
        return "Copper Supply & Mining"

    if _contains_any(
        name,
        [
            "chile_",
            "peru_",
            "cochilco_",
            "chilean_",
            "peruvian_",
        ],
    ):
        return "Latin America Mining & Economy"

    if (
        name.startswith(
            "china_"
        )
        or _contains_any(
            name,
            [
                "china_industrial",
                "china_ppi",
                "china_gdp",
                "china_electricity",
                "china_fixed_asset",
                "china_real_estate",
                "china_refined_copper",
                "gfcf_china",
            ],
        )
    ):
        return "China Economy & Copper Demand"

    if _contains_any(
        name,
        [
            "ev_",
            "vehicle",
            "renewable",
            "solar",
            "wind",
            "battery",
            "grid_",
            "energy_transition",
            "copper_demand",
        ],
    ):
        return "Demand & Energy Transition"

    if _contains_any(
        name,
        [
            "ecb",
            "euro_area",
            "eurozone",
            "europe_",
            "eu_",
        ],
    ):
        return "Europe Economy & Rates"

    if (
        name.startswith(
            "us_"
        )
        or name.startswith(
            "usa_"
        )
        or _contains_any(
            name,
            [
                "fed_interest",
                "fedfund",
                "treasury",
                "us_cpi",
                "us_inflation",
                "m2_money",
                "unit_labor",
                "gfcf_us",
                "us_pmi",
            ],
        )
    ):
        return "US Economy & Financial Conditions"

    if _contains_any(
        name,
        [
            "global_",
            "world_gdp",
            "gfcf_world",
            "cli_g20",
            "composite_leading",
            "leading_indicator",
            "oecd",
        ],
    ):
        return "Global Economy & Leading Indicators"

    if _contains_any(
        name,
        [
            "aluminum",
            "aluminium",
            "lead_cash",
            "lead_price",
            "nickel",
            "tin_cash",
            "tin_price",
            "zinc",
            "gold",
            "silver",
            "metal_price",
            "metal_cash",
            "ppi_copper",
            "gold_copper_ratio",
        ],
    ):
        return "Metal Prices"

    return "Macro & Economic Activity"


def build_categories(
    numeric_columns: List[str],
) -> Dict[str, List[str]]:
    grouped: Dict[
        str,
        List[str],
    ] = {}

    for column in numeric_columns:
        category = classify_column(
            column
        )

        grouped.setdefault(
            category,
            [],
        ).append(
            column
        )

    result: Dict[
        str,
        List[str],
    ] = {
        "All Indicators": sorted(
            numeric_columns,
            key=pretty_name,
        )
    }

    for category in CATEGORY_ORDER:
        values = grouped.get(
            category,
            [],
        )

        if values:
            result[
                category
            ] = sorted(
                values,
                key=pretty_name,
            )

    return result


# =============================================================================
# TIME SERIES HELPERS
# =============================================================================

def calculate_change(
    series: pd.Series,
    periods: int,
) -> float:
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(clean) <= periods:
        return np.nan

    previous = clean.iloc[
        -(periods + 1)
    ]

    latest = clean.iloc[-1]

    if previous == 0:
        return np.nan

    return (
        latest / previous - 1
    ) * 100


def filter_period(
    df: pd.DataFrame,
    date_column: str,
    period: str,
) -> pd.DataFrame:
    if period == "All":
        return df.copy()

    years = {
        "1Y": 1,
        "3Y": 3,
        "5Y": 5,
        "10Y": 10,
    }[period]

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


def _transparent_layout(
    fig: go.Figure,
    *,
    height: int,
    y_title: str | None = None,
) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=25,
            r=55,
            t=115,
            b=25,
        ),
        hovermode="x unified",
        dragmode="zoom",
        xaxis_title="Month",
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            x=1,
            y=1.16,
            xanchor="right",
            yanchor="bottom",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        showspikes=True,
        spikethickness=1,
        spikedash="dot",
        spikesnap="cursor",
        rangeslider=dict(
            visible=True,
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


# =============================================================================
# SIGNAL ANALYSIS
# =============================================================================

def _transform_for_signal_analysis(
    series: pd.Series,
    basis: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    if basis == "Raw levels":
        return numeric

    if basis == "Monthly difference":
        return numeric.diff()

    if basis == "Monthly % change":
        result = numeric.pct_change(
            fill_method=None,
        ) * 100

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    raise ValueError(
        f"Unsupported analysis basis: {basis}"
    )


def calculate_lead_lag_table(
    df: pd.DataFrame,
    *,
    target_column: str,
    numeric_columns: List[str],
    max_lead_months: int,
    min_observations: int,
    basis: str,
) -> pd.DataFrame:
    target = _transform_for_signal_analysis(
        df[target_column],
        basis,
    )

    rows = []

    for feature in numeric_columns:
        if feature == target_column:
            continue

        feature_series = (
            _transform_for_signal_analysis(
                df[feature],
                basis,
            )
        )

        for lead in range(
            max_lead_months + 1
        ):
            future_target = target.shift(
                -lead
            )

            aligned = pd.concat(
                [
                    feature_series.rename(
                        "feature"
                    ),
                    future_target.rename(
                        "future_copper"
                    ),
                ],
                axis=1,
            ).dropna()

            observations = len(
                aligned
            )

            if observations < min_observations:
                continue

            correlation = aligned[
                "feature"
            ].corr(
                aligned[
                    "future_copper"
                ]
            )

            if pd.isna(correlation):
                continue

            rows.append(
                {
                    "column": feature,
                    "indicator":
                        pretty_name(
                            feature
                        ),
                    "category":
                        classify_column(
                            feature
                        ),
                    "lead_months":
                        lead,
                    "correlation":
                        float(
                            correlation
                        ),
                    "abs_correlation":
                        abs(
                            float(
                                correlation
                            )
                        ),
                    "observations":
                        observations,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


def build_best_leading_signals(
    lag_table: pd.DataFrame,
) -> pd.DataFrame:
    if lag_table.empty:
        return lag_table

    leading = lag_table.loc[
        lag_table[
            "lead_months"
        ] > 0
    ].copy()

    if leading.empty:
        return pd.DataFrame()

    leading = leading.sort_values(
        [
            "column",
            "abs_correlation",
            "lead_months",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    best = (
        leading
        .drop_duplicates(
            subset=[
                "column"
            ],
            keep="first",
        )
        .copy()
    )

    same_month = (
        lag_table.loc[
            lag_table[
                "lead_months"
            ] == 0,
            [
                "column",
                "correlation",
            ],
        ]
        .rename(
            columns={
                "correlation":
                    "same_month_correlation"
            }
        )
    )

    best = best.merge(
        same_month,
        on="column",
        how="left",
    )

    best[
        "correlation_gain"
    ] = (
        best[
            "abs_correlation"
        ]
        - best[
            "same_month_correlation"
        ].abs()
    )

    best[
        "signal_strength"
    ] = best[
        "abs_correlation"
    ].apply(
        correlation_strength
    )

    return (
        best
        .sort_values(
            "abs_correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def correlation_strength(
    value: float,
) -> str:
    value = abs(
        float(value)
    )

    if value >= 0.70:
        return "Very strong"

    if value >= 0.50:
        return "Strong"

    if value >= 0.30:
        return "Moderate"

    if value >= 0.15:
        return "Weak"

    return "Very weak"


# =============================================================================
# MARKET DATA EXPLORER
# =============================================================================

def render_market_data_explorer(
    master_df: pd.DataFrame,
) -> None:
    st.markdown(
        """
        <div class="page-hero market-hero">
            <div class="hero-kicker">
                MARKET INTELLIGENCE
            </div>
            <div class="hero-title">
                Market Data Explorer
            </div>
            <div class="hero-copy">
                Browse the monthly data warehouse using clear market
                categories and compare any indicator directly with copper.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        df
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
            df,
            date_column,
        )
    )

    categories = build_categories(
        numeric_columns
    )

    controls = st.columns(
        [
            1.55,
            2.25,
            0.75,
            1.0,
        ]
    )

    with controls[0]:
        category = st.selectbox(
            "Market category",
            list(
                categories.keys()
            ),
            format_func=category_label,
            key="market_category",
        )

    with controls[1]:
        selected_variable = (
            st.selectbox(
                "Indicator",
                categories[
                    category
                ],
                format_func=pretty_name,
                key="market_indicator",
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
            key="market_period",
        )

    with controls[3]:
        overlay_copper = (
            st.checkbox(
                "Show copper overlay",
                value=True,
                disabled=(
                    selected_variable
                    == TARGET_COLUMN
                ),
                key="market_overlay",
            )
        )

    selected_category = (
        classify_column(
            selected_variable
        )
    )

    if category == "All Indicators":
        shown_description = (
            f"{category_label(selected_category)}  "
            f"Selected indicator category."
        )
    else:
        shown_description = (
            f"{category_label(category)}  "
            f"{CATEGORY_DESCRIPTION.get(category, '')}"
        )

    st.caption(
        shown_description
    )

    comparison_mode = st.radio(
        "Chart mode",
        [
            "Raw values",
            "Indexed comparison (Base = 100)",
        ],
        horizontal=True,
        disabled=(
            not overlay_copper
            or selected_variable
            == TARGET_COLUMN
        ),
        key="market_chart_mode",
    )

    df[
        selected_variable
    ] = pd.to_numeric(
        df[
            selected_variable
        ],
        errors="coerce",
    )

    if TARGET_COLUMN in df.columns:
        df[
            TARGET_COLUMN
        ] = pd.to_numeric(
            df[
                TARGET_COLUMN
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
            "Selected indicator has no numeric data "
            "for this period."
        )
        return

    unit = infer_unit(
        selected_variable
    )

    latest_value = (
        clean_series.iloc[-1]
    )

    change_1m = calculate_change(
        series,
        1,
    )

    change_3m = calculate_change(
        series,
        3,
    )

    change_12m = calculate_change(
        series,
        12,
    )

    missing_pct = (
        series.isna().mean()
        * 100
    )

    k1, k2, k3, k4, k5, k6 = (
        st.columns(
            6
        )
    )

    k1.metric(
        "Latest value",
        f"{latest_value:,.2f}",
        unit,
    )

    k2.metric(
        "1M change",
        (
            f"{change_1m:+.2f}%"
            if pd.notna(
                change_1m
            )
            else "N/A"
        ),
    )

    k3.metric(
        "3M change",
        (
            f"{change_3m:+.2f}%"
            if pd.notna(
                change_3m
            )
            else "N/A"
        ),
    )

    k4.metric(
        "12M change",
        (
            f"{change_12m:+.2f}%"
            if pd.notna(
                change_12m
            )
            else "N/A"
        ),
    )

    k5.metric(
        "Observations",
        f"{len(clean_series):,}",
    )

    k6.metric(
        "Missing",
        f"{missing_pct:.1f}%",
    )

    st.caption(
        f"Source column: {selected_variable}"
        f"  |  Unit: {unit}"
    )

    chart_col, stats_col = (
        st.columns(
            [
                3.2,
                1,
            ]
        )
    )

    use_overlay = (
        overlay_copper
        and TARGET_COLUMN
        in filtered_df.columns
        and selected_variable
        != TARGET_COLUMN
    )

    with chart_col:
        st.markdown(
            f"### {pretty_name(selected_variable)}"
        )

        fig = go.Figure()

        if (
            use_overlay
            and comparison_mode
            == "Indexed comparison (Base = 100)"
        ):
            comparison_df = (
                filtered_df[
                    [
                        date_column,
                        selected_variable,
                        TARGET_COLUMN,
                    ]
                ]
                .dropna()
                .copy()
            )

            if not comparison_df.empty:
                selected_base = (
                    comparison_df[
                        selected_variable
                    ].iloc[0]
                )

                copper_base = (
                    comparison_df[
                        TARGET_COLUMN
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
                            TARGET_COLUMN
                        ]
                        / copper_base
                        * 100
                    )

                    fig.add_trace(
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
                                width=2.8,
                                color="#22d3ee",
                            ),
                            marker=dict(
                                size=5,
                            ),
                            hovertemplate=(
                                "<b>%{x|%b %Y}</b>"
                                "<br>"
                                "Index: %{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=comparison_df[
                                date_column
                            ],
                            y=comparison_df[
                                "copper_index"
                            ],
                            mode="lines+markers",
                            name="LME Copper",
                            line=dict(
                                width=2.8,
                                dash="dash",
                                color="#f59e0b",
                            ),
                            marker=dict(
                                size=5,
                            ),
                            hovertemplate=(
                                "<b>%{x|%b %Y}</b>"
                                "<br>"
                                "Copper index: %{y:.2f}"
                                "<extra></extra>"
                            ),
                        )
                    )

            y_title = (
                "Index (Base = 100)"
            )

        else:
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
                        width=2.8,
                        color="#22d3ee",
                    ),
                    marker=dict(
                        size=5,
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

            y_title = unit

            if use_overlay:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df[
                            date_column
                        ],
                        y=filtered_df[
                            TARGET_COLUMN
                        ],
                        mode="lines",
                        name="LME Copper",
                        yaxis="y2",
                        line=dict(
                            width=2.7,
                            dash="dash",
                            color="#f59e0b",
                        ),
                        hovertemplate=(
                            "<b>%{x|%b %Y}</b>"
                            "<br>"
                            "Copper: $%{y:,.2f}/t"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    yaxis2=dict(
                        title=(
                            "Copper USD / ton"
                        ),
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )

        _transparent_layout(
            fig,
            height=650,
            y_title=y_title,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        if use_overlay:
            st.caption(
                "Click a legend item to hide/show a series. "
                "Double-click to isolate one series."
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
            "Std. deviation",
            f"{clean_series.std():,.2f}",
        )

        valid_dates = (
            filtered_df.loc[
                filtered_df[
                    selected_variable
                ].notna(),
                date_column,
            ]
        )

        if not valid_dates.empty:
            st.metric(
                "Latest data month",
                valid_dates.iloc[
                    -1
                ].strftime(
                    "%b %Y"
                ),
            )

        if use_overlay:
            aligned = (
                filtered_df[
                    [
                        selected_variable,
                        TARGET_COLUMN,
                    ]
                ]
                .dropna()
            )

            if len(aligned) >= 3:
                correlation = (
                    aligned[
                        selected_variable
                    ]
                    .corr(
                        aligned[
                            TARGET_COLUMN
                        ]
                    )
                )

                st.metric(
                    "Same-month copper corr.",
                    f"{correlation:+.3f}",
                )

    raw_tab, stats_tab, missing_tab = (
        st.tabs(
            [
                "Raw data",
                "Descriptive statistics",
                "Missing data",
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
            and TARGET_COLUMN
            not in raw_columns
        ):
            raw_columns.append(
                TARGET_COLUMN
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
                TARGET_COLUMN
            ] = (
                "LME Copper USD/t"
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
            height=430,
        )

        st.download_button(
            "Download selected data",
            raw_table.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            file_name=(
                f"{selected_variable}"
                "_analysis.csv"
            ),
            mime="text/csv",
        )

    with stats_tab:
        stats = (
            clean_series
            .describe()
            .to_frame(
                name="Value"
            )
        )

        stats.loc[
            "missing_count"
        ] = (
            series.isna().sum()
        )

        stats.loc[
            "missing_pct"
        ] = missing_pct

        stats.loc[
            "1m_change_pct"
        ] = change_1m

        stats.loc[
            "3m_change_pct"
        ] = change_3m

        stats.loc[
            "12m_change_pct"
        ] = change_12m

        st.dataframe(
            stats,
            use_container_width=True,
        )

    with missing_tab:
        valid_dates = (
            filtered_df.loc[
                filtered_df[
                    selected_variable
                ].notna(),
                date_column,
            ]
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total rows",
                    "Valid observations",
                    "Missing observations",
                    "Missing %",
                    "First valid month",
                    "Last valid month",
                    "Unit",
                    "Market category",
                    "Source column",
                ],
                "Value": [
                    len(series),
                    int(
                        series.notna().sum()
                    ),
                    int(
                        series.isna().sum()
                    ),
                    f"{missing_pct:.2f}%",
                    (
                        valid_dates.iloc[
                            0
                        ].strftime(
                            "%Y-%m-%d"
                        )
                        if not valid_dates.empty
                        else "N/A"
                    ),
                    (
                        valid_dates.iloc[
                            -1
                        ].strftime(
                            "%Y-%m-%d"
                        )
                        if not valid_dates.empty
                        else "N/A"
                    ),
                    unit,
                    selected_category,
                    selected_variable,
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# LEAD-LAG SIGNAL LAB
# =============================================================================

def render_lead_lag_signal_lab(
    master_df: pd.DataFrame,
) -> None:
    st.markdown(
        """
        <div class="page-hero signal-hero">
            <div class="hero-kicker">
                LEADING INDICATOR RESEARCH
            </div>
            <div class="hero-title">
                Copper Lead-Lag Signal Lab
            </div>
            <div class="hero-copy">
                Search the full monthly data warehouse for indicators
                that move before copper and measure the relationship
                at different monthly leads.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    if TARGET_COLUMN not in df.columns:
        st.error(
            f"Target column not found: "
            f"{TARGET_COLUMN}"
        )
        return

    df[
        date_column
    ] = pd.to_datetime(
        df[
            date_column
        ],
        errors="coerce",
    )

    df = (
        df
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
            df,
            date_column,
        )
    )

    signal_categories = (
        build_categories(
            [
                column
                for column in numeric_columns
                if column
                != TARGET_COLUMN
            ]
        )
    )

    c1, c2, c3, c4, c5 = (
        st.columns(
            [
                1.25,
                1.15,
                0.85,
                0.95,
                0.85,
            ]
        )
    )

    with c1:
        signal_category = (
            st.selectbox(
                "Indicator group",
                list(
                    signal_categories.keys()
                ),
                format_func=category_label,
                key="signal_category",
            )
        )

    with c2:
        basis = st.selectbox(
            "Analysis basis",
            [
                "Monthly % change",
                "Monthly difference",
                "Raw levels",
            ],
            index=0,
            key="signal_basis",
        )

    with c3:
        max_lead = st.slider(
            "Max lead",
            min_value=1,
            max_value=12,
            value=6,
            key="signal_max_lead",
        )

    with c4:
        min_observations = (
            st.slider(
                "Minimum obs.",
                min_value=12,
                max_value=80,
                value=36,
                step=6,
                key="signal_min_obs",
            )
        )

    with c5:
        top_n = st.slider(
            "Top signals",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            key="signal_top_n",
        )

    st.info(
        "Example: Lead +3M means the indicator observed in "
        "month t is compared with copper in month t+3. "
        "The default Monthly % change view is usually more useful "
        "for signal research than raw price levels."
    )

    selected_signal_columns = (
        signal_categories[
            signal_category
        ]
    )

    with st.spinner(
        "Calculating lead-lag correlations..."
    ):
        lag_table = (
            calculate_lead_lag_table(
                df,
                target_column=TARGET_COLUMN,
                numeric_columns=[
                    TARGET_COLUMN,
                    *selected_signal_columns,
                ],
                max_lead_months=max_lead,
                min_observations=(
                    min_observations
                ),
                basis=basis,
            )
        )

    if lag_table.empty:
        st.warning(
            "No indicators have enough aligned "
            "observations for these settings."
        )
        return

    best_rows = (
        build_best_leading_signals(
            lag_table
        )
    )

    if best_rows.empty:
        st.warning(
            "No leading signals were available."
        )
        return

    top = (
        best_rows
        .head(
            top_n
        )
        .copy()
    )

    strongest = (
        top.iloc[0]
    )

    k1, k2, k3, k4, k5 = (
        st.columns(
            5
        )
    )

    k1.metric(
        "Strongest signal",
        strongest[
            "indicator"
        ],
    )

    k2.metric(
        "Best lead",
        (
            f"+{int(strongest['lead_months'])}M"
        ),
    )

    k3.metric(
        "Lead correlation",
        (
            f"{strongest['correlation']:+.3f}"
        ),
    )

    same_corr = strongest[
        "same_month_correlation"
    ]

    k4.metric(
        "Same-month corr.",
        (
            f"{same_corr:+.3f}"
            if pd.notna(
                same_corr
            )
            else "N/A"
        ),
    )

    k5.metric(
        "Signal strength",
        strongest[
            "signal_strength"
        ],
    )

    st.markdown(
        "### Strongest copper-leading indicators"
    )

    table_view = top[
        [
            "category",
            "indicator",
            "lead_months",
            "correlation",
            "same_month_correlation",
            "correlation_gain",
            "signal_strength",
            "observations",
            "column",
        ]
    ].copy()

    table_view[
        "category"
    ] = table_view[
        "category"
    ].map(
        category_label
    )

    table_view.columns = [
        "Category",
        "Indicator",
        "Best Lead",
        "Lead Correlation",
        "Same-Month Correlation",
        "Lead Advantage",
        "Strength",
        "Observations",
        "Source Column",
    ]

    table_view[
        "Best Lead"
    ] = table_view[
        "Best Lead"
    ].apply(
        lambda value:
            f"+{int(value)}M"
    )

    st.dataframe(
        table_view.style.format(
            {
                "Lead Correlation":
                    "{:+.3f}",
                "Same-Month Correlation":
                    "{:+.3f}",
                "Lead Advantage":
                    "{:+.3f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
        height=480,
    )

    st.caption(
        "Lead Advantage = |best leading correlation| "
        "- |same-month correlation|. Positive values mean "
        "the indicator becomes more informative when shifted "
        "ahead of copper."
    )

    left, right = st.columns(
        [
            1.1,
            1.0,
        ]
    )

    with left:
        ranking_data = (
            top
            .sort_values(
                "abs_correlation",
                ascending=True,
            )
        )

        ranking_fig = go.Figure()

        ranking_fig.add_trace(
            go.Bar(
                x=ranking_data[
                    "correlation"
                ],
                y=ranking_data[
                    "indicator"
                ],
                orientation="h",
                customdata=np.column_stack(
                    [
                        ranking_data[
                            "lead_months"
                        ],
                        ranking_data[
                            "observations"
                        ],
                        ranking_data[
                            "category"
                        ],
                    ]
                ),
                marker=dict(
                    color=ranking_data[
                        "correlation"
                    ],
                    colorscale="RdBu",
                    cmid=0,
                ),
                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>"
                    "Correlation: %{x:+.3f}"
                    "<br>"
                    "Lead: +%{customdata[0]}M"
                    "<br>"
                    "Obs: %{customdata[1]}"
                    "<br>"
                    "Category: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

        ranking_fig.update_layout(
            title=(
                "Best leading correlation "
                "by indicator"
            ),
            height=max(
                430,
                32 * len(
                    ranking_data
                ),
            ),
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=25,
            ),
            xaxis_title=(
                "Correlation with future copper"
            ),
            yaxis_title="",
        )

        ranking_fig.update_xaxes(
            zeroline=True,
            zerolinecolor=(
                "rgba(148,163,184,0.5)"
            ),
            gridcolor=(
                "rgba(148,163,184,0.12)"
            ),
        )

        st.plotly_chart(
            ranking_fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        heatmap_features = (
            top[
                "column"
            ]
            .tolist()[
                :12
            ]
        )

        heatmap_rows = (
            lag_table.loc[
                lag_table[
                    "column"
                ].isin(
                    heatmap_features
                )
            ]
            .copy()
        )

        pivot = (
            heatmap_rows
            .pivot_table(
                index="column",
                columns="lead_months",
                values="correlation",
                aggfunc="first",
            )
            .reindex(
                heatmap_features
            )
        )

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[
                    f"+{int(value)}M"
                    for value
                    in pivot.columns
                ],
                y=[
                    pretty_name(
                        value
                    )
                    for value
                    in pivot.index
                ],
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                colorbar=dict(
                    title="Corr.",
                ),
                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>"
                    "Lead: %{x}"
                    "<br>"
                    "Correlation: %{z:+.3f}"
                    "<extra></extra>"
                ),
            )
        )

        heatmap_fig.update_layout(
            title=(
                "Correlation across lead months"
            ),
            height=max(
                430,
                34 * len(
                    pivot
                ),
            ),
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=25,
            ),
        )

        st.plotly_chart(
            heatmap_fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.markdown(
        "### Inspect one leading signal"
    )

    selected_column = (
        st.selectbox(
            "Indicator",
            top[
                "column"
            ].tolist(),
            format_func=pretty_name,
            key="signal_detail_indicator",
        )
    )

    selected_best = (
        best_rows.loc[
            best_rows[
                "column"
            ] == selected_column
        ]
        .iloc[0]
    )

    selected_lead = int(
        selected_best[
            "lead_months"
        ]
    )

    detail_metrics = st.columns(
        4
    )

    detail_metrics[0].metric(
        "Selected lead",
        f"+{selected_lead}M",
    )

    detail_metrics[1].metric(
        "Lead correlation",
        (
            f"{selected_best['correlation']:+.3f}"
        ),
    )

    selected_same = selected_best[
        "same_month_correlation"
    ]

    detail_metrics[2].metric(
        "Same-month corr.",
        (
            f"{selected_same:+.3f}"
            if pd.notna(
                selected_same
            )
            else "N/A"
        ),
    )

    detail_metrics[3].metric(
        "Category",
        category_label(
            selected_best[
                "category"
            ]
        ),
    )

    profile = (
        lag_table.loc[
            lag_table[
                "column"
            ] == selected_column
        ]
        .sort_values(
            "lead_months"
        )
    )

    profile_fig = go.Figure()

    profile_fig.add_trace(
        go.Scatter(
            x=profile[
                "lead_months"
            ],
            y=profile[
                "correlation"
            ],
            mode="lines+markers",
            name="Lead correlation",
            line=dict(
                width=3,
                color="#f59e0b",
            ),
            marker=dict(
                size=8,
            ),
            hovertemplate=(
                "Lead +%{x}M"
                "<br>"
                "Correlation: %{y:+.3f}"
                "<extra></extra>"
            ),
        )
    )

    profile_fig.add_hline(
        y=0,
        line_width=1,
        line_color=(
            "rgba(148,163,184,0.5)"
        ),
    )

    profile_fig.update_layout(
        title=(
            f"Lead profile - "
            f"{pretty_name(selected_column)}"
        ),
        height=350,
        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=25,
        ),
        xaxis_title=(
            "Months before copper"
        ),
        yaxis_title=(
            "Correlation"
        ),
    )

    profile_fig.update_xaxes(
        dtick=1,
        gridcolor=(
            "rgba(148,163,184,0.12)"
        ),
    )

    profile_fig.update_yaxes(
        range=[
            -1,
            1,
        ],
        gridcolor=(
            "rgba(148,163,184,0.12)"
        ),
    )

    st.plotly_chart(
        profile_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    feature_series = (
        _transform_for_signal_analysis(
            df[
                selected_column
            ],
            basis,
        )
    )

    copper_series = (
        _transform_for_signal_analysis(
            df[
                TARGET_COLUMN
            ],
            basis,
        )
    )

    detail = pd.DataFrame(
        {
            "date":
                df[
                    date_column
                ],
            "feature":
                feature_series,
            "future_copper":
                copper_series.shift(
                    -selected_lead
                ),
        }
    ).dropna()

    if detail.empty:
        return

    detail_left, detail_right = (
        st.columns(
            2
        )
    )

    with detail_left:
        comparison = detail.copy()

        feature_std = (
            comparison[
                "feature"
            ].std()
        )

        copper_std = (
            comparison[
                "future_copper"
            ].std()
        )

        if (
            pd.notna(
                feature_std
            )
            and feature_std != 0
            and pd.notna(
                copper_std
            )
            and copper_std != 0
        ):
            comparison[
                "feature_z"
            ] = (
                comparison[
                    "feature"
                ]
                - comparison[
                    "feature"
                ].mean()
            ) / feature_std

            comparison[
                "copper_z"
            ] = (
                comparison[
                    "future_copper"
                ]
                - comparison[
                    "future_copper"
                ].mean()
            ) / copper_std

            signal_fig = (
                go.Figure()
            )

            signal_fig.add_trace(
                go.Scatter(
                    x=comparison[
                        "date"
                    ],
                    y=comparison[
                        "feature_z"
                    ],
                    mode="lines",
                    name=pretty_name(
                        selected_column
                    ),
                    line=dict(
                        width=2.5,
                        color="#22d3ee",
                    ),
                )
            )

            signal_fig.add_trace(
                go.Scatter(
                    x=comparison[
                        "date"
                    ],
                    y=comparison[
                        "copper_z"
                    ],
                    mode="lines",
                    name=(
                        f"Copper "
                        f"{selected_lead}M later"
                    ),
                    line=dict(
                        width=2.5,
                        dash="dash",
                        color="#f59e0b",
                    ),
                )
            )

            _transparent_layout(
                signal_fig,
                height=470,
                y_title=(
                    "Standardized value "
                    "(z-score)"
                ),
            )

            signal_fig.update_layout(
                title=(
                    "Aligned signal view "
                    f"| Lead +{selected_lead}M"
                )
            )

            st.plotly_chart(
                signal_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    with detail_right:
        scatter_fig = (
            go.Figure()
        )

        scatter_fig.add_trace(
            go.Scatter(
                x=detail[
                    "feature"
                ],
                y=detail[
                    "future_copper"
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.75,
                    color="#f59e0b",
                ),
                text=detail[
                    "date"
                ].dt.strftime(
                    "%Y-%m"
                ),
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<br>"
                    "Indicator: %{x:,.4f}"
                    "<br>"
                    "Future copper: %{y:,.4f}"
                    "<extra></extra>"
                ),
            )
        )

        if len(detail) >= 3:
            x_values = detail[
                "feature"
            ].to_numpy()

            y_values = detail[
                "future_copper"
            ].to_numpy()

            if np.nanstd(
                x_values
            ) > 0:
                slope, intercept = (
                    np.polyfit(
                        x_values,
                        y_values,
                        1,
                    )
                )

                x_line = (
                    np.linspace(
                        np.nanmin(
                            x_values
                        ),
                        np.nanmax(
                            x_values
                        ),
                        100,
                    )
                )

                y_line = (
                    slope
                    * x_line
                    + intercept
                )

                scatter_fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Linear fit",
                        line=dict(
                            width=2,
                            dash="dot",
                            color="#22d3ee",
                        ),
                    )
                )

        scatter_fig.update_layout(
            title=(
                f"Indicator t vs "
                f"Copper t+{selected_lead}"
            ),
            height=470,
            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=30,
            ),
            xaxis_title=(
                pretty_name(
                    selected_column
                )
            ),
            yaxis_title=(
                f"Copper "
                f"{selected_lead} months later"
            ),
        )

        scatter_fig.update_xaxes(
            gridcolor=(
                "rgba(148,163,184,0.12)"
            ),
        )

        scatter_fig.update_yaxes(
            gridcolor=(
                "rgba(148,163,184,0.12)"
            ),
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.caption(
        "Lead-lag correlation is exploratory evidence, "
        "not proof of causality. Common trends, macro regimes, "
        "seasonality and publication timing can create apparent "
        "relationships. Production model decisions should still "
        "be validated with walk-forward backtests."
    )
