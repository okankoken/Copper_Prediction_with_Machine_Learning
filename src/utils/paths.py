from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ------------------------------------------------------------------
# Project-level directories
# ------------------------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

CONFIG_DIR = (
    PROJECT_ROOT
    / "config"
)

LOGS_DIR = (
    PROJECT_ROOT
    / "logs"
)

NOTEBOOKS_DIR = (
    PROJECT_ROOT
    / "notebooks"
)


# ------------------------------------------------------------------
# Main data directories
# ------------------------------------------------------------------

RAW_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

PREDICTIONS_DIR = (
    DATA_DIR
    / "predictions"
)

SOURCE_DIR = (
    DATA_DIR
    / "source"
)


# ------------------------------------------------------------------
# Raw data category directories
# ------------------------------------------------------------------

MARKET_RAW_DIR = (
    RAW_DIR
    / "market"
)

MACRO_RAW_DIR = (
    RAW_DIR
    / "macro"
)

MINING_RAW_DIR = (
    RAW_DIR
    / "mining"
)

SHIPPING_RAW_DIR = (
    RAW_DIR
    / "shipping"
)

RISK_RAW_DIR = (
    RAW_DIR
    / "risk"
)

EQUITIES_RAW_DIR = (
    RAW_DIR
    / "equities"
)

ENERGY_TRANSITION_RAW_DIR = (
    RAW_DIR
    / "energy_transition"
)


# ------------------------------------------------------------------
# Source data directories
# ------------------------------------------------------------------

COCHILCO_SOURCE_DIR = (
    SOURCE_DIR
    / "cochilco"
)

PERU_SOURCE_DIR = (
    SOURCE_DIR
    / "peru"
)


# ------------------------------------------------------------------
# Processed data directories
# ------------------------------------------------------------------

QUALITY_DIR = (
    PROCESSED_DIR
    / "quality"
)

DIAGNOSTICS_DIR = (
    PROCESSED_DIR
    / "diagnostics"
)

MONTHLY_DIR = (
    PROCESSED_DIR
    / "monthly"
)

FEATURES_DIR = (
    PROCESSED_DIR
    / "features"
)


def ensure_directories():
    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        PREDICTIONS_DIR,
        SOURCE_DIR,
        MARKET_RAW_DIR,
        MACRO_RAW_DIR,
        MINING_RAW_DIR,
        SHIPPING_RAW_DIR,
        RISK_RAW_DIR,
        EQUITIES_RAW_DIR,
        ENERGY_TRANSITION_RAW_DIR,
        COCHILCO_SOURCE_DIR,
        PERU_SOURCE_DIR,
        QUALITY_DIR,
        DIAGNOSTICS_DIR,
        MONTHLY_DIR,
        FEATURES_DIR,
        CONFIG_DIR,
        LOGS_DIR,
        NOTEBOOKS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
