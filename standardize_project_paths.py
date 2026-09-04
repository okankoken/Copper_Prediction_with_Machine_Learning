from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(
    "/home/train/Copper_Prediction_with_Machine_Learning"
)

BACKUP_DIR = (
    PROJECT_ROOT
    / ".path_backups"
    / "path_standardization_backup"
)

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def backup(path):
    relative = path.relative_to(
        PROJECT_ROOT
    )

    destination = (
        BACKUP_DIR
        / relative
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        path,
        destination,
    )


def add_import(
    text,
    import_line,
):
    if import_line in text:
        return text

    matches = list(
        re.finditer(
            r'^(?:from|import)\s+.*$',
            text,
            flags=re.MULTILINE,
        )
    )

    if not matches:
        raise RuntimeError(
            "Import block not found"
        )

    position = matches[-1].end()

    return (
        text[:position]
        + "\n"
        + import_line
        + "\n"
        + text[position:]
    )


def remove_multiline_assignment(
    text,
    variable,
):
    pattern = (
        rf'^\s*{re.escape(variable)}\s*=\s*'
        rf'(?:Path\([^\n]*\)|'
        rf'Path\(\s*\n(?:.*\n)*?\)|'
        rf'\(\s*\n(?:.*\n)*?\)|'
        rf'[^\n]+)'
        rf'\s*\n'
    )

    return re.sub(
        pattern,
        "",
        text,
        flags=re.MULTILINE,
    )


def set_simple_path(
    text,
    variable,
    constant,
    filename,
):
    pattern = (
        rf'^\s*{re.escape(variable)}\s*=\s*'
        rf'(?:Path\([^\n]*\)|'
        rf'Path\(\s*\n(?:.*\n)*?\)|'
        rf'\(\s*\n(?:.*\n)*?\)|'
        rf'[^\n]+)'
        rf'\s*\n'
    )

    replacement = (
        f'{variable} = (\n'
        f'    {constant}\n'
        f'    / "{filename}"\n'
        f')\n'
    )

    new_text, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if count == 0:
        raise RuntimeError(
            f"Could not replace {variable}"
        )

    return new_text


def write_file(path, text):
    path.write_text(
        text,
        encoding="utf-8",
    )


# ================================================================
# INGESTION
# ================================================================

INGESTION_SIMPLE = {
    "src/ingestion/ingest_economic_policy_uncertainty.py": (
        "RISK_RAW_DIR",
        "OUTPUT_FILE",
        "economic_policy_uncertainty_monthly.csv",
    ),
    "src/ingestion/ingest_fred.py": (
        "MACRO_RAW_DIR",
        "OUTPUT_FILE",
        "fred_macro_monthly.csv",
    ),
    "src/ingestion/ingest_geopolitical_risk.py": (
        "RISK_RAW_DIR",
        "OUTPUT_FILE",
        "geopolitical_risk_monthly.csv",
    ),
    "src/ingestion/ingest_global_leading_indicators.py": (
        "MACRO_RAW_DIR",
        "OUTPUT_FILE",
        "global_leading_indicators_monthly.csv",
    ),
    "src/ingestion/ingest_global_macro.py": (
        "MACRO_RAW_DIR",
        "OUTPUT_FILE",
        "global_macro_monthly.csv",
    ),
    "src/ingestion/ingest_peru_copper_cost_drivers.py": (
        "MINING_RAW_DIR",
        "OUTPUT_FILE",
        "peru_copper_cost_drivers_monthly.csv",
    ),
    "src/ingestion/ingest_portwatch.py": (
        "SHIPPING_RAW_DIR",
        "OUTPUT_FILE",
        "portwatch_shipping_activity_daily.csv",
    ),
    "src/ingestion/ingest_shfe.py": (
        "MARKET_RAW_DIR",
        "OUTPUT_PATH",
        "shfe_copper_daily.csv",
    ),
    "src/ingestion/ingest_worldbank_commodities.py": (
        "MARKET_RAW_DIR",
        "OUTPUT_FILE",
        "worldbank_commodities_monthly.csv",
    ),
}


for filename, (
    constant,
    variable,
    csv_name,
) in INGESTION_SIMPLE.items():

    path = PROJECT_ROOT / filename
    backup(path)

    text = path.read_text(
        encoding="utf-8"
    )

    text = add_import(
        text,
        f"from src.utils.paths import {constant}",
    )

    text = set_simple_path(
        text,
        variable,
        constant,
        csv_name,
    )

    text = remove_multiline_assignment(
        text,
        "PROJECT_ROOT",
    )

    text = remove_multiline_assignment(
        text,
        "BASE_DIR",
    )

    text = remove_multiline_assignment(
        text,
        "RAW_DIR",
    )

    write_file(
        path,
        text,
    )

    print(
        f"[OK] {filename}"
    )


# ================================================================
# ENERGY TRANSITION
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_energy_transition.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import ENERGY_TRANSITION_RAW_DIR",
)

text = set_simple_path(
    text,
    "OUTPUT_PATH",
    "ENERGY_TRANSITION_RAW_DIR",
    "energy_transition_annual.csv",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_energy_transition.py"
)


# ================================================================
# TURKEY
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_turkey.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import MACRO_RAW_DIR",
)

text = set_simple_path(
    text,
    "OUTPUT_PATH",
    "MACRO_RAW_DIR",
    "turkey_monthly.csv",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_turkey.py"
)


# ================================================================
# METALS
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_metals.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import MARKET_RAW_DIR",
)

text = set_simple_path(
    text,
    "PALLADIUM_OUTPUT_FILE",
    "MARKET_RAW_DIR",
    "palladium_daily.csv",
)

text = set_simple_path(
    text,
    "OUTPUT_FILE",
    "MARKET_RAW_DIR",
    "lme_other_metals_daily.csv",
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_metals.py"
)


# ================================================================
# CHILE
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_chile_cochilco.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    (
        "from src.utils.paths import "
        "COCHILCO_SOURCE_DIR, MINING_RAW_DIR"
    ),
)

text = set_simple_path(
    text,
    "SOURCE_FILE",
    "COCHILCO_SOURCE_DIR",
    "chile_cochilco_yearbook_2005_2024.xlsx",
)

text = set_simple_path(
    text,
    "OUTPUT_FILE",
    "MINING_RAW_DIR",
    "chile_cochilco_copper_cost_annual.csv",
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_chile_cochilco.py"
)


# ================================================================
# PERU MINING
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_peru_copper_mining.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    (
        "from src.utils.paths import "
        "MINING_RAW_DIR, PERU_SOURCE_DIR"
    ),
)

text = set_simple_path(
    text,
    "EMPLOYMENT_FILE",
    "PERU_SOURCE_DIR",
    "peru_mining_employment_2020_2026.xlsx",
)

text = set_simple_path(
    text,
    "OUTPUT_FILE",
    "MINING_RAW_DIR",
    "peru_copper_mining_annual.csv",
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_peru_copper_mining.py"
)


# ================================================================
# COUNTRY STOCK INDICES
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_country_stock_indices.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import EQUITIES_RAW_DIR",
)

text = set_simple_path(
    text,
    "OUTPUT_FILE",
    "EQUITIES_RAW_DIR",
    "country_stock_indices_monthly.csv",
)

text = remove_multiline_assignment(
    text,
    "BASE_DIR",
)

text = remove_multiline_assignment(
    text,
    "RAW_DIR",
)

text = text.replace(
    "RAW_DIR\n        / \"Chile_IPSA_Historical_Data.csv\"",
    "EQUITIES_RAW_DIR\n        / \"Chile_IPSA_Historical_Data.csv\"",
)

text = text.replace(
    "RAW_DIR\n        / \"China_CSI300_Historical_Data.csv\"",
    "EQUITIES_RAW_DIR\n        / \"China_CSI300_Historical_Data.csv\"",
)

text = text.replace(
    "RAW_DIR\n        / \"Poland_WIG20_Historical_Data.csv\"",
    "EQUITIES_RAW_DIR\n        / \"Poland_WIG20_Historical_Data.csv\"",
)

text = text.replace(
    "RAW_DIR.mkdir(",
    "EQUITIES_RAW_DIR.mkdir(",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_country_stock_indices.py"
)


# ================================================================
# BDI
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_bdi.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import SHIPPING_RAW_DIR",
)

text = set_simple_path(
    text,
    "SOURCE_FILE",
    "SHIPPING_RAW_DIR",
    "Baltic Dry Index Geçmiş Verileri.csv",
)

text = set_simple_path(
    text,
    "OUTPUT_FILE",
    "SHIPPING_RAW_DIR",
    "baltic_dry_index_daily.csv",
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

text = remove_multiline_assignment(
    text,
    "RAW_DIR",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_bdi.py"
)


# ================================================================
# CHINA
# ================================================================

path = (
    PROJECT_ROOT
    / "src/ingestion/ingest_china.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    (
        "from src.utils.paths import "
        "MACRO_RAW_DIR, MARKET_RAW_DIR"
    ),
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

text = remove_multiline_assignment(
    text,
    "RAW_DIR",
)

# Add explicit target directory to each dataset.
text = text.replace(
    '"output_file": "china_refined_copper_monthly.csv",',
    (
        '"output_file": "china_refined_copper_monthly.csv",\n'
        '        "output_dir": MARKET_RAW_DIR,'
    ),
)

for csv_name in [
    "china_industrial_output_monthly.csv",
    "china_ppi_monthly.csv",
    "china_electricity_generation_monthly.csv",
    "china_fixed_asset_investment_monthly.csv",
    "china_real_estate_investment_monthly.csv",
]:
    text = text.replace(
        f'"output_file": "{csv_name}",',
        (
            f'"output_file": "{csv_name}",\n'
            '        "output_dir": MACRO_RAW_DIR,'
        ),
    )

# Replace any RAW_DIR based output construction.
text = re.sub(
    (
        r'RAW_DIR\s*/\s*'
        r'(?:config|dataset_config|dataset_info|info)'
        r'\["output_file"\]'
    ),
    (
        r'\1["output_dir"]'
        r' / \1["output_file"]'
    ),
    text,
)

# Handle a generic loop variable if used.
text = text.replace(
    'RAW_DIR / dataset["output_file"]',
    'dataset["output_dir"] / dataset["output_file"]',
)

text = text.replace(
    'RAW_DIR / config["output_file"]',
    'config["output_dir"] / config["output_file"]',
)

text = text.replace(
    'RAW_DIR / info["output_file"]',
    'info["output_dir"] / info["output_file"]',
)

write_file(
    path,
    text,
)

print(
    "[OK] src/ingestion/ingest_china.py"
)


# ================================================================
# QUALITY SIMPLE INPUTS
# ================================================================

QUALITY_SIMPLE = {
    "src/quality/quality_bdi.py": (
        "SHIPPING_RAW_DIR",
        "BDI_FILE",
        "baltic_dry_index_daily.csv",
    ),
    "src/quality/quality_copper_company_stocks.py": (
        "EQUITIES_RAW_DIR",
        "DATA_FILE",
        "copper_company_stocks_monthly.csv",
    ),
    "src/quality/quality_country_stock_indices.py": (
        "EQUITIES_RAW_DIR",
        "DATA_FILE",
        "country_stock_indices_monthly.csv",
    ),
    "src/quality/quality_economic_policy_uncertainty.py": (
        "RISK_RAW_DIR",
        "DATA_FILE",
        "economic_policy_uncertainty_monthly.csv",
    ),
    "src/quality/quality_geopolitical_risk.py": (
        "RISK_RAW_DIR",
        "DATA_FILE",
        "geopolitical_risk_monthly.csv",
    ),
    "src/quality/quality_global_leading_indicators.py": (
        "MACRO_RAW_DIR",
        "DATA_FILE",
        "global_leading_indicators_monthly.csv",
    ),
    "src/quality/quality_global_macro.py": (
        "MACRO_RAW_DIR",
        "INPUT_FILE",
        "global_macro_monthly.csv",
    ),
    "src/quality/quality_portwatch.py": (
        "SHIPPING_RAW_DIR",
        "DATA_FILE",
        "portwatch_shipping_activity_daily.csv",
    ),
    "src/quality/quality_shfe.py": (
        "MARKET_RAW_DIR",
        "INPUT_PATH",
        "shfe_copper_daily.csv",
    ),
    "src/quality/quality_turkey.py": (
        "MACRO_RAW_DIR",
        "INPUT_PATH",
        "turkey_monthly.csv",
    ),
    "src/quality/quality_worldbank_commodities.py": (
        "MARKET_RAW_DIR",
        "INPUT_FILE",
        "worldbank_commodities_monthly.csv",
    ),
    "src/quality/quality_energy_transition.py": (
        "ENERGY_TRANSITION_RAW_DIR",
        "INPUT_PATH",
        "energy_transition_annual.csv",
    ),
}


for filename, (
    constant,
    variable,
    csv_name,
) in QUALITY_SIMPLE.items():

    path = PROJECT_ROOT / filename
    backup(path)

    text = path.read_text(
        encoding="utf-8"
    )

    text = add_import(
        text,
        f"from src.utils.paths import {constant}",
    )

    text = set_simple_path(
        text,
        variable,
        constant,
        csv_name,
    )

    text = remove_multiline_assignment(
        text,
        "PROJECT_ROOT",
    )

    text = remove_multiline_assignment(
        text,
        "BASE_DIR",
    )

    write_file(
        path,
        text,
    )

    print(
        f"[OK] {filename}"
    )


# ================================================================
# QUALITY METALS
# ================================================================

path = (
    PROJECT_ROOT
    / "src/quality/quality_metals.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    "from src.utils.paths import MARKET_RAW_DIR",
)

text = set_simple_path(
    text,
    "INPUT_FILE",
    "MARKET_RAW_DIR",
    "lme_other_metals_daily.csv",
)

text = set_simple_path(
    text,
    "PALLADIUM_FILE",
    "MARKET_RAW_DIR",
    "palladium_daily.csv",
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

write_file(
    path,
    text,
)

print(
    "[OK] src/quality/quality_metals.py"
)


# ================================================================
# QUALITY CHINA
# ================================================================

path = (
    PROJECT_ROOT
    / "src/quality/quality_china.py"
)

backup(path)

text = path.read_text(
    encoding="utf-8"
)

text = add_import(
    text,
    (
        "from src.utils.paths import "
        "MACRO_RAW_DIR, MARKET_RAW_DIR"
    ),
)

text = remove_multiline_assignment(
    text,
    "PROJECT_ROOT",
)

text = remove_multiline_assignment(
    text,
    "RAW_DIR",
)

replacements = {
    (
        'RAW_DIR\n'
        '        / "china_refined_copper_monthly.csv"'
    ): (
        'MARKET_RAW_DIR\n'
        '        / "china_refined_copper_monthly.csv"'
    ),
    (
        'RAW_DIR\n'
        '        / "china_industrial_output_monthly.csv"'
    ): (
        'MACRO_RAW_DIR\n'
        '        / "china_industrial_output_monthly.csv"'
    ),
    (
        'RAW_DIR\n'
        '        / "china_ppi_monthly.csv"'
    ): (
        'MACRO_RAW_DIR\n'
        '        / "china_ppi_monthly.csv"'
    ),
    (
        'RAW_DIR\n'
        '        / "china_electricity_generation_monthly.csv"'
    ): (
        'MACRO_RAW_DIR\n'
        '        / "china_electricity_generation_monthly.csv"'
    ),
    (
        'RAW_DIR\n'
        '        / "china_fixed_asset_investment_monthly.csv"'
    ): (
        'MACRO_RAW_DIR\n'
        '        / "china_fixed_asset_investment_monthly.csv"'
    ),
    (
        'RAW_DIR\n'
        '        / "china_real_estate_investment_monthly.csv"'
    ): (
        'MACRO_RAW_DIR\n'
        '        / "china_real_estate_investment_monthly.csv"'
    ),
}


for old, new in replacements.items():
    text = text.replace(
        old,
        new,
    )


write_file(
    path,
    text,
)

print(
    "[OK] src/quality/quality_china.py"
)


print()
print("=" * 80)
print("PATH STANDARDIZATION COMPLETED")
print("=" * 80)
print(
    "Backup directory:",
    BACKUP_DIR,
)
