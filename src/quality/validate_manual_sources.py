import sys

import pandas as pd

from src.utils.paths import PROJECT_ROOT

REGISTRY_FILE = (
    PROJECT_ROOT
    / "config"
    / "manual_data_sources.csv"
)


REQUIRED_REGISTRY_COLUMNS = [
    "source_name",
    "file_path",
    "category",
    "source_mode",
    "frequency",
    "required",
    "ingestion_script",
    "official_source",
    "source_url",
    "notes",
]


def parse_required(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
    }


def load_registry():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Registry file not found: {REGISTRY_FILE}"
        )

    df = pd.read_csv(
        REGISTRY_FILE
    )

    missing_columns = [
        column
        for column in REQUIRED_REGISTRY_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Registry is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Manual source registry is empty."
        )

    return df


def validate_source(row):
    source_name = str(
        row["source_name"]
    ).strip()

    relative_path = str(
        row["file_path"]
    ).strip()

    source_mode = str(
        row["source_mode"]
    ).strip()

    required = parse_required(
        row["required"]
    )

    full_path = (
        PROJECT_ROOT
        / relative_path
    )

    result = {
        "source_name": source_name,
        "source_mode": source_mode,
        "file_path": relative_path,
        "required": required,
        "exists": False,
        "readable": False,
        "size_bytes": None,
        "status": "PASS",
        "detail": "",
    }

    if not full_path.exists():
        result["status"] = (
            "FAIL"
            if required
            else "WARNING"
        )

        result["detail"] = (
            "Required source file does not exist."
            if required
            else "Optional source file does not exist."
        )

        return result

    result["exists"] = True

    if not full_path.is_file():
        result["status"] = "FAIL"

        result["detail"] = (
            "Configured path exists but is not a file."
        )

        return result

    file_size = full_path.stat().st_size

    result["size_bytes"] = file_size

    if file_size <= 0:
        result["status"] = "FAIL"

        result["detail"] = (
            "Source file exists but is empty."
        )

        return result

    try:
        with full_path.open("rb") as file_handle:
            file_handle.read(1)

        result["readable"] = True

    except Exception as exc:
        result["status"] = "FAIL"

        result["detail"] = (
            f"Source file is not readable: {exc}"
        )

        return result

    suffix = full_path.suffix.lower()

    try:
        if suffix == ".csv":
            sample_df = pd.read_csv(
                full_path,
                nrows=5,
            )

        elif suffix in {
            ".xlsx",
            ".xls",
        }:
            sample_df = pd.read_excel(
                full_path,
                nrows=5,
            )

        else:
            result["status"] = "WARNING"

            result["detail"] = (
                f"Unsupported validation extension: {suffix}"
            )

            return result

    except Exception as exc:
        result["status"] = "FAIL"

        result["detail"] = (
            f"File cannot be parsed: {exc}"
        )

        return result

    if sample_df.empty:
        result["status"] = "FAIL"

        result["detail"] = (
            "File can be opened but contains no data rows."
        )

        return result

    result["detail"] = (
        "Source file exists, is readable, and can be parsed."
    )

    return result


def main():
    print(
        "=" * 100
    )

    print(
        "MANUAL SOURCE VALIDATION"
    )

    print(
        "=" * 100
    )

    registry_df = load_registry()

    results = []

    for _, row in registry_df.iterrows():
        result = validate_source(
            row
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n[INFO] Validation results:\n"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    fail_count = (
        result_df["status"]
        .eq("FAIL")
        .sum()
    )

    warning_count = (
        result_df["status"]
        .eq("WARNING")
        .sum()
    )

    pass_count = (
        result_df["status"]
        .eq("PASS")
        .sum()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"PASS    : {pass_count}"
    )

    print(
        f"WARNING : {warning_count}"
    )

    print(
        f"FAIL    : {fail_count}"
    )

    if fail_count > 0:
        print(
            "\n[FAIL] Manual source validation failed."
        )

        sys.exit(1)

    print(
        "\n[OK] Manual source validation passed."
    )


if __name__ == "__main__":
    main()
