from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.paths import PROJECT_ROOT


MONTHLY_MASTER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly"
    / "copper_monthly_master.csv"
)

MODEL_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "copper_model_features.csv"
)

MULTI_HORIZON_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "copper_multi_horizon_features.csv"
)

MODEL_FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "copper_model_feature_manifest.csv"
)

MONTHLY_ARCHIVE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "monthly"
    / "archive"
)

FEATURE_ARCHIVE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "archive"
)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_source_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Input path is not a file: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Input file is empty: {file_path}"
        )


def get_forecast_origin() -> str:
    validate_source_file(
        MONTHLY_MASTER
    )

    df = pd.read_csv(
        MONTHLY_MASTER,
        usecols=["date"],
    )

    if df.empty:
        raise ValueError(
            "Monthly master is empty."
        )

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    dates = dates.dropna()

    if dates.empty:
        raise ValueError(
            "No valid dates found in monthly master."
        )

    latest_date = dates.max()

    forecast_origin = (
        latest_date
        .to_period("M")
        .to_timestamp(how="end")
        .normalize()
    )

    return forecast_origin.strftime(
        "%Y-%m-%d"
    )


def archive_file(
    source_file: Path,
    archive_dir: Path,
    forecast_origin: str,
    timestamp: str,
) -> Path:
    validate_source_file(
        source_file
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_file = (
        archive_dir
        / (
            f"{source_file.stem}_"
            f"{forecast_origin}_"
            f"{timestamp}"
            f"{source_file.suffix}"
        )
    )

    shutil.copy2(
        source_file,
        destination_file,
    )

    validate_source_file(
        destination_file
    )

    source_size = (
        source_file.stat().st_size
    )

    destination_size = (
        destination_file.stat().st_size
    )

    if source_size != destination_size:
        raise RuntimeError(
            "Archive size validation failed: "
            f"{source_file.name}"
        )

    source_hash = calculate_sha256(
        source_file
    )

    destination_hash = calculate_sha256(
        destination_file
    )

    if source_hash != destination_hash:
        raise RuntimeError(
            "Archive SHA256 validation failed: "
            f"{source_file.name}"
        )

    print(
        f"[OK] Archived: {destination_file}"
    )

    print(
        f"[INFO] Size bytes: {destination_size}"
    )

    print(
        f"[INFO] SHA256: {destination_hash}"
    )

    return destination_file


def main() -> None:
    print(
        "=" * 100
    )

    print(
        "MODEL INPUT ARCHIVE"
    )

    print(
        "=" * 100
    )

    forecast_origin = (
        get_forecast_origin()
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S"
        )
    )

    print(
        f"[INFO] Forecast origin: "
        f"{forecast_origin}"
    )

    print(
        f"[INFO] Archive timestamp: "
        f"{timestamp}"
    )

    archive_jobs = [
        (
            MONTHLY_MASTER,
            MONTHLY_ARCHIVE_DIR,
        ),
        (
            MODEL_FEATURES,
            FEATURE_ARCHIVE_DIR,
        ),
        (
            MULTI_HORIZON_FEATURES,
            FEATURE_ARCHIVE_DIR,
        ),
        (
            MODEL_FEATURE_MANIFEST,
            FEATURE_ARCHIVE_DIR,
        ),
    ]

    archived_files = []

    for (
        source_file,
        archive_dir,
    ) in archive_jobs:
        archived_file = archive_file(
            source_file=source_file,
            archive_dir=archive_dir,
            forecast_origin=forecast_origin,
            timestamp=timestamp,
        )

        archived_files.append(
            archived_file
        )

    print()

    print(
        "=" * 100
    )

    print(
        "ARCHIVE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"[OK] Archived files: "
        f"{len(archived_files)}"
    )

    for file_path in archived_files:
        print(
            f"[OK] {file_path}"
        )

    print()

    print(
        "[DONE] Model input archive completed successfully."
    )


if __name__ == "__main__":
    main()