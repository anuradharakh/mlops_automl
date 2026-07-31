"""Export the fixed local splits for Databricks AutoML."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from athlete_automl.databricks.export import (
    combine_splits,
    find_identifier_column,
    sha256_file,
    validate_export_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "databricks.yaml"


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def main() -> None:
    """Create CSV and Parquet exports with a fixed split column."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    schema_config = config["schema"]
    split_config = config["splits"]

    split_paths = {
        split_config["train_label"]: resolve_path(data_config["train_path"]),
        split_config["validation_label"]: resolve_path(data_config["validation_path"]),
        split_config["test_label"]: resolve_path(data_config["test_path"]),
    }

    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in split_paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Run Phase 1C first. Missing split files: " + ", ".join(missing_files)
        )

    split_frames = {label: pd.read_parquet(path) for label, path in split_paths.items()}

    combined = combine_splits(
        split_frames=split_frames,
        split_column=schema_config["split_column"],
    )

    identifier_column = find_identifier_column(
        columns=list(combined.columns),
        candidates=list(schema_config["identifier_candidates"]),
    )

    validate_export_dataset(
        dataframe=combined,
        identifier_column=identifier_column,
        target_column=schema_config["target_column"],
        split_column=schema_config["split_column"],
    )

    csv_path = resolve_path(data_config["export_csv_path"])
    parquet_path = resolve_path(data_config["export_parquet_path"])
    report_path = resolve_path(config["reports"]["export_summary_path"])

    for path in (csv_path, parquet_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    excluded_from_training = {
        identifier_column,
        schema_config["target_column"],
        schema_config["split_column"],
    }
    model_features = [
        column for column in combined.columns if column not in excluded_from_training
    ]

    split_counts = {
        str(label): int(count)
        for label, count in combined[schema_config["split_column"]]
        .value_counts()
        .sort_index()
        .items()
    }

    summary = {
        "status": "PASS",
        "row_count": int(len(combined)),
        "column_count": int(combined.shape[1]),
        "identifier_column": identifier_column,
        "target_column": schema_config["target_column"],
        "split_column": schema_config["split_column"],
        "split_counts": split_counts,
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
        "csv_sha256": sha256_file(csv_path),
        "csv_size_bytes": int(csv_path.stat().st_size),
        "parquet_path": str(parquet_path.relative_to(PROJECT_ROOT)),
        "parquet_sha256": sha256_file(parquet_path),
        "parquet_size_bytes": int(parquet_path.stat().st_size),
    }

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Databricks AutoML dataset export completed.")
    print(f"Rows: {len(combined):,}")
    print(f"Model features: {len(model_features)}")
    print(f"Split counts: {split_counts}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Report: {report_path}")
    print("PHASE 2A STATUS: PASS")


if __name__ == "__main__":
    main()
