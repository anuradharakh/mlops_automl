"""Remove corrupted target outliers and rebuild fixed splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from athlete_automl.data.target_quality import (
    filter_valid_target_rows,
    identify_target_outliers,
    split_dataset,
    verify_split_integrity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "target_quality.yaml"


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def main() -> None:
    """Run the target-quality gate."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    target_config = config["target"]
    data_config = config["data"]
    split_config = config["split"]
    reports_config = config["reports"]

    processed_path = resolve_path(
        data_config["processed_parquet_path"]
    )
    if not processed_path.exists():
        raise FileNotFoundError(
            "Run scripts/prepare_data.py first."
        )

    dataframe = pd.read_parquet(processed_path)

    identifier_column = next(
        (
            column
            for column in ("athlete_id", "record_id")
            if column in dataframe.columns
        ),
        None,
    )
    if identifier_column is None:
        raise ValueError(
            "Expected athlete_id or record_id."
        )

    target_column = target_config["name"]
    minimum = float(target_config["minimum"])
    maximum = float(target_config["maximum"])

    audit = identify_target_outliers(
        dataframe,
        target_column,
        minimum,
        maximum,
    )
    cleaned = filter_valid_target_rows(
        dataframe,
        target_column,
        minimum,
        maximum,
    )

    splits = split_dataset(
        cleaned,
        float(split_config["train_size"]),
        float(split_config["validation_size"]),
        float(split_config["test_size"]),
        int(split_config["random_state"]),
    )
    verify_split_integrity(
        splits,
        identifier_column,
    )

    processed_csv_path = resolve_path(
        data_config["processed_csv_path"]
    )
    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cleaned.to_parquet(
        processed_path,
        index=False,
    )
    cleaned.to_csv(
        processed_csv_path,
        index=False,
    )

    split_paths = {
        "train": resolve_path(data_config["train_path"]),
        "validation": resolve_path(
            data_config["validation_path"]
        ),
        "test": resolve_path(data_config["test_path"]),
    }

    for name, split in splits.items():
        split_paths[name].parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        split.to_parquet(
            split_paths[name],
            index=False,
        )

    audit_path = resolve_path(
        reports_config["audit_path"]
    )
    summary_path = resolve_path(
        reports_config["summary_path"]
    )
    split_summary_path = resolve_path(
        reports_config["split_summary_path"]
    )
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    summary = {
        "status": "PASS",
        "target_column": target_column,
        "valid_range": {
            "minimum": minimum,
            "maximum": maximum,
        },
        "rows_before": int(len(dataframe)),
        "rows_removed": int(len(audit)),
        "rows_after": int(len(cleaned)),
        "removal_reason_counts": {
            str(reason): int(count)
            for reason, count in audit[
                "target_quality_reason"
            ].value_counts().items()
        },
        "target_statistics_after_cleaning": {
            "minimum": float(cleaned[target_column].min()),
            "maximum": float(cleaned[target_column].max()),
            "mean": float(cleaned[target_column].mean()),
            "median": float(cleaned[target_column].median()),
            "standard_deviation": float(
                cleaned[target_column].std()
            ),
            "p99": float(
                cleaned[target_column].quantile(0.99)
            ),
        },
        "split_counts": {
            name: int(len(split))
            for name, split in splits.items()
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "split": name,
                "row_count": int(len(split)),
                "target_mean": float(
                    split[target_column].mean()
                ),
                "target_median": float(
                    split[target_column].median()
                ),
                "target_std": float(
                    split[target_column].std()
                ),
                "target_maximum": float(
                    split[target_column].max()
                ),
            }
            for name, split in splits.items()
        ]
    ).to_csv(
        split_summary_path,
        index=False,
    )

    print("Target-quality cleanup completed.")
    print(f"Rows before: {len(dataframe):,}")
    print(f"Rows removed: {len(audit):,}")
    print(f"Rows after: {len(cleaned):,}")
    print(
        "Clean target range: "
        f"{cleaned[target_column].min():,.2f} to "
        f"{cleaned[target_column].max():,.2f}"
    )
    print(
        "Split counts: "
        + ", ".join(
            f"{name}={len(split):,}"
            for name, split in splits.items()
        )
    )
    print("PHASE 1D STATUS: PASS")


if __name__ == "__main__":
    main()
