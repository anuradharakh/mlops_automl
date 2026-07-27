"""Dataset profiling utilities for the athlete AutoML workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def calculate_file_hash(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def convert_to_json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values to JSON-compatible values."""
    if value is None or value is pd.NA:
        return None

    if isinstance(value, dict):
        return {str(key): convert_to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [convert_to_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return convert_to_json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and np.isnan(value):
        return None

    return value


def classify_column(
    column: str,
    target_name: str,
    target_components: set[str],
    identifier_columns: set[str],
    metadata_columns: set[str],
) -> str:
    """Assign the initial role for one raw column."""
    if column == target_name:
        return "target"

    if column in target_components:
        return "target_component"

    if column in identifier_columns:
        return "identifier"

    if column in metadata_columns:
        return "metadata"

    return "candidate_feature"


def build_column_profile(
    dataframe: pd.DataFrame,
    target_name: str,
    target_components: list[str],
    identifier_columns: list[str],
    metadata_columns: list[str],
    sample_value_count: int = 3,
) -> pd.DataFrame:
    """Create a column-level schema and quality profile."""
    records: list[dict[str, Any]] = []

    target_component_set = set(target_components)

    identifier_set = set(identifier_columns)

    metadata_set = set(metadata_columns)

    for column in dataframe.columns:
        series = dataframe[column]

        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        unique_count = int(non_null.nunique(dropna=True))

        unique_ratio = unique_count / non_null_count if non_null_count else 0.0

        if pd.api.types.is_numeric_dtype(series):
            numeric_parse_rate = 1.0
        else:
            numeric_values = pd.to_numeric(
                series,
                errors="coerce",
            )

            numeric_parse_rate = float(numeric_values.notna().mean())

        sample_values = [
            str(value)[:100]
            for value in non_null.drop_duplicates().head(sample_value_count).tolist()
        ]

        role = classify_column(
            column=column,
            target_name=target_name,
            target_components=(target_component_set),
            identifier_columns=(identifier_set),
            metadata_columns=metadata_set,
        )

        is_constant = unique_count <= 1

        is_high_cardinality = bool(unique_count > 100 and unique_ratio > 0.50)

        records.append(
            {
                "column": column,
                "raw_dtype": str(series.dtype),
                "role": role,
                "row_count": int(len(series)),
                "non_null_count": (non_null_count),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(
                    float(series.isna().mean() * 100),
                    4,
                ),
                "unique_count": unique_count,
                "unique_ratio": round(
                    unique_ratio,
                    6,
                ),
                "numeric_parse_rate": round(
                    numeric_parse_rate,
                    6,
                ),
                "is_constant": is_constant,
                "is_high_cardinality": (is_high_cardinality),
                "sample_values": " | ".join(sample_values),
            }
        )

    return pd.DataFrame(records)


def build_target_readiness_summary(
    dataframe: pd.DataFrame,
    target_components: list[str],
    sentinel_values: list[float | int],
    identifier_column: str | None,
) -> dict[str, Any]:
    """Assess how many rows can produce a valid target."""
    missing_components = set(target_components).difference(dataframe.columns)

    if missing_components:
        raise ValueError(
            f"Raw dataset is missing target components: {sorted(missing_components)}"
        )

    numeric_components = dataframe[target_components].apply(
        pd.to_numeric,
        errors="coerce",
    )

    sentinel_counts = {
        column: int(numeric_components[column].isin(sentinel_values).sum())
        for column in target_components
    }

    cleaned_components = numeric_components.mask(
        numeric_components.isin(sentinel_values)
    )

    complete_target_mask = cleaned_components.notna().all(axis=1)

    if identifier_column and identifier_column in dataframe.columns:
        identifier_valid_mask = dataframe[identifier_column].notna()
    else:
        identifier_valid_mask = pd.Series(
            True,
            index=dataframe.index,
        )

    eligible_mask = identifier_valid_mask & complete_target_mask

    target_values = cleaned_components.sum(
        axis=1,
        min_count=len(target_components),
    )

    eligible_target = target_values[eligible_mask]

    invalid_target_rows = int((identifier_valid_mask & ~complete_target_mask).sum())

    return {
        "initial_rows": int(len(dataframe)),
        "missing_identifier_rows": int((~identifier_valid_mask).sum()),
        "missing_or_invalid_target_rows": (invalid_target_rows),
        "eligible_rows": int(eligible_mask.sum()),
        "sentinel_values": list(sentinel_values),
        "sentinel_counts": (sentinel_counts),
        "total_sentinel_replacements": int(sum(sentinel_counts.values())),
        "target_statistics": {
            "count": int(eligible_target.notna().sum()),
            "minimum": float(eligible_target.min()),
            "maximum": float(eligible_target.max()),
            "mean": float(eligible_target.mean()),
            "median": float(eligible_target.median()),
            "standard_deviation": float(eligible_target.std()),
        },
    }


def build_feature_contract_draft(
    column_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Build an initial feature inclusion review table."""
    records: list[dict[str, Any]] = []

    for row in column_profile.to_dict(orient="records"):
        role = row["role"]

        if role in {
            "target",
            "target_component",
            "identifier",
            "metadata",
        }:
            action = "exclude"
            reason = f"Excluded based on its {role} role."

        elif row["is_constant"]:
            action = "exclude"
            reason = "Constant or effectively constant column."

        elif row["is_high_cardinality"] and row["numeric_parse_rate"] < 0.90:
            action = "review"
            reason = "High-cardinality non-numeric column requires manual review."

        else:
            action = "candidate"
            reason = "Potential AutoML feature; confirm semantic meaning."

        records.append(
            {
                "column": row["column"],
                "role": role,
                "raw_dtype": row["raw_dtype"],
                "missing_percentage": row["missing_percentage"],
                "unique_count": row["unique_count"],
                "numeric_parse_rate": row["numeric_parse_rate"],
                "recommended_action": action,
                "reason": reason,
            }
        )

    return pd.DataFrame(records)
