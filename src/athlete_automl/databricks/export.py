"""Utilities for exporting fixed AutoML splits to Databricks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

EXPECTED_SPLIT_LABELS = ("train", "validate", "test")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def find_identifier_column(
    columns: list[str],
    candidates: list[str],
) -> str:
    """Return the first configured identifier present in the dataset."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(f"No identifier column found. Expected one of: {candidates}")


def combine_splits(
    split_frames: Mapping[str, pd.DataFrame],
    split_column: str = "data_split",
) -> pd.DataFrame:
    """Combine train, validation, and test frames with fixed labels."""
    if tuple(split_frames.keys()) != EXPECTED_SPLIT_LABELS:
        raise ValueError("split_frames must be ordered as train, validate, and test.")

    reference_columns = list(split_frames["train"].columns)

    if len(reference_columns) != len(set(reference_columns)):
        raise ValueError("Training data contains duplicate columns.")

    combined_parts: list[pd.DataFrame] = []

    for split_label in EXPECTED_SPLIT_LABELS:
        frame = split_frames[split_label]

        if list(frame.columns) != reference_columns:
            raise ValueError(f"Column mismatch in '{split_label}' split.")

        part = frame.copy()
        part[split_column] = split_label
        combined_parts.append(part)

    combined = pd.concat(
        combined_parts,
        axis=0,
        ignore_index=True,
    )

    observed = set(combined[split_column].dropna().unique())
    expected = set(EXPECTED_SPLIT_LABELS)

    if observed != expected:
        raise ValueError(f"Unexpected split labels: {sorted(observed)}")

    return combined


def validate_export_dataset(
    dataframe: pd.DataFrame,
    identifier_column: str,
    target_column: str,
    split_column: str,
) -> None:
    """Validate schema, target, identifiers, and split isolation."""
    required_columns = {
        identifier_column,
        target_column,
        split_column,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("The Databricks export dataset is empty.")

    if dataframe[target_column].isna().any():
        raise ValueError("Target column contains missing values.")

    if dataframe[identifier_column].isna().any():
        raise ValueError("Identifier column contains missing values.")

    split_counts_per_identifier = dataframe.groupby(identifier_column, dropna=False)[
        split_column
    ].nunique()

    overlap_count = int((split_counts_per_identifier > 1).sum())

    if overlap_count:
        raise ValueError(f"{overlap_count} identifiers occur in multiple splits.")

    if dataframe.duplicated(subset=[identifier_column]).any():
        raise ValueError("Duplicate identifiers remain in the export dataset.")

    observed = set(dataframe[split_column].unique())
    expected = set(EXPECTED_SPLIT_LABELS)

    if observed != expected:
        raise ValueError(
            f"Expected split labels {sorted(expected)}, found {sorted(observed)}."
        )
