"""Top-feature selection utilities for AutoGluon experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_top_features(
    path: Path,
    feature_column: str,
    rank_column: str,
    count: int,
) -> list[str]:
    """Read the highest-ranked unique features from a CSV artifact."""
    if count <= 0:
        raise ValueError("Feature count must be greater than zero.")

    if not path.exists():
        raise FileNotFoundError(f"Feature-importance artifact not found: {path}")

    dataframe = pd.read_csv(path)

    if feature_column not in dataframe.columns:
        raise ValueError(f"Missing feature column: {feature_column}")

    if rank_column in dataframe.columns:
        dataframe = dataframe.sort_values(
            rank_column,
            ascending=True,
            kind="stable",
        )

    selected: list[str] = []

    for value in dataframe[feature_column].dropna():
        feature = str(value).strip()

        if feature and feature not in selected:
            selected.append(feature)

        if len(selected) == count:
            break

    if len(selected) < count:
        raise ValueError(f"Expected {count} unique features, found {len(selected)}.")

    return selected


def validate_selected_features(
    selected_features: list[str],
    available_columns: list[str],
    prohibited_columns: set[str] | None = None,
) -> None:
    """Validate existence, uniqueness, and leakage safety."""
    if len(selected_features) != len(set(selected_features)):
        raise ValueError("Selected features must be unique.")

    missing = sorted(set(selected_features).difference(available_columns))

    if missing:
        raise ValueError(f"Selected features are missing from the dataset: {missing}")

    prohibited = prohibited_columns or set()
    leaked = sorted(set(selected_features).intersection(prohibited))

    if leaked:
        raise ValueError(f"Prohibited leakage columns selected: {leaked}")


def reduce_dataset(
    dataframe: pd.DataFrame,
    identifier_column: str,
    target_column: str,
    selected_features: list[str],
) -> pd.DataFrame:
    """Return only identifier, selected features, and target."""
    required = {
        identifier_column,
        target_column,
        *selected_features,
    }

    missing = sorted(required.difference(dataframe.columns))

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return dataframe[
        [
            identifier_column,
            *selected_features,
            target_column,
        ]
    ].copy()
