"""Leakage-safe data preparation for the athlete AutoML workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_column_name(value: str) -> str:
    """Convert a raw column name to lowercase snake case."""
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    return normalized.strip("_").lower()


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized and unique column names."""
    result = dataframe.copy()
    result.columns = [
        normalize_column_name(str(column))
        for column in result.columns
    ]

    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"Duplicate columns after normalization: {duplicated}"
        )

    return result


def _parse_time_value(value: str) -> float:
    """Convert MM:SS or HH:MM:SS text to seconds."""
    parts = [float(part) for part in value.split(":")]

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    return np.nan


def parse_numeric_or_time(series: pd.Series) -> pd.Series:
    """Parse numeric values and clock-style duration values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    text = series.astype("string").str.strip()
    text = text.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
        }
    )

    numeric = pd.to_numeric(
        text.str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)

    time_mask = text.str.match(
        r"^\d+(?::\d{1,2}){1,2}$",
        na=False,
    )

    if time_mask.any():
        numeric.loc[time_mask] = text.loc[time_mask].map(
            _parse_time_value
        )

    return numeric


def clean_categorical(series: pd.Series) -> pd.Series:
    """Normalize categorical text while preserving missing values."""
    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
        }
    )
    return cleaned


def apply_valid_range(
    series: pd.Series,
    minimum: float | None,
    maximum: float | None,
) -> pd.Series:
    """Replace values outside a configured valid range with missing."""
    result = series.copy()

    if minimum is not None:
        result = result.mask(result < minimum)

    if maximum is not None:
        result = result.mask(result > maximum)

    return result


def build_target(
    dataframe: pd.DataFrame,
    target_name: str,
    components: list[str],
    sentinel_values: list[float | int],
) -> tuple[pd.Series, dict[str, Any]]:
    """Create the regression target from all required lift components."""
    missing = sorted(set(components).difference(dataframe.columns))
    if missing:
        raise ValueError(
            f"Missing target component columns: {missing}"
        )

    numeric_components = dataframe[components].apply(
        pd.to_numeric,
        errors="coerce",
    )

    sentinel_counts = {
        column: int(
            numeric_components[column].isin(sentinel_values).sum()
        )
        for column in components
    }

    cleaned_components = numeric_components.mask(
        numeric_components.isin(sentinel_values)
    )
    cleaned_components = cleaned_components.mask(
        cleaned_components <= 0
    )

    target = cleaned_components.sum(
        axis=1,
        min_count=len(components),
    )
    target.name = target_name

    summary = {
        "sentinel_values": list(sentinel_values),
        "sentinel_counts": sentinel_counts,
        "total_sentinel_replacements": int(
            sum(sentinel_counts.values())
        ),
        "rows_with_complete_target": int(target.notna().sum()),
        "rows_with_incomplete_target": int(target.isna().sum()),
    }

    return target, summary


def select_available_features(
    dataframe: pd.DataFrame,
    required_numeric: list[str],
    optional_numeric: list[str],
    required_categorical: list[str],
    optional_categorical: list[str],
) -> dict[str, list[str]]:
    """Validate required features and retain available optional features."""
    required = required_numeric + required_categorical
    missing_required = sorted(set(required).difference(dataframe.columns))

    if missing_required:
        raise ValueError(
            f"Missing required feature columns: {missing_required}"
        )

    return {
        "required_numeric": required_numeric,
        "optional_numeric": [
            column
            for column in optional_numeric
            if column in dataframe.columns
        ],
        "required_categorical": required_categorical,
        "optional_categorical": [
            column
            for column in optional_categorical
            if column in dataframe.columns
        ],
        "missing_optional_numeric": sorted(
            set(optional_numeric).difference(dataframe.columns)
        ),
        "missing_optional_categorical": sorted(
            set(optional_categorical).difference(dataframe.columns)
        ),
    }


def add_engineered_features(
    dataframe: pd.DataFrame,
    engineered_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Create configured engineered features when dependencies exist."""
    result = dataframe.copy()
    created: list[str] = []

    if "bmi" in engineered_config and {
        "weight",
        "height",
    }.issubset(result.columns):
        valid_height = result["height"].where(result["height"] > 0)
        result["bmi"] = 703.0 * result["weight"] / valid_height.pow(2)
        created.append("bmi")

    if "age_squared" in engineered_config and "age" in result.columns:
        result["age_squared"] = result["age"].pow(2)
        created.append("age_squared")

    if "weight_height_ratio" in engineered_config and {
        "weight",
        "height",
    }.issubset(result.columns):
        valid_height = result["height"].where(result["height"] > 0)
        result["weight_height_ratio"] = (
            result["weight"] / valid_height
        )
        created.append("weight_height_ratio")

    return result, created


def build_feature_contract(
    numeric_features: list[str],
    categorical_features: list[str],
    engineered_features: list[str],
    target_name: str,
    target_components: list[str],
    identifier_column: str,
    all_raw_columns: list[str],
) -> pd.DataFrame:
    """Create the final documented feature inclusion contract."""
    included = set(
        numeric_features
        + categorical_features
        + engineered_features
    )

    records: list[dict[str, Any]] = []

    for column in all_raw_columns:
        if column == identifier_column:
            role = "identifier"
            action = "exclude"
            reason = "Identifier retained only for traceability."
        elif column == target_name:
            role = "target"
            action = "exclude"
            reason = "Regression target."
        elif column in target_components:
            role = "target_component"
            action = "exclude"
            reason = "Excluded to prevent direct target leakage."
        elif column in numeric_features:
            role = "numeric_feature"
            action = "include"
            reason = "Validated numeric model feature."
        elif column in categorical_features:
            role = "categorical_feature"
            action = "include"
            reason = "Validated categorical model feature."
        else:
            role = "unused_raw_column"
            action = "exclude"
            reason = (
                "Not selected for the reproducible tabular AutoML feature set."
            )

        records.append(
            {
                "column": column,
                "role": role,
                "source": "raw",
                "action": action,
                "reason": reason,
            }
        )

    for column in engineered_features:
        if column not in included:
            continue

        records.append(
            {
                "column": column,
                "role": "numeric_feature",
                "source": "engineered",
                "action": "include",
                "reason": "Deterministic feature engineering from raw inputs.",
            }
        )

    return pd.DataFrame(records)


def split_dataset(
    dataframe: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Create deterministic train, validation, and test partitions."""
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1.0."
        )

    train_validation, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    validation_fraction_of_remaining = validation_size / (
        train_size + validation_size
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_remaining,
        random_state=random_state,
        shuffle=True,
    )

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def verify_split_integrity(
    splits: dict[str, pd.DataFrame],
    identifier_column: str,
) -> None:
    """Confirm that split identifiers do not overlap."""
    identifier_sets = {
        name: set(split[identifier_column].astype(str))
        for name, split in splits.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left, right in pairs:
        overlap = identifier_sets[left].intersection(
            identifier_sets[right]
        )
        if overlap:
            raise ValueError(
                f"Identifier overlap between {left} and {right}: "
                f"{len(overlap)} records"
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
