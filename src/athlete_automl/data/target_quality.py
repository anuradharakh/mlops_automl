"""Target-quality checks and deterministic resplitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def identify_target_outliers(
    dataframe: pd.DataFrame,
    target_column: str,
    minimum: float,
    maximum: float,
) -> pd.DataFrame:
    """Return rows with missing or out-of-range targets."""
    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    reason = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="string",
    )
    reason = reason.mask(target.isna(), "missing_target")
    reason = reason.mask(target < minimum, "below_minimum")
    reason = reason.mask(target > maximum, "above_maximum")

    audit = dataframe.loc[reason.notna()].copy()
    audit["target_quality_reason"] = reason.loc[reason.notna()]
    return audit


def filter_valid_target_rows(
    dataframe: pd.DataFrame,
    target_column: str,
    minimum: float,
    maximum: float,
) -> pd.DataFrame:
    """Keep only rows with valid targets."""
    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )
    mask = (
        target.notna()
        & target.ge(minimum)
        & target.le(maximum)
    )

    cleaned = dataframe.loc[mask].copy()
    cleaned[target_column] = target.loc[mask]
    return cleaned.reset_index(drop=True)


def split_dataset(
    dataframe: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Create deterministic train, validation, and test splits."""
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")

    train_validation, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    validation_fraction = validation_size / (
        train_size + validation_size
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
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
    """Verify unique, non-overlapping identifiers."""
    identifier_sets: dict[str, set[str]] = {}

    for name, split in splits.items():
        if split[identifier_column].isna().any():
            raise ValueError(
                f"Missing identifiers in {name}."
            )

        if split[identifier_column].duplicated().any():
            raise ValueError(
                f"Duplicate identifiers in {name}."
            )

        identifier_sets[name] = set(
            split[identifier_column].astype(str)
        )

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = identifier_sets[left] & identifier_sets[right]
        if overlap:
            raise ValueError(
                f"Identifier overlap between {left} and {right}."
            )
