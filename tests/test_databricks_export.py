"""Tests for Databricks AutoML dataset export."""

from __future__ import annotations

import pandas as pd
import pytest

from athlete_automl.databricks.export import (
    combine_splits,
    find_identifier_column,
    validate_export_dataset,
)


def build_frame(
    start_identifier: int,
    row_count: int,
) -> pd.DataFrame:
    """Create a representative prepared split."""
    return pd.DataFrame(
        {
            "athlete_id": range(
                start_identifier,
                start_identifier + row_count,
            ),
            "age": range(20, 20 + row_count),
            "weight": range(150, 150 + row_count),
            "total_lift": range(500, 500 + row_count),
        }
    )


def test_combine_splits_creates_fixed_labels() -> None:
    """The export should preserve train/validate/test partitions."""
    combined = combine_splits(
        {
            "train": build_frame(0, 6),
            "validate": build_frame(10, 2),
            "test": build_frame(20, 2),
        }
    )

    assert len(combined) == 10
    assert set(combined["data_split"]) == {
        "train",
        "validate",
        "test",
    }


def test_combine_splits_rejects_schema_mismatch() -> None:
    """Each local partition must use the same schema."""
    validation = build_frame(10, 2).drop(columns=["weight"])

    with pytest.raises(
        ValueError,
        match="Column mismatch",
    ):
        combine_splits(
            {
                "train": build_frame(0, 6),
                "validate": validation,
                "test": build_frame(20, 2),
            }
        )


def test_find_identifier_column_uses_candidate_order() -> None:
    """The first configured identifier present should be selected."""
    selected = find_identifier_column(
        columns=["record_id", "athlete_id", "age"],
        candidates=["athlete_id", "record_id"],
    )

    assert selected == "athlete_id"


def test_export_validation_detects_split_overlap() -> None:
    """One athlete must not appear across multiple partitions."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": [1, 1],
            "age": [25, 25],
            "total_lift": [700, 700],
            "data_split": ["train", "test"],
        }
    )

    with pytest.raises(
        ValueError,
        match="multiple splits",
    ):
        validate_export_dataset(
            dataframe=dataframe,
            identifier_column="athlete_id",
            target_column="total_lift",
            split_column="data_split",
        )


def test_export_validation_accepts_clean_dataset() -> None:
    """A valid fixed-split dataset should pass validation."""
    dataframe = combine_splits(
        {
            "train": build_frame(0, 6),
            "validate": build_frame(10, 2),
            "test": build_frame(20, 2),
        }
    )

    validate_export_dataset(
        dataframe=dataframe,
        identifier_column="athlete_id",
        target_column="total_lift",
        split_column="data_split",
    )
