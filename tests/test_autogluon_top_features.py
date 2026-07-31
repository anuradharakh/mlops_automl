"""Tests for AutoGluon top-feature utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from athlete_automl.autogluon.top_features import (
    read_top_features,
    reduce_dataset,
    validate_selected_features,
)


def test_read_top_features_uses_rank_order(
    tmp_path: Path,
) -> None:
    """The three highest-ranked unique features should be selected."""
    path = tmp_path / "top5_features.csv"

    pd.DataFrame(
        {
            "importance_rank": [3, 1, 2, 4],
            "feature": [
                "height",
                "weight",
                "age",
                "gender",
            ],
        }
    ).to_csv(path, index=False)

    selected = read_top_features(
        path=path,
        feature_column="feature",
        rank_column="importance_rank",
        count=3,
    )

    assert selected == [
        "weight",
        "age",
        "height",
    ]


def test_read_top_features_rejects_short_file(
    tmp_path: Path,
) -> None:
    """The source must contain enough unique features."""
    path = tmp_path / "features.csv"

    pd.DataFrame(
        {
            "feature": ["age", "age"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Expected 3 unique features",
    ):
        read_top_features(
            path=path,
            feature_column="feature",
            rank_column="importance_rank",
            count=3,
        )


def test_validate_selected_features_rejects_leakage() -> None:
    """A target component cannot be selected as a feature."""
    with pytest.raises(
        ValueError,
        match="Prohibited leakage",
    ):
        validate_selected_features(
            selected_features=[
                "age",
                "weight",
                "deadlift",
            ],
            available_columns=[
                "age",
                "weight",
                "deadlift",
            ],
            prohibited_columns={
                "deadlift",
            },
        )


def test_reduce_dataset_keeps_only_required_columns() -> None:
    """The reduced frame should contain only ID, top features, and target."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": [1, 2],
            "age": [20, 30],
            "weight": [150, 160],
            "height": [65, 70],
            "gender": ["male", "female"],
            "total_lift": [500, 600],
        }
    )

    reduced = reduce_dataset(
        dataframe=dataframe,
        identifier_column="athlete_id",
        target_column="total_lift",
        selected_features=[
            "age",
            "weight",
            "height",
        ],
    )

    assert list(reduced.columns) == [
        "athlete_id",
        "age",
        "weight",
        "height",
        "total_lift",
    ]
