"""Tests for target-quality filtering."""

from __future__ import annotations

import pandas as pd

from athlete_automl.data.target_quality import (
    filter_valid_target_rows,
    identify_target_outliers,
    split_dataset,
    verify_split_integrity,
)


def test_outlier_filtering() -> None:
    """Missing and extreme targets should be removed."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": range(6),
            "total_lift": [
                8,
                1000,
                2367,
                2500,
                2501,
                None,
            ],
        }
    )

    audit = identify_target_outliers(
        dataframe,
        "total_lift",
        8,
        2500,
    )
    cleaned = filter_valid_target_rows(
        dataframe,
        "total_lift",
        8,
        2500,
    )

    assert len(audit) == 2
    assert len(cleaned) == 4
    assert cleaned["total_lift"].min() == 8
    assert cleaned["total_lift"].max() == 2500


def test_resplit_is_deterministic() -> None:
    """The cleaned data should retain fixed split logic."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": range(100),
            "total_lift": range(500, 600),
        }
    )

    first = split_dataset(
        dataframe,
        0.64,
        0.16,
        0.20,
        42,
    )
    second = split_dataset(
        dataframe,
        0.64,
        0.16,
        0.20,
        42,
    )

    assert len(first["train"]) == 64
    assert len(first["validation"]) == 16
    assert len(first["test"]) == 20

    for name in first:
        assert (
            first[name]["athlete_id"].tolist()
            == second[name]["athlete_id"].tolist()
        )

    verify_split_integrity(
        first,
        "athlete_id",
    )
