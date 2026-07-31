"""Tests for H2O top-feature extraction and comparison."""

from __future__ import annotations

import pytest

from athlete_automl.h2o_workflow.feature_selection import (
    compare_h2o_run_summaries,
    extract_h2o_top_features,
)


def test_extract_h2o_top_features() -> None:
    """Exactly three unique H2O features should be returned."""
    result = extract_h2o_top_features(
        {
            "top_three_features": [
                "weight",
                "bmi",
                "age",
            ]
        }
    )

    assert result == [
        "weight",
        "bmi",
        "age",
    ]


def test_extract_h2o_top_features_rejects_duplicates() -> None:
    """Duplicate H2O top features should fail."""
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        extract_h2o_top_features(
            {
                "top_three_features": [
                    "weight",
                    "weight",
                    "age",
                ]
            }
        )


def test_compare_h2o_runs() -> None:
    """H2O deltas should use top minus all."""
    all_features = {
        "feature_count": 13,
        "model_features": [
            "age",
            "weight",
        ],
        "best_model": "GBM_1",
        "validation_rmse": 160.0,
        "test_rmse": 165.0,
        "test_mae": 125.0,
        "test_r2": 0.65,
        "training_wall_clock_seconds": 900.0,
        "prediction_seconds": 1.0,
    }

    top_features = {
        "feature_count": 3,
        "model_features": [
            "weight",
            "bmi",
            "age",
        ],
        "best_model": "GBM_2",
        "validation_rmse": 158.0,
        "test_rmse": 162.0,
        "test_mae": 122.0,
        "test_r2": 0.68,
        "training_wall_clock_seconds": 500.0,
        "prediction_seconds": 0.5,
    }

    result = compare_h2o_run_summaries(
        all_features,
        top_features,
    )

    assert (
        result[
            "validation_rmse_delta_top_minus_all"
        ]
        == -2.0
    )
    assert (
        result[
            "training_seconds_delta_top_minus_all"
        ]
        == -400.0
    )
    assert (
        result[
            "validation_performance_assessment"
        ]
        == "improved"
    )
