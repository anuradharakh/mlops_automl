"""Tests for top-feature extraction and comparison."""

from __future__ import annotations

import pytest

from athlete_automl.automl.feature_selection import (
    compare_run_summaries,
    extract_top_features,
)


def test_extract_top_features() -> None:
    """Exactly three unique features should be returned."""
    features = extract_top_features(
        {
            "top_three_features": [
                "weight",
                "bmi",
                "age",
            ]
        }
    )

    assert features == [
        "weight",
        "bmi",
        "age",
    ]


def test_extract_top_features_rejects_duplicates() -> None:
    """Duplicate top features should fail validation."""
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        extract_top_features(
            {
                "top_three_features": [
                    "weight",
                    "weight",
                    "age",
                ]
            }
        )


def test_compare_run_summaries() -> None:
    """Comparison deltas should use top minus all."""
    all_features = {
        "feature_count": 13,
        "model_features": [
            "age",
            "weight",
        ],
        "best_model": "ModelA",
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
        "best_model": "ModelB",
        "validation_rmse": 162.0,
        "test_rmse": 167.0,
        "test_mae": 127.0,
        "test_r2": 0.63,
        "training_wall_clock_seconds": 500.0,
        "prediction_seconds": 0.5,
    }

    result = compare_run_summaries(
        all_features,
        top_features,
    )

    assert (
        result[
            "validation_rmse_delta_top_minus_all"
        ]
        == 2.0
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
        == "degraded"
    )
