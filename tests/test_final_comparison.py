"""Tests for final AutoML comparison utilities."""

from __future__ import annotations

import pandas as pd

from athlete_automl.comparison.final_comparison import (
    add_random_forest_improvement,
    build_recommendation,
    feature_overlap,
    percent_change,
    rank_comparison,
)


def build_comparison() -> pd.DataFrame:
    """Create representative baseline and AutoML results."""
    return pd.DataFrame(
        [
            {
                "platform": "scikit-learn",
                "feature_set": "all_features",
                "feature_count": 13,
                "best_model": (
                    "RandomForestRegressor"
                ),
                "validation_rmse": 170.0,
                "test_rmse": 175.0,
                "test_mae": 130.0,
                "test_r2": 0.60,
                "training_seconds": 30.0,
                "prediction_seconds": 1.0,
                "is_automl": False,
            },
            {
                "platform": "AutoGluon",
                "feature_set": "all_features",
                "feature_count": 13,
                "best_model": "WeightedEnsemble",
                "validation_rmse": 155.0,
                "test_rmse": 160.0,
                "test_mae": 120.0,
                "test_r2": 0.70,
                "training_seconds": 900.0,
                "prediction_seconds": 2.0,
                "is_automl": True,
            },
            {
                "platform": "AutoGluon",
                "feature_set": "top_features",
                "feature_count": 3,
                "best_model": "GBM",
                "validation_rmse": 157.0,
                "test_rmse": 162.0,
                "test_mae": 122.0,
                "test_r2": 0.68,
                "training_seconds": 400.0,
                "prediction_seconds": 0.5,
                "is_automl": True,
            },
        ]
    )


def test_percent_change() -> None:
    """Percentage change should use the reference denominator."""
    assert round(
        percent_change(90.0, 100.0),
        4,
    ) == -10.0


def test_rank_and_baseline_improvement() -> None:
    """Validation ranking and RF improvement should be correct."""
    ranked = rank_comparison(
        build_comparison()
    )
    enriched = (
        add_random_forest_improvement(
            ranked
        )
    )

    best = enriched.iloc[0]

    assert best["platform"] == "AutoGluon"
    assert best["validation_rank"] == 1
    assert (
        best[
            "test_rmse_change_vs_rf_percent"
        ]
        < 0
    )


def test_recommendation_accepts_reduced_run() -> None:
    """A reduced run within tolerance should be identified."""
    comparison = rank_comparison(
        build_comparison()
    )
    recommendation = build_recommendation(
        comparison,
        reduced_feature_tolerance_percent=2.0,
    )

    assert (
        recommendation[
            "best_predictive_run"
        ]["feature_set"]
        == "all_features"
    )
    assert (
        recommendation[
            "recommended_reduced_feature_run"
        ]["feature_count"]
        if "feature_count"
        in recommendation[
            "recommended_reduced_feature_run"
        ]
        else recommendation[
            "recommended_reduced_feature_run"
        ]["features"]
    ) == 3


def test_feature_overlap() -> None:
    """Feature overlap should report intersection and Jaccard."""
    result = feature_overlap(
        ["weight", "age", "bmi"],
        ["weight", "bmi", "region"],
    )

    assert result["shared_features"] == [
        "bmi",
        "weight",
    ]
    assert round(
        result["jaccard_similarity"],
        4,
    ) == 0.5
