"""Tests for AutoML evaluation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from athlete_automl.automl.evaluation import (
    normalize_autogluon_leaderboard,
    regression_metrics,
    top_models_by_score,
    top_models_by_speed,
    validate_feature_contract,
)


def build_leaderboard() -> pd.DataFrame:
    """Create a representative AutoGluon leaderboard."""
    return pd.DataFrame(
        {
            "model": [
                "ModelB",
                "ModelA",
                "ModelC",
            ],
            "score_val": [
                -170.0,
                -160.0,
                -180.0,
            ],
            "score_test": [
                -175.0,
                -165.0,
                -185.0,
            ],
            "fit_time_marginal": [
                4.0,
                8.0,
                2.0,
            ],
        }
    )


def test_regression_metrics() -> None:
    """Regression metrics should be calculated correctly."""
    metrics = regression_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.0, 4.0],
    )

    assert round(metrics["rmse"], 6) == 0.577350
    assert round(metrics["mae"], 6) == 0.333333
    assert round(metrics["r2"], 6) == 0.5


def test_leaderboard_converts_negative_rmse() -> None:
    """AutoGluon scores should become positive RMSE values."""
    result = normalize_autogluon_leaderboard(
        build_leaderboard()
    )

    assert result.iloc[0]["model"] == "ModelA"
    assert (
        result.iloc[0]["validation_rmse"]
        == 160.0
    )
    assert result.iloc[0]["test_rmse"] == 165.0


def test_top_models_by_score_and_speed() -> None:
    """Score and speed rankings should use different criteria."""
    leaderboard = normalize_autogluon_leaderboard(
        build_leaderboard()
    )

    top_score = top_models_by_score(
        leaderboard,
        model_count=2,
    )
    top_speed, speed_column = (
        top_models_by_speed(
            leaderboard,
            model_count=2,
        )
    )

    assert top_score["model"].tolist() == [
        "ModelA",
        "ModelB",
    ]
    assert top_speed["model"].tolist() == [
        "ModelC",
        "ModelB",
    ]
    assert speed_column == "fit_time_marginal"


def test_feature_contract_rejects_leakage() -> None:
    """Target components must not enter the AutoML feature set."""
    with pytest.raises(
        ValueError,
        match="Prohibited",
    ):
        validate_feature_contract(
            model_features=[
                "age",
                "deadlift",
            ],
            prohibited_features=[
                "deadlift",
                "total_lift",
            ],
        )
