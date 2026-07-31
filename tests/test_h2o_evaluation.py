"""Tests for H2O AutoML evaluation utilities."""

from __future__ import annotations

import pandas as pd

from athlete_automl.h2o_workflow.evaluation import (
    build_data_insights,
    normalize_h2o_leaderboard,
    normalize_variable_importance,
    top_models_by_score,
    top_models_by_training_speed,
)


def build_leaderboard() -> pd.DataFrame:
    """Create a representative H2O leaderboard."""
    return pd.DataFrame(
        {
            "model_id": [
                "GBM_1",
                "GLM_1",
                "DRF_1",
            ],
            "rmse": [
                170.0,
                160.0,
                180.0,
            ],
            "mae": [
                130.0,
                120.0,
                140.0,
            ],
            "training_time_ms": [
                4000,
                1000,
                2000,
            ],
            "predict_time_per_row_ms": [
                0.03,
                0.01,
                0.02,
            ],
        }
    )


def test_h2o_leaderboard_ranking() -> None:
    """The lowest validation RMSE should rank first."""
    leaderboard = normalize_h2o_leaderboard(
        build_leaderboard()
    )

    assert (
        leaderboard.iloc[0]["model_id"]
        == "GLM_1"
    )
    assert (
        leaderboard.iloc[0]["rmse"]
        == 160.0
    )

    top_score = top_models_by_score(
        leaderboard,
        model_count=2,
    )
    top_speed = (
        top_models_by_training_speed(
            leaderboard,
            model_count=2,
        )
    )

    assert top_score[
        "model_id"
    ].tolist() == [
        "GLM_1",
        "GBM_1",
    ]
    assert top_speed[
        "model_id"
    ].tolist() == [
        "GLM_1",
        "DRF_1",
    ]


def test_variable_importance_normalization() -> None:
    """Variable importance should be sorted descending."""
    importance = pd.DataFrame(
        {
            "variable": [
                "age",
                "weight",
                "bmi",
            ],
            "relative_importance": [
                1.0,
                3.0,
                2.0,
            ],
        }
    )

    result = normalize_variable_importance(
        importance
    )

    assert result[
        "variable"
    ].tolist() == [
        "weight",
        "bmi",
        "age",
    ]


def test_data_insights() -> None:
    """Data insights should summarize splits and missingness."""
    train = pd.DataFrame(
        {
            "age": [
                20.0,
                30.0,
                40.0,
            ],
            "fran": [
                100.0,
                None,
                120.0,
            ],
            "gender": [
                "male",
                "female",
                "male",
            ],
            "total_lift": [
                500.0,
                700.0,
                900.0,
            ],
        }
    )

    insights = build_data_insights(
        train=train,
        validation=train.iloc[:1],
        test=train.iloc[1:],
        model_features=[
            "age",
            "fran",
            "gender",
        ],
        numeric_features=[
            "age",
            "fran",
        ],
        categorical_features=[
            "gender",
        ],
        target_column="total_lift",
    )

    assert (
        insights["row_counts"]["train"]
        == 3
    )
    assert (
        round(
            insights[
                "missing_percentage_train"
            ]["fran"],
            4,
        )
        == 33.3333
    )
    assert (
        insights[
            "categorical_cardinality_train"
        ]["gender"]
        == 2
    )
