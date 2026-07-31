"""AutoML leaderboard and regression evaluation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    """Calculate RMSE, MAE, and R-squared."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape."
        )

    return {
        "rmse": float(
            mean_squared_error(actual, predicted) ** 0.5
        ),
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


def normalize_autogluon_leaderboard(
    leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Add positive RMSE columns and stable model rankings.

    AutoGluon presents metrics in higher-is-better form. For RMSE,
    validation and test scores therefore appear as negative values.
    """
    if "model" not in leaderboard.columns:
        raise ValueError(
            "AutoGluon leaderboard is missing the model column."
        )

    result = leaderboard.copy()

    if "score_val" in result.columns:
        result["validation_rmse"] = (
            -pd.to_numeric(
                result["score_val"],
                errors="coerce",
            )
        )

    if "score_test" in result.columns:
        result["test_rmse"] = (
            -pd.to_numeric(
                result["score_test"],
                errors="coerce",
            )
        )

    if "validation_rmse" not in result.columns:
        raise ValueError(
            "AutoGluon leaderboard is missing score_val."
        )

    result = result.sort_values(
        ["validation_rmse", "model"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    result.insert(
        0,
        "validation_rank",
        range(1, len(result) + 1),
    )

    return result


def top_models_by_score(
    leaderboard: pd.DataFrame,
    model_count: int = 3,
) -> pd.DataFrame:
    """Return the strongest models by validation RMSE."""
    if "validation_rmse" not in leaderboard.columns:
        raise ValueError(
            "Leaderboard is missing validation_rmse."
        )

    return (
        leaderboard.sort_values(
            ["validation_rmse", "model"],
            ascending=[True, True],
            kind="stable",
        )
        .head(model_count)
        .reset_index(drop=True)
    )


def select_speed_column(
    leaderboard: pd.DataFrame,
) -> str:
    """Select the best available model training-time column."""
    for column in (
        "fit_time_marginal",
        "fit_time",
    ):
        if column in leaderboard.columns:
            return column

    raise ValueError(
        "Leaderboard does not contain a model fit-time column."
    )


def top_models_by_speed(
    leaderboard: pd.DataFrame,
    model_count: int = 3,
) -> tuple[pd.DataFrame, str]:
    """Return the fastest models by AutoGluon-reported fit time."""
    speed_column = select_speed_column(leaderboard)

    ranked = (
        leaderboard.dropna(
            subset=[speed_column]
        )
        .sort_values(
            [speed_column, "validation_rmse", "model"],
            ascending=[True, True, True],
            kind="stable",
        )
        .head(model_count)
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "speed_rank",
        range(1, len(ranked) + 1),
    )

    return ranked, speed_column


def validate_feature_contract(
    model_features: list[str],
    prohibited_features: list[str],
) -> None:
    """Reject target, target components, and identifiers as features."""
    overlap = sorted(
        set(model_features).intersection(prohibited_features)
    )

    if overlap:
        raise ValueError(
            f"Prohibited model features detected: {overlap}"
        )

    if not model_features:
        raise ValueError(
            "The model feature list cannot be empty."
        )
