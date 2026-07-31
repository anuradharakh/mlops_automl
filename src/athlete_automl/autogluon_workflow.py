"""Helpers for the AutoGluon AutoML experiment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_identifier(columns: list[str], candidates: list[str]) -> str:
    """Return the first configured identifier present."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No identifier found. Expected one of: {candidates}")


def validate_dataset(
    dataframe: pd.DataFrame,
    target: str,
    identifier: str,
) -> None:
    """Validate prepared data and reject direct target leakage."""
    missing = sorted({target, identifier}.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("Dataset is empty.")
    if dataframe[target].isna().any():
        raise ValueError("Target contains missing values.")
    if dataframe[identifier].isna().any():
        raise ValueError("Identifier contains missing values.")

    leakage = sorted(
        {"deadlift", "candj", "snatch", "backsq"}.intersection(dataframe.columns)
    )
    if leakage:
        raise ValueError(f"Target leakage columns found: {leakage}")


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Return RMSE, MAE, and R-squared."""
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted).reshape(-1)
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted lengths differ.")

    return {
        "rmse": float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def rank_models(
    leaderboard: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    predict: Callable[[pd.DataFrame, str], np.ndarray],
) -> pd.DataFrame:
    """Rank every trained model using comparable validation metrics."""
    required = {"model", "fit_time", "pred_time_val"}
    missing = sorted(required.difference(leaderboard.columns))
    if missing:
        raise ValueError(f"Leaderboard is missing columns: {missing}")

    records = []
    for row in leaderboard.to_dict(orient="records"):
        model = str(row["model"])
        metrics = regression_metrics(
            validation_target,
            predict(validation_features, model),
        )
        records.append(
            {
                "model": model,
                "validation_rmse": metrics["rmse"],
                "validation_mae": metrics["mae"],
                "validation_r2": metrics["r2"],
                "fit_time_seconds": row.get("fit_time"),
                "validation_prediction_seconds": row.get("pred_time_val"),
            }
        )

    result = (
        pd.DataFrame(records)
        .sort_values("validation_rmse", ascending=True, kind="stable")
        .reset_index(drop=True)
    )
    result.insert(0, "validation_rank", range(1, len(result) + 1))
    return result


def rank_by_speed(model_ranking: pd.DataFrame) -> pd.DataFrame:
    """Rank models by training time, using RMSE as the tie breaker."""
    result = (
        model_ranking.dropna(subset=["fit_time_seconds"])
        .sort_values(
            ["fit_time_seconds", "validation_rmse"],
            ascending=[True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    result.insert(0, "speed_rank", range(1, len(result) + 1))
    return result
