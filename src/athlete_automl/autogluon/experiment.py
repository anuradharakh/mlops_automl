"""Reusable evaluation utilities for the AutoGluon workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PredictionFunction = Callable[
    [pd.DataFrame, str],
    np.ndarray | pd.Series,
]


def find_identifier_column(
    columns: list[str],
    candidates: list[str],
) -> str:
    """Return the first configured identifier found in the dataset."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        "No identifier column found. Expected one of: "
        f"{candidates}"
    )


def validate_split_schemas(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Confirm that all fixed partitions use the same schema."""
    train_columns = list(train.columns)

    if list(validation.columns) != train_columns:
        raise ValueError(
            "Validation schema does not match the training schema."
        )

    if list(test.columns) != train_columns:
        raise ValueError(
            "Test schema does not match the training schema."
        )


def validate_modeling_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    identifier_column: str,
) -> None:
    """Validate target, identifier, and leakage-safe prepared data."""
    required = {
        target_column,
        identifier_column,
    }
    missing = sorted(
        required.difference(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if dataframe.empty:
        raise ValueError(
            "The modeling dataset is empty."
        )

    if dataframe[target_column].isna().any():
        raise ValueError(
            "The target contains missing values."
        )

    if dataframe[identifier_column].isna().any():
        raise ValueError(
            "The identifier contains missing values."
        )

    prohibited_features = {
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    }
    leaked = sorted(
        prohibited_features.intersection(
            dataframe.columns
        )
    )

    if leaked:
        raise ValueError(
            "Target-component leakage columns are present: "
            f"{leaked}"
        )


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
    identifier_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate model features, target values, and identifiers."""
    identifiers = dataframe[
        identifier_column
    ].copy()
    target = dataframe[
        target_column
    ].copy()

    features = dataframe.drop(
        columns=[
            identifier_column,
            target_column,
        ]
    )

    if features.empty:
        raise ValueError(
            "No model features remain."
        )

    return features, target, identifiers


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate RMSE, MAE, and R-squared."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(
        predicted
    ).reshape(-1)

    if len(actual_array) != len(
        predicted_array
    ):
        raise ValueError(
            "Actual and predicted arrays have different lengths."
        )

    return {
        "rmse": float(
            mean_squared_error(
                actual_array,
                predicted_array,
            )
            ** 0.5
        ),
        "mae": float(
            mean_absolute_error(
                actual_array,
                predicted_array,
            )
        ),
        "r2": float(
            r2_score(
                actual_array,
                predicted_array,
            )
        ),
    }


def build_model_comparison(
    leaderboard: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    predict_for_model: PredictionFunction,
) -> pd.DataFrame:
    """Calculate comparable validation metrics for each trained model."""
    required_columns = {
        "model",
        "fit_time",
        "pred_time_val",
    }
    missing = sorted(
        required_columns.difference(
            leaderboard.columns
        )
    )

    if missing:
        raise ValueError(
            f"Leaderboard is missing columns: {missing}"
        )

    records: list[dict[str, Any]] = []

    for row in leaderboard.to_dict(
        orient="records"
    ):
        model_name = str(
            row["model"]
        )
        predictions = predict_for_model(
            validation_features,
            model_name,
        )
        metrics = regression_metrics(
            validation_target,
            predictions,
        )

        records.append(
            {
                "model": model_name,
                "validation_rmse": (
                    metrics["rmse"]
                ),
                "validation_mae": (
                    metrics["mae"]
                ),
                "validation_r2": (
                    metrics["r2"]
                ),
                "fit_time_seconds": (
                    _safe_float(
                        row.get(
                            "fit_time"
                        )
                    )
                ),
                "validation_prediction_seconds": (
                    _safe_float(
                        row.get(
                            "pred_time_val"
                        )
                    )
                ),
                "stack_level": row.get(
                    "stack_level"
                ),
                "can_infer": row.get(
                    "can_infer"
                ),
            }
        )

    result = (
        pd.DataFrame(records)
        .sort_values(
            "validation_rmse",
            ascending=True,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "validation_rank",
        range(1, len(result) + 1),
    )

    return result


def build_speed_ranking(
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Rank models by individual fit time."""
    result = (
        model_comparison.dropna(
            subset=[
                "fit_time_seconds"
            ]
        )
        .sort_values(
            [
                "fit_time_seconds",
                "validation_rmse",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "speed_rank",
        range(1, len(result) + 1),
    )

    return result


def create_data_insights(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    identifier_column: str,
) -> dict[str, Any]:
    """Create reproducible data insights for the report."""
    feature_columns = [
        column
        for column in train.columns
        if column
        not in {
            target_column,
            identifier_column,
        }
    ]

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            train[column]
        )
    ]

    categorical_features = [
        column
        for column in feature_columns
        if column
        not in numeric_features
    ]

    missing_percentages = (
        train[
            feature_columns
        ]
        .isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    return {
        "row_counts": {
            "train": int(
                len(train)
            ),
            "validation": int(
                len(validation)
            ),
            "test": int(
                len(test)
            ),
            "total": int(
                len(train)
                + len(validation)
                + len(test)
            ),
        },
        "feature_count": len(
            feature_columns
        ),
        "numeric_feature_count": len(
            numeric_features
        ),
        "categorical_feature_count": len(
            categorical_features
        ),
        "numeric_features": (
            numeric_features
        ),
        "categorical_features": (
            categorical_features
        ),
        "target_statistics": {
            "train_mean": float(
                train[
                    target_column
                ].mean()
            ),
            "train_median": float(
                train[
                    target_column
                ].median()
            ),
            "train_standard_deviation": float(
                train[
                    target_column
                ].std()
            ),
            "train_minimum": float(
                train[
                    target_column
                ].min()
            ),
            "train_maximum": float(
                train[
                    target_column
                ].max()
            ),
        },
        "features_with_missing_values": {
            str(column): float(value)
            for column, value
            in missing_percentages.items()
            if value > 0
        },
    }


def json_safe(
    value: Any,
) -> Any:
    """Convert NumPy and pandas values to JSON-compatible values."""
    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if (
        isinstance(value, float)
        and np.isnan(value)
    ):
        return None

    return value


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a stable JSON artifact."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
        ),
        encoding="utf-8",
    )


def _safe_float(
    value: Any,
) -> float | None:
    """Convert finite numeric values to float."""
    if value is None:
        return None

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if np.isnan(number):
        return None

    return number
