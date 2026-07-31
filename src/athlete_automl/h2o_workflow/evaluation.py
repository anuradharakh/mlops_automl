"""H2O leaderboard, feature-importance, and data-insight utilities."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def normalize_h2o_leaderboard(
    leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and rank an H2O regression leaderboard."""
    required = {
        "model_id",
        "rmse",
    }
    missing = sorted(
        required.difference(leaderboard.columns)
    )

    if missing:
        raise ValueError(
            f"H2O leaderboard is missing columns: {missing}"
        )

    result = leaderboard.copy()
    result["rmse"] = pd.to_numeric(
        result["rmse"],
        errors="coerce",
    )

    if result["rmse"].isna().all():
        raise ValueError(
            "H2O leaderboard contains no numeric RMSE values."
        )

    result = result.sort_values(
        ["rmse", "model_id"],
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
    """Return the strongest H2O models by validation RMSE."""
    return (
        leaderboard.sort_values(
            ["rmse", "model_id"],
            ascending=[True, True],
            kind="stable",
        )
        .head(model_count)
        .reset_index(drop=True)
    )


def top_models_by_training_speed(
    leaderboard: pd.DataFrame,
    model_count: int = 3,
) -> pd.DataFrame:
    """Return the fastest models by H2O training time."""
    if "training_time_ms" not in leaderboard.columns:
        raise ValueError(
            "H2O leaderboard is missing training_time_ms."
        )

    result = leaderboard.copy()
    result["training_time_ms"] = pd.to_numeric(
        result["training_time_ms"],
        errors="coerce",
    )

    ranked = (
        result.dropna(
            subset=["training_time_ms"]
        )
        .sort_values(
            [
                "training_time_ms",
                "rmse",
                "model_id",
            ],
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

    return ranked


def top_models_by_prediction_speed(
    leaderboard: pd.DataFrame,
    model_count: int = 3,
) -> pd.DataFrame:
    """Return the fastest models by prediction time per row."""
    column = "predict_time_per_row_ms"

    if column not in leaderboard.columns:
        raise ValueError(
            "H2O leaderboard is missing "
            "predict_time_per_row_ms."
        )

    result = leaderboard.copy()
    result[column] = pd.to_numeric(
        result[column],
        errors="coerce",
    )

    ranked = (
        result.dropna(subset=[column])
        .sort_values(
            [
                column,
                "rmse",
                "model_id",
            ],
            ascending=[True, True, True],
            kind="stable",
        )
        .head(model_count)
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "prediction_speed_rank",
        range(1, len(ranked) + 1),
    )

    return ranked


def normalize_variable_importance(
    variable_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize H2O variable-importance output."""
    if variable_importance.empty:
        raise ValueError(
            "Variable-importance output is empty."
        )

    result = variable_importance.copy()

    if "variable" not in result.columns:
        first_column = result.columns[0]
        result = result.rename(
            columns={
                first_column: "variable",
            }
        )

    importance_column = None
    for candidate in (
        "relative_importance",
        "scaled_importance",
        "percentage",
    ):
        if candidate in result.columns:
            importance_column = candidate
            break

    if importance_column is None:
        raise ValueError(
            "No supported importance column was found."
        )

    result[importance_column] = pd.to_numeric(
        result[importance_column],
        errors="coerce",
    )

    result = (
        result.dropna(
            subset=[importance_column]
        )
        .sort_values(
            importance_column,
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return result


def find_feature_importance_model(
    leaderboard: pd.DataFrame,
    model_loader: Callable[[str], Any],
    maximum_models_to_check: int,
) -> tuple[str, pd.DataFrame]:
    """Find the highest-ranked model exposing variable importance."""
    model_ids = (
        leaderboard["model_id"]
        .head(maximum_models_to_check)
        .astype(str)
        .tolist()
    )

    errors: list[str] = []

    for model_id in model_ids:
        try:
            model = model_loader(model_id)
            variable_importance = model.varimp(
                use_pandas=True
            )

            if (
                variable_importance is not None
                and not variable_importance.empty
            ):
                return (
                    model_id,
                    normalize_variable_importance(
                        variable_importance
                    ),
                )
        except Exception as error:  # pragma: no cover
            errors.append(
                f"{model_id}: {type(error).__name__}"
            )

    raise RuntimeError(
        "No checked H2O model exposed variable importance. "
        f"Checked models: {model_ids}. Errors: {errors}"
    )


def build_data_insights(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    model_features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Build reproducible data insights for the H2O report."""
    missing_percentages = (
        train[model_features]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    categorical_cardinality = {
        column: int(
            train[column].nunique(
                dropna=True
            )
        )
        for column in categorical_features
    }

    numeric_correlations: dict[str, float] = {}

    if numeric_features:
        correlation_frame = train[
            [*numeric_features, target_column]
        ].corr(
            numeric_only=True
        )

        if target_column in correlation_frame.columns:
            numeric_correlations = {
                str(feature): float(value)
                for feature, value in (
                    correlation_frame[
                        target_column
                    ]
                    .drop(
                        labels=[target_column],
                        errors="ignore",
                    )
                    .abs()
                    .sort_values(
                        ascending=False
                    )
                    .items()
                )
                if pd.notna(value)
            }

    target = train[target_column]

    return {
        "row_counts": {
            "train": int(len(train)),
            "validation": int(
                len(validation)
            ),
            "test": int(len(test)),
        },
        "feature_count": int(
            len(model_features)
        ),
        "numeric_feature_count": int(
            len(numeric_features)
        ),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "target_statistics_train": {
            "minimum": float(target.min()),
            "maximum": float(target.max()),
            "mean": float(target.mean()),
            "median": float(target.median()),
            "standard_deviation": float(
                target.std()
            ),
        },
        "missing_percentage_train": {
            str(feature): float(value)
            for feature, value
            in missing_percentages.items()
        },
        "categorical_cardinality_train": (
            categorical_cardinality
        ),
        "absolute_numeric_correlation_with_target": (
            numeric_correlations
        ),
    }
