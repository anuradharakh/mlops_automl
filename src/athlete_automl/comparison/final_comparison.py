"""Utilities for building the final AutoML comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_RUN_FIELDS = (
    "platform",
    "feature_set",
    "feature_count",
    "best_model",
    "validation_rmse",
    "test_rmse",
    "test_mae",
    "test_r2",
    "training_wall_clock_seconds",
    "prediction_seconds",
)


def percent_change(
    new_value: float,
    reference_value: float,
) -> float:
    """Calculate percentage change from a reference value."""
    if reference_value == 0:
        raise ValueError(
            "Reference value cannot be zero."
        )

    return (
        (new_value - reference_value)
        / abs(reference_value)
        * 100.0
    )


def validate_run_summary(
    summary: dict[str, Any],
    source_name: str,
) -> None:
    """Validate the shared AutoML run-summary contract."""
    missing = [
        field
        for field in REQUIRED_RUN_FIELDS
        if field not in summary
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing fields: {missing}"
        )


def automl_summary_to_row(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Convert an AutoML summary into a comparison row."""
    return {
        "platform": str(summary["platform"]),
        "feature_set": str(
            summary["feature_set"]
        ),
        "feature_count": int(
            summary["feature_count"]
        ),
        "best_model": str(
            summary["best_model"]
        ),
        "validation_rmse": float(
            summary["validation_rmse"]
        ),
        "test_rmse": float(
            summary["test_rmse"]
        ),
        "test_mae": float(
            summary["test_mae"]
        ),
        "test_r2": float(
            summary["test_r2"]
        ),
        "training_seconds": float(
            summary[
                "training_wall_clock_seconds"
            ]
        ),
        "prediction_seconds": float(
            summary["prediction_seconds"]
        ),
        "is_automl": True,
    }


def baseline_summary_to_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert baseline-model results into comparison rows."""
    models = summary.get("models")

    if not isinstance(models, list) or not models:
        raise ValueError(
            "Baseline summary must contain a non-empty models list."
        )

    rows: list[dict[str, Any]] = []

    for model in models:
        rows.append(
            {
                "platform": "scikit-learn",
                "feature_set": (
                    model["feature_set"]
                ),
                "feature_count": int(
                    model["feature_count"]
                ),
                "best_model": (
                    model["model_name"]
                ),
                "validation_rmse": float(
                    model[
                        "validation_rmse"
                    ]
                ),
                "test_rmse": float(
                    model["test_rmse"]
                ),
                "test_mae": float(
                    model["test_mae"]
                ),
                "test_r2": float(
                    model["test_r2"]
                ),
                "training_seconds": float(
                    model[
                        "training_wall_clock_seconds"
                    ]
                ),
                "prediction_seconds": float(
                    model[
                        "prediction_seconds"
                    ]
                ),
                "is_automl": False,
            }
        )

    return rows


def rank_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Add validation, test, and speed rankings."""
    result = comparison.copy()

    for column in (
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "training_seconds",
        "prediction_seconds",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    result["validation_rank"] = (
        result["validation_rmse"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )
    result["test_rmse_rank"] = (
        result["test_rmse"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )
    result["test_r2_rank"] = (
        result["test_r2"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )
    result["training_speed_rank"] = (
        result["training_seconds"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )
    result["prediction_speed_rank"] = (
        result["prediction_seconds"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return result.sort_values(
        [
            "validation_rank",
            "test_rmse_rank",
            "training_speed_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)


def add_random_forest_improvement(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate AutoML improvements over the RF baseline."""
    result = comparison.copy()

    baseline_rows = result.loc[
        result["best_model"]
        == "RandomForestRegressor"
    ]

    if len(baseline_rows) != 1:
        raise ValueError(
            "Exactly one RandomForestRegressor baseline is required."
        )

    baseline = baseline_rows.iloc[0]

    result[
        "validation_rmse_change_vs_rf_percent"
    ] = result["validation_rmse"].apply(
        lambda value: percent_change(
            float(value),
            float(
                baseline["validation_rmse"]
            ),
        )
    )
    result[
        "test_rmse_change_vs_rf_percent"
    ] = result["test_rmse"].apply(
        lambda value: percent_change(
            float(value),
            float(baseline["test_rmse"]),
        )
    )
    result[
        "test_mae_change_vs_rf_percent"
    ] = result["test_mae"].apply(
        lambda value: percent_change(
            float(value),
            float(baseline["test_mae"]),
        )
    )
    result[
        "test_r2_change_vs_rf_percent"
    ] = result["test_r2"].apply(
        lambda value: percent_change(
            float(value),
            float(baseline["test_r2"]),
        )
        if float(baseline["test_r2"]) != 0
        else float("nan")
    )

    return result


def build_recommendation(
    comparison: pd.DataFrame,
    reduced_feature_tolerance_percent: float,
) -> dict[str, Any]:
    """Select predictive and efficiency-oriented recommendations."""
    automl = comparison.loc[
        comparison["is_automl"]
    ].copy()

    if automl.empty:
        raise ValueError(
            "At least one AutoML result is required."
        )

    predictive = automl.sort_values(
        [
            "validation_rmse",
            "test_rmse",
            "training_seconds",
        ],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]

    fastest = automl.sort_values(
        [
            "training_seconds",
            "validation_rmse",
        ],
        ascending=[True, True],
        kind="stable",
    ).iloc[0]

    reduced_candidates = automl.loc[
        automl["feature_set"]
        == "top_features"
    ].copy()

    acceptable_reduced: list[dict[str, Any]] = []

    for _, reduced in reduced_candidates.iterrows():
        platform_all = automl.loc[
            (
                automl["platform"]
                == reduced["platform"]
            )
            & (
                automl["feature_set"]
                == "all_features"
            )
        ]

        if len(platform_all) != 1:
            continue

        full = platform_all.iloc[0]
        degradation_percent = percent_change(
            float(reduced["validation_rmse"]),
            float(full["validation_rmse"]),
        )
        training_change_percent = percent_change(
            float(reduced["training_seconds"]),
            float(full["training_seconds"]),
        )

        if (
            degradation_percent
            <= reduced_feature_tolerance_percent
        ):
            acceptable_reduced.append(
                {
                    "platform": str(
                        reduced["platform"]
                    ),
                    "best_model": str(
                        reduced["best_model"]
                    ),
                    "features": int(
                        reduced["feature_count"]
                    ),
                    "validation_rmse": float(
                        reduced[
                            "validation_rmse"
                        ]
                    ),
                    "validation_degradation_percent": (
                        degradation_percent
                    ),
                    "training_time_change_percent": (
                        training_change_percent
                    ),
                }
            )

    efficiency_choice = None

    if acceptable_reduced:
        efficiency_choice = sorted(
            acceptable_reduced,
            key=lambda item: (
                item[
                    "validation_degradation_percent"
                ],
                item["validation_rmse"],
                item[
                    "training_time_change_percent"
                ],
            ),
        )[0]

    return {
        "status": "PASS",
        "selection_basis": (
            "Primary recommendation is selected by the lowest "
            "validation RMSE. Test metrics are reported as final "
            "unbiased evaluation, not used as the first selection key."
        ),
        "best_predictive_run": {
            "platform": str(
                predictive["platform"]
            ),
            "feature_set": str(
                predictive["feature_set"]
            ),
            "feature_count": int(
                predictive["feature_count"]
            ),
            "best_model": str(
                predictive["best_model"]
            ),
            "validation_rmse": float(
                predictive["validation_rmse"]
            ),
            "test_rmse": float(
                predictive["test_rmse"]
            ),
            "test_mae": float(
                predictive["test_mae"]
            ),
            "test_r2": float(
                predictive["test_r2"]
            ),
        },
        "fastest_automl_run": {
            "platform": str(
                fastest["platform"]
            ),
            "feature_set": str(
                fastest["feature_set"]
            ),
            "training_seconds": float(
                fastest["training_seconds"]
            ),
            "validation_rmse": float(
                fastest["validation_rmse"]
            ),
        },
        "reduced_feature_tolerance_percent": float(
            reduced_feature_tolerance_percent
        ),
        "recommended_reduced_feature_run": (
            efficiency_choice
        ),
    }


def feature_overlap(
    first: list[str],
    second: list[str],
) -> dict[str, Any]:
    """Calculate top-feature overlap between two platforms."""
    first_set = set(first)
    second_set = set(second)
    union = first_set.union(second_set)
    intersection = sorted(
        first_set.intersection(second_set)
    )

    jaccard = (
        len(intersection) / len(union)
        if union
        else 0.0
    )

    return {
        "first_features": first,
        "second_features": second,
        "shared_features": intersection,
        "shared_feature_count": len(
            intersection
        ),
        "jaccard_similarity": float(
            jaccard
        ),
    }
