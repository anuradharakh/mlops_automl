"""H2O feature-selection and reduced-model comparison utilities."""

from __future__ import annotations

from typing import Any


def extract_h2o_top_features(
    run_summary: dict[str, Any],
    expected_count: int = 3,
) -> list[str]:
    """Extract the ordered H2O top-feature list."""
    features = run_summary.get(
        "top_three_features"
    )

    if not isinstance(features, list):
        raise ValueError(
            "The H2O all-features summary does not contain "
            "'top_three_features'."
        )

    normalized = [
        str(feature).strip()
        for feature in features
        if str(feature).strip()
    ]

    if len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} H2O top features, "
            f"found {len(normalized)}."
        )

    if len(set(normalized)) != expected_count:
        raise ValueError(
            "H2O top features must be unique."
        )

    return normalized


def compare_h2o_run_summaries(
    all_features: dict[str, Any],
    top_features: dict[str, Any],
) -> dict[str, Any]:
    """Compare H2O all-feature and top-feature experiments."""
    required = (
        "feature_count",
        "model_features",
        "best_model",
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "training_wall_clock_seconds",
        "prediction_seconds",
    )

    missing = [
        field
        for field in required
        if field not in all_features
        or field not in top_features
    ]

    if missing:
        raise ValueError(
            "Missing H2O comparison fields: "
            + ", ".join(missing)
        )

    validation_rmse_delta = (
        float(top_features["validation_rmse"])
        - float(all_features["validation_rmse"])
    )
    test_rmse_delta = (
        float(top_features["test_rmse"])
        - float(all_features["test_rmse"])
    )
    test_mae_delta = (
        float(top_features["test_mae"])
        - float(all_features["test_mae"])
    )
    test_r2_delta = (
        float(top_features["test_r2"])
        - float(all_features["test_r2"])
    )
    training_seconds_delta = (
        float(
            top_features[
                "training_wall_clock_seconds"
            ]
        )
        - float(
            all_features[
                "training_wall_clock_seconds"
            ]
        )
    )
    prediction_seconds_delta = (
        float(top_features["prediction_seconds"])
        - float(all_features["prediction_seconds"])
    )

    if validation_rmse_delta < 0:
        validation_assessment = "improved"
    elif validation_rmse_delta > 0:
        validation_assessment = "degraded"
    else:
        validation_assessment = "maintained"

    return {
        "status": "PASS",
        "platform": "H2O AutoML",
        "all_features_count": int(
            all_features["feature_count"]
        ),
        "top_features_count": int(
            top_features["feature_count"]
        ),
        "selected_top_features": list(
            top_features["model_features"]
        ),
        "all_features_best_model": (
            all_features["best_model"]
        ),
        "top_features_best_model": (
            top_features["best_model"]
        ),
        "all_features_validation_rmse": float(
            all_features["validation_rmse"]
        ),
        "top_features_validation_rmse": float(
            top_features["validation_rmse"]
        ),
        "validation_rmse_delta_top_minus_all": (
            validation_rmse_delta
        ),
        "all_features_test_rmse": float(
            all_features["test_rmse"]
        ),
        "top_features_test_rmse": float(
            top_features["test_rmse"]
        ),
        "test_rmse_delta_top_minus_all": (
            test_rmse_delta
        ),
        "all_features_test_mae": float(
            all_features["test_mae"]
        ),
        "top_features_test_mae": float(
            top_features["test_mae"]
        ),
        "test_mae_delta_top_minus_all": (
            test_mae_delta
        ),
        "all_features_test_r2": float(
            all_features["test_r2"]
        ),
        "top_features_test_r2": float(
            top_features["test_r2"]
        ),
        "test_r2_delta_top_minus_all": (
            test_r2_delta
        ),
        "all_features_training_seconds": float(
            all_features[
                "training_wall_clock_seconds"
            ]
        ),
        "top_features_training_seconds": float(
            top_features[
                "training_wall_clock_seconds"
            ]
        ),
        "training_seconds_delta_top_minus_all": (
            training_seconds_delta
        ),
        "all_features_prediction_seconds": float(
            all_features["prediction_seconds"]
        ),
        "top_features_prediction_seconds": float(
            top_features["prediction_seconds"]
        ),
        "prediction_seconds_delta_top_minus_all": (
            prediction_seconds_delta
        ),
        "validation_performance_assessment": (
            validation_assessment
        ),
        "interpretation": (
            "Negative RMSE and MAE deltas favor the H2O "
            "top-feature run. Positive R-squared deltas favor "
            "the top-feature run. Negative timing deltas "
            "indicate faster execution."
        ),
    }
