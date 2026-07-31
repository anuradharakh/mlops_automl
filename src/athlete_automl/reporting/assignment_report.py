"""Utilities for producing the Assignment 3 written report."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_number(
    value: Any,
    digits: int = 4,
) -> str:
    """Format numeric values consistently."""
    if value is None:
        return "Not available"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    digits: int = 4,
) -> str:
    """Render a DataFrame as a dependency-free Markdown table."""
    data = dataframe.copy()

    if columns is not None:
        available = [
            column
            for column in columns
            if column in data.columns
        ]
        data = data[available]

    if data.empty:
        return "_No rows were available._"

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            data[column] = data[column].map(
                lambda value: (
                    format_number(
                        value,
                        digits=digits,
                    )
                    if pd.notna(value)
                    else ""
                )
            )

    header = (
        "| "
        + " | ".join(
            str(column)
            for column in data.columns
        )
        + " |"
    )
    separator = (
        "| "
        + " | ".join(
            ["---"] * len(data.columns)
        )
        + " |"
    )
    rows = [
        "| "
        + " | ".join(
            str(value)
            for value in row
        )
        + " |"
        for row in data.itertuples(
            index=False,
            name=None,
        )
    ]

    return "\n".join(
        [header, separator, *rows]
    )


def comparison_assessment(
    comparison: dict[str, Any],
) -> str:
    """Describe the reduced-feature validation result."""
    assessment = str(
        comparison.get(
            "validation_performance_assessment",
            "not available",
        )
    )
    delta = comparison.get(
        "validation_rmse_delta_top_minus_all"
    )

    if delta is None:
        return assessment

    return (
        f"{assessment.capitalize()} validation performance; "
        f"top-features minus all-features RMSE = "
        f"{format_number(delta)}."
    )


def historical_baseline_text(
    baseline_config: dict[str, Any],
) -> str:
    """Build the Assignment 1 baseline disclosure."""
    baseline = baseline_config.get(
        "assignment1_baseline",
        {}
    )

    if not baseline.get("available", False):
        return (
            "Exact Assignment 1 validation and runtime artifacts were not "
            "entered in `configs/assignment1_baseline.yaml`. Therefore, "
            "the numerical comparison below uses a same-split reconstructed "
            "Random Forest baseline from Phase 4. This improves metric "
            "comparability but is not presented as the original Assignment "
            "1 result. The historical configuration file should be updated "
            "before submission when the original metrics are available."
        )

    values = [
        (
            "model",
            baseline.get("model_name"),
        ),
        (
            "version",
            baseline.get(
                "feature_or_dataset_version"
            ),
        ),
        (
            "validation metric",
            baseline.get(
                "validation_metric_name"
            ),
        ),
        (
            "validation value",
            format_number(
                baseline.get(
                    "validation_metric_value"
                )
            ),
        ),
        (
            "test RMSE",
            format_number(
                baseline.get("test_rmse")
            ),
        ),
        (
            "training seconds",
            format_number(
                baseline.get(
                    "training_seconds"
                )
            ),
        ),
    ]

    return (
        "The original Assignment 1 baseline was available: "
        + "; ".join(
            f"{name}={value}"
            for name, value in values
        )
        + "."
    )


def bullet_list(items: list[str]) -> str:
    """Render Markdown bullets."""
    return "\n".join(
        f"- {item}"
        for item in items
    )
