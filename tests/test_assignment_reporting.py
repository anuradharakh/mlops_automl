"""Tests for Assignment 3 report helpers."""

from __future__ import annotations

import pandas as pd

from athlete_automl.reporting.assignment_report import (
    comparison_assessment,
    historical_baseline_text,
    markdown_table,
)


def test_markdown_table() -> None:
    """Markdown rendering should include headers and rows."""
    result = markdown_table(
        pd.DataFrame(
            {
                "model": ["A"],
                "rmse": [123.45678],
            }
        )
    )

    assert "| model | rmse |" in result
    assert "| A | 123.4568 |" in result


def test_comparison_assessment() -> None:
    """Feature-reduction assessment should include its delta."""
    result = comparison_assessment(
        {
            "validation_performance_assessment": (
                "degraded"
            ),
            "validation_rmse_delta_top_minus_all": (
                2.5
            ),
        }
    )

    assert "Degraded" in result
    assert "2.5000" in result


def test_missing_historical_baseline_is_disclosed() -> None:
    """Unavailable Assignment 1 evidence should be explicit."""
    result = historical_baseline_text(
        {
            "assignment1_baseline": {
                "available": False,
            }
        }
    )

    assert "not entered" in result
    assert "reconstructed" in result
