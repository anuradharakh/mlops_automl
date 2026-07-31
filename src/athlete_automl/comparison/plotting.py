"""Plots for the final baseline and AutoML comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _labels(data: pd.DataFrame) -> pd.Series:
    """Build concise labels for comparison plots."""
    return (
        data["platform"].astype(str)
        + " | "
        + data["feature_set"].astype(str)
    )


def plot_test_rmse(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot test RMSE across baseline and AutoML runs."""
    data = comparison.sort_values(
        "test_rmse",
        ascending=True,
    ).copy()
    data["label"] = _labels(data)

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )
    axis.barh(
        data["label"],
        data["test_rmse"],
    )
    axis.set_title(
        "Baseline and AutoML test RMSE"
    )
    axis.set_xlabel(
        "Test RMSE — lower is better"
    )
    axis.set_ylabel("Run")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_test_r2(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot test R-squared across runs."""
    data = comparison.sort_values(
        "test_r2",
        ascending=True,
    ).copy()
    data["label"] = _labels(data)

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )
    axis.barh(
        data["label"],
        data["test_r2"],
    )
    axis.set_title(
        "Baseline and AutoML test R-squared"
    )
    axis.set_xlabel(
        "Test R-squared — higher is better"
    )
    axis.set_ylabel("Run")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_training_time(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot wall-clock training time."""
    data = comparison.sort_values(
        "training_seconds",
        ascending=True,
    ).copy()
    data["label"] = _labels(data)

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )
    axis.barh(
        data["label"],
        data["training_seconds"],
    )
    axis.set_title(
        "Baseline and AutoML training time"
    )
    axis.set_xlabel(
        "Wall-clock training seconds — lower is better"
    )
    axis.set_ylabel("Run")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
