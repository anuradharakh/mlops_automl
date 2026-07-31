"""Plotting helpers for AutoML evidence artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_leaderboard_rmse(
    leaderboard: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot top models by validation RMSE."""
    data = (
        leaderboard.head(top_n)
        .sort_values(
            "validation_rmse",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )
    axis.barh(
        data["model"],
        data["validation_rmse"],
    )
    axis.set_title(
        "AutoGluon models by validation RMSE"
    )
    axis.set_xlabel("Validation RMSE")
    axis.set_ylabel("Model")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot the strongest permutation-importance features."""
    data = (
        feature_importance.head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )
    axis.barh(
        data.index.astype(str),
        data["importance"],
    )
    axis.set_title(
        "AutoGluon permutation feature importance"
    )
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot actual versus predicted target values."""
    figure, axis = plt.subplots(
        figsize=(7, 7)
    )
    axis.scatter(
        predictions["actual_total_lift"],
        predictions["predicted_total_lift"],
        alpha=0.35,
        s=14,
    )

    lower = min(
        predictions["actual_total_lift"].min(),
        predictions["predicted_total_lift"].min(),
    )
    upper = max(
        predictions["actual_total_lift"].max(),
        predictions["predicted_total_lift"].max(),
    )

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )
    axis.set_title(
        "AutoGluon actual versus predicted total lift"
    )
    axis.set_xlabel("Actual total lift")
    axis.set_ylabel("Predicted total lift")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
