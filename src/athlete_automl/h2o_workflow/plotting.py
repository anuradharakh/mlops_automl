"""Plots for H2O AutoML evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_leaderboard_rmse(
    leaderboard: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot top H2O models by validation RMSE."""
    data = (
        leaderboard.head(top_n)
        .sort_values(
            "rmse",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )
    axis.barh(
        data["model_id"],
        data["rmse"],
    )
    axis.set_title(
        "H2O AutoML models by validation RMSE"
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
    variable_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot H2O variable importance."""
    importance_column = next(
        column
        for column in (
            "relative_importance",
            "scaled_importance",
            "percentage",
        )
        if column in variable_importance.columns
    )

    data = (
        variable_importance.head(top_n)
        .sort_values(
            importance_column,
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )
    axis.barh(
        data["variable"],
        data[importance_column],
    )
    axis.set_title(
        "H2O AutoML feature importance"
    )
    axis.set_xlabel(
        importance_column.replace(
            "_",
            " ",
        ).title()
    )
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
        predictions[
            "actual_total_lift"
        ].min(),
        predictions[
            "predicted_total_lift"
        ].min(),
    )
    upper = max(
        predictions[
            "actual_total_lift"
        ].max(),
        predictions[
            "predicted_total_lift"
        ].max(),
    )

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )
    axis.set_title(
        "H2O actual versus predicted total lift"
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
