"""Build the final baseline and cross-platform AutoML comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from athlete_automl.comparison.final_comparison import (
    add_random_forest_improvement,
    automl_summary_to_row,
    baseline_summary_to_rows,
    build_recommendation,
    feature_overlap,
    rank_comparison,
    validate_run_summary,
)
from athlete_automl.comparison.plotting import (
    plot_test_r2,
    plot_test_rmse,
    plot_training_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "final_comparison.yaml"
)


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def markdown_table(
    dataframe: pd.DataFrame,
) -> str:
    """Render a compact markdown table without optional dependencies."""
    columns = [
        "platform",
        "feature_set",
        "feature_count",
        "best_model",
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "training_seconds",
    ]

    display = dataframe[columns].copy()

    for column in (
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "training_seconds",
    ):
        display[column] = display[
            column
        ].map(
            lambda value: f"{float(value):.4f}"
        )

    header = (
        "| "
        + " | ".join(display.columns)
        + " |"
    )
    separator = (
        "| "
        + " | ".join(
            ["---"] * len(display.columns)
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
        for row in display.itertuples(
            index=False,
            name=None,
        )
    ]

    return "\n".join(
        [header, separator, *rows]
    )


def main() -> None:
    """Create tables, plots, feature overlap, and recommendation."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    input_paths = {
        key: resolve_path(value)
        for key, value
        in config["inputs"].items()
    }

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in input_paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required experiment results are missing: "
            + ", ".join(missing)
        )

    baseline = load_json(
        input_paths["baseline_summary"]
    )
    autogluon_all = load_json(
        input_paths["autogluon_all"]
    )
    autogluon_top = load_json(
        input_paths["autogluon_top"]
    )
    h2o_all = load_json(
        input_paths["h2o_all"]
    )
    h2o_top = load_json(
        input_paths["h2o_top"]
    )

    for name, summary in {
        "autogluon_all": autogluon_all,
        "autogluon_top": autogluon_top,
        "h2o_all": h2o_all,
        "h2o_top": h2o_top,
    }.items():
        validate_run_summary(
            summary,
            name,
        )

    rows = baseline_summary_to_rows(
        baseline
    )
    rows.extend(
        automl_summary_to_row(summary)
        for summary in (
            autogluon_all,
            autogluon_top,
            h2o_all,
            h2o_top,
        )
    )

    comparison = pd.DataFrame(rows)
    comparison = rank_comparison(
        comparison
    )
    comparison = (
        add_random_forest_improvement(
            comparison
        )
    )

    recommendation = build_recommendation(
        comparison,
        reduced_feature_tolerance_percent=float(
            config["selection"][
                "reduced_feature_tolerance_percent"
            ]
        ),
    )

    overlap = {
        "autogluon_vs_h2o_top_five": (
            feature_overlap(
                list(
                    autogluon_all[
                        "top_five_features"
                    ]
                ),
                list(
                    h2o_all[
                        "top_five_features"
                    ]
                ),
            )
        ),
        "autogluon_vs_random_forest_top_five": (
            feature_overlap(
                list(
                    autogluon_all[
                        "top_five_features"
                    ]
                ),
                list(
                    baseline[
                        "top_five_features"
                    ]
                ),
            )
        ),
        "h2o_vs_random_forest_top_five": (
            feature_overlap(
                list(
                    h2o_all[
                        "top_five_features"
                    ]
                ),
                list(
                    baseline[
                        "top_five_features"
                    ]
                ),
            )
        ),
    }

    output_dir = resolve_path(
        config["reports"]["output_dir"]
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_dir
        / "platform_comparison.csv",
        index=False,
    )
    (
        output_dir / "recommendation.json"
    ).write_text(
        json.dumps(
            recommendation,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        output_dir / "feature_overlap.json"
    ).write_text(
        json.dumps(
            overlap,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_test_rmse(
        comparison,
        output_dir
        / "test_rmse_comparison.png",
    )
    plot_test_r2(
        comparison,
        output_dir
        / "test_r2_comparison.png",
    )
    plot_training_time(
        comparison,
        output_dir
        / "training_time_comparison.png",
    )

    best = recommendation[
        "best_predictive_run"
    ]
    reduced = recommendation[
        "recommended_reduced_feature_run"
    ]

    reduced_text = (
        "No reduced-feature run stayed within the "
        "configured validation-RMSE tolerance."
        if reduced is None
        else (
            f"{reduced['platform']} top-features run "
            f"with {reduced['features']} features; "
            f"validation change "
            f"{reduced['validation_degradation_percent']:.2f}% "
            f"and training-time change "
            f"{reduced['training_time_change_percent']:.2f}%."
        )
    )

    report = f"""# Final AutoML Comparison

## Experimental controls

All current-split models use the same cleaned train, validation, and test
partitions. Target components and identifiers are excluded from model
features. Model selection is ranked primarily by validation RMSE; test
metrics are retained for final evaluation.

## Comparison

{markdown_table(comparison)}

## Recommendation

The strongest predictive run by validation RMSE is
**{best['platform']} — {best['feature_set']}** using
**{best['feature_count']} features** and model
`{best['best_model']}`.

- Validation RMSE: {best['validation_rmse']:.4f}
- Test RMSE: {best['test_rmse']:.4f}
- Test MAE: {best['test_mae']:.4f}
- Test R-squared: {best['test_r2']:.4f}

## Reduced-feature assessment

{reduced_text}

## Feature agreement

AutoGluon and H2O share
**{overlap['autogluon_vs_h2o_top_five']['shared_feature_count']}**
of their top-five features. Their Jaccard similarity is
**{overlap['autogluon_vs_h2o_top_five']['jaccard_similarity']:.4f}**.

## Interpretation notes

- Negative RMSE or MAE change versus Random Forest indicates improvement.
- Positive R-squared change versus Random Forest indicates improvement.
- Wall-clock training time is machine-dependent and should be interpreted
  together with predictive quality.
- A reduced-feature model is recommended only when its validation RMSE
  remains within the configured tolerance of its platform's all-features
  run.
"""

    (
        output_dir
        / "final_comparison_report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        "Final comparison generated."
    )
    print(
        "Best predictive run: "
        f"{best['platform']} | "
        f"{best['feature_set']} | "
        f"{best['best_model']}"
    )
    print(
        f"Reports: {output_dir}"
    )
    print(
        "PHASE 4B STATUS: PASS"
    )


if __name__ == "__main__":
    main()
