"""Build the final Assignment 3 report and evidence manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from athlete_automl.reporting.assignment_report import (
    bullet_list,
    comparison_assessment,
    format_number,
    historical_baseline_text,
    markdown_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "submission.yaml"
)


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV artifact."""
    return pd.read_csv(path)


def main() -> None:
    """Generate the written report from completed experiment outputs."""
    config = load_yaml(CONFIG_PATH)
    assignment = config["assignment"]
    paths = {
        key: resolve_path(value)
        for key, value in config[
            "paths"
        ].items()
    }
    output_path = resolve_path(
        config["outputs"]["report"]
    )
    manifest_path = resolve_path(
        config["outputs"][
            "evidence_manifest"
        ]
    )

    required = [
        path
        for key, path in paths.items()
        if key != "assignment1_baseline"
    ]
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Cannot build the final report. Missing artifacts: "
            + ", ".join(missing)
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_list = load_json(
        paths["feature_list"]
    )
    target_quality = load_json(
        paths["target_quality"]
    )
    split_summary = read_csv(
        paths["split_summary"]
    )

    ag_all = load_json(
        paths["autogluon_all_summary"]
    )
    ag_top = load_json(
        paths["autogluon_top_summary"]
    )
    ag_compare = load_json(
        paths["autogluon_comparison"]
    )
    ag_all_score = read_csv(
        paths["autogluon_all_score"]
    )
    ag_all_speed = read_csv(
        paths["autogluon_all_speed"]
    )
    ag_top_score = read_csv(
        paths["autogluon_top_score"]
    )
    ag_top_speed = read_csv(
        paths["autogluon_top_speed"]
    )

    h2o_insights = load_json(
        paths["h2o_data_insights"]
    )
    h2o_all = load_json(
        paths["h2o_all_summary"]
    )
    h2o_top = load_json(
        paths["h2o_top_summary"]
    )
    h2o_compare = load_json(
        paths["h2o_comparison"]
    )
    h2o_all_score = read_csv(
        paths["h2o_all_score"]
    )
    h2o_all_speed = read_csv(
        paths["h2o_all_speed"]
    )
    h2o_top_score = read_csv(
        paths["h2o_top_score"]
    )
    h2o_top_speed = read_csv(
        paths["h2o_top_speed"]
    )

    reconstructed_baseline = load_json(
        paths["reconstructed_baseline"]
    )
    final_comparison = read_csv(
        paths["final_comparison"]
    )
    recommendation = load_json(
        paths["recommendation"]
    )
    overlap = load_json(
        paths["feature_overlap"]
    )
    historical_baseline = load_yaml(
        paths["assignment1_baseline"]
    )

    data_rows_after = target_quality.get(
        "rows_after",
        target_quality.get(
            "clean_rows",
            target_quality.get(
                "final_rows",
                "Not available",
            ),
        ),
    )
    rows_removed = target_quality.get(
        "rows_removed",
        "Not available",
    )
    clean_minimum = target_quality.get(
        "clean_target_min",
        target_quality.get(
            "target_min",
            8,
        ),
    )
    clean_maximum = target_quality.get(
        "clean_target_max",
        target_quality.get(
            "target_max",
            2330,
        ),
    )

    best = recommendation[
        "best_predictive_run"
    ]
    ag_feature_list = ", ".join(
        str(feature)
        for feature in ag_all[
            "top_five_features"
        ]
    )
    h2o_feature_list = ", ".join(
        str(feature)
        for feature in h2o_all[
            "top_five_features"
        ]
    )

    discussion = config["discussion"]

    report = f"""# {assignment['title']}

**Course:** {assignment['course']}  
**Author:** {assignment['author']}  
**Primary AutoML platform:** {assignment['primary_platform']}  
**Platform mode:** {assignment['primary_platform_mode']}  
**Primary validation metric:** {assignment['validation_metric']}  
**Random seed:** {assignment['random_seed']}

## Executive summary

The workflow compares a primary AutoGluon plus MLflow implementation with
the required H2O AutoML repeat. Both platforms use the same cleaned,
deterministic train, validation, and test partitions. Each platform is run
first with all approved features and then with its own top three features.
The strongest predictive run by validation RMSE was
**{best['platform']} — {best['feature_set']}**, using model
`{best['best_model']}`. Its test RMSE was
**{format_number(best['test_rmse'])}**, test MAE was
**{format_number(best['test_mae'])}**, and test R-squared was
**{format_number(best['test_r2'])}**.

## 1. Dataset loading and setup

- Dataset source: `{assignment['dataset_source']}`
- Target: `{assignment['target']}`
- Processed rows after target-quality filtering: {data_rows_after}
- Rows removed by the target-quality gate: {rows_removed}
- Clean target range: {format_number(clean_minimum, 2)} to
  {format_number(clean_maximum, 2)}
- Approved model features: {feature_list['feature_count']}
- Identifier excluded from modeling: `{feature_list['identifier']}`
- Target components excluded from modeling to prevent leakage:
  `deadlift`, `candj`, `snatch`, and `backsq`

{assignment['processed_dataset_description']}

### Fixed split summary

{markdown_table(split_summary)}

### Assumptions and preprocessing choices

- Lift-component sentinel value `1` was treated as missing before the target
  was calculated.
- Target totals outside the documented project-quality range were audited
  and removed rather than capped.
- Features with more than 70% missingness were removed before AutoML.
- The same fixed partitions were reused across AutoGluon, H2O, and the
  reconstructed baseline.

## 2. Chosen platform configuration

The selected primary workflow is **AutoGluon with MLflow**, classified as
**full-code AutoML**.

{discussion['primary_platform_reason']}

The full-code classification is appropriate because datasets, feature
contracts, runtime budgets, experiment execution, metric extraction, and
artifact generation are controlled through Python and YAML. AutoGluon
automates major modeling steps, but the workflow still requires code and
human decisions.

### AutoGluon all-features configuration

- Feature count: {ag_all['feature_count']}
- Runtime budget: recorded in MLflow and the run configuration
- Problem type: regression
- Validation metric: RMSE
- Best model: `{ag_all['best_model']}`
- Validation RMSE: {format_number(ag_all['validation_rmse'])}
- Test RMSE: {format_number(ag_all['test_rmse'])}
- Test MAE: {format_number(ag_all['test_mae'])}
- Test R-squared: {format_number(ag_all['test_r2'])}

## 3. AutoGluon run using all features

### Top three models by validation score

{markdown_table(
    ag_all_score,
    columns=[
        'validation_rank',
        'model',
        'validation_rmse',
        'test_rmse',
        'fit_time_marginal',
        'pred_time_test_marginal',
    ],
)}

### Top three models by speed

{markdown_table(
    ag_all_speed,
    columns=[
        'speed_rank',
        'model',
        ag_all.get('speed_measure', 'fit_time_marginal'),
        'validation_rmse',
        'test_rmse',
    ],
)}

## 4. AutoGluon data insights and feature importance

AutoGluon's top five features were: **{ag_feature_list}**.

Feature importance indicates predictive contribution within the fitted
model; it is not evidence of causality. Engineered features can also share
information with their source variables, so correlated-feature rankings
should be interpreted together rather than independently.

## 5. AutoGluon top-features experiment

Selected features: **{', '.join(ag_top['model_features'])}**

### Top three models by validation score

{markdown_table(
    ag_top_score,
    columns=[
        'validation_rank',
        'model',
        'validation_rmse',
        'test_rmse',
        'fit_time_marginal',
        'pred_time_test_marginal',
    ],
)}

### Top three models by speed

{markdown_table(
    ag_top_speed,
    columns=[
        'speed_rank',
        'model',
        ag_top.get('speed_measure', 'fit_time_marginal'),
        'validation_rmse',
        'test_rmse',
    ],
)}

### Feature-reduction assessment

{comparison_assessment(ag_compare)}

The all-features and top-features runs used the same split strategy and
metric. This makes the validation comparison direct, although individual
AutoML components can still show small run-to-run variation.

## 6. Speed definition and tradeoffs

**Primary platform:** {discussion['speed_definition_primary']}

The fastest model is not automatically the best production choice. A
slightly slower model may be justified when its validation improvement is
meaningful and prediction latency remains acceptable. Conversely, a
top-three-feature model may be preferable when it preserves validation
quality while reducing training, scoring, monitoring, and explanation
complexity.

## 7. Comparison with the Assignment 1 baseline

{historical_baseline_text(historical_baseline)}

### Same-split baseline and AutoML comparison

{markdown_table(
    final_comparison,
    columns=[
        'platform',
        'feature_set',
        'feature_count',
        'best_model',
        'validation_rmse',
        'test_rmse',
        'test_mae',
        'test_r2',
        'training_seconds',
        'prediction_seconds',
        'validation_rmse_change_vs_rf_percent',
        'test_rmse_change_vs_rf_percent',
    ],
)}

AutoML reduces manual algorithm selection and hyperparameter experimentation,
but introduces additional dependencies, compute use, artifact volume, and
governance needs. The reconstructed Random Forest provides a controlled
same-split comparison. Any comparison with the original Assignment 1 run is
limited when preprocessing, feature definitions, split logic, or hardware
differ.

## 8. Platform AutoML mode assessment

### Automated

{bullet_list(discussion['automated_steps'])}

### Manual decisions

{bullet_list(discussion['manual_decisions'])}

### Operational strengths

{bullet_list(discussion['operational_strengths'])}

### Operational risks

{bullet_list(discussion['operational_risks'])}

Screenshots are not mandatory for this primary workflow because it is
full-code rather than no-code or low-code. MLflow screenshots may still be
included as supplementary execution evidence.

## 9. H2O AutoML repeat

### Data insights

- Train rows: {h2o_insights['row_counts']['train']}
- Validation rows: {h2o_insights['row_counts']['validation']}
- Test rows: {h2o_insights['row_counts']['test']}
- Numeric features: {h2o_insights['numeric_feature_count']}
- Categorical features: {h2o_insights['categorical_feature_count']}
- H2O top five features: **{h2o_feature_list}**
- Best all-features model: `{h2o_all['best_model']}`
- Validation RMSE: {format_number(h2o_all['validation_rmse'])}
- Test RMSE: {format_number(h2o_all['test_rmse'])}
- Test MAE: {format_number(h2o_all['test_mae'])}
- Test R-squared: {format_number(h2o_all['test_r2'])}

### H2O top three models by validation score — all features

{markdown_table(
    h2o_all_score,
    columns=[
        'validation_rank',
        'model_id',
        'rmse',
        'mae',
        'r2',
        'training_time_ms',
        'predict_time_per_row_ms',
    ],
)}

### H2O top three models by training speed — all features

{markdown_table(
    h2o_all_speed,
    columns=[
        'speed_rank',
        'model_id',
        'training_time_ms',
        'rmse',
        'mae',
        'r2',
    ],
)}

### H2O top-features run

Selected features: **{', '.join(h2o_top['model_features'])}**

{markdown_table(
    h2o_top_score,
    columns=[
        'validation_rank',
        'model_id',
        'rmse',
        'mae',
        'r2',
        'training_time_ms',
    ],
)}

### H2O top-features speed

{markdown_table(
    h2o_top_speed,
    columns=[
        'speed_rank',
        'model_id',
        'training_time_ms',
        'rmse',
        'mae',
        'r2',
    ],
)}

### H2O feature-reduction assessment

{comparison_assessment(h2o_compare)}

## 10. Cross-platform findings

AutoGluon and H2O shared
**{overlap['autogluon_vs_h2o_top_five']['shared_feature_count']}**
top-five features, with Jaccard similarity
**{format_number(overlap['autogluon_vs_h2o_top_five']['jaccard_similarity'])}**.

The platforms may select different leaders because they search different
algorithm families, hyperparameter spaces, ensembles, and stopping
strategies. Their time measurements are also not perfectly identical:
AutoGluon model-level speed uses its fit-time columns, while H2O uses
platform-reported `training_time_ms`. Wall-clock run times are retained for
broader comparison.

## 11. Final recommendation

The final model should be selected primarily using validation RMSE, followed
by test-set confirmation, execution cost, interpretability, and deployment
requirements. Based on the completed artifacts, the recommended predictive
run is **{best['platform']} — {best['feature_set']}** with model
`{best['best_model']}`.

A reduced-feature model is operationally attractive only when its validation
performance remains within the configured tolerance. The final comparison
artifact records whether such a reduced model qualified.

## 12. Reproducibility

From a clean Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-automl.txt
python -m pip install -r requirements-h2o.txt
python -m pip install -e .

python scripts/profile_data.py
python scripts/prepare_data.py
python scripts/clean_target_and_resplit.py

python scripts/run_autogluon_all_features.py --overwrite
python scripts/run_autogluon_top_features.py --overwrite

python scripts/check_h2o_environment.py
python scripts/run_h2o_all_features.py --overwrite
python scripts/run_h2o_top_features.py --overwrite

python scripts/run_baselines.py
python scripts/build_final_comparison.py
python scripts/build_assignment_report.py
python scripts/validate_assignment_submission.py
```

Reproduction requires the original `athletes.csv` file under `data/raw/`.
Generated datasets, model binaries, and the local MLflow SQLite database are
not required in Git when the repository includes the code, dependency files,
configuration, saved report artifacts, and execution instructions.

## 13. Limitations

- Runtime budgets can cause small differences across machines.
- The 70% sparse-feature threshold is a project decision rather than a
  universal rule.
- The target-quality maximum is based on the project data history and is
  documented as an auditable assumption.
- Feature importance is model-dependent and does not establish causality.
- Ensemble leaders can be more difficult to explain and deploy than a single
  model.
- The historical Assignment 1 comparison remains incomplete until exact
  original metrics and speed evidence are entered in
  `configs/assignment1_baseline.yaml`.
"""

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    evidence_rows = []
    for key, path in paths.items():
        evidence_rows.append(
            {
                "artifact_key": key,
                "repository_path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "exists": path.exists(),
                "file_type": path.suffix.lstrip(
                    "."
                ),
            }
        )

    pd.DataFrame(
        evidence_rows
    ).to_csv(
        manifest_path,
        index=False,
    )

    print(
        f"Report: {output_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        "PHASE 5A STATUS: PASS"
    )


if __name__ == "__main__":
    main()
