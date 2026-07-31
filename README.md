# ADSP 31021 Assignment 3 - AutoML

## Athlete Total-Lift Prediction with AutoGluon, MLflow, and H2O AutoML

This repository implements a reproducible AutoML workflow for predicting an athlete's
combined lift total using the `athletes.csv` dataset.

```text
total_lift = deadlift + candj + snatch + backsq
```

The project compares:

- AutoGluon with MLflow as the primary full-code AutoML workflow
- H2O AutoML as the required repeat workflow
- All approved features versus each platform's top three features
- AutoML models versus a same-split Random Forest baseline
- Current AutoML results versus the original Assignment 1 baseline where comparable
- Validation performance, test performance, feature importance, and execution speed

## Workflow

```text
athletes.csv
  -> profiling and schema validation
  -> sentinel handling and target construction
  -> target-quality audit
  -> leakage-safe feature engineering
  -> deterministic 64/16/20 train-validation-test split
  -> AutoGluon all-features run
  -> AutoGluon top-three-features run
  -> H2O all-features run
  -> H2O top-three-features run
  -> conventional baselines
  -> cross-platform comparison
  -> detailed report, high-level PDF, validation, and submission ZIP
```

## Dataset and Modeling Contract

| Item | Value |
|---|---:|
| Raw rows | 423,006 |
| Final modeling rows | 53,505 |
| Corrupted target rows removed | 29 |
| Final target range | 8 to 2,330 |
| Approved model features | 13 |
| Train rows | 34,243 |
| Validation rows | 8,561 |
| Test rows | 10,701 |
| Random seed | 42 |
| Primary selection metric | Validation RMSE |
| Additional metrics | Test RMSE, MAE, and R-squared |

The target components are excluded because they directly define `total_lift`:

```text
deadlift
candj
snatch
backsq
```

The target, athlete identifier, and other non-model metadata are also excluded from the
predictor matrix. Features with more than 70% missing observations were removed before
AutoML.

## Platforms

### AutoGluon with MLflow

AutoGluon is the primary full-code AutoML engine. It automates preprocessing, candidate
model training, algorithm selection, hyperparameter exploration, leaderboard generation,
and ensembling where supported.

MLflow records parameters, validation and test metrics, package versions, leaderboards,
feature-importance outputs, and comparison artifacts.

### H2O AutoML

H2O repeats the workflow using the same fixed train, validation, and test partitions. It
provides leaderboard metrics, model-level training and prediction speed, model persistence,
and variable importance where supported.

## Experiment Matrix

| Experiment | Platform | Feature set |
|---|---|---|
| `autogluon_all_features` | AutoGluon + MLflow | All 13 approved features |
| `autogluon_top_features` | AutoGluon + MLflow | AutoGluon's top 3 features |
| `h2o_all_features` | H2O AutoML | All 13 approved features |
| `h2o_top_features` | H2O AutoML | H2O's top 3 features |
| `sklearn_baselines` | Scikit-learn | Mean and Random Forest baselines |

## Environment Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-automl.txt
python -m pip install -r requirements-h2o.txt
python -m pip install -r requirements-report.txt
python -m pip install -e .
```

H2O also requires Java:

```bash
java -version
python scripts/check_h2o_environment.py
```

Place the source dataset at:

```text
data/raw/athletes.csv
```

## Reproduce the Complete Workflow

```bash
source .venv/bin/activate

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
python scripts/build_high_level_report.py
python scripts/validate_assignment_submission.py
python scripts/create_submission_zip.py
```

Run the final reporting and packaging steps together:

```bash
python scripts/finalize_submission.py
```

## Main Outputs

```text
reports/autogluon/
reports/h2o/
reports/baseline/
reports/final_comparison/

submission/assignment3_report.md
submission/Assignment3_AutoML_High_Level_Report.md
submission/Assignment3_AutoML_High_Level_Report.pdf
submission/evidence_manifest.csv
submission/validation_report.json

dist/mlops_automl_assignment3_submission.zip
```

## Assignment 1 Baseline

The original Assignment 1 processed Dataset v2 Random Forest reported:

| Metric | Result |
|---|---:|
| Test RMSE | 152.55167 |
| Test MAE | 114.88727 |
| Test R-squared | 0.70624 |
| Processed rows | 30,190 |

Assignment 1 used an 80/20 train-test split and did not preserve a directly comparable
validation score or reliable runtime artifact. The report therefore treats it as historical
context and uses the Phase 4 same-split Random Forest for the controlled comparison.

## Tests and Validation

```bash
ruff format --check src scripts tests
ruff check src scripts tests
python -m pytest -v
python scripts/validate_assignment_submission.py
```

## Final Submission

Submit separately:

1. `submission/Assignment3_AutoML_High_Level_Report.pdf`
2. `dist/mlops_automl_assignment3_submission.zip`
3. GitHub repository URL

The detailed technical report remains inside the ZIP.

## Limitations

- Runtime-limited AutoML searches can vary across machines.
- Feature importance is model-dependent and does not establish causality.
- The 70% missingness threshold is a documented project choice.
- The target-quality range is based on project data history and requires monitoring.
- Ensembles can improve predictive performance while increasing deployment complexity.
- Assignment 1 used a different processed dataset and split strategy.
