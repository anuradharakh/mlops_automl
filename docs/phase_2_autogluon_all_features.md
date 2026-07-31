# Phase 2 — AutoGluon + MLflow All-Features Run

The primary AutoML workflow is now **AutoGluon**, tracked with **MLflow**.
It is classified as a **full-code AutoML platform**. Databricks is no
longer required for this assignment workflow.

## 1. Extract the ZIP

Extract this package into the `mlops_automl` repository root and allow
folders to merge.

## 2. Confirm Phase 1C outputs

```bash
ls data/splits/train.parquet
ls data/splits/validation.parquet
ls data/splits/test.parquet
```

When missing:

```bash
python scripts/profile_data.py
python scripts/prepare_data.py
```

## 3. Install AutoGluon and MLflow

```bash
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-automl.txt
python -m pip install -e .
```

## 4. Validate

```bash
ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v
```

## 5. Run the all-features AutoML experiment

```bash
python scripts/run_autogluon_all_features.py
```

Expected final line:

```text
PHASE 2 AUTOGLUON STATUS: PASS
```

The script produces:

```text
reports/autogluon/all_features/
├── autogluon_raw_leaderboard.csv
├── leaderboard_by_validation_score.csv
├── top3_models_by_validation_score.csv
├── leaderboard_by_training_speed.csv
├── top3_models_by_training_speed.csv
├── best_model_metrics.json
├── best_model_test_predictions.csv
├── feature_importance.csv
├── top5_features.csv
├── predictor_info.json
└── data_insights.json
```

## 6. Open MLflow

In a second terminal:

```bash
source .venv/bin/activate
python scripts/start_mlflow.py
```

Open:

```text
http://127.0.0.1:5000
```

Capture the experiment, run parameters, metrics, and artifacts.

## 7. Add these lines to `.gitignore`

```text
models/
mlruns/
mlflow.db
artifacts/
```

## 8. Commit

```bash
git add   requirements-automl.txt   configs/autogluon.yaml   src/athlete_automl/autogluon_workflow.py   scripts/run_autogluon_all_features.py   scripts/start_mlflow.py   tests/test_autogluon_workflow.py   docs/phase_2_autogluon_all_features.md

git commit -m "feat: Add AutoGluon AutoML workflow with MLflow tracking"
git push origin master
```

The next phase will read `top5_features.csv`, select the first three
features, and run the reduced-feature AutoGluon experiment.
