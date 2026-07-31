# Phase 2A — Databricks AutoML All-Features Run

## 1. Extract this ZIP

Copy the package into the `mlops_automl` repository root and allow the
folders to merge.

## 2. Confirm Phase 1C output

```bash
ls data/splits/train.parquet
ls data/splits/validation.parquet
ls data/splits/test.parquet
```

## 3. Install and validate

```bash
source .venv/bin/activate
python -m pip install -e .

ruff format src scripts tests
ruff check src scripts tests --fix
pytest -v
```

## 4. Export the Databricks dataset

```bash
python scripts/export_databricks_dataset.py
```

Expected final line:

```text
PHASE 2A STATUS: PASS
```

Generated files:

```text
data/processed/databricks_automl_dataset.csv
data/processed/databricks_automl_dataset.parquet
reports/databricks/all_features/dataset_export_summary.json
```

## 5. Upload to Databricks

Upload:

```text
data/processed/databricks_automl_dataset.csv
```

Create a managed table such as:

```text
workspace.default.athletes_automl
```

## 6. Import the notebook

Import:

```text
notebooks/databricks/01_automl_all_features.py
```

Update the widgets:

```text
table_name: workspace.default.athletes_automl
experiment_dir: /Users/<your-email>/databricks_automl
experiment_name: athletes_databricks_all_features
timeout_minutes: 15
```

Run all cells.

## 7. Capture evidence

Save screenshots of:

- Split counts
- AutoML configuration
- MLflow experiment
- Leaderboard
- Top three models by validation RMSE
- Top three models by duration
- Best-model test metrics
- SHAP feature importance
- Top five features
