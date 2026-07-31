# Phase 2B — AutoGluon + MLflow All-Features Run

This phase implements the primary full-code AutoML workflow locally.

## Outputs

```text
reports/autogluon/all_features/
├── leaderboard.csv
├── top3_by_score.csv
├── top3_by_speed.csv
├── feature_importance.csv
├── top5_features.csv
├── test_predictions.parquet
├── leaderboard_rmse.png
├── feature_importance.png
├── actual_vs_predicted.png
└── run_summary.json
```

The trained AutoGluon predictor is stored under:

```text
artifacts/autogluon/all_features/
```

MLflow tracking metadata is stored in:

```text
mlflow.db
```

## Install

Use the existing Python 3.11 environment:

```bash
source .venv/bin/activate

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-automl.txt
python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import autogluon.tabular
import mlflow

print("AutoGluon import: PASS")
print("MLflow version:", mlflow.__version__)
PY
```

## Test

```bash
ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v
```

## Run

The configured training budget is 15 minutes:

```bash
python scripts/run_autogluon_all_features.py
```

To replace an earlier predictor:

```bash
python scripts/run_autogluon_all_features.py --overwrite
```

For an initial five-minute smoke run:

```bash
python scripts/run_autogluon_all_features.py   --overwrite   --time-limit 300
```

Expected final line:

```text
PHASE 2B STATUS: PASS
```

## Open MLflow

In a second terminal:

```bash
cd /Users/sanketmayekar/AnuradhaM/MLOPS/mlops_automl
source .venv/bin/activate

mlflow server   --backend-store-uri sqlite:///mlflow.db   --host 127.0.0.1   --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Select the experiment:

```text
athletes_automl_autogluon
```

## Evidence to capture

Save screenshots of:

- MLflow experiment and run
- AutoML configuration parameters
- Validation and test metrics
- Full leaderboard
- Top three models by validation RMSE
- Top three models by `fit_time_marginal`
- Feature-importance table and plot
- Top five features
- Actual-versus-predicted plot

The next phase will use the top three features from
`reports/autogluon/all_features/run_summary.json`.
