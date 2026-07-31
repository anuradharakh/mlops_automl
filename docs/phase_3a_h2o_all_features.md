# Phase 3A — H2O AutoML All-Features Run

This phase completes the required H2O AutoML repeat using the same fixed
train, validation, and test partitions as the primary AutoGluon workflow.

## Configuration

```text
Feature set: all 13 approved features
Primary ranking metric: RMSE
Train rows: fixed Phase 1 split
Validation rows: fixed Phase 1 split
Test rows: fixed Phase 1 split
nfolds: 0
max_models: 20
max_runtime_secs: 900
seed: 42
excluded algorithm: DeepLearning
```

With `nfolds=0`, the configured validation frame is used for individual
model stopping and the leaderboard frame is used for ranking. This also
disables stacked ensembles. The untouched test frame is evaluated only
after the leader is selected.

## Install Java 17

Check Java:

```bash
java -version
```

Install Java 17 on macOS when it is missing:

```bash
brew install openjdk@17

sudo ln -sfn   "$(brew --prefix openjdk@17)/libexec/openjdk.jdk"   /Library/Java/JavaVirtualMachines/openjdk-17.jdk
```

Open a new terminal and confirm:

```bash
java -version
```

## Install Python requirements

```bash
source .venv/bin/activate

python -m pip install -r requirements-h2o.txt
python -m pip install -e .
python scripts/check_h2o_environment.py
```

Expected:

```text
H2O ENVIRONMENT STATUS: PASS
```

## Test

```bash
ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v
```

## Smoke run

```bash
python scripts/run_h2o_all_features.py   --overwrite   --time-limit 300   --max-models 5
```

## Final run

```bash
python scripts/run_h2o_all_features.py   --overwrite
```

Expected:

```text
PHASE 3A STATUS: PASS
```

## Outputs

```text
reports/h2o/all_features/
├── data_insights.json
├── leaderboard.csv
├── top3_by_score.csv
├── top3_by_training_speed.csv
├── top3_by_prediction_speed.csv
├── feature_importance.csv
├── top5_features.csv
├── test_predictions.parquet
├── leaderboard_rmse.png
├── feature_importance.png
├── actual_vs_predicted.png
└── run_summary.json
```

The trained H2O leader is saved under:

```text
artifacts/h2o/all_features/
```

## Review

```bash
cat reports/h2o/all_features/run_summary.json
column -s, -t reports/h2o/all_features/top3_by_score.csv
column -s, -t reports/h2o/all_features/top3_by_training_speed.csv
column -s, -t reports/h2o/all_features/top5_features.csv
```

## Commit

```bash
git add   requirements-h2o.txt   configs/h2o.yaml   src/athlete_automl/h2o_workflow   scripts/check_h2o_environment.py   scripts/run_h2o_all_features.py   tests/test_h2o_evaluation.py   docs/phase_3a_h2o_all_features.md   reports/h2o/all_features

git commit -m "feat: Add H2O AutoML all-features workflow"
git push origin master
```
