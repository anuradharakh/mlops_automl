# Phase 3B — H2O AutoML Top-Three-Features Run

This phase reads H2O's selected features from:

```text
reports/h2o/all_features/run_summary.json
```

It repeats H2O AutoML with exactly those three features while preserving
the same fixed train, validation, test, RMSE, seed, model-count, and
runtime strategy.

## Prerequisite

Phase 3A must have completed successfully:

```bash
cat reports/h2o/all_features/run_summary.json
```

The summary must contain:

```text
top_three_features
```

## Install and test

```bash
source .venv/bin/activate
python -m pip install -r requirements-h2o.txt
python -m pip install -e .

python scripts/check_h2o_environment.py
ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v
```

## Smoke run

```bash
python scripts/run_h2o_top_features.py   --overwrite   --time-limit 300   --max-models 5
```

## Final run

```bash
python scripts/run_h2o_top_features.py   --overwrite
```

Expected:

```text
PHASE 3B STATUS: PASS
```

## Outputs

```text
reports/h2o/top_features/
├── leaderboard.csv
├── top3_by_score.csv
├── top3_by_training_speed.csv
├── top3_by_prediction_speed.csv
├── feature_importance.csv
├── test_predictions.parquet
├── leaderboard_rmse.png
├── feature_importance.png
├── actual_vs_predicted.png
└── run_summary.json

reports/h2o/comparison/
├── all_vs_top_features.csv
└── all_vs_top_features.json
```

The H2O leader model is saved under:

```text
artifacts/h2o/top_features/
```

## Review

```bash
cat reports/h2o/top_features/run_summary.json
cat reports/h2o/comparison/all_vs_top_features.json
column -s, -t reports/h2o/top_features/top3_by_score.csv
column -s, -t reports/h2o/top_features/top3_by_training_speed.csv
```

## Commit

```bash
git add   configs/h2o_top_features.yaml   src/athlete_automl/h2o_workflow/feature_selection.py   scripts/run_h2o_top_features.py   tests/test_h2o_feature_selection.py   docs/phase_3b_h2o_top_features.md   reports/h2o/top_features   reports/h2o/comparison

git commit -m "feat: Add H2O AutoML top-features experiment"
git push origin master
```
