# Phase 2C — AutoGluon Top-Three-Features Run

This phase automatically reads the selected features from:

```text
reports/autogluon/all_features/run_summary.json
```

It then repeats AutoGluon using exactly those three features and the same
train, validation, test, metric, preset, and time-limit strategy.

## Prerequisite

Phase 2B must have completed successfully:

```bash
cat reports/autogluon/all_features/run_summary.json
```

The JSON must contain:

```text
top_three_features
```

## Install and test

```bash
source .venv/bin/activate
python -m pip install -r requirements-automl.txt
python -m pip install -e .

ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v
```

## Smoke run

```bash
python scripts/run_autogluon_top_features.py   --overwrite   --time-limit 300
```

## Final run

```bash
python scripts/run_autogluon_top_features.py   --overwrite
```

Expected:

```text
PHASE 2C STATUS: PASS
```

## Outputs

```text
reports/autogluon/top_features/
├── leaderboard.csv
├── top3_by_score.csv
├── top3_by_speed.csv
├── feature_importance.csv
├── test_predictions.parquet
├── leaderboard_rmse.png
├── feature_importance.png
├── actual_vs_predicted.png
└── run_summary.json

reports/autogluon/comparison/
├── all_vs_top_features.csv
└── all_vs_top_features.json
```

## Review

```bash
cat reports/autogluon/top_features/run_summary.json
cat reports/autogluon/comparison/all_vs_top_features.json
```

## Commit

```bash
git add   configs/autogluon_top_features.yaml   src/athlete_automl/automl/feature_selection.py   scripts/run_autogluon_top_features.py   tests/test_feature_selection.py   docs/phase_2c_autogluon_top_features.md   reports/autogluon/top_features   reports/autogluon/comparison

git commit -m "feat: Add AutoGluon top-features experiment"
git push origin master
```
