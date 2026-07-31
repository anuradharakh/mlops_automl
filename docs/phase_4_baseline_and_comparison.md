# Phase 4 — Baselines and Final Cross-Platform Comparison

This phase adds two reproducible baselines and compares them with:

- AutoGluon all features
- AutoGluon top three features
- H2O AutoML all features
- H2O AutoML top three features

The baselines are:

1. Mean `DummyRegressor`, representing a no-skill regression floor.
2. `RandomForestRegressor`, representing a conventional non-AutoML model
   trained on the same cleaned splits and 13 approved features.

## Prerequisites

The following runs must already have passed:

```text
PHASE 2B STATUS: PASS
PHASE 2C STATUS: PASS
PHASE 3A STATUS: PASS
PHASE 3B STATUS: PASS
```

Confirm:

```bash
ls   reports/autogluon/all_features/run_summary.json   reports/autogluon/top_features/run_summary.json   reports/h2o/all_features/run_summary.json   reports/h2o/top_features/run_summary.json
```

## Install and test

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pytest -v
```

## Run baselines

```bash
python scripts/run_baselines.py
```

Expected:

```text
PHASE 4A STATUS: PASS
```

Review:

```bash
cat reports/baseline/run_summary.json
column -s, -t reports/baseline/baseline_leaderboard.csv
```

## Build the final comparison

```bash
python scripts/build_final_comparison.py
```

Expected:

```text
PHASE 4B STATUS: PASS
```

Review:

```bash
column -s, -t   reports/final_comparison/platform_comparison.csv

cat reports/final_comparison/recommendation.json

cat reports/final_comparison/feature_overlap.json

cat reports/final_comparison/final_comparison_report.md
```

## Outputs

```text
reports/baseline/
├── baseline_leaderboard.csv
├── feature_importance.csv
├── random_forest_test_predictions.parquet
└── run_summary.json

reports/final_comparison/
├── platform_comparison.csv
├── recommendation.json
├── feature_overlap.json
├── final_comparison_report.md
├── test_rmse_comparison.png
├── test_r2_comparison.png
└── training_time_comparison.png
```

## Commit

Generated model files and the SQLite database should remain ignored.

```bash
git add   configs/baseline.yaml   configs/final_comparison.yaml   src/athlete_automl/comparison   scripts/run_baselines.py   scripts/build_final_comparison.py   tests/test_final_comparison.py   docs/phase_4_baseline_and_comparison.md   reports/baseline   reports/final_comparison

git commit -m "feat: Add baselines and final AutoML comparison"
git push origin master
```
