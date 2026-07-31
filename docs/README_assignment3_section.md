## Assignment 3 — AutoML

This repository implements a reproducible regression AutoML workflow using
the athletes dataset.

### Platforms

- Primary workflow: AutoGluon with MLflow experiment tracking
- Required repeat: H2O AutoML
- Platform mode: full-code

### Experiments

1. AutoGluon with all approved features
2. AutoGluon with the top three AutoGluon-ranked features
3. H2O AutoML with all approved features
4. H2O AutoML with the top three H2O-ranked features
5. Mean and Random Forest baselines
6. Cross-platform comparison and final recommendation

### Environment

Use Python 3.11. H2O also requires Java.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-automl.txt
python -m pip install -r requirements-h2o.txt
python -m pip install -e .
```

Place the original dataset at:

```text
data/raw/athletes.csv
```

### Full execution

```bash
chmod +x scripts/reproduce_assignment3.sh
./scripts/reproduce_assignment3.sh
```

### Individual execution

```bash
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
python scripts/create_submission_zip.py
```

### Main deliverables

```text
submission/assignment3_report.md
submission/evidence_manifest.csv
submission/validation_report.json
reports/final_comparison/
dist/mlops_automl_assignment3_submission.zip
```

### Important baseline note

Update `configs/assignment1_baseline.yaml` with exact Assignment 1 metrics
and runtime evidence when available. Until then, the report clearly labels
the Phase 4 Random Forest as a same-split reconstructed baseline rather than
the original Assignment 1 result.
