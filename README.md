# MLOps AutoML

An end-to-end AutoML comparison project using the athletes dataset.

This repository compares:

- Databricks AutoML
- H2O AutoML
- A manually developed baseline model from the earlier assignment

The project focuses on model performance, feature importance, validation rankings,
execution speed, reproducibility, and the practical strengths and limitations of
AutoML platforms.

---

## Assignment Objective

Predict an athlete's combined lift total:

```text
total_lift = deadlift + candj + snatch + backsq
```

The workflow will evaluate:

- AutoML using all appropriate features
- AutoML using only the top three features
- Top models by validation score
- Top models by execution speed
- Top five feature-importance results
- Databricks AutoML versus H2O AutoML
- AutoML results versus the previous manual baseline

---

## Platforms

### Primary platform

**Databricks AutoML**

Databricks is treated as a low-code AutoML platform because it automates model
selection, training, tuning, and leaderboard generation, while data preparation,
target definition, leakage prevention, evaluation design, and final model approval
still require manual decisions.

### Required comparison platform

**H2O AutoML**

H2O AutoML will run locally through Python and will repeat the all-feature and
top-feature experiments.

---

## Planned Experiment Matrix

| Experiment | Platform | Feature set |
|---|---|---|
| `databricks_all_features` | Databricks AutoML | All appropriate features |
| `databricks_top_features` | Databricks AutoML | Top three features |
| `h2o_all_features` | H2O AutoML | All appropriate features |
| `h2o_top_features` | H2O AutoML | Top three features |

The same target definition and comparable train, validation, and test logic will be
used across platforms.

---

## Data

Place the original dataset at:

```text
data/raw/athletes.csv
```

The raw file is treated as immutable and is intentionally excluded from Git.

Generated datasets will be saved under:

```text
data/processed/
data/splits/
```

---

## Leakage Prevention

The following columns must not be used as model features:

```text
total_lift
deadlift
candj
snatch
backsq
```

Additional identifiers and metadata such as `athlete_id` and `event_timestamp` will
also be excluded from the model feature matrix.

---

## Evaluation Strategy

The planned split is:

| Partition | Fraction |
|---|---:|
| Train | 64% |
| Validation | 16% |
| Test | 20% |

Configuration:

```text
Random seed: 42
Primary metric: RMSE
Additional metrics: MAE and R²
```

The final test set will remain untouched during AutoML model selection.

Validation metrics will be used to select models. Test metrics will be used only for
final generalization assessment.

---

## Concise Rollout Plan

### Phase 1 — Repository and Data Preparation

- Initialize the GitHub repository
- Create the Python environment
- Add the project structure and configuration
- Profile `athletes.csv`
- Create a leakage-safe feature contract
- Build reproducible train, validation, and test datasets

### Phase 2 — Databricks AutoML

Run Databricks AutoML with:

- All appropriate features
- Top three features

Save configuration, leaderboard, metrics, feature importance, speed rankings, and
screenshots.

### Phase 3 — H2O AutoML

Run H2O AutoML with:

- All appropriate features
- Top three features

Save leaderboards, metrics, feature importance, predictions, and execution times.

### Phase 4 — Evaluation and Comparison

Compare:

- Databricks versus H2O
- All features versus top three features
- AutoML versus the previous manual baseline
- Validation performance versus speed
- Automation benefits versus operational complexity

### Phase 5 — Reporting and Reproducibility

Complete:

- README
- Assignment report
- HTML report
- Screenshots
- Tests
- Dependency files
- Reproduction instructions
- Final recommendation
- GitHub validation

---

## Repository Structure

```text
mlops-automl/
├── configs/
│   └── automl.yaml
├── data/
│   ├── raw/
│   │   └── athletes.csv
│   ├── processed/
│   └── splits/
├── docs/
│   └── assets/
├── notebooks/
│   └── databricks/
├── reports/
│   ├── data/
│   ├── databricks/
│   │   ├── all_features/
│   │   └── top_features/
│   ├── h2o/
│   │   ├── all_features/
│   │   └── top_features/
│   └── comparison/
├── scripts/
├── src/
│   └── athlete_automl/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Environment Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## Phase 1 Validation

Run formatting:

```bash
ruff format src tests
```

Run linting:

```bash
ruff check src tests
```

Run tests:

```bash
pytest -v
```

Verify the raw dataset:

```bash
python - <<'PY'
from pathlib import Path

path = Path("data/raw/athletes.csv")

assert path.exists(), "data/raw/athletes.csv is missing"
assert path.stat().st_size > 0, "athletes.csv is empty"

print("PASS:", path)
print("Size:", f"{path.stat().st_size / 1024 / 1024:.2f} MB")
PY
```

---

## Accounts and Local Requirements

| Tool | Account required? |
|---|---:|
| GitHub | Existing account |
| Databricks AutoML | Yes |
| H2O AutoML | No |
| MLflow | No |
| AWS, Azure, or GCP | Not required for the initial local workflow |

H2O AutoML will also require a supported Java installation.

---

## Expected Outputs

### Data preparation

```text
data/processed/athletes_automl.parquet
data/splits/train.parquet
data/splits/validation.parquet
data/splits/test.parquet
reports/data/data_profile.json
reports/data/feature_contract.csv
```

### Databricks AutoML

```text
reports/databricks/all_features/
reports/databricks/top_features/
docs/assets/databricks/
```

### H2O AutoML

```text
reports/h2o/all_features/
reports/h2o/top_features/
```

### Final comparison

```text
reports/comparison/model_comparison.csv
reports/comparison/speed_comparison.csv
reports/comparison/feature_importance_comparison.csv
reports/comparison/final_recommendation.json
```

---

## Current Status

- [x] Repository name selected: `mlops-automl`
- [x] Repository structure defined
- [x] Python 3.11 environment planned
- [x] AutoML configuration defined
- [x] Raw-data location defined
- [ ] Dataset profiling completed
- [ ] AutoML-ready dataset created
- [ ] Databricks AutoML completed
- [ ] H2O AutoML completed
- [ ] Final comparison completed
- [ ] Final report completed
- [ ] Clean-environment validation completed