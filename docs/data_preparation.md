# Phase 1C — AutoML Dataset Preparation

This phase converts the immutable raw athlete CSV into a leakage-safe,
reproducible AutoML dataset.

## Included features

Required demographic features:

- age
- height
- weight
- gender
- region

Optional performance features are included only when present:

- fran
- helen
- grace
- filthy50
- fgonebad
- run400
- run5k
- pullups

Engineered features:

- bmi
- age_squared
- weight_height_ratio

## Leakage controls

The target is calculated as:

```text
total_lift = deadlift + candj + snatch + backsq
```

The four target components are excluded from all model feature sets.

The sentinel value `1` is treated as missing in each target component.
A row is eligible only when all target components are valid.

## Split strategy

- Train: 64%
- Validation: 16%
- Test: 20%
- Random state: 42

The test partition remains untouched during AutoML model selection.

## Run

```bash
python scripts/prepare_data.py
```

## Validate

```bash
ruff format src scripts tests
ruff check src scripts tests --fix
pytest -v
```

## Generated artifacts

```text
data/processed/athletes_automl.parquet
data/processed/athletes_automl.csv
data/splits/train.parquet
data/splits/validation.parquet
data/splits/test.parquet
reports/data/feature_contract.csv
reports/data/feature_list.json
reports/data/preparation_summary.json
reports/data/split_summary.csv
```
