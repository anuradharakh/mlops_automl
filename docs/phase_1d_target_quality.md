# Phase 1D — Target Quality Gate

The target review found corrupted extreme values such as 33,554,428,
16,787,215, 8,390,604, and 39,995. These values distort the training
mean and standard deviation and should not enter AutoML.

The quality gate keeps this inclusive range:

```text
8 <= total_lift <= 2500
```

The upper limit is slightly above the previous clean project maximum of
2,367. Rows are removed rather than capped because capping would create
invented target labels.

## Run

```bash
source .venv/bin/activate
python -m pip install -e .

ruff format src scripts tests
ruff check src scripts tests --fix
python -m pytest -v

python scripts/clean_target_and_resplit.py
python scripts/export_databricks_dataset.py
```

Expected:

```text
PHASE 1D STATUS: PASS
PHASE 2A STATUS: PASS
```
