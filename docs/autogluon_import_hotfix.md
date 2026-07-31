# AutoGluon Import Hotfix

This package restores the shared AutoGluon module required by both:

```text
scripts/run_autogluon_all_features.py
scripts/run_autogluon_top_features.py
```

Extract the ZIP into the repository root, then run:

```bash
source .venv/bin/activate
python -m pip install -e .

python -c "from athlete_automl.autogluon.experiment import regression_metrics; print('IMPORT PASS')"

python -m pytest -v
python scripts/run_autogluon_top_features.py
```
