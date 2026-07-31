#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create a Python 3.11 virtual environment first."
  exit 1
fi

source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -r requirements-automl.txt
python -m pip install -r requirements-h2o.txt
python -m pip install -e .

python -m pytest -v

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

echo "ASSIGNMENT 3 REPRODUCTION STATUS: PASS"
