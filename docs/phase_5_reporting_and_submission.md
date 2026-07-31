# Phase 5 — Final Report, Validation, and Submission Bundle

This phase maps the completed repository artifacts to the Assignment 3
deliverables, generates the written discussion, validates the evidence, and
creates a clean submission ZIP.

## 1. Extract

```bash
unzip -o ~/Downloads/mlops_automl_phase5.zip -d .
chmod +x scripts/reproduce_assignment3.sh
```

## 2. Update the author

Edit:

```text
configs/submission.yaml
```

Replace:

```yaml
author: UPDATE_ME
```

with the name used for the course submission.

## 3. Add exact Assignment 1 evidence

Edit:

```text
configs/assignment1_baseline.yaml
```

Set `available: true` and enter the exact Assignment 1 model, metric,
runtime, and source artifact values when available.

When the original values are not available, leave `available: false`.
The report will use the reconstructed same-split Random Forest and explicitly
document the comparison limitation.

## 4. Add the README section

Copy the contents of:

```text
docs/README_assignment3_section.md
```

into the repository's main `README.md`.

## 5. Test

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pytest -v
```

## 6. Generate the report

```bash
python scripts/build_assignment_report.py
```

Expected:

```text
PHASE 5A STATUS: PASS
```

Review:

```bash
cat submission/assignment3_report.md
```

## 7. Validate

```bash
python scripts/validate_assignment_submission.py
```

Expected:

```text
PHASE 5B STATUS: PASS
```

`PASS_WITH_WARNINGS` is possible when exact Assignment 1 metrics are not
configured. The warning is acceptable only when the report clearly discloses
that limitation.

## 8. Create the clean submission ZIP

```bash
python scripts/create_submission_zip.py
```

Expected:

```text
PHASE 5C STATUS: PASS
```

Output:

```text
dist/mlops_automl_assignment3_submission.zip
```

The ZIP intentionally excludes raw and processed datasets, model binaries,
prediction Parquet files, local virtual environments, MLflow SQLite
databases, and Git metadata.

## 9. Evidence review

Confirm these deliverables before submission:

- README setup and execution instructions
- AutoGluon configuration and results
- AutoGluon all-feature and top-feature leaderboards
- AutoGluon top five feature summary
- AutoGluon top three models by validation and speed for both feature sets
- H2O configuration and results
- H2O all-feature and top-feature leaderboards
- H2O top five feature summary
- H2O top three models by validation and speed for both feature sets
- Baseline comparison
- Full-code platform-mode assessment
- Assumptions, limitations, runtime choices, and operational implications
- Final recommendation
- Reproducibility commands
- Passing tests and submission validator

Because the selected primary platform is full-code, no-code or low-code
screenshots are not required. MLflow screenshots can be included as
supplementary evidence.

## 10. Commit

```bash
git add   configs/submission.yaml   configs/assignment1_baseline.yaml   src/athlete_automl/reporting   scripts/build_assignment_report.py   scripts/validate_assignment_submission.py   scripts/create_submission_zip.py   scripts/reproduce_assignment3.sh   tests/test_assignment_reporting.py   docs/README_assignment3_section.md   docs/phase_5_reporting_and_submission.md   submission

git commit -m "docs: Add final AutoML report and submission workflow"
git push origin master
```

Do not commit the generated file under `dist/` unless the instructor
specifically asks for the ZIP in Git.
