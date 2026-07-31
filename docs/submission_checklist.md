# Assignment 3 Final Submission Checklist

## Replace or add the files from this update package

- `README.md`
- `requirements-report.txt`
- `configs/submission.yaml`
- `configs/assignment1_baseline.yaml`
- `scripts/build_high_level_report.py`
- `scripts/create_submission_zip.py`
- `scripts/finalize_submission.py`
- `docs/submission_checklist.md`
- `docs/canvas_submission_note.txt`

## Update before running

Open `configs/submission.yaml` and replace:

```yaml
repository_url: UPDATE_WITH_GITHUB_URL
```

Verify the author name:

```yaml
author: Anuradha Rakh
```

## Install the PDF-report dependency

```bash
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python -m pip install -e .
```

## Generate final deliverables

```bash
python scripts/finalize_submission.py
```

Expected:

```text
FINAL SUBMISSION STATUS: PASS
```

## Inspect

Open:

```text
submission/Assignment3_AutoML_High_Level_Report.pdf
```

Check that the author, repository URL, metrics, models, features, charts, and final
recommendation are correct.

## Submit separately

1. `submission/Assignment3_AutoML_High_Level_Report.pdf`
2. `dist/mlops_automl_assignment3_submission.zip`
3. GitHub repository URL in the Canvas comment

## Commit

```bash
git add   README.md   requirements-report.txt   configs/submission.yaml   configs/assignment1_baseline.yaml   scripts/build_high_level_report.py   scripts/create_submission_zip.py   scripts/finalize_submission.py   docs/submission_checklist.md   docs/canvas_submission_note.txt   submission

git commit -m "docs: Finalize AutoML report and submission package"
git push origin master
```
