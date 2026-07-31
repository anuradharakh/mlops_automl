"""Build reports, validate evidence, and package Assignment 3."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script)]
    print("\nRunning:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("build_assignment_report.py")
    run("build_high_level_report.py")
    run("validate_assignment_submission.py")
    run("create_submission_zip.py")

    print("\nFINAL SUBMISSION STATUS: PASS")
    print("Submit:")
    print("1. submission/Assignment3_AutoML_High_Level_Report.pdf")
    print("2. dist/mlops_automl_assignment3_submission.zip")
    print("3. GitHub repository URL")


if __name__ == "__main__":
    main()
