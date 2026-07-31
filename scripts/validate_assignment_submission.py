"""Validate Assignment 3 evidence and reproducibility artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "submission.yaml"
)


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main() -> None:
    """Validate required deliverables and write a machine-readable report."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []

    required_files = {
        **{
            key: resolve_path(value)
            for key, value in config[
                "paths"
            ].items()
            if key != "assignment1_baseline"
        },
        "generated_report": resolve_path(
            config["outputs"]["report"]
        ),
        "evidence_manifest": resolve_path(
            config["outputs"][
                "evidence_manifest"
            ]
        ),
        "requirements": (
            PROJECT_ROOT / "requirements.txt"
        ),
        "requirements_automl": (
            PROJECT_ROOT
            / "requirements-automl.txt"
        ),
        "requirements_h2o": (
            PROJECT_ROOT
            / "requirements-h2o.txt"
        ),
        "pyproject": (
            PROJECT_ROOT / "pyproject.toml"
        ),
    }

    for name, path in required_files.items():
        passed = path.exists()
        checks.append(
            {
                "check": (
                    f"file:{name}"
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "details": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

        if not passed:
            failures.append(
                f"Missing required file: "
                f"{path.relative_to(PROJECT_ROOT)}"
            )

    run_summary_keys = [
        "autogluon_all_summary",
        "autogluon_top_summary",
        "h2o_all_summary",
        "h2o_top_summary",
    ]

    for key in run_summary_keys:
        path = resolve_path(
            config["paths"][key]
        )

        if not path.exists():
            continue

        summary = load_json(path)
        status = summary.get("status")
        passed = status == "PASS"
        checks.append(
            {
                "check": (
                    f"run_status:{key}"
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "details": (
                    f"status={status}"
                ),
            }
        )

        if not passed:
            failures.append(
                f"{key} status was not PASS."
            )

    table_requirements = {
        "autogluon_all_score": 3,
        "autogluon_all_speed": 3,
        "autogluon_top_score": 3,
        "autogluon_top_speed": 3,
        "h2o_all_score": 3,
        "h2o_all_speed": 3,
        "h2o_top_score": 3,
        "h2o_top_speed": 3,
        "autogluon_all_features": 5,
        "h2o_all_features": 5,
    }

    for key, minimum_rows in table_requirements.items():
        path = resolve_path(
            config["paths"][key]
        )

        if not path.exists():
            continue

        rows = len(pd.read_csv(path))
        passed = rows >= minimum_rows
        checks.append(
            {
                "check": (
                    f"row_count:{key}"
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "details": (
                    f"rows={rows}, "
                    f"minimum={minimum_rows}"
                ),
            }
        )

        if not passed:
            failures.append(
                f"{key} contains {rows} rows; "
                f"expected at least {minimum_rows}."
            )

    assignment1_path = resolve_path(
        config["paths"][
            "assignment1_baseline"
        ]
    )

    if assignment1_path.exists():
        with assignment1_path.open(
            encoding="utf-8"
        ) as file:
            baseline = yaml.safe_load(file)[
                "assignment1_baseline"
            ]

        if not baseline.get(
            "available",
            False,
        ):
            warnings.append(
                "Original Assignment 1 metrics are not "
                "configured. The report uses the reconstructed "
                "same-split Random Forest comparison and "
                "discloses this limitation."
            )

    author = config["assignment"].get(
        "author"
    )

    if not author or author == "UPDATE_ME":
        failures.append(
            "Update assignment.author in "
            "configs/submission.yaml."
        )
        checks.append(
            {
                "check": "assignment_author",
                "status": "FAIL",
                "details": (
                    "Author is still UPDATE_ME."
                ),
            }
        )
    else:
        checks.append(
            {
                "check": "assignment_author",
                "status": "PASS",
                "details": author,
            }
        )

    overall_status = (
        "FAIL"
        if failures
        else (
            "PASS_WITH_WARNINGS"
            if warnings
            else "PASS"
        )
    )

    result = {
        "status": overall_status,
        "failure_count": len(
            failures
        ),
        "warning_count": len(
            warnings
        ),
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }

    output_path = resolve_path(
        config["outputs"][
            "validation_report"
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    print(
        f"Validation report: {output_path}"
    )

    if failures:
        raise SystemExit(1)

    print(
        "PHASE 5B STATUS: PASS"
    )


if __name__ == "__main__":
    main()
