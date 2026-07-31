"""Create a clean Assignment 3 submission ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "submission.yaml"

INCLUDED_ROOTS = ("configs", "src", "scripts", "tests", "docs", "reports", "submission")
INCLUDED_ROOT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-automl.txt",
    "requirements-h2o.txt",
    "requirements-report.txt",
)
EXCLUDED_SUFFIXES = (".db", ".parquet", ".joblib", ".zip")
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "artifacts", "data", "dist"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def main() -> None:
    with CONFIG.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = [
        ROOT / config["outputs"]["validation_report"],
        ROOT / config["outputs"]["report"],
        ROOT / config["outputs"]["high_level_report_pdf"],
        ROOT / config["outputs"]["high_level_report_markdown"],
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing submission artifact: {path.relative_to(ROOT)}")

    validation = json.loads(required[0].read_text(encoding="utf-8"))
    if validation["status"] == "FAIL":
        raise RuntimeError("Submission validation failed.")

    candidates: set[Path] = set()
    for filename in INCLUDED_ROOT_FILES:
        path = ROOT / filename
        if path.exists() and eligible(path):
            candidates.add(path)

    for root_name in INCLUDED_ROOTS:
        directory = ROOT / root_name
        if directory.exists():
            for path in directory.rglob("*"):
                if eligible(path):
                    candidates.add(path)

    manifest_path = ROOT / "submission" / "submission_file_manifest.json"
    manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": checksum(path),
        }
        for path in sorted(candidates)
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidates.add(manifest_path)

    output = ROOT / config["outputs"]["submission_zip"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, path.relative_to(ROOT))

    print(f"Submission ZIP: {output}")
    print(f"Files included: {len(candidates)}")
    print("PHASE 5C STATUS: PASS")


if __name__ == "__main__":
    main()
