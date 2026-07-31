"""Profile the raw athletes dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from athlete_automl.data.profiling import (
    build_column_profile,
    build_feature_contract_draft,
    build_target_readiness_summary,
    calculate_file_hash,
    convert_to_json_safe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "automl.yaml"


def resolve_path(
    path_value: str,
) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / path_value


def main() -> None:
    """Run raw-data profiling."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_path = resolve_path(config["data"]["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    dataframe = pd.read_csv(
        raw_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError("Raw dataset is empty.")

    if dataframe.columns.duplicated().any():
        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()

        raise ValueError(f"Duplicate raw columns found: {duplicate_columns}")

    target_config = config["target"]
    profiling_config = config["profiling"]

    identifier_columns = list(config["excluded_columns"]["identifiers"])

    metadata_columns = list(config["excluded_columns"]["metadata"])

    target_components = list(target_config["components"])

    column_profile = build_column_profile(
        dataframe=dataframe,
        target_name=target_config["name"],
        target_components=(target_components),
        identifier_columns=(identifier_columns),
        metadata_columns=(metadata_columns),
        sample_value_count=int(profiling_config["sample_value_count"]),
    )

    feature_contract = build_feature_contract_draft(column_profile)

    identifier_column = identifier_columns[0] if identifier_columns else None

    target_readiness = build_target_readiness_summary(
        dataframe=dataframe,
        target_components=(target_components),
        sentinel_values=list(target_config["sentinel_values"]),
        identifier_column=(identifier_column),
    )

    role_counts = {
        str(role): int(count)
        for role, count in column_profile["role"].value_counts().items()
    }

    action_counts = {
        str(action): int(count)
        for action, count in feature_contract["recommended_action"]
        .value_counts()
        .items()
    }

    profile = {
        "status": "PASS",
        "source": {
            "path": str(raw_path.relative_to(PROJECT_ROOT)),
            "sha256": (calculate_file_hash(raw_path)),
            "size_bytes": int(raw_path.stat().st_size),
        },
        "dataset": {
            "row_count": int(len(dataframe)),
            "column_count": int(dataframe.shape[1]),
            "duplicate_row_count": int(dataframe.duplicated().sum()),
            "memory_usage_mb": round(
                float(dataframe.memory_usage(deep=True).sum() / 1024 / 1024),
                4,
            ),
            "columns": (dataframe.columns.tolist()),
        },
        "column_roles": role_counts,
        "feature_contract_actions": (action_counts),
        "target_readiness": (target_readiness),
    }

    profile_path = resolve_path(profiling_config["dataset_profile"])

    column_profile_path = resolve_path(profiling_config["column_profile"])

    contract_path = resolve_path(profiling_config["feature_contract_draft"])

    for output_path in [
        profile_path,
        column_profile_path,
        contract_path,
    ]:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    profile_path.write_text(
        json.dumps(
            convert_to_json_safe(profile),
            indent=2,
        ),
        encoding="utf-8",
    )

    column_profile.to_csv(
        column_profile_path,
        index=False,
    )

    feature_contract.to_csv(
        contract_path,
        index=False,
    )

    print("Raw athlete dataset profiling completed successfully.")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]:,}")
    print(f"Eligible target rows: {target_readiness['eligible_rows']:,}")
    print(f"Sentinel replacements: {target_readiness['total_sentinel_replacements']:,}")
    print(f"Dataset profile: {profile_path}")
    print(f"Column profile: {column_profile_path}")
    print(f"Feature contract: {contract_path}")
    print("PHASE 1B STATUS: PASS")


if __name__ == "__main__":
    main()
