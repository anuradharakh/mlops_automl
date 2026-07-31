"""Prepare leakage-safe AutoML datasets and reproducible splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from athlete_automl.data.preparation import (
    add_engineered_features,
    apply_valid_range,
    build_feature_contract,
    build_target,
    clean_categorical,
    normalize_columns,
    parse_numeric_or_time,
    select_available_features,
    split_dataset,
    verify_split_integrity,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "feature_contract.yaml"


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative configuration path."""
    return PROJECT_ROOT / value


def main() -> None:
    """Build the processed dataset and fixed data splits."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_path = resolve_path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw = pd.read_csv(raw_path, low_memory=False)
    raw = normalize_columns(raw)

    raw_rows = len(raw)
    raw_columns = raw.columns.tolist()

    target_config = config["target"]
    target, target_summary = build_target(
        dataframe=raw,
        target_name=target_config["name"],
        components=list(target_config["components"]),
        sentinel_values=list(target_config["sentinel_values"]),
    )

    feature_config = config["features"]
    selected = select_available_features(
        dataframe=raw,
        required_numeric=list(feature_config["required_numeric"]),
        optional_numeric=list(feature_config["optional_numeric"]),
        required_categorical=list(feature_config["required_categorical"]),
        optional_categorical=list(feature_config["optional_categorical"]),
    )

    numeric_features = selected["required_numeric"] + selected["optional_numeric"]
    categorical_features = (
        selected["required_categorical"] + selected["optional_categorical"]
    )

    identifier_config = config["identifiers"]
    preferred_identifier = identifier_config["preferred_column"]

    if preferred_identifier in raw.columns:
        identifier_column = preferred_identifier
        identifier = raw[preferred_identifier].astype("string")
    else:
        identifier_column = identifier_config["fallback_name"]
        identifier = pd.Series(
            [f"row_{index}" for index in raw.index],
            index=raw.index,
            dtype="string",
        )

    prepared = pd.DataFrame(
        {
            identifier_column: identifier,
            target_config["name"]: target,
        }
    )

    for column in numeric_features:
        prepared[column] = parse_numeric_or_time(raw[column])

    valid_ranges = config["cleaning"].get("valid_ranges", {})
    for column, limits in valid_ranges.items():
        if column in prepared.columns:
            prepared[column] = apply_valid_range(
                prepared[column],
                minimum=limits.get("minimum"),
                maximum=limits.get("maximum"),
            )

    for column in config["cleaning"].get("positive_only", []):
        if column in prepared.columns:
            prepared[column] = prepared[column].mask(prepared[column] <= 0)

    for column in categorical_features:
        prepared[column] = clean_categorical(raw[column])

    prepared, engineered_features = add_engineered_features(
        dataframe=prepared,
        engineered_config=dict(feature_config["engineered"]),
    )

    rows_after_target_filter = int(prepared[target_config["name"]].notna().sum())
    prepared = prepared.loc[prepared[target_config["name"]].notna()].copy()

    missing_identifier_rows = int(prepared[identifier_column].isna().sum())
    prepared = prepared.loc[prepared[identifier_column].notna()].copy()

    required_features = selected["required_numeric"] + selected["required_categorical"]

    rows_before_required_filter = len(prepared)
    if config["cleaning"]["drop_rows_missing_required_features"]:
        prepared = prepared.dropna(subset=required_features)
    rows_removed_for_required_features = rows_before_required_filter - len(prepared)

    duplicate_identifier_rows = int(
        prepared.duplicated(
            subset=[identifier_column],
            keep=False,
        ).sum()
    )

    duplicates_removed = 0
    if config["cleaning"]["drop_duplicate_identifiers"]:
        before_deduplication = len(prepared)

        feature_columns_for_completeness = (
            numeric_features + categorical_features + engineered_features
        )
        prepared["_feature_completeness"] = (
            prepared[feature_columns_for_completeness].notna().sum(axis=1)
        )

        prepared = (
            prepared.sort_values(
                by=[
                    identifier_column,
                    "_feature_completeness",
                ],
                ascending=[True, False],
                kind="stable",
            )
            .drop_duplicates(
                subset=[identifier_column],
                keep="first",
            )
            .drop(columns=["_feature_completeness"])
        )

        duplicates_removed = before_deduplication - len(prepared)

    final_feature_columns = (
        numeric_features + categorical_features + engineered_features
    )

    ordered_columns = [
        identifier_column,
        *final_feature_columns,
        target_config["name"],
    ]

    prepared = prepared[ordered_columns].reset_index(drop=True)

    if prepared.empty:
        raise ValueError("No rows remain after data preparation.")

    splits = split_dataset(
        dataframe=prepared,
        train_size=float(config["split"]["train_size"]),
        validation_size=float(config["split"]["validation_size"]),
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
    )
    verify_split_integrity(
        splits=splits,
        identifier_column=identifier_column,
    )

    processed_parquet_path = resolve_path(config["data"]["processed_parquet_path"])
    processed_csv_path = resolve_path(config["data"]["processed_csv_path"])

    processed_parquet_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    processed_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared.to_parquet(
        processed_parquet_path,
        index=False,
    )
    prepared.to_csv(
        processed_csv_path,
        index=False,
    )

    split_paths = {
        "train": resolve_path(config["data"]["train_path"]),
        "validation": resolve_path(config["data"]["validation_path"]),
        "test": resolve_path(config["data"]["test_path"]),
    }

    for name, split in splits.items():
        path = split_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_parquet(path, index=False)

    contract = build_feature_contract(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        engineered_features=engineered_features,
        target_name=target_config["name"],
        target_components=list(target_config["components"]),
        identifier_column=identifier_column,
        all_raw_columns=raw_columns,
    )

    feature_contract_path = resolve_path(config["reports"]["feature_contract_path"])
    feature_contract_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    contract.to_csv(
        feature_contract_path,
        index=False,
    )

    feature_list = {
        "identifier": identifier_column,
        "target": target_config["name"],
        "numeric_features": numeric_features + engineered_features,
        "categorical_features": categorical_features,
        "all_model_features": final_feature_columns,
        "feature_count": len(final_feature_columns),
        "missing_optional_numeric": selected["missing_optional_numeric"],
        "missing_optional_categorical": selected["missing_optional_categorical"],
    }
    write_json(
        resolve_path(config["reports"]["feature_list_path"]),
        feature_list,
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "row_count": len(split),
                "percentage": round(
                    len(split) / len(prepared) * 100,
                    4,
                ),
                "target_mean": round(
                    float(split[target_config["name"]].mean()),
                    6,
                ),
                "target_median": round(
                    float(split[target_config["name"]].median()),
                    6,
                ),
                "target_std": round(
                    float(split[target_config["name"]].std()),
                    6,
                ),
            }
            for name, split in splits.items()
        ]
    )
    split_summary_path = resolve_path(config["reports"]["split_summary_path"])
    split_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    summary = {
        "status": "PASS",
        "raw_rows": raw_rows,
        "raw_columns": len(raw_columns),
        "rows_after_target_filter": rows_after_target_filter,
        "missing_identifier_rows_removed": missing_identifier_rows,
        "rows_removed_for_missing_required_features": (
            rows_removed_for_required_features
        ),
        "duplicate_identifier_rows_detected": (duplicate_identifier_rows),
        "duplicate_identifiers_removed": duplicates_removed,
        "final_rows": len(prepared),
        "identifier_column": identifier_column,
        "target": target_config["name"],
        "target_summary": target_summary,
        "feature_count": len(final_feature_columns),
        "numeric_features": numeric_features + engineered_features,
        "categorical_features": categorical_features,
        "split_counts": {name: len(split) for name, split in splits.items()},
        "random_state": int(config["split"]["random_state"]),
        "target_statistics": {
            "minimum": float(prepared[target_config["name"]].min()),
            "maximum": float(prepared[target_config["name"]].max()),
            "mean": float(prepared[target_config["name"]].mean()),
            "median": float(prepared[target_config["name"]].median()),
            "standard_deviation": float(prepared[target_config["name"]].std()),
        },
        "output_files": {
            "processed_parquet": str(processed_parquet_path.relative_to(PROJECT_ROOT)),
            "processed_csv": str(processed_csv_path.relative_to(PROJECT_ROOT)),
            "train": str(split_paths["train"].relative_to(PROJECT_ROOT)),
            "validation": str(split_paths["validation"].relative_to(PROJECT_ROOT)),
            "test": str(split_paths["test"].relative_to(PROJECT_ROOT)),
        },
    }

    write_json(
        resolve_path(config["reports"]["preparation_summary_path"]),
        summary,
    )

    print("AutoML data preparation completed successfully.")
    print(f"Raw rows: {raw_rows:,}")
    print(f"Final rows: {len(prepared):,}")
    print(f"Model features: {len(final_feature_columns)}")
    print(
        "Split counts: "
        + ", ".join(f"{name}={len(split):,}" for name, split in splits.items())
    )
    print(f"Sentinel replacements: {target_summary['total_sentinel_replacements']:,}")
    print("PHASE 1C STATUS: PASS")


if __name__ == "__main__":
    main()
