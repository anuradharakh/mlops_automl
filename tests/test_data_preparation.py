"""Tests for leakage-safe athlete data preparation."""

from __future__ import annotations

import pandas as pd

from athlete_automl.data.preparation import (
    add_engineered_features,
    build_feature_contract,
    build_target,
    parse_numeric_or_time,
    split_dataset,
    verify_split_integrity,
)


def test_parse_numeric_or_time() -> None:
    """Numeric and MM:SS values should be converted consistently."""
    values = pd.Series(["120", "4:30", "1:02:03", None])
    parsed = parse_numeric_or_time(values)

    assert parsed.iloc[0] == 120.0
    assert parsed.iloc[1] == 270.0
    assert parsed.iloc[2] == 3723.0
    assert pd.isna(parsed.iloc[3])


def test_target_creation_excludes_sentinel_values() -> None:
    """A sentinel in any target component should invalidate the row."""
    dataframe = pd.DataFrame(
        {
            "deadlift": [100, 1, 100],
            "candj": [50, 50, None],
            "snatch": [40, 40, 40],
            "backsq": [80, 80, 80],
        }
    )

    target, summary = build_target(
        dataframe=dataframe,
        target_name="total_lift",
        components=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
        sentinel_values=[1],
    )

    assert target.iloc[0] == 270
    assert pd.isna(target.iloc[1])
    assert pd.isna(target.iloc[2])
    assert summary["total_sentinel_replacements"] == 1


def test_engineered_features_are_created() -> None:
    """Configured deterministic features should be reproducible."""
    dataframe = pd.DataFrame(
        {
            "age": [30.0],
            "height": [70.0],
            "weight": [200.0],
        }
    )

    result, created = add_engineered_features(
        dataframe=dataframe,
        engineered_config={
            "bmi": {},
            "age_squared": {},
            "weight_height_ratio": {},
        },
    )

    assert set(created) == {
        "bmi",
        "age_squared",
        "weight_height_ratio",
    }
    assert round(result.loc[0, "bmi"], 4) == 28.6939
    assert result.loc[0, "age_squared"] == 900.0


def test_target_components_are_excluded_from_contract() -> None:
    """Lift components must never be included as predictors."""
    contract = build_feature_contract(
        numeric_features=["age", "weight"],
        categorical_features=["gender"],
        engineered_features=["bmi"],
        target_name="total_lift",
        target_components=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
        identifier_column="athlete_id",
        all_raw_columns=[
            "athlete_id",
            "age",
            "weight",
            "gender",
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
    )

    component_actions = contract.loc[
        contract["column"].isin(
            ["deadlift", "candj", "snatch", "backsq"]
        ),
        "action",
    ]

    assert set(component_actions) == {"exclude"}


def test_split_is_deterministic_and_has_no_overlap() -> None:
    """Repeated split calls should produce identical partitions."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": range(100),
            "age": range(100),
            "total_lift": range(100, 200),
        }
    )

    first = split_dataset(
        dataframe=dataframe,
        train_size=0.64,
        validation_size=0.16,
        test_size=0.20,
        random_state=42,
    )
    second = split_dataset(
        dataframe=dataframe,
        train_size=0.64,
        validation_size=0.16,
        test_size=0.20,
        random_state=42,
    )

    assert len(first["train"]) == 64
    assert len(first["validation"]) == 16
    assert len(first["test"]) == 20

    for split_name in first:
        assert first[split_name]["athlete_id"].tolist() == second[
            split_name
        ]["athlete_id"].tolist()

    verify_split_integrity(
        splits=first,
        identifier_column="athlete_id",
    )
