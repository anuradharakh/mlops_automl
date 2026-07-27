"""Tests for raw athlete dataset profiling."""

from __future__ import annotations

import pandas as pd

from athlete_automl.data.profiling import (
    build_column_profile,
    build_feature_contract_draft,
    build_target_readiness_summary,
    classify_column,
)


def build_sample_data() -> pd.DataFrame:
    """Create a representative raw athlete dataset."""
    return pd.DataFrame(
        {
            "athlete_id": [
                1,
                2,
                3,
                None,
                5,
            ],
            "age": [
                25,
                30,
                35,
                40,
                45,
            ],
            "gender": [
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "deadlift": [
                100,
                1,
                100,
                100,
                100,
            ],
            "candj": [
                50,
                50,
                None,
                50,
                50,
            ],
            "snatch": [
                40,
                40,
                40,
                40,
                40,
            ],
            "backsq": [
                80,
                80,
                80,
                80,
                80,
            ],
        }
    )


def test_column_roles_are_classified() -> None:
    """Known raw columns should receive the expected roles."""
    assert (
        classify_column(
            column="athlete_id",
            target_name="total_lift",
            target_components={
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            },
            identifier_columns={"athlete_id"},
            metadata_columns=set(),
        )
        == "identifier"
    )

    assert (
        classify_column(
            column="deadlift",
            target_name="total_lift",
            target_components={
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            },
            identifier_columns={"athlete_id"},
            metadata_columns=set(),
        )
        == "target_component"
    )

    assert (
        classify_column(
            column="age",
            target_name="total_lift",
            target_components={
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            },
            identifier_columns={"athlete_id"},
            metadata_columns=set(),
        )
        == "candidate_feature"
    )


def test_target_readiness_handles_sentinel_values() -> None:
    """Sentinel and incomplete targets should be excluded."""
    summary = build_target_readiness_summary(
        dataframe=build_sample_data(),
        target_components=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
        sentinel_values=[1],
        identifier_column="athlete_id",
    )

    assert summary["initial_rows"] == 5

    assert summary["missing_identifier_rows"] == 1

    assert summary["missing_or_invalid_target_rows"] == 2

    assert summary["eligible_rows"] == 2

    assert summary["sentinel_counts"]["deadlift"] == 1


def test_column_profile_contains_quality_metrics() -> None:
    """The column profile should contain one row per column."""
    dataframe = build_sample_data()

    profile = build_column_profile(
        dataframe=dataframe,
        target_name="total_lift",
        target_components=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
        identifier_columns=["athlete_id"],
        metadata_columns=[],
    )

    assert len(profile) == len(dataframe.columns)

    assert {
        "column",
        "raw_dtype",
        "role",
        "missing_percentage",
        "unique_count",
        "numeric_parse_rate",
    }.issubset(profile.columns)


def test_contract_excludes_target_components() -> None:
    """Target components should never be model candidates."""
    profile = build_column_profile(
        dataframe=build_sample_data(),
        target_name="total_lift",
        target_components=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
        identifier_columns=["athlete_id"],
        metadata_columns=[],
    )

    contract = build_feature_contract_draft(profile)

    component_actions = contract.loc[
        contract["column"].isin(
            [
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            ]
        ),
        "recommended_action",
    ]

    assert set(component_actions) == {"exclude"}
