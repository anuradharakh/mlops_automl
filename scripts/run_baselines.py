"""Train reproducible dummy and Random Forest baselines."""

from __future__ import annotations

import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from athlete_automl.automl.evaluation import (
    regression_metrics,
    validate_feature_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "baseline.yaml"
)


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def find_identifier_column(
    columns: list[str],
    candidates: list[str],
) -> str:
    """Return the first supported identifier."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        "No supported identifier column was found."
    )


def evaluate_model(
    model: Any,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Evaluate a fitted model on validation and test sets."""
    validation_predictions = model.predict(
        validation_features
    )

    prediction_start = time.perf_counter()
    test_predictions = model.predict(
        test_features
    )
    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    return (
        regression_metrics(
            validation_target,
            validation_predictions,
        ),
        regression_metrics(
            test_target,
            test_predictions,
        ),
        prediction_seconds,
    )


def main() -> None:
    """Train, evaluate, report, save, and track baselines."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    schema_config = config["schema"]
    random_forest_config = config[
        "random_forest"
    ]
    importance_config = config[
        "permutation_importance"
    ]
    mlflow_config = config["mlflow"]
    artifact_config = config["artifacts"]
    reports_config = config["reports"]

    paths = {
        "train": resolve_path(
            data_config["train_path"]
        ),
        "validation": resolve_path(
            data_config["validation_path"]
        ),
        "test": resolve_path(
            data_config["test_path"]
        ),
        "features": resolve_path(
            data_config["feature_list_path"]
        ),
    }

    missing_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in paths.values()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required baseline inputs are missing: "
            + ", ".join(missing_paths)
        )

    train = pd.read_parquet(
        paths["train"]
    )
    validation = pd.read_parquet(
        paths["validation"]
    )
    test = pd.read_parquet(
        paths["test"]
    )
    feature_list = load_json(
        paths["features"]
    )

    model_features = list(
        feature_list["all_model_features"]
    )
    numeric_features = list(
        feature_list["numeric_features"]
    )
    categorical_features = list(
        feature_list[
            "categorical_features"
        ]
    )

    validate_feature_contract(
        model_features=model_features,
        prohibited_features=list(
            schema_config[
                "prohibited_features"
            ]
        ),
    )

    target_column = schema_config[
        "target_column"
    ]
    identifier_column = find_identifier_column(
        list(train.columns),
        list(
            schema_config[
                "identifier_candidates"
            ]
        ),
    )

    x_train = train[model_features]
    y_train = train[target_column]
    x_validation = validation[
        model_features
    ]
    y_validation = validation[
        target_column
    ]
    x_test = test[model_features]
    y_test = test[target_column]

    report_dir = resolve_path(
        reports_config["output_dir"]
    )
    model_path = resolve_path(
        artifact_config["model_path"]
    )

    if report_dir.exists():
        shutil.rmtree(report_dir)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_state = int(
        random_forest_config[
            "random_state"
        ]
    )
    os.environ["PYTHONHASHSEED"] = str(
        random_state
    )
    random.seed(random_state)
    np.random.seed(random_state)

    tracking_database = resolve_path(
        mlflow_config[
            "tracking_database"
        ]
    )
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_database.resolve()}"
    )
    mlflow.set_experiment(
        mlflow_config[
            "experiment_name"
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy=(
                                    "most_frequent"
                                )
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown=(
                                    "ignore"
                                ),
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=int(
                        random_forest_config[
                            "n_estimators"
                        ]
                    ),
                    max_depth=int(
                        random_forest_config[
                            "max_depth"
                        ]
                    ),
                    min_samples_leaf=int(
                        random_forest_config[
                            "min_samples_leaf"
                        ]
                    ),
                    max_features=(
                        random_forest_config[
                            "max_features"
                        ]
                    ),
                    random_state=(
                        random_state
                    ),
                    n_jobs=int(
                        random_forest_config[
                            "n_jobs"
                        ]
                    ),
                ),
            ),
        ]
    )

    with mlflow.start_run(
        run_name=mlflow_config["run_name"]
    ) as run:
        dummy_start = time.perf_counter()
        dummy = DummyRegressor(
            strategy="mean"
        )
        dummy.fit(x_train, y_train)
        dummy_training_seconds = (
            time.perf_counter()
            - dummy_start
        )
        (
            dummy_validation,
            dummy_test,
            dummy_prediction_seconds,
        ) = evaluate_model(
            dummy,
            x_validation,
            y_validation,
            x_test,
            y_test,
        )

        rf_start = time.perf_counter()
        random_forest.fit(
            x_train,
            y_train,
        )
        rf_training_seconds = (
            time.perf_counter()
            - rf_start
        )
        (
            rf_validation,
            rf_test,
            rf_prediction_seconds,
        ) = evaluate_model(
            random_forest,
            x_validation,
            y_validation,
            x_test,
            y_test,
        )

        sample_size = min(
            int(
                importance_config[
                    "sample_size"
                ]
            ),
            len(x_validation),
        )
        sample = x_validation.sample(
            n=sample_size,
            random_state=int(
                importance_config[
                    "random_state"
                ]
            ),
        )
        sample_target = y_validation.loc[
            sample.index
        ]

        importance = permutation_importance(
            random_forest,
            sample,
            sample_target,
            scoring=(
                "neg_root_mean_squared_error"
            ),
            n_repeats=int(
                importance_config[
                    "n_repeats"
                ]
            ),
            random_state=int(
                importance_config[
                    "random_state"
                ]
            ),
            n_jobs=int(
                importance_config[
                    "n_jobs"
                ]
            ),
        )

        feature_importance = (
            pd.DataFrame(
                {
                    "feature": model_features,
                    "importance_mean": (
                        importance.importances_mean
                    ),
                    "importance_std": (
                        importance.importances_std
                    ),
                }
            )
            .sort_values(
                "importance_mean",
                ascending=False,
                kind="stable",
            )
            .reset_index(drop=True)
        )

        top_five_features = (
            feature_importance[
                "feature"
            ].head(5).tolist()
        )

        test_predictions = (
            random_forest.predict(x_test)
        )
        predictions = pd.DataFrame(
            {
                identifier_column: (
                    test[
                        identifier_column
                    ].astype(str)
                ),
                "actual_total_lift": (
                    y_test.to_numpy()
                ),
                "predicted_total_lift": (
                    test_predictions
                ),
            }
        )
        predictions["residual"] = (
            predictions[
                "actual_total_lift"
            ]
            - predictions[
                "predicted_total_lift"
            ]
        )

        models = [
            {
                "model_name": (
                    "DummyRegressorMean"
                ),
                "feature_set": (
                    "dummy_mean"
                ),
                "feature_count": 0,
                "validation_rmse": (
                    dummy_validation["rmse"]
                ),
                "validation_mae": (
                    dummy_validation["mae"]
                ),
                "validation_r2": (
                    dummy_validation["r2"]
                ),
                "test_rmse": (
                    dummy_test["rmse"]
                ),
                "test_mae": (
                    dummy_test["mae"]
                ),
                "test_r2": (
                    dummy_test["r2"]
                ),
                "training_wall_clock_seconds": (
                    dummy_training_seconds
                ),
                "prediction_seconds": (
                    dummy_prediction_seconds
                ),
            },
            {
                "model_name": (
                    "RandomForestRegressor"
                ),
                "feature_set": (
                    "all_features"
                ),
                "feature_count": len(
                    model_features
                ),
                "validation_rmse": (
                    rf_validation["rmse"]
                ),
                "validation_mae": (
                    rf_validation["mae"]
                ),
                "validation_r2": (
                    rf_validation["r2"]
                ),
                "test_rmse": (
                    rf_test["rmse"]
                ),
                "test_mae": (
                    rf_test["mae"]
                ),
                "test_r2": (
                    rf_test["r2"]
                ),
                "training_wall_clock_seconds": (
                    rf_training_seconds
                ),
                "prediction_seconds": (
                    rf_prediction_seconds
                ),
            },
        ]

        summary = {
            "status": "PASS",
            "mlflow_run_id": run.info.run_id,
            "platform": "scikit-learn",
            "data_split": (
                "Same fixed train, validation, and "
                "test partitions as AutoML."
            ),
            "models": models,
            "top_five_features": (
                top_five_features
            ),
            "model_features": model_features,
            "package_versions": {
                "python": (
                    sys.version.split()[0]
                ),
                "scikit-learn": (
                    importlib.metadata.version(
                        "scikit-learn"
                    )
                ),
                "mlflow": (
                    importlib.metadata.version(
                        "mlflow"
                    )
                ),
                "pandas": (
                    importlib.metadata.version(
                        "pandas"
                    )
                ),
                "numpy": (
                    importlib.metadata.version(
                        "numpy"
                    )
                ),
            },
        }

        (
            report_dir / "run_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        pd.DataFrame(models).to_csv(
            report_dir
            / "baseline_leaderboard.csv",
            index=False,
        )
        feature_importance.to_csv(
            report_dir
            / "feature_importance.csv",
            index=False,
        )
        predictions.to_parquet(
            report_dir
            / "random_forest_test_predictions.parquet",
            index=False,
        )
        joblib.dump(
            random_forest,
            model_path,
        )

        mlflow.log_params(
            {
                "rf_n_estimators": int(
                    random_forest_config[
                        "n_estimators"
                    ]
                ),
                "rf_max_depth": int(
                    random_forest_config[
                        "max_depth"
                    ]
                ),
                "rf_min_samples_leaf": int(
                    random_forest_config[
                        "min_samples_leaf"
                    ]
                ),
                "rf_max_features": (
                    random_forest_config[
                        "max_features"
                    ]
                ),
                "random_state": (
                    random_state
                ),
                "feature_count": len(
                    model_features
                ),
            }
        )
        mlflow.log_metrics(
            {
                "dummy_validation_rmse": (
                    dummy_validation["rmse"]
                ),
                "dummy_test_rmse": (
                    dummy_test["rmse"]
                ),
                "rf_validation_rmse": (
                    rf_validation["rmse"]
                ),
                "rf_test_rmse": (
                    rf_test["rmse"]
                ),
                "rf_test_mae": (
                    rf_test["mae"]
                ),
                "rf_test_r2": (
                    rf_test["r2"]
                ),
                "rf_training_seconds": (
                    rf_training_seconds
                ),
            }
        )
        mlflow.log_artifacts(
            str(report_dir),
            artifact_path="reports",
        )

        print(
            "Baseline experiments completed."
        )
        print(
            "Random Forest validation RMSE: "
            f"{rf_validation['rmse']:.6f}"
        )
        print(
            "Random Forest test RMSE: "
            f"{rf_test['rmse']:.6f}"
        )
        print(
            "Random Forest test MAE: "
            f"{rf_test['mae']:.6f}"
        )
        print(
            "Random Forest test R2: "
            f"{rf_test['r2']:.6f}"
        )
        print(
            "Top five baseline features: "
            + ", ".join(
                top_five_features
            )
        )
        print(
            "PHASE 4A STATUS: PASS"
        )


if __name__ == "__main__":
    main()
