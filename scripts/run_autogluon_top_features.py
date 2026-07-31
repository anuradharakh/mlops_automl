"""Run the AutoGluon experiment using the top three features."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from athlete_automl.autogluon.experiment import (
    build_model_comparison,
    build_speed_ranking,
    find_identifier_column,
    regression_metrics,
    split_features_target,
    validate_modeling_dataset,
    validate_split_schemas,
    write_json,
)

from athlete_automl.autogluon.top_features import (
    read_top_features,
    reduce_dataset,
    validate_selected_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "autogluon_top_features.yaml"


def resolve_path(value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / value


def import_autogluon() -> Any:
    """Import AutoGluon with a useful installation error."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise RuntimeError(
            "AutoGluon is not installed. Run: "
            "python -m pip install -r requirements-automl.txt"
        ) from error

    return TabularPredictor


def save_feature_importance(
    predictor: Any,
    validation_data: pd.DataFrame,
    model_name: str,
    output_path: Path,
    subsample_size: int,
    num_shuffle_sets: int,
    time_limit_seconds: int,
) -> pd.DataFrame:
    """Calculate permutation importance for the reduced feature set."""
    importance = predictor.feature_importance(
        data=validation_data,
        model=model_name,
        subsample_size=min(
            subsample_size,
            len(validation_data),
        ),
        num_shuffle_sets=num_shuffle_sets,
        time_limit=time_limit_seconds,
        silent=True,
    )

    importance = importance.reset_index()

    if "feature" not in importance.columns:
        importance = importance.rename(
            columns={
                importance.columns[0]: "feature",
            }
        )

    importance.insert(
        0,
        "importance_rank",
        range(1, len(importance) + 1),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    importance.to_csv(
        output_path,
        index=False,
    )

    return importance


def create_platform_comparison(
    all_features_metrics_path: Path,
    top_features_payload: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    """Compare AutoGluon all-feature and top-feature results."""
    if not all_features_metrics_path.exists():
        raise FileNotFoundError(
            "Run the all-features experiment first. Missing: "
            f"{all_features_metrics_path}"
        )

    all_payload = json.loads(all_features_metrics_path.read_text(encoding="utf-8"))

    all_metrics = all_payload["metrics"]
    top_metrics = top_features_payload["metrics"]

    comparison = pd.DataFrame(
        [
            {
                "platform": "AutoGluon",
                "feature_set": "all_features",
                "feature_count": all_payload.get("feature_count"),
                "best_model": all_payload["best_model"],
                "validation_rmse": all_metrics["best_validation_rmse"],
                "test_rmse": all_metrics["test_rmse"],
                "test_mae": all_metrics["test_mae"],
                "test_r2": all_metrics["test_r2"],
                "training_wall_clock_seconds": all_metrics[
                    "training_wall_clock_seconds"
                ],
                "test_prediction_seconds": all_metrics["test_prediction_seconds"],
            },
            {
                "platform": "AutoGluon",
                "feature_set": "top_features",
                "feature_count": top_features_payload["feature_count"],
                "best_model": top_features_payload["best_model"],
                "validation_rmse": top_metrics["best_validation_rmse"],
                "test_rmse": top_metrics["test_rmse"],
                "test_mae": top_metrics["test_mae"],
                "test_r2": top_metrics["test_r2"],
                "training_wall_clock_seconds": top_metrics[
                    "training_wall_clock_seconds"
                ],
                "test_prediction_seconds": top_metrics["test_prediction_seconds"],
            },
        ]
    )

    all_row = comparison.loc[comparison["feature_set"] == "all_features"].iloc[0]
    top_row = comparison.loc[comparison["feature_set"] == "top_features"].iloc[0]

    comparison["test_rmse_change_vs_all"] = [
        0.0,
        float(top_row["test_rmse"] - all_row["test_rmse"]),
    ]
    comparison["training_time_change_vs_all"] = [
        0.0,
        float(
            top_row["training_wall_clock_seconds"]
            - all_row["training_wall_clock_seconds"]
        ),
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison.to_csv(
        output_path,
        index=False,
    )

    return comparison


def main() -> None:
    """Execute the top-three-feature AutoGluon experiment."""
    TabularPredictor = import_autogluon()

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    schema_config = config["schema"]
    selection_config = config["feature_selection"]
    experiment_config = config["experiment"]
    importance_config = config["feature_importance"]
    output_config = config["outputs"]

    split_paths = {
        "train": resolve_path(data_config["train_path"]),
        "validation": resolve_path(data_config["validation_path"]),
        "test": resolve_path(data_config["test_path"]),
    }

    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in split_paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Run scripts/prepare_data.py first. Missing: " + ", ".join(missing_files)
        )

    train = pd.read_parquet(split_paths["train"])
    validation = pd.read_parquet(split_paths["validation"])
    test = pd.read_parquet(split_paths["test"])

    validate_split_schemas(
        train=train,
        validation=validation,
        test=test,
    )

    target_column = schema_config["target_column"]
    identifier_column = find_identifier_column(
        columns=list(train.columns),
        candidates=list(schema_config["identifier_candidates"]),
    )

    selected_features = read_top_features(
        path=resolve_path(selection_config["source_path"]),
        feature_column=selection_config["feature_column"],
        rank_column=selection_config["rank_column"],
        count=int(selection_config["top_feature_count"]),
    )

    validate_selected_features(
        selected_features=selected_features,
        available_columns=list(train.columns),
        prohibited_columns={
            target_column,
            identifier_column,
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        },
    )

    train = reduce_dataset(
        dataframe=train,
        identifier_column=identifier_column,
        target_column=target_column,
        selected_features=selected_features,
    )
    validation = reduce_dataset(
        dataframe=validation,
        identifier_column=identifier_column,
        target_column=target_column,
        selected_features=selected_features,
    )
    test = reduce_dataset(
        dataframe=test,
        identifier_column=identifier_column,
        target_column=target_column,
        selected_features=selected_features,
    )

    for dataframe in (
        train,
        validation,
        test,
    ):
        validate_modeling_dataset(
            dataframe=dataframe,
            target_column=target_column,
            identifier_column=identifier_column,
        )

    train_features, train_target, _ = split_features_target(
        train,
        target_column=target_column,
        identifier_column=identifier_column,
    )
    validation_features, validation_target, _ = split_features_target(
        validation,
        target_column=target_column,
        identifier_column=identifier_column,
    )
    test_features, test_target, test_identifiers = split_features_target(
        test,
        target_column=target_column,
        identifier_column=identifier_column,
    )

    train_automl = train_features.copy()
    train_automl[target_column] = train_target.to_numpy()

    validation_automl = validation_features.copy()
    validation_automl[target_column] = validation_target.to_numpy()

    random_state = int(experiment_config["random_state"])
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    predictor_path = resolve_path(output_config["predictor_path"])
    report_dir = resolve_path(output_config["report_dir"])
    comparison_path = resolve_path(output_config["comparison_path"])

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if predictor_path.exists():
        shutil.rmtree(predictor_path)

    selected_features_path = report_dir / "selected_top3_features.json"
    write_json(
        selected_features_path,
        {
            "source": selection_config["source_path"],
            "selected_features": selected_features,
            "feature_count": len(selected_features),
        },
    )

    mlflow.set_tracking_uri(config["tracking"]["uri"])
    mlflow.set_experiment(experiment_config["name"])

    with mlflow.start_run(run_name=experiment_config["run_name"]) as run:
        mlflow.log_params(
            {
                "platform": (config["project"]["platform"]),
                "platform_mode": (config["project"]["mode"]),
                "feature_set": (experiment_config["feature_set"]),
                "selected_features": ",".join(selected_features),
                "target": target_column,
                "primary_metric": (experiment_config["primary_metric"]),
                "presets": (experiment_config["presets"]),
                "time_limit_seconds": int(experiment_config["time_limit_seconds"]),
                "random_state": random_state,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "feature_count": len(selected_features),
            }
        )

        training_started = time.perf_counter()

        predictor = TabularPredictor(
            label=target_column,
            problem_type="regression",
            eval_metric=("root_mean_squared_error"),
            path=str(predictor_path),
            verbosity=2,
        ).fit(
            train_data=train_automl,
            tuning_data=validation_automl,
            presets=experiment_config["presets"],
            time_limit=int(experiment_config["time_limit_seconds"]),
            num_bag_folds=0,
            num_stack_levels=0,
            fit_weighted_ensemble=True,
        )

        training_wall_clock_seconds = time.perf_counter() - training_started

        raw_leaderboard = predictor.leaderboard(
            validation_automl,
            silent=True,
            extra_info=True,
        )

        raw_leaderboard_path = report_dir / "autogluon_raw_leaderboard.csv"
        raw_leaderboard.to_csv(
            raw_leaderboard_path,
            index=False,
        )

        def predict_for_model(
            features: pd.DataFrame,
            model_name: str,
        ) -> np.ndarray:
            return np.asarray(
                predictor.predict(
                    features,
                    model=model_name,
                )
            )

        model_comparison = build_model_comparison(
            leaderboard=raw_leaderboard,
            validation_features=validation_features,
            validation_target=validation_target,
            predict_for_model=predict_for_model,
        )

        leaderboard_path = report_dir / "leaderboard_by_validation_score.csv"
        model_comparison.to_csv(
            leaderboard_path,
            index=False,
        )

        top_three_score_path = report_dir / "top3_models_by_validation_score.csv"
        model_comparison.head(3).to_csv(
            top_three_score_path,
            index=False,
        )

        speed_ranking = build_speed_ranking(model_comparison)
        speed_ranking_path = report_dir / "leaderboard_by_training_speed.csv"
        speed_ranking.to_csv(
            speed_ranking_path,
            index=False,
        )

        top_three_speed_path = report_dir / "top3_models_by_training_speed.csv"
        speed_ranking.head(3).to_csv(
            top_three_speed_path,
            index=False,
        )

        best_model = str(model_comparison.iloc[0]["model"])
        best_validation_rmse = float(model_comparison.iloc[0]["validation_rmse"])

        prediction_started = time.perf_counter()
        test_predictions = predictor.predict(
            test_features,
            model=best_model,
        )
        prediction_seconds = time.perf_counter() - prediction_started

        test_metrics = regression_metrics(
            test_target,
            test_predictions,
        )

        metrics = {
            "best_validation_rmse": (best_validation_rmse),
            "test_rmse": (test_metrics["rmse"]),
            "test_mae": (test_metrics["mae"]),
            "test_r2": test_metrics["r2"],
            "training_wall_clock_seconds": (training_wall_clock_seconds),
            "test_prediction_seconds": (prediction_seconds),
            "test_prediction_rows": len(test),
        }

        mlflow.log_metrics(metrics)

        metrics_payload = {
            "status": "PASS",
            "mlflow_run_id": run.info.run_id,
            "platform": "AutoGluon",
            "platform_mode": "full-code",
            "feature_set": "top_features",
            "selected_features": selected_features,
            "feature_count": len(selected_features),
            "best_model": best_model,
            "metrics": metrics,
            "model_count": int(len(model_comparison)),
            "predictor_path": str(predictor_path.relative_to(PROJECT_ROOT)),
        }

        metrics_path = report_dir / "best_model_metrics.json"
        write_json(
            metrics_path,
            metrics_payload,
        )

        predictions_path = report_dir / "best_model_test_predictions.csv"
        prediction_array = np.asarray(test_predictions).reshape(-1)

        pd.DataFrame(
            {
                identifier_column: (test_identifiers.astype(str)),
                "actual_total_lift": (test_target.to_numpy()),
                "predicted_total_lift": (prediction_array),
                "residual": (test_target.to_numpy() - prediction_array),
            }
        ).to_csv(
            predictions_path,
            index=False,
        )

        importance_path = report_dir / "feature_importance.csv"
        feature_importance = save_feature_importance(
            predictor=predictor,
            validation_data=validation_automl,
            model_name=best_model,
            output_path=importance_path,
            subsample_size=int(importance_config["subsample_size"]),
            num_shuffle_sets=int(importance_config["num_shuffle_sets"]),
            time_limit_seconds=int(importance_config["time_limit_seconds"]),
        )

        predictor_info_path = report_dir / "predictor_info.json"
        predictor_info_path.write_text(
            json.dumps(
                predictor.info(),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        comparison = create_platform_comparison(
            all_features_metrics_path=(
                PROJECT_ROOT
                / "reports"
                / "autogluon"
                / "all_features"
                / "best_model_metrics.json"
            ),
            top_features_payload=metrics_payload,
            output_path=comparison_path,
        )

        mlflow.log_artifact(
            str(CONFIG_PATH),
            artifact_path="configuration",
        )

        for artifact_path in [
            selected_features_path,
            raw_leaderboard_path,
            leaderboard_path,
            top_three_score_path,
            speed_ranking_path,
            top_three_speed_path,
            metrics_path,
            predictions_path,
            importance_path,
            predictor_info_path,
            comparison_path,
        ]:
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="reports",
            )

        print("AutoGluon top-features run completed.")
        print("Selected features: " + ", ".join(selected_features))
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Best model: {best_model}")
        print(f"Best validation RMSE: {best_validation_rmse:.6f}")
        print(f"Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        print(f"Test R²: {test_metrics['r2']:.6f}")
        print(
            "Reduced feature importance: "
            + ", ".join(feature_importance["feature"].astype(str).tolist())
        )
        print("Comparison:")
        print(comparison.to_string(index=False))
        print("PHASE 2B AUTOGLUON STATUS: PASS")


if __name__ == "__main__":
    main()
