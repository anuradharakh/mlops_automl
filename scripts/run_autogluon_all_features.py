"""Run the primary AutoGluon all-features workflow with MLflow."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from athlete_automl.autogluon_workflow import (
    find_identifier,
    rank_by_speed,
    rank_models,
    regression_metrics,
    validate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "autogluon.yaml"


def resolve(value: str) -> Path:
    return PROJECT_ROOT / value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise RuntimeError(
            "Install AutoGluon first: python -m pip install -r requirements-automl.txt"
        ) from error

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_path = resolve(config["data"]["train_path"])
    validation_path = resolve(config["data"]["validation_path"])
    test_path = resolve(config["data"]["test_path"])

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (train_path, validation_path, test_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run scripts/prepare_data.py first. Missing: " + ", ".join(missing)
        )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)

    if list(train.columns) != list(validation.columns) or list(train.columns) != list(
        test.columns
    ):
        raise ValueError("Train, validation, and test schemas do not match.")

    target = config["schema"]["target_column"]
    identifier = find_identifier(
        list(train.columns),
        list(config["schema"]["identifier_candidates"]),
    )

    for frame in (train, validation, test):
        validate_dataset(frame, target, identifier)

    feature_columns = [
        column for column in train.columns if column not in {target, identifier}
    ]

    train_automl = train[feature_columns + [target]].copy()
    validation_automl = validation[feature_columns + [target]].copy()
    validation_features = validation[feature_columns].copy()
    validation_target = validation[target].copy()
    test_features = test[feature_columns].copy()
    test_target = test[target].copy()

    seed = int(config["experiment"]["random_state"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    predictor_path = resolve(config["outputs"]["predictor_path"])
    report_dir = resolve(config["outputs"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    if predictor_path.exists():
        shutil.rmtree(predictor_path)

    data_insights = {
        "row_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "feature_count": len(feature_columns),
        "numeric_features": [
            column
            for column in feature_columns
            if pd.api.types.is_numeric_dtype(train[column])
        ],
        "categorical_features": [
            column
            for column in feature_columns
            if not pd.api.types.is_numeric_dtype(train[column])
        ],
        "target_statistics": train[target].describe().to_dict(),
        "missing_percentage": (train[feature_columns].isna().mean().mul(100).to_dict()),
    }
    data_insights_path = report_dir / "data_insights.json"
    write_json(data_insights_path, data_insights)

    mlflow.set_tracking_uri(config["tracking"]["uri"])
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["experiment"]["run_name"]) as run:
        mlflow.log_params(
            {
                "platform": config["project"]["platform"],
                "platform_mode": config["project"]["mode"],
                "feature_set": "all_features",
                "target": target,
                "presets": config["experiment"]["presets"],
                "time_limit_seconds": int(config["experiment"]["time_limit_seconds"]),
                "random_state": seed,
                "feature_count": len(feature_columns),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }
        )

        started = time.perf_counter()
        predictor = TabularPredictor(
            label=target,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(predictor_path),
            verbosity=2,
        ).fit(
            train_data=train_automl,
            tuning_data=validation_automl,
            presets=config["experiment"]["presets"],
            time_limit=int(config["experiment"]["time_limit_seconds"]),
            num_bag_folds=0,
            num_stack_levels=0,
        )
        training_seconds = time.perf_counter() - started

        raw_leaderboard = predictor.leaderboard(
            validation_automl,
            silent=True,
            extra_info=True,
        )
        raw_leaderboard_path = report_dir / "autogluon_raw_leaderboard.csv"
        raw_leaderboard.to_csv(raw_leaderboard_path, index=False)

        model_ranking = rank_models(
            raw_leaderboard,
            validation_features,
            validation_target,
            lambda data, model: np.asarray(predictor.predict(data, model=model)),
        )
        model_ranking_path = report_dir / "leaderboard_by_validation_score.csv"
        model_ranking.to_csv(model_ranking_path, index=False)
        top3_score_path = report_dir / "top3_models_by_validation_score.csv"
        model_ranking.head(3).to_csv(top3_score_path, index=False)

        speed_ranking = rank_by_speed(model_ranking)
        speed_ranking_path = report_dir / "leaderboard_by_training_speed.csv"
        speed_ranking.to_csv(speed_ranking_path, index=False)
        top3_speed_path = report_dir / "top3_models_by_training_speed.csv"
        speed_ranking.head(3).to_csv(top3_speed_path, index=False)

        best_model = str(model_ranking.iloc[0]["model"])
        prediction_started = time.perf_counter()
        test_predictions = predictor.predict(test_features, model=best_model)
        prediction_seconds = time.perf_counter() - prediction_started
        test_metrics = regression_metrics(test_target, test_predictions)

        metrics = {
            "best_validation_rmse": float(model_ranking.iloc[0]["validation_rmse"]),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
            "training_wall_clock_seconds": training_seconds,
            "test_prediction_seconds": prediction_seconds,
        }
        mlflow.log_metrics(metrics)

        metrics_path = report_dir / "best_model_metrics.json"
        write_json(
            metrics_path,
            {
                "status": "PASS",
                "mlflow_run_id": run.info.run_id,
                "best_model": best_model,
                "metrics": metrics,
            },
        )

        predictions_path = report_dir / "best_model_test_predictions.csv"
        pd.DataFrame(
            {
                identifier: test[identifier].astype(str),
                "actual_total_lift": test_target.to_numpy(),
                "predicted_total_lift": np.asarray(test_predictions).reshape(-1),
                "residual": test_target.to_numpy()
                - np.asarray(test_predictions).reshape(-1),
            }
        ).to_csv(predictions_path, index=False)

        importance = predictor.feature_importance(
            data=validation_automl,
            model=best_model,
            subsample_size=min(
                int(config["feature_importance"]["subsample_size"]),
                len(validation_automl),
            ),
            num_shuffle_sets=int(config["feature_importance"]["num_shuffle_sets"]),
            time_limit=int(config["feature_importance"]["time_limit_seconds"]),
            silent=True,
        ).reset_index()
        importance = importance.rename(columns={importance.columns[0]: "feature"})
        importance.insert(0, "importance_rank", range(1, len(importance) + 1))

        importance_path = report_dir / "feature_importance.csv"
        top5_path = report_dir / "top5_features.csv"
        importance.to_csv(importance_path, index=False)
        importance.head(5).to_csv(top5_path, index=False)

        predictor_info_path = report_dir / "predictor_info.json"
        write_json(predictor_info_path, predictor.info())

        for artifact in (
            CONFIG_PATH,
            data_insights_path,
            raw_leaderboard_path,
            model_ranking_path,
            top3_score_path,
            speed_ranking_path,
            top3_speed_path,
            metrics_path,
            predictions_path,
            importance_path,
            top5_path,
            predictor_info_path,
        ):
            mlflow.log_artifact(str(artifact), artifact_path="evidence")

        print("AutoGluon all-features run completed.")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Best model: {best_model}")
        print(f"Validation RMSE: {metrics['best_validation_rmse']:.6f}")
        print(f"Test RMSE: {metrics['test_rmse']:.6f}")
        print(f"Test MAE: {metrics['test_mae']:.6f}")
        print(f"Test R²: {metrics['test_r2']:.6f}")
        print("PHASE 2 AUTOGLUON STATUS: PASS")


if __name__ == "__main__":
    main()
