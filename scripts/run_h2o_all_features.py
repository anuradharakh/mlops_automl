"""Run the required H2O AutoML all-features experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h2o
import mlflow
import numpy as np
import pandas as pd
import yaml
from h2o.automl import H2OAutoML

from athlete_automl.h2o_workflow.evaluation import (
    build_data_insights,
    find_feature_importance_model,
    normalize_h2o_leaderboard,
    top_models_by_prediction_speed,
    top_models_by_score,
    top_models_by_training_speed,
)
from athlete_automl.h2o_workflow.plotting import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_leaderboard_rmse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "h2o.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run H2O AutoML using all approved features."
        )
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help=(
            "Override the total AutoML runtime in seconds."
        ),
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help=(
            "Override the maximum number of non-ensemble models."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing H2O report and model directories."
        ),
    )
    parser.add_argument(
        "--keep-cluster-running",
        action="store_true",
        help=(
            "Do not shut down the local H2O cluster after the run."
        ),
    )
    return parser.parse_args()


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


def to_h2o_frame(
    dataframe: pd.DataFrame,
    categorical_features: list[str],
    frame_id: str,
) -> h2o.H2OFrame:
    """Convert pandas data to an H2OFrame and set categorical types."""
    frame = h2o.H2OFrame(
        dataframe,
        destination_frame=frame_id,
    )

    for column in categorical_features:
        frame[column] = frame[column].asfactor()

    return frame


def main() -> None:
    """Train, evaluate, report, save, and track H2O AutoML."""
    args = parse_args()

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    schema_config = config["schema"]
    cluster_config = config["cluster"]
    automl_config = config["automl"]
    importance_config = config[
        "feature_importance"
    ]
    mlflow_config = config["mlflow"]
    artifact_config = config["artifacts"]
    report_config = config["reports"]

    train_path = resolve_path(
        data_config["train_path"]
    )
    validation_path = resolve_path(
        data_config["validation_path"]
    )
    test_path = resolve_path(
        data_config["test_path"]
    )
    feature_list_path = resolve_path(
        data_config["feature_list_path"]
    )

    missing_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (
            train_path,
            validation_path,
            test_path,
            feature_list_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required data artifacts are missing: "
            + ", ".join(missing_paths)
        )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(
        validation_path
    )
    test = pd.read_parquet(test_path)
    feature_list = load_json(
        feature_list_path
    )

    model_features = list(
        feature_list["all_model_features"]
    )
    numeric_features = list(
        feature_list["numeric_features"]
    )
    categorical_features = list(
        feature_list["categorical_features"]
    )

    prohibited = set(
        schema_config["prohibited_features"]
    )
    leakage = sorted(
        set(model_features).intersection(
            prohibited
        )
    )

    if leakage:
        raise ValueError(
            f"Prohibited features detected: {leakage}"
        )

    target_column = schema_config[
        "target_column"
    ]
    identifier_column = find_identifier_column(
        columns=list(train.columns),
        candidates=list(
            schema_config[
                "identifier_candidates"
            ]
        ),
    )

    required_columns = {
        identifier_column,
        target_column,
        *model_features,
    }

    for split_name, dataframe in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        missing_columns = sorted(
            required_columns.difference(
                dataframe.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: "
                f"{missing_columns}"
            )

        if dataframe[target_column].isna().any():
            raise ValueError(
                f"{split_name} contains missing targets."
            )

    train_model = train[
        [*model_features, target_column]
    ].copy()
    validation_model = validation[
        [*model_features, target_column]
    ].copy()
    test_model = test[
        [*model_features, target_column]
    ].copy()

    report_dir = resolve_path(
        report_config["output_dir"]
    )
    model_dir = resolve_path(
        artifact_config["model_dir"]
    )

    if args.overwrite:
        for directory in (
            report_dir,
            model_dir,
        ):
            if directory.exists():
                shutil.rmtree(directory)

    if report_dir.exists() and any(
        report_dir.iterdir()
    ):
        raise FileExistsError(
            f"Report directory is not empty: {report_dir}. "
            "Use --overwrite."
        )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_state = int(
        automl_config["seed"]
    )
    os.environ["PYTHONHASHSEED"] = str(
        random_state
    )
    random.seed(random_state)
    np.random.seed(random_state)

    time_limit_seconds = (
        args.time_limit
        if args.time_limit is not None
        else int(
            automl_config[
                "max_runtime_seconds"
            ]
        )
    )
    max_models = (
        args.max_models
        if args.max_models is not None
        else int(
            automl_config["max_models"]
        )
    )

    tracking_database = resolve_path(
        mlflow_config["tracking_database"]
    )
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_database.resolve()}"
    )
    mlflow.set_experiment(
        mlflow_config["experiment_name"]
    )

    package_versions = {
        "python": sys.version.split()[0],
        "h2o": importlib.metadata.version(
            "h2o"
        ),
        "mlflow": importlib.metadata.version(
            "mlflow"
        ),
        "pandas": importlib.metadata.version(
            "pandas"
        ),
        "numpy": importlib.metadata.version(
            "numpy"
        ),
    }

    data_insights = build_data_insights(
        train=train,
        validation=validation,
        test=test,
        model_features=model_features,
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
        target_column=target_column,
    )

    (
        report_dir / "data_insights.json"
    ).write_text(
        json.dumps(
            data_insights,
            indent=2,
        ),
        encoding="utf-8",
    )

    h2o.init(
        max_mem_size=cluster_config[
            "max_memory"
        ],
        nthreads=int(
            cluster_config["nthreads"]
        ),
        port=int(cluster_config["port"]),
    )
    h2o.remove_all()

    try:
        train_h2o = to_h2o_frame(
            train_model,
            categorical_features,
            "athletes_h2o_train",
        )
        validation_h2o = to_h2o_frame(
            validation_model,
            categorical_features,
            "athletes_h2o_validation",
        )
        test_h2o = to_h2o_frame(
            test_model,
            categorical_features,
            "athletes_h2o_test",
        )

        with mlflow.start_run(
            run_name=mlflow_config["run_name"]
        ) as run:
            mlflow.log_params(
                {
                    "platform": (
                        automl_config[
                            "platform"
                        ]
                    ),
                    "feature_set": (
                        automl_config[
                            "feature_set"
                        ]
                    ),
                    "sort_metric": (
                        automl_config[
                            "sort_metric"
                        ]
                    ),
                    "stopping_metric": (
                        automl_config[
                            "stopping_metric"
                        ]
                    ),
                    "max_models": max_models,
                    "max_runtime_seconds": (
                        time_limit_seconds
                    ),
                    "seed": random_state,
                    "nfolds": int(
                        automl_config["nfolds"]
                    ),
                    "excluded_algorithms": (
                        ",".join(
                            automl_config[
                                "exclude_algorithms"
                            ]
                        )
                    ),
                    "feature_count": len(
                        model_features
                    ),
                    "train_rows": len(
                        train_model
                    ),
                    "validation_rows": len(
                        validation_model
                    ),
                    "test_rows": len(
                        test_model
                    ),
                }
            )

            mlflow.log_dict(
                config,
                "configuration/h2o.yaml.json",
            )
            mlflow.log_dict(
                feature_list,
                "configuration/feature_list.json",
            )
            mlflow.log_dict(
                package_versions,
                "environment/package_versions.json",
            )

            automl = H2OAutoML(
                project_name=automl_config[
                    "project_name"
                ],
                sort_metric=automl_config[
                    "sort_metric"
                ],
                stopping_metric=automl_config[
                    "stopping_metric"
                ],
                max_models=max_models,
                max_runtime_secs=(
                    time_limit_seconds
                ),
                seed=random_state,
                nfolds=int(
                    automl_config["nfolds"]
                ),
                exclude_algos=list(
                    automl_config[
                        "exclude_algorithms"
                    ]
                ),
                keep_cross_validation_predictions=False,
                verbosity="info",
            )

            training_start = time.perf_counter()

            automl.train(
                x=model_features,
                y=target_column,
                training_frame=train_h2o,
                validation_frame=(
                    validation_h2o
                ),
                leaderboard_frame=(
                    validation_h2o
                ),
            )

            training_wall_clock_seconds = (
                time.perf_counter()
                - training_start
            )

            raw_leaderboard = (
                h2o.automl.get_leaderboard(
                    automl,
                    extra_columns="ALL",
                ).as_data_frame()
            )

            leaderboard = (
                normalize_h2o_leaderboard(
                    raw_leaderboard
                )
            )
            top_score = top_models_by_score(
                leaderboard,
                model_count=3,
            )
            top_training_speed = (
                top_models_by_training_speed(
                    leaderboard,
                    model_count=3,
                )
            )
            top_prediction_speed = (
                top_models_by_prediction_speed(
                    leaderboard,
                    model_count=3,
                )
            )

            leader = automl.leader
            leader_model_id = str(
                leader.model_id
            )

            prediction_start = (
                time.perf_counter()
            )
            prediction_frame = leader.predict(
                test_h2o
            )
            prediction_seconds = (
                time.perf_counter()
                - prediction_start
            )

            prediction_values = (
                prediction_frame.as_data_frame()[
                    "predict"
                ].to_numpy()
            )

            performance = (
                leader.model_performance(
                    test_h2o
                )
            )
            test_rmse = float(
                performance.rmse()
            )
            test_mae = float(
                performance.mae()
            )
            test_r2 = float(
                performance.r2()
            )

            predictions = pd.DataFrame(
                {
                    identifier_column: (
                        test[
                            identifier_column
                        ].astype(str)
                    ),
                    "actual_total_lift": (
                        test[target_column]
                        .to_numpy()
                    ),
                    "predicted_total_lift": (
                        prediction_values
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

            (
                feature_importance_model_id,
                variable_importance,
            ) = find_feature_importance_model(
                leaderboard=leaderboard,
                model_loader=h2o.get_model,
                maximum_models_to_check=int(
                    importance_config[
                        "maximum_models_to_check"
                    ]
                ),
            )

            top_five_features = (
                variable_importance[
                    "variable"
                ]
                .head(5)
                .astype(str)
                .tolist()
            )
            top_three_features = (
                top_five_features[:3]
            )

            leaderboard.to_csv(
                report_dir
                / "leaderboard.csv",
                index=False,
            )
            top_score.to_csv(
                report_dir
                / "top3_by_score.csv",
                index=False,
            )
            top_training_speed.to_csv(
                report_dir
                / "top3_by_training_speed.csv",
                index=False,
            )
            top_prediction_speed.to_csv(
                report_dir
                / "top3_by_prediction_speed.csv",
                index=False,
            )
            variable_importance.to_csv(
                report_dir
                / "feature_importance.csv",
                index=False,
            )
            variable_importance.head(5).to_csv(
                report_dir
                / "top5_features.csv",
                index=False,
            )
            predictions.to_parquet(
                report_dir
                / "test_predictions.parquet",
                index=False,
            )

            plot_leaderboard_rmse(
                leaderboard=leaderboard,
                output_path=(
                    report_dir
                    / "leaderboard_rmse.png"
                ),
            )
            plot_feature_importance(
                variable_importance=(
                    variable_importance
                ),
                output_path=(
                    report_dir
                    / "feature_importance.png"
                ),
            )
            plot_actual_vs_predicted(
                predictions=predictions,
                output_path=(
                    report_dir
                    / "actual_vs_predicted.png"
                ),
            )

            saved_model_path = h2o.save_model(
                model=leader,
                path=str(model_dir),
                force=True,
            )

            run_summary = {
                "status": "PASS",
                "mlflow_run_id": (
                    run.info.run_id
                ),
                "platform": "H2O AutoML",
                "platform_mode": (
                    "full-code AutoML"
                ),
                "feature_set": (
                    "all_features"
                ),
                "feature_count": len(
                    model_features
                ),
                "model_features": (
                    model_features
                ),
                "best_model": (
                    leader_model_id
                ),
                "validation_rmse": float(
                    top_score.iloc[0]["rmse"]
                ),
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "training_wall_clock_seconds": float(
                    training_wall_clock_seconds
                ),
                "prediction_seconds": float(
                    prediction_seconds
                ),
                "prediction_rows": int(
                    len(test_model)
                ),
                "speed_measure": (
                    "training_time_ms"
                ),
                "feature_importance_model": (
                    feature_importance_model_id
                ),
                "top_five_features": (
                    top_five_features
                ),
                "top_three_features": (
                    top_three_features
                ),
                "saved_model_path": str(
                    Path(saved_model_path)
                    .resolve()
                    .relative_to(PROJECT_ROOT)
                ),
                "package_versions": (
                    package_versions
                ),
                "validation_strategy": (
                    "nfolds=0 with the fixed validation "
                    "partition used as validation_frame and "
                    "leaderboard_frame. The fixed test set is "
                    "used only for final evaluation."
                ),
                "reproducibility_note": (
                    "Seed and max_models are fixed, and "
                    "DeepLearning is excluded. The 900-second "
                    "runtime cap remains a resource-dependent "
                    "secondary stopping condition."
                ),
            }

            summary_path = (
                report_dir
                / "run_summary.json"
            )
            summary_path.write_text(
                json.dumps(
                    run_summary,
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_metrics(
                {
                    "validation_rmse": (
                        run_summary[
                            "validation_rmse"
                        ]
                    ),
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "test_r2": test_r2,
                    "training_wall_clock_seconds": (
                        training_wall_clock_seconds
                    ),
                    "prediction_seconds": (
                        prediction_seconds
                    ),
                }
            )
            mlflow.set_tags(
                {
                    "assignment": (
                        "ADSP 31021 Assignment 3"
                    ),
                    "platform_mode": (
                        "full-code"
                    ),
                    "feature_set": (
                        "all_features"
                    ),
                    "best_model": (
                        leader_model_id
                    ),
                    "feature_importance_model": (
                        feature_importance_model_id
                    ),
                }
            )
            mlflow.log_artifacts(
                str(report_dir),
                artifact_path=(
                    "reports/all_features"
                ),
            )

            print(
                "H2O AutoML all-features run completed."
            )
            print(
                f"MLflow run ID: {run.info.run_id}"
            )
            print(
                f"Best model: {leader_model_id}"
            )
            print(
                "Validation RMSE: "
                f"{run_summary['validation_rmse']:.6f}"
            )
            print(
                f"Test RMSE: {test_rmse:.6f}"
            )
            print(
                f"Test MAE: {test_mae:.6f}"
            )
            print(
                f"Test R2: {test_r2:.6f}"
            )
            print(
                "Top five features: "
                + ", ".join(
                    top_five_features
                )
            )
            print(
                "Top three features: "
                + ", ".join(
                    top_three_features
                )
            )
            print(
                f"Reports: {report_dir}"
            )
            print(
                "PHASE 3A STATUS: PASS"
            )
    finally:
        if not args.keep_cluster_running:
            h2o.cluster().shutdown(
                prompt=False
            )


if __name__ == "__main__":
    main()
