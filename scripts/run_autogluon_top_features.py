"""Run AutoGluon using the top three all-features results."""

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

import mlflow
import numpy as np
import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor

from athlete_automl.automl.evaluation import (
    normalize_autogluon_leaderboard,
    regression_metrics,
    top_models_by_score,
    top_models_by_speed,
    validate_feature_contract,
)
from athlete_automl.automl.feature_selection import (
    compare_run_summaries,
    extract_top_features,
)
from athlete_automl.automl.plotting import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_leaderboard_rmse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "autogluon_top_features.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AutoGluon using the top three features."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing top-feature predictor."
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help=(
            "Override the configured training time limit."
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


def main() -> None:
    """Train, evaluate, compare, and track top-feature AutoML."""
    args = parse_args()

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    schema_config = config["schema"]
    automl_config = config["automl"]
    importance_config = config[
        "feature_importance"
    ]
    mlflow_config = config["mlflow"]
    reports_config = config["reports"]

    train_path = resolve_path(
        data_config["train_path"]
    )
    validation_path = resolve_path(
        data_config["validation_path"]
    )
    test_path = resolve_path(
        data_config["test_path"]
    )
    all_summary_path = resolve_path(
        data_config[
            "all_features_summary_path"
        ]
    )

    missing_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (
            train_path,
            validation_path,
            test_path,
            all_summary_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required prior artifacts are missing: "
            + ", ".join(missing_paths)
        )

    all_summary = load_json(
        all_summary_path
    )
    model_features = extract_top_features(
        all_summary,
        expected_count=int(
            automl_config[
                "expected_feature_count"
            ]
        ),
    )

    validate_feature_contract(
        model_features=model_features,
        prohibited_features=list(
            schema_config[
                "prohibited_features"
            ]
        ),
    )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(
        validation_path
    )
    test = pd.read_parquet(test_path)

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

    predictor_path = resolve_path(
        automl_config["predictor_path"]
    )
    report_dir = resolve_path(
        reports_config["output_dir"]
    )
    comparison_dir = resolve_path(
        reports_config["comparison_dir"]
    )

    if predictor_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Predictor directory already exists: "
                f"{predictor_path}. Use --overwrite."
            )

        shutil.rmtree(predictor_path)

    predictor_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_state = int(
        automl_config["random_state"]
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
                "time_limit_seconds"
            ]
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
        "autogluon.tabular": (
            importlib.metadata.version(
                "autogluon.tabular"
            )
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
        "scikit-learn": (
            importlib.metadata.version(
                "scikit-learn"
            )
        ),
    }

    with mlflow.start_run(
        run_name=mlflow_config["run_name"]
    ) as run:
        mlflow.log_params(
            {
                "platform": automl_config[
                    "platform"
                ],
                "feature_set": automl_config[
                    "feature_set"
                ],
                "problem_type": automl_config[
                    "problem_type"
                ],
                "eval_metric": automl_config[
                    "eval_metric"
                ],
                "presets": automl_config[
                    "presets"
                ],
                "time_limit_seconds": (
                    time_limit_seconds
                ),
                "fit_strategy": automl_config[
                    "fit_strategy"
                ],
                "random_state": random_state,
                "feature_count": len(
                    model_features
                ),
                "selected_features": ",".join(
                    model_features
                ),
                "train_rows": len(train_model),
                "validation_rows": len(
                    validation_model
                ),
                "test_rows": len(test_model),
            }
        )

        mlflow.log_dict(
            config,
            "configuration/"
            "autogluon_top_features.yaml.json",
        )
        mlflow.log_dict(
            all_summary,
            "configuration/"
            "all_features_run_summary.json",
        )
        mlflow.log_dict(
            package_versions,
            "environment/package_versions.json",
        )

        training_start = time.perf_counter()

        predictor = TabularPredictor(
            label=target_column,
            problem_type=automl_config[
                "problem_type"
            ],
            eval_metric=automl_config[
                "eval_metric"
            ],
            path=str(predictor_path),
            verbosity=2,
        ).fit(
            train_data=train_model,
            tuning_data=validation_model,
            time_limit=time_limit_seconds,
            presets=automl_config["presets"],
            fit_strategy=automl_config[
                "fit_strategy"
            ],
        )

        training_wall_clock_seconds = (
            time.perf_counter()
            - training_start
        )

        raw_leaderboard = predictor.leaderboard(
            test_model,
            extra_info=True,
            silent=True,
        )
        leaderboard = (
            normalize_autogluon_leaderboard(
                raw_leaderboard
            )
        )

        top_score = top_models_by_score(
            leaderboard,
            model_count=3,
        )
        top_speed, speed_metric = (
            top_models_by_speed(
                leaderboard,
                model_count=3,
            )
        )

        test_features = test_model[
            model_features
        ]
        y_test = test_model[target_column]

        prediction_start = time.perf_counter()
        y_pred = predictor.predict(
            test_features
        )
        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        metrics = regression_metrics(
            y_true=y_test,
            y_pred=y_pred,
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
                    np.asarray(y_pred)
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

        importance = predictor.feature_importance(
            data=validation_model,
            model=predictor.model_best,
            feature_stage="original",
            subsample_size=min(
                int(
                    importance_config[
                        "subsample_size"
                    ]
                ),
                len(validation_model),
            ),
            num_shuffle_sets=int(
                importance_config[
                    "num_shuffle_sets"
                ]
            ),
            silent=True,
        ).sort_values(
            "importance",
            ascending=False,
        )

        leaderboard.to_csv(
            report_dir / "leaderboard.csv",
            index=False,
        )
        top_score.to_csv(
            report_dir / "top3_by_score.csv",
            index=False,
        )
        top_speed.to_csv(
            report_dir / "top3_by_speed.csv",
            index=False,
        )
        importance.to_csv(
            report_dir
            / "feature_importance.csv",
            index=True,
            index_label="feature",
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
            feature_importance=importance,
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

        run_summary = {
            "status": "PASS",
            "mlflow_run_id": run.info.run_id,
            "platform": "AutoGluon",
            "platform_mode": (
                "full-code AutoML"
            ),
            "feature_set": "top_features",
            "feature_count": len(
                model_features
            ),
            "model_features": model_features,
            "best_model": predictor.model_best,
            "validation_rmse": float(
                top_score.iloc[0][
                    "validation_rmse"
                ]
            ),
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "test_r2": metrics["r2"],
            "training_wall_clock_seconds": float(
                training_wall_clock_seconds
            ),
            "prediction_seconds": float(
                prediction_seconds
            ),
            "prediction_rows": int(
                len(test_model)
            ),
            "speed_measure": speed_metric,
            "predictor_path": str(
                predictor_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "package_versions": (
                package_versions
            ),
            "reproducibility_note": (
                "The same fixed data partitions and AutoML "
                "configuration are used as in the all-features run."
            ),
        }

        summary_path = (
            report_dir / "run_summary.json"
        )
        summary_path.write_text(
            json.dumps(
                run_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        comparison = compare_run_summaries(
            all_features=all_summary,
            top_features=run_summary,
        )

        comparison_json_path = (
            comparison_dir
            / "all_vs_top_features.json"
        )
        comparison_json_path.write_text(
            json.dumps(
                comparison,
                indent=2,
            ),
            encoding="utf-8",
        )

        comparison_csv_path = (
            comparison_dir
            / "all_vs_top_features.csv"
        )
        pd.DataFrame(
            [
                {
                    "feature_set": (
                        "all_features"
                    ),
                    "feature_count": (
                        comparison[
                            "all_features_count"
                        ]
                    ),
                    "best_model": (
                        comparison[
                            "all_features_best_model"
                        ]
                    ),
                    "validation_rmse": (
                        comparison[
                            "all_features_validation_rmse"
                        ]
                    ),
                    "test_rmse": (
                        comparison[
                            "all_features_test_rmse"
                        ]
                    ),
                    "test_mae": (
                        comparison[
                            "all_features_test_mae"
                        ]
                    ),
                    "test_r2": (
                        comparison[
                            "all_features_test_r2"
                        ]
                    ),
                    "training_seconds": (
                        comparison[
                            "all_features_training_seconds"
                        ]
                    ),
                    "prediction_seconds": (
                        comparison[
                            "all_features_prediction_seconds"
                        ]
                    ),
                },
                {
                    "feature_set": (
                        "top_features"
                    ),
                    "feature_count": (
                        comparison[
                            "top_features_count"
                        ]
                    ),
                    "best_model": (
                        comparison[
                            "top_features_best_model"
                        ]
                    ),
                    "validation_rmse": (
                        comparison[
                            "top_features_validation_rmse"
                        ]
                    ),
                    "test_rmse": (
                        comparison[
                            "top_features_test_rmse"
                        ]
                    ),
                    "test_mae": (
                        comparison[
                            "top_features_test_mae"
                        ]
                    ),
                    "test_r2": (
                        comparison[
                            "top_features_test_r2"
                        ]
                    ),
                    "training_seconds": (
                        comparison[
                            "top_features_training_seconds"
                        ]
                    ),
                    "prediction_seconds": (
                        comparison[
                            "top_features_prediction_seconds"
                        ]
                    ),
                },
            ]
        ).to_csv(
            comparison_csv_path,
            index=False,
        )

        mlflow.log_metrics(
            {
                "validation_rmse": (
                    run_summary[
                        "validation_rmse"
                    ]
                ),
                "test_rmse": metrics[
                    "rmse"
                ],
                "test_mae": metrics[
                    "mae"
                ],
                "test_r2": metrics["r2"],
                "training_wall_clock_seconds": (
                    training_wall_clock_seconds
                ),
                "prediction_seconds": (
                    prediction_seconds
                ),
                "validation_rmse_delta_top_minus_all": (
                    comparison[
                        "validation_rmse_delta_top_minus_all"
                    ]
                ),
                "test_rmse_delta_top_minus_all": (
                    comparison[
                        "test_rmse_delta_top_minus_all"
                    ]
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
                    "top_features"
                ),
                "best_model": (
                    predictor.model_best
                ),
                "selected_features": (
                    ",".join(model_features)
                ),
            }
        )
        mlflow.log_artifacts(
            str(report_dir),
            artifact_path=(
                "reports/top_features"
            ),
        )
        mlflow.log_artifacts(
            str(comparison_dir),
            artifact_path=(
                "reports/comparison"
            ),
        )

        print(
            "AutoGluon top-features run completed."
        )
        print(
            f"MLflow run ID: {run.info.run_id}"
        )
        print(
            "Selected features: "
            + ", ".join(model_features)
        )
        print(
            f"Best model: {predictor.model_best}"
        )
        print(
            "Validation RMSE: "
            f"{run_summary['validation_rmse']:.6f}"
        )
        print(
            f"Test RMSE: {metrics['rmse']:.6f}"
        )
        print(
            f"Test MAE: {metrics['mae']:.6f}"
        )
        print(
            f"Test R2: {metrics['r2']:.6f}"
        )
        print(
            "Feature-reduction assessment: "
            + comparison[
                "validation_performance_assessment"
            ]
        )
        print(
            f"Reports: {report_dir}"
        )
        print(
            f"Comparison: {comparison_json_path}"
        )
        print(
            "PHASE 2C STATUS: PASS"
        )


if __name__ == "__main__":
    main()
