# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks AutoML — All Features
# MAGIC
# MAGIC **Target:** `total_lift`
# MAGIC
# MAGIC **Primary metric:** RMSE
# MAGIC
# MAGIC **Fixed split column:** `data_split`

# COMMAND ----------

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd
from databricks import automl
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# COMMAND ----------

dbutils.widgets.text(
    "table_name",
    "workspace.default.athletes_automl",
)
dbutils.widgets.text(
    "experiment_dir",
    "/Users/<your-email>/databricks_automl",
)
dbutils.widgets.text(
    "experiment_name",
    "athletes_databricks_all_features",
)
dbutils.widgets.text("timeout_minutes", "15")

table_name = dbutils.widgets.get("table_name").strip()
experiment_dir = dbutils.widgets.get(
    "experiment_dir"
).strip()
experiment_name = dbutils.widgets.get(
    "experiment_name"
).strip()
timeout_minutes = int(
    dbutils.widgets.get("timeout_minutes")
)

if "<your-email>" in experiment_dir:
    raise ValueError(
        "Replace <your-email> in the experiment_dir widget."
    )

# COMMAND ----------

dataset = spark.table(table_name)

required_columns = {
    "total_lift",
    "data_split",
}
missing_columns = sorted(
    required_columns.difference(dataset.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

identifier_column = next(
    (
        column
        for column in ("athlete_id", "record_id")
        if column in dataset.columns
    ),
    None,
)

if identifier_column is None:
    raise ValueError(
        "Expected athlete_id or record_id."
    )

observed_splits = {
    row["data_split"]
    for row in dataset.select(
        "data_split"
    ).distinct().collect()
}

if observed_splits != {
    "train",
    "validate",
    "test",
}:
    raise ValueError(
        f"Unexpected split labels: {observed_splits}"
    )

display(
    dataset.groupBy("data_split")
    .count()
    .orderBy("data_split")
)

# COMMAND ----------

start_time = time.perf_counter()

summary = automl.regress(
    dataset=dataset,
    target_col="total_lift",
    primary_metric="rmse",
    split_col="data_split",
    exclude_cols=[identifier_column],
    timeout_minutes=timeout_minutes,
    experiment_dir=experiment_dir,
    experiment_name=experiment_name,
)

automl_wall_clock_seconds = (
    time.perf_counter() - start_time
)

print(
    "Experiment ID:",
    summary.experiment.experiment_id,
)
print(
    "Best run ID:",
    summary.best_trial.mlflow_run_id,
)
print(
    "Best validation RMSE:",
    summary.best_trial.evaluation_metric_score,
)
print(
    "Best model:",
    summary.best_trial.model_description,
)
print(
    "Best model path:",
    summary.best_trial.model_path,
)
print(
    "Best trial notebook:",
    summary.best_trial.notebook_url,
)
print(
    "AutoML wall-clock seconds:",
    round(automl_wall_clock_seconds, 3),
)

# COMMAND ----------

client = MlflowClient()
records = []

for trial in summary.trials:
    run = client.get_run(trial.mlflow_run_id)

    duration_seconds = None
    if (
        run.info.start_time is not None
        and run.info.end_time is not None
    ):
        duration_seconds = (
            run.info.end_time
            - run.info.start_time
        ) / 1000.0

    records.append(
        {
            "run_id": trial.mlflow_run_id,
            "validation_rmse": float(
                trial.evaluation_metric_score
            ),
            "duration_seconds": duration_seconds,
            "model": trial.model_description,
            "model_path": trial.model_path,
            "notebook_url": trial.notebook_url,
            "metrics_json": json.dumps(
                trial.metrics,
                sort_keys=True,
                default=str,
            ),
            "params_json": json.dumps(
                trial.params,
                sort_keys=True,
                default=str,
            ),
        }
    )

leaderboard = (
    pd.DataFrame(records)
    .sort_values(
        "validation_rmse",
        ascending=True,
        kind="stable",
    )
    .reset_index(drop=True)
)

leaderboard.insert(
    0,
    "validation_rank",
    range(1, len(leaderboard) + 1),
)

display(leaderboard)

top_three_by_score = leaderboard.head(3)
display(top_three_by_score)

top_three_by_speed = (
    leaderboard.dropna(
        subset=["duration_seconds"]
    )
    .sort_values(
        "duration_seconds",
        ascending=True,
        kind="stable",
    )
    .head(3)
    .reset_index(drop=True)
)
top_three_by_speed.insert(
    0,
    "speed_rank",
    range(1, len(top_three_by_speed) + 1),
)

display(top_three_by_speed)

# COMMAND ----------

test_pdf = (
    dataset.filter("data_split = 'test'")
    .drop("data_split")
    .toPandas()
)

test_identifiers = test_pdf.pop(
    identifier_column
)
y_test = test_pdf.pop("total_lift")

best_model = mlflow.pyfunc.load_model(
    summary.best_trial.model_path
)

prediction_start = time.perf_counter()
predictions = best_model.predict(test_pdf)
prediction_seconds = (
    time.perf_counter() - prediction_start
)

y_pred = np.asarray(predictions).reshape(-1)

test_rmse = float(
    mean_squared_error(y_test, y_pred) ** 0.5
)
test_mae = float(
    mean_absolute_error(y_test, y_pred)
)
test_r2 = float(
    r2_score(y_test, y_pred)
)

test_metrics = pd.DataFrame(
    [
        {
            "platform": "databricks",
            "feature_set": "all_features",
            "experiment_id": (
                summary.experiment.experiment_id
            ),
            "best_run_id": (
                summary.best_trial.mlflow_run_id
            ),
            "validation_rmse": float(
                summary.best_trial.evaluation_metric_score
            ),
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "prediction_seconds": float(
                prediction_seconds
            ),
            "test_rows": int(len(y_test)),
            "automl_wall_clock_seconds": float(
                automl_wall_clock_seconds
            ),
        }
    ]
)

display(test_metrics)

test_predictions = pd.DataFrame(
    {
        identifier_column: (
            test_identifiers.astype(str)
        ),
        "actual_total_lift": (
            y_test.to_numpy()
        ),
        "predicted_total_lift": y_pred,
        "residual": (
            y_test.to_numpy() - y_pred
        ),
    }
)

display(test_predictions.head(20))

# COMMAND ----------

catalog_schema = ".".join(
    table_name.split(".")[:-1]
)

spark.createDataFrame(
    leaderboard
).write.mode("overwrite").saveAsTable(
    f"{catalog_schema}."
    "databricks_all_features_leaderboard"
)

spark.createDataFrame(
    top_three_by_score
).write.mode("overwrite").saveAsTable(
    f"{catalog_schema}."
    "databricks_all_features_top3_score"
)

spark.createDataFrame(
    top_three_by_speed
).write.mode("overwrite").saveAsTable(
    f"{catalog_schema}."
    "databricks_all_features_top3_speed"
)

spark.createDataFrame(
    test_metrics
).write.mode("overwrite").saveAsTable(
    f"{catalog_schema}."
    "databricks_all_features_test_metrics"
)

spark.createDataFrame(
    test_predictions
).write.mode("overwrite").saveAsTable(
    f"{catalog_schema}."
    "databricks_all_features_test_predictions"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC Open the best generated trial notebook and run its SHAP
# MAGIC feature-importance section. Record the top five features and
# MAGIC use the top three in the next experiment.
