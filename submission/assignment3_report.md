# Assignment 3 - AutoML

**Course:** ADSP 31021 Machine Learning Operations  
**Author:** Anuradha Rakh  
**Primary AutoML platform:** AutoGluon with MLflow  
**Platform mode:** full-code  
**Primary validation metric:** RMSE  
**Random seed:** 42

## Executive summary

The workflow compares a primary AutoGluon plus MLflow implementation with
the required H2O AutoML repeat. Both platforms use the same cleaned,
deterministic train, validation, and test partitions. Each platform is run
first with all approved features and then with its own top three features.
The strongest predictive run by validation RMSE was
**AutoGluon — all_features**, using model
`WeightedEnsemble_L2`. Its test RMSE was
**124.6788**, test MAE was
**91.3759**, and test R-squared was
**0.7956**.

## 1. Dataset loading and setup

- Dataset source: `athletes.csv`
- Target: `total_lift`
- Processed rows after target-quality filtering: 53505
- Rows removed by the target-quality gate: 29
- Clean target range: 8.00 to
  2330.00
- Approved model features: 13
- Identifier excluded from modeling: `athlete_id`
- Target components excluded from modeling to prevent leakage:
  `deadlift`, `candj`, `snatch`, and `backsq`

Cleaned and engineered athlete dataset produced by the reproducible Phase 1 pipeline. Target-component sentinel values and corrupted target totals are audited before deterministic train, validation, and test splitting.

### Fixed split summary

| split | row_count | target_mean | target_median | target_std | target_maximum |
| --- | --- | --- | --- | --- | --- |
| train | 34243.0000 | 1002.4892 | 1020.0000 | 279.0570 | 2330.0000 |
| validation | 8561.0000 | 1011.2949 | 1030.0000 | 277.1584 | 2155.0000 |
| test | 10701.0000 | 1002.4015 | 1020.0000 | 275.8003 | 2040.0000 |

### Assumptions and preprocessing choices

- Lift-component sentinel value `1` was treated as missing before the target
  was calculated.
- Target totals outside the documented project-quality range were audited
  and removed rather than capped.
- Features with more than 70% missingness were removed before AutoML.
- The same fixed partitions were reused across AutoGluon, H2O, and the
  reconstructed baseline.

## 2. Chosen platform configuration

The selected primary workflow is **AutoGluon with MLflow**, classified as
**full-code AutoML**.

AutoGluon automates preprocessing, candidate-model training, algorithm selection, hyperparameter exploration, leaderboard generation, and ensembling while remaining executable locally. MLflow adds experiment tracking and artifact management without paid cloud compute.


The full-code classification is appropriate because datasets, feature
contracts, runtime budgets, experiment execution, metric extraction, and
artifact generation are controlled through Python and YAML. AutoGluon
automates major modeling steps, but the workflow still requires code and
human decisions.

### AutoGluon all-features configuration

- Feature count: 13
- Runtime budget: recorded in MLflow and the run configuration
- Problem type: regression
- Validation metric: RMSE
- Best model: `WeightedEnsemble_L2`
- Validation RMSE: 125.7833
- Test RMSE: 124.6788
- Test MAE: 91.3759
- Test R-squared: 0.7956

## 3. AutoGluon run using all features

### Top three models by validation score

| validation_rank | model | validation_rmse | test_rmse | fit_time_marginal | pred_time_test_marginal |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | WeightedEnsemble_L2 | 125.7833 | 124.6788 | 0.0103 | 0.0138 |
| 2.0000 | CatBoost | 125.9429 | 124.9162 | 4.5256 | 0.0186 |
| 3.0000 | ExtraTreesMSE | 131.0075 | 129.5629 | 1.4833 | 0.2169 |

### Top three models by speed

| speed_rank | model | fit_time_marginal | validation_rmse | test_rmse |
| --- | --- | --- | --- | --- |
| 1.0000 | WeightedEnsemble_L2 | 0.0103 | 125.7833 | 124.6788 |
| 2.0000 | ExtraTreesMSE | 1.4833 | 131.0075 | 129.5629 |
| 3.0000 | RandomForestMSE | 4.4548 | 132.2859 | 130.5045 |

## 4. AutoGluon data insights and feature importance

AutoGluon's top five features were: **gender, fran, weight_height_ratio, grace, pullups**.

Feature importance indicates predictive contribution within the fitted
model; it is not evidence of causality. Engineered features can also share
information with their source variables, so correlated-feature rankings
should be interpreted together rather than independently.

## 5. AutoGluon top-features experiment

Selected features: **gender, fran, weight_height_ratio**

### Top three models by validation score

| validation_rank | model | validation_rmse | test_rmse | fit_time_marginal | pred_time_test_marginal |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | WeightedEnsemble_L2 | 142.5675 | 141.9736 | 0.0057 | 0.0307 |
| 2.0000 | CatBoost | 142.6002 | 142.0445 | 1.0271 | 0.0267 |
| 3.0000 | ExtraTreesMSE | 153.1315 | 150.7506 | 0.6621 | 0.1862 |

### Top three models by speed

| speed_rank | model | fit_time_marginal | validation_rmse | test_rmse |
| --- | --- | --- | --- | --- |
| 1.0000 | WeightedEnsemble_L2 | 0.0057 | 142.5675 | 141.9736 |
| 2.0000 | ExtraTreesMSE | 0.6621 | 153.1315 | 150.7506 |
| 3.0000 | CatBoost | 1.0271 | 142.6002 | 142.0445 |

### Feature-reduction assessment

Degraded validation performance; top-features minus all-features RMSE = 16.7842.

The all-features and top-features runs used the same split strategy and
metric. This makes the validation comparison direct, although individual
AutoML components can still show small run-to-run variation.

## 6. Speed definition and tradeoffs

**Primary platform:** AutoGluon model speed uses platform fit-time columns. Wall-clock experiment time is retained for run-level comparison.


The fastest model is not automatically the best production choice. A
slightly slower model may be justified when its validation improvement is
meaningful and prediction latency remains acceptable. Conversely, a
top-three-feature model may be preferable when it preserves validation
quality while reducing training, scoring, monitoring, and explanation
complexity.

## 7. Comparison with the Assignment 1 baseline

The original Assignment 1 baseline was available: model=RandomForestRegressor; version=Dataset v2 - processed; validation metric=Not available in the original Assignment 1 artifacts; validation value=Not available; test RMSE=152.5517; training seconds=Not available.

### Same-split baseline and AutoML comparison

| platform | feature_set | feature_count | best_model | validation_rmse | test_rmse | test_mae | test_r2 | training_seconds | prediction_seconds | validation_rmse_change_vs_rf_percent | test_rmse_change_vs_rf_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AutoGluon | all_features | 13.0000 | WeightedEnsemble_L2 | 125.7833 | 124.6788 | 91.3759 | 0.7956 | 11.5731 | 0.2372 | -1.9662 | -2.3388 |
| scikit-learn | all_features | 13.0000 | RandomForestRegressor | 128.3061 | 127.6647 | 94.3746 | 0.7857 | 0.9110 | 0.0657 | 0.0000 | 0.0000 |
| H2O AutoML | all_features | 13.0000 | GBM_2_AutoML_1_20260731_161446 | 130.7070 | 129.4366 | 96.1941 | 0.7797 | 63.4568 | 0.2793 | 1.8712 | 1.3880 |
| AutoGluon | top_features | 3.0000 | WeightedEnsemble_L2 | 142.5675 | 141.9736 | 105.2106 | 0.7350 | 3.9903 | 0.2073 | 11.1151 | 11.2082 |
| H2O AutoML | top_features | 3.0000 | GBM_1_AutoML_1_20260731_162003 | 143.5787 | 142.4882 | 105.7187 | 0.7331 | 3.4200 | 0.2529 | 11.9032 | 11.6113 |
| scikit-learn | dummy_mean | 0.0000 | DummyRegressorMean | 277.2820 | 275.7874 | 227.8270 | -0.0000 | 0.0010 | 0.0000 | 116.1097 | 116.0249 |

AutoML reduces manual algorithm selection and hyperparameter experimentation,
but introduces additional dependencies, compute use, artifact volume, and
governance needs. The reconstructed Random Forest provides a controlled
same-split comparison. Any comparison with the original Assignment 1 run is
limited when preprocessing, feature definitions, split logic, or hardware
differ.

## 8. Platform AutoML mode assessment

### Automated

- Missing-value handling supported by the AutoML engines
- Candidate algorithm training
- Hyperparameter exploration
- Model evaluation and leaderboard ranking
- Ensemble construction where supported
- Feature-importance generation
- Experiment metric and artifact logging

### Manual decisions

- Target definition and quality thresholds
- Feature eligibility and leakage exclusions
- Sparse-feature removal threshold
- Fixed split strategy
- Runtime and model-count budgets
- Validation metric selection
- Interpretation of importance and speed tradeoffs

### Operational strengths

- Reproducible configurations and fixed partitions
- Broad model search with consistent evaluation
- Saved metrics, leaderboards, plots, and model artifacts
- Local execution improves portability and avoids cloud billing

### Operational risks

- AutoML can consume substantial compute and storage
- Runtime-limited searches may vary across machines
- Ensembles can be harder to explain and deploy
- Feature importance is model-dependent and not causal
- Schema, data-quality, and drift monitoring remain manual responsibilities

Screenshots are not mandatory for this primary workflow because it is
full-code rather than no-code or low-code. MLflow screenshots may still be
included as supplementary execution evidence.

## 9. H2O AutoML repeat

### Data insights

- Train rows: 34243
- Validation rows: 8561
- Test rows: 10701
- Numeric features: 11
- Categorical features: 2
- H2O top five features: **gender, weight_height_ratio, fran, weight, grace**
- Best all-features model: `GBM_2_AutoML_1_20260731_161446`
- Validation RMSE: 130.7070
- Test RMSE: 129.4366
- Test MAE: 96.1941
- Test R-squared: 0.7797

### H2O top three models by validation score — all features

| validation_rank | model_id | rmse | mae | training_time_ms | predict_time_per_row_ms |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | GBM_2_AutoML_1_20260731_161446 | 130.7070 | 95.9590 | 716.0000 | 0.0133 |
| 2.0000 | GBM_3_AutoML_1_20260731_161446 | 130.7231 | 96.4438 | 840.0000 | 0.0136 |
| 3.0000 | GBM_4_AutoML_1_20260731_161446 | 131.0095 | 96.2505 | 1556.0000 | 0.0170 |

### H2O top three models by training speed — all features

| speed_rank | model_id | training_time_ms | rmse | mae |
| --- | --- | --- | --- | --- |
| 1.0000 | GBM_grid_1_AutoML_1_20260731_161446_model_1 | 599.0000 | 132.3753 | 97.6898 |
| 2.0000 | GBM_5_AutoML_1_20260731_161446 | 618.0000 | 131.5648 | 96.9279 |
| 3.0000 | GLM_1_AutoML_1_20260731_161446 | 627.0000 | 277.2326 | 229.9503 |

### H2O top-features run

Selected features: **gender, weight_height_ratio, fran**

| validation_rank | model_id | rmse | mae | training_time_ms |
| --- | --- | --- | --- | --- |
| 1.0000 | GBM_1_AutoML_1_20260731_162003 | 143.5787 | 105.7738 | 807.0000 |
| 2.0000 | GBM_2_AutoML_1_20260731_162003 | 143.7116 | 105.9787 | 217.0000 |
| 3.0000 | GBM_3_AutoML_1_20260731_162003 | 143.8901 | 106.1359 | 192.0000 |

### H2O top-features speed

| speed_rank | model_id | training_time_ms | rmse | mae |
| --- | --- | --- | --- | --- |
| 1.0000 | GLM_1_AutoML_1_20260731_162003 | 146.0000 | 277.2865 | 230.0558 |
| 2.0000 | GBM_3_AutoML_1_20260731_162003 | 192.0000 | 143.8901 | 106.1359 |
| 3.0000 | GBM_2_AutoML_1_20260731_162003 | 217.0000 | 143.7116 | 105.9787 |

### H2O feature-reduction assessment

Degraded validation performance; top-features minus all-features RMSE = 12.8716.

## 10. Cross-platform findings

AutoGluon and H2O shared
**4**
top-five features, with Jaccard similarity
**0.6667**.

The platforms may select different leaders because they search different
algorithm families, hyperparameter spaces, ensembles, and stopping
strategies. Their time measurements are also not perfectly identical:
AutoGluon model-level speed uses its fit-time columns, while H2O uses
platform-reported `training_time_ms`. Wall-clock run times are retained for
broader comparison.

## 11. Final recommendation

The final model should be selected primarily using validation RMSE, followed
by test-set confirmation, execution cost, interpretability, and deployment
requirements. Based on the completed artifacts, the recommended predictive
run is **AutoGluon — all_features** with model
`WeightedEnsemble_L2`.

A reduced-feature model is operationally attractive only when its validation
performance remains within the configured tolerance. The final comparison
artifact records whether such a reduced model qualified.

## 12. Reproducibility

From a clean Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-automl.txt
python -m pip install -r requirements-h2o.txt
python -m pip install -e .

python scripts/profile_data.py
python scripts/prepare_data.py
python scripts/clean_target_and_resplit.py

python scripts/run_autogluon_all_features.py --overwrite
python scripts/run_autogluon_top_features.py --overwrite

python scripts/check_h2o_environment.py
python scripts/run_h2o_all_features.py --overwrite
python scripts/run_h2o_top_features.py --overwrite

python scripts/run_baselines.py
python scripts/build_final_comparison.py
python scripts/build_assignment_report.py
python scripts/validate_assignment_submission.py
```

Reproduction requires the original `athletes.csv` file under `data/raw/`.
Generated datasets, model binaries, and the local MLflow SQLite database are
not required in Git when the repository includes the code, dependency files,
configuration, saved report artifacts, and execution instructions.

## 13. Limitations

- Runtime budgets can cause small differences across machines.
- The 70% sparse-feature threshold is a project decision rather than a
  universal rule.
- The target-quality maximum is based on the project data history and is
  documented as an auditable assumption.
- Feature importance is model-dependent and does not establish causality.
- Ensemble leaders can be more difficult to explain and deploy than a single
  model.
- The historical Assignment 1 comparison remains incomplete until exact
  original metrics and speed evidence are entered in
  `configs/assignment1_baseline.yaml`.
