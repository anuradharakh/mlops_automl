# Assignment 3 - AutoML High-Level Report

**Course:** ADSP 31021 Machine Learning Operations  
**Author:** Anuradha Rakh  
**Primary platform:** AutoGluon with MLflow  
**Repository:** https://github.com/anuradharakh/mlops_automl

## Executive Summary

The strongest run by validation RMSE was **AutoGluon -
all_features**, using model `WeightedEnsemble_L2`.

- Validation RMSE: 125.783
- Test RMSE: 124.679
- Test MAE: 91.376
- Test R-squared: 0.796

## Dataset

- Raw rows: 423,006
- Final modeling rows: 53505
- Target-quality rows removed: 29
- Approved features: 13
- Split: 64% train / 16% validation / 20% test
- Random seed: 42

## Comparison

| platform     | feature_set   |   feature_count | best_model                     |   validation_rmse |   test_rmse |   test_mae |      test_r2 |   training_seconds |   prediction_seconds | is_automl   |   validation_rank |   test_rmse_rank |   test_r2_rank |   training_speed_rank |   prediction_speed_rank |   validation_rmse_change_vs_rf_percent |   test_rmse_change_vs_rf_percent |   test_mae_change_vs_rf_percent |   test_r2_change_vs_rf_percent |
|:-------------|:--------------|----------------:|:-------------------------------|------------------:|------------:|-----------:|-------------:|-------------------:|---------------------:|:------------|------------------:|-----------------:|---------------:|----------------------:|------------------------:|---------------------------------------:|---------------------------------:|--------------------------------:|-------------------------------:|
| AutoGluon    | all_features  |              13 | WeightedEnsemble_L2            |           125.783 |     124.679 |    91.3759 |  0.795621    |       11.5731      |           0.237155   | True        |                 1 |                1 |              1 |                     5 |                       4 |                               -1.96623 |                         -2.3388  |                        -3.17741 |                        1.26079 |
| scikit-learn | all_features  |              13 | RandomForestRegressor          |           128.306 |     127.665 |    94.3746 |  0.785715    |        0.911024    |           0.0657051  | False       |                 2 |                2 |              2 |                     2 |                       2 |                                0       |                          0       |                         0       |                        0       |
| H2O AutoML   | all_features  |              13 | GBM_2_AutoML_1_20260731_161446 |           130.707 |     129.437 |    96.1941 |  0.779725    |       63.4568      |           0.279309   | True        |                 3 |                3 |              3 |                     6 |                       6 |                                1.87124 |                          1.38799 |                         1.92796 |                       -0.76234 |
| AutoGluon    | top_features  |               3 | WeightedEnsemble_L2            |           142.568 |     141.974 |   105.211  |  0.734988    |        3.99031     |           0.207343   | True        |                 4 |                4 |              4 |                     4 |                       3 |                               11.1151  |                         11.2082  |                        11.4819  |                       -6.45616 |
| H2O AutoML   | top_features  |               3 | GBM_1_AutoML_1_20260731_162003 |           143.579 |     142.488 |   105.719  |  0.733063    |        3.41996     |           0.252909   | True        |                 5 |                5 |              5 |                     3 |                       5 |                               11.9032  |                         11.6113  |                        12.0203  |                       -6.70112 |
| scikit-learn | dummy_mean    |               0 | DummyRegressorMean             |           277.282 |     275.787 |   227.827  | -1.01108e-07 |        0.000962583 |           3.5125e-05 | False       |                 6 |                6 |              6 |                     1 |                       1 |                              116.11    |                        116.025   |                       141.407   |                     -100       |

## Top Features

- AutoGluon: gender, fran, weight_height_ratio, grace, pullups
- H2O: gender, weight_height_ratio, fran, weight, grace

## Assignment 1 Baseline

Assignment 1 Dataset v2 Random Forest reported test RMSE
152.552, test MAE
114.887, and test R-squared
0.706. It used a different processed dataset and
80/20 split, and did not retain directly comparable validation or runtime evidence.

## Recommendation

Select the model primarily by validation RMSE and confirm performance on the
untouched test set. Prefer a reduced-feature run only when its validation
performance remains within the documented tolerance and operational simplicity is
valuable.
