# Final AutoML Comparison

## Experimental controls

All current-split models use the same cleaned train, validation, and test
partitions. Target components and identifiers are excluded from model
features. Model selection is ranked primarily by validation RMSE; test
metrics are retained for final evaluation.

## Comparison

| platform | feature_set | feature_count | best_model | validation_rmse | test_rmse | test_mae | test_r2 | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AutoGluon | all_features | 13 | WeightedEnsemble_L2 | 125.7833 | 124.6788 | 91.3759 | 0.7956 | 11.5731 |
| scikit-learn | all_features | 13 | RandomForestRegressor | 128.3061 | 127.6647 | 94.3746 | 0.7857 | 0.9110 |
| H2O AutoML | all_features | 13 | GBM_2_AutoML_1_20260731_161446 | 130.7070 | 129.4366 | 96.1941 | 0.7797 | 63.4568 |
| AutoGluon | top_features | 3 | WeightedEnsemble_L2 | 142.5675 | 141.9736 | 105.2106 | 0.7350 | 3.9903 |
| H2O AutoML | top_features | 3 | GBM_1_AutoML_1_20260731_162003 | 143.5787 | 142.4882 | 105.7187 | 0.7331 | 3.4200 |
| scikit-learn | dummy_mean | 0 | DummyRegressorMean | 277.2820 | 275.7874 | 227.8270 | -0.0000 | 0.0010 |

## Recommendation

The strongest predictive run by validation RMSE is
**AutoGluon — all_features** using
**13 features** and model
`WeightedEnsemble_L2`.

- Validation RMSE: 125.7833
- Test RMSE: 124.6788
- Test MAE: 91.3759
- Test R-squared: 0.7956

## Reduced-feature assessment

No reduced-feature run stayed within the configured validation-RMSE tolerance.

## Feature agreement

AutoGluon and H2O share
**4**
of their top-five features. Their Jaccard similarity is
**0.6667**.

## Interpretation notes

- Negative RMSE or MAE change versus Random Forest indicates improvement.
- Positive R-squared change versus Random Forest indicates improvement.
- Wall-clock training time is machine-dependent and should be interpreted
  together with predictive quality.
- A reduced-feature model is recommended only when its validation RMSE
  remains within the configured tolerance of its platform's all-features
  run.
