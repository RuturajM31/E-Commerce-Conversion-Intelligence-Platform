# Comment Preservation Baseline

## Purpose

This file records the code-comment and docstring baseline before project remediation.
Useful explanations must be preserved, updated, or moved with the related logic.

## Baseline totals

- Python files inspected: 48
- Total code lines: 17253
- Comment tokens: 1691
- Module, class, and function docstrings: 312

## Files with the most comments

| File | Lines | Comments | Docstrings | Syntax |
|---|---:|---:|---:|---|
| `src/models/automl_model_registry.py` | 489 | 180 | 1 | OK |
| `src/models/model_evaluation.py` | 526 | 155 | 7 | OK |
| `src/forecasting/build_business_forecasts.py` | 1268 | 119 | 23 | OK |
| `src/models/model_selection.py` | 348 | 105 | 7 | OK |
| `src/models/finalize_true_champion.py` | 1387 | 103 | 24 | OK |
| `src/anomaly/build_anomaly_signals.py` | 658 | 99 | 12 | OK |
| `src/monitoring/prediction_logger.py` | 198 | 74 | 6 | OK |
| `src/models/build_mvd_method_coverage.py` | 854 | 71 | 15 | OK |
| `app/pages/5_Anomaly_Outlier.py` | 987 | 63 | 13 | OK |
| `src/models/manual_model_registry.py` | 213 | 62 | 1 | OK |
| `app/pages/2_Batch_Scoring.py` | 991 | 61 | 13 | OK |
| `src/models/run_model_selection.py` | 241 | 60 | 3 | OK |
| `app/pages/3_Model_Benchmark_Selection.py` | 1007 | 60 | 16 | OK |
| `app/pages/6_Monitoring_Drift_Health.py` | 996 | 60 | 18 | OK |
| `app/pages/7_MLOps_Architecture.py` | 679 | 60 | 8 | OK |
| `app/pages/8_MVD_Coverage_Proof.py` | 927 | 57 | 12 | OK |
| `app/pages/4_Business_KPI_Forecasting.py` | 899 | 56 | 15 | OK |
| `app/app_utils.py` | 776 | 54 | 31 | OK |
| `app/pages/1_Visitor_Intent_Predictor.py` | 963 | 54 | 12 | OK |
| `app/Executive_Overview.py` | 955 | 53 | 11 | OK |

## Preservation rules

- Do not remove useful learning comments merely to shorten files.
- Update comments when the associated code changes.
- Move explanations with logic that is moved into shared modules.
- Preserve business context, inputs, outputs, and variable-flow notes.
- Review the Git diff before every remediation commit.
