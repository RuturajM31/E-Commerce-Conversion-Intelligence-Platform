# Dependency Upgrade Procedure

## Purpose

Upgrade dependencies without breaking the application, MLflow, Evidently,
Docker, or the production XGBoost model.

## Safe workflow

1. Keep the live requirement files unchanged.
2. Build candidate requirement files under
   `reports/qa/dependency_preflight/`.
3. Use exact version pins.
4. Test each candidate in a disposable Python 3.10 Linux container.
5. Run `pip check`.
6. Run `pip-audit`.
7. Load the production champion model and compare predictions.
8. Run focused tests for the affected service.
9. For MLflow, use native `mlflow.xgboost` logging and JSON model format.
10. For Evidently, generate real feature and prediction drift reports.
11. Promote pins only after all isolated checks pass.
12. Run the full pytest suite, security checks, Docker Compose, Streamlit,
    Helm, and Kubernetes validation.
13. Review the Git diff before committing.

## Rollback

Use Git to restore the requirement files if final QA fails. Do not delete the
validated candidate files because they are reproducibility evidence.

## Current validated service versions

- PyArrow `23.0.1`
- scikit-learn `1.5.0`
- Streamlit `1.54.0`
- Pillow `12.2.0`
- MLflow `3.14.0`
- Evidently `0.7.21`
- XGBoost `3.2.0`
- cryptography `48.0.1`
