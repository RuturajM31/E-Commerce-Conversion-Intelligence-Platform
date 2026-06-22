# Baseline Test Report

## Purpose

This report records the project state before full remediation begins.

## Git baseline

- Branch: fix/full-project-remediation
- Baseline commit: 5fd0097
- Protected tag: baseline-submitted-2026-06-19
- Remote branch: origin/fix/full-project-remediation

## Python environment

- Python version: 3.10.9
- Python executable source: pyenv
- Pytest version: 9.1.0

## Baseline test command

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -ra -p no:cacheprovider

## Baseline result

- Passed: 30
- Skipped: 3
- Failed: 0
- Errors: 0

## Skipped tests

1. tests/test_forecasting_smoke.py
   - Strict Prophet-best test skipped by default.

2. tests/test_model_comparison_logic.py
   - Strict XGBoost greater-than Logistic Regression test skipped by default.

3. tests/test_model_training_smoke.py
   - Full retraining test skipped by default.
   - Enable with RUN_FULL_PIPELINE_TESTS=1.

## Working-tree check

The Git working tree remained clean after baseline testing.

## Interpretation

The submitted baseline is syntactically and functionally stable under the
current default test suite. The remediation work must preserve these passing
tests while adding coverage for the confirmed audit issues.
