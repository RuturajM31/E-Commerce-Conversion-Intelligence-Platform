import os
from pathlib import Path

import pandas as pd
import pytest


def test_xgboost_and_logistic_are_compared_when_available():
    """The benchmark should compare XGBoost-style models against Logistic Regression when results exist.

    We do not force XGBoost to beat Logistic Regression every time by default.
    That would be a weak test because model ranking can change by data split, features, tuning, and metric.
    """

    candidate_paths = [
        Path("reports/tables/final_true_champion_comparison.csv"),
        Path("reports/tables/manual_model_comparison.csv"),
        Path("reports/tables/automl_benchmark_results.csv"),
    ]

    available = [path for path in candidate_paths if path.exists()]

    if not available:
        pytest.skip("No model comparison table is available yet.")

    comparison = pd.concat(
        [pd.read_csv(path) for path in available],
        ignore_index=True,
    )

    if "model_name" not in comparison.columns:
        pytest.skip("Comparison table has no model_name column.")

    names = comparison["model_name"].astype(str).str.lower()

    has_logistic = names.str.contains("logistic").any()
    has_xgboost = names.str.contains("xgboost|xgb").any()

    assert has_logistic, "Logistic Regression was not found in model comparison results."

    if not has_xgboost:
        pytest.skip("XGBoost was not available or not included in this run.")

    assert has_xgboost


def test_xgboost_beats_logistic_only_when_strict_mode_enabled():
    """Optional strict check for XGBoost beating Logistic Regression.

    This is not enabled by default because it is not always a valid expectation.
    To force this check:
        REQUIRE_XGB_BEATS_LR=1 pytest tests/test_model_comparison_logic.py -q
    """

    if os.getenv("REQUIRE_XGB_BEATS_LR") != "1":
        pytest.skip("Strict XGBoost > Logistic test skipped by default.")

    path = Path("reports/tables/final_true_champion_comparison.csv")

    if not path.exists():
        pytest.skip("Final champion comparison table is not available.")

    comparison = pd.read_csv(path)

    if "model_name" not in comparison.columns or "business_score" not in comparison.columns:
        pytest.skip("Required comparison columns are missing.")

    names = comparison["model_name"].astype(str).str.lower()

    logistic = comparison[names.str.contains("logistic")]
    xgb = comparison[names.str.contains("xgboost|xgb")]

    if logistic.empty or xgb.empty:
        pytest.skip("Need both Logistic and XGBoost rows for strict comparison.")

    assert xgb["business_score"].max() > logistic["business_score"].max()
