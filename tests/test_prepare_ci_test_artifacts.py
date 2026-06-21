"""Tests for the CI-only artifact preparation script."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from scripts.prepare_ci_test_artifacts import (
    prepare_ci_test_artifacts,
)
from src.data.feature_engineering import (
    BASE_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    prepare_model_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "visitor_features_sample.csv"


def test_ci_artifact_bundle_is_complete_and_test_only(
    tmp_path: Path,
) -> None:
    """The builder must create one loadable final pair and required tables."""

    result = prepare_ci_test_artifacts(
        project_root=tmp_path,
        sample_path=SAMPLE_PATH,
    )

    assert result["artifact_scope"] == "ci_test_only"
    assert result["output_count"] == 7

    model_path = tmp_path / "models" / "trained" / "final_champion_model.joblib"
    metadata_path = tmp_path / "models" / "metadata" / "final_champion_metadata.json"

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["artifact_scope"] == "ci_test_only"
    assert metadata["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert metadata["best_threshold"] == 0.50

    sample = pd.read_csv(SAMPLE_PATH).head(3)
    features = prepare_model_features(sample[BASE_FEATURE_COLUMNS])
    probabilities = model.predict_proba(features)[:, 1]

    assert len(probabilities) == 3
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_ci_report_tables_have_required_business_schemas(
    tmp_path: Path,
) -> None:
    """Forecast and model-selection fixtures must satisfy real test schemas."""

    prepare_ci_test_artifacts(
        project_root=tmp_path,
        sample_path=SAMPLE_PATH,
    )

    table_dir = tmp_path / "reports" / "tables"

    future = pd.read_csv(table_dir / "business_forecast_future.csv")
    forecast_comparison = pd.read_csv(table_dir / "business_forecast_comparison.csv")
    model_comparison = pd.read_csv(table_dir / "final_true_champion_comparison.csv")

    assert {
        "date",
        "target_name",
        "predicted_value",
    }.issubset(future.columns)

    winner_count = forecast_comparison.groupby("target_name")["is_best_model"].sum()
    assert (winner_count == 1).all()

    assert {
        "model_name",
        "business_score",
        "is_selected",
    }.issubset(model_comparison.columns)

    assert model_comparison["is_selected"].sum() == 1

    names = model_comparison["model_name"].astype(str).str.lower()
    assert names.str.contains("logistic").any()
    assert names.str.contains("xgboost|xgb", regex=True).any()
