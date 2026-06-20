# test_forecasting_production_model.py
# Confirm that forecasting uses the same production model, metadata,
# threshold, score file, and manifest as the other workflows.

import json

import numpy as np
import pandas as pd

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.forecasting import build_business_forecasts


class FakeModel:
    """Small model replacement used for fast scoring tests."""

    def predict_proba(self, features):
        scores = np.linspace(0.30, 0.70, len(features))

        return np.column_stack([1 - scores, scores])


def create_test_bundle(tmp_path):
    """Create a complete temporary production artifact pair."""

    model_path = tmp_path / "final_champion_model.joblib"
    metadata_path = tmp_path / "final_champion_metadata.json"
    score_path = tmp_path / "final_champion_visitor_scores.csv"
    manifest_path = tmp_path / "final_champion_score_manifest.json"

    model_path.write_bytes(b"temporary-forecast-model")

    metadata = {
        "final_model_name": "Test Final Champion",
        "best_threshold": 0.97,
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    bundle = {
        "model": FakeModel(),
        "metadata": metadata,
        "generation": "final_champion",
        "model_path": model_path,
        "metadata_path": metadata_path,
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    return bundle, score_path, manifest_path


def test_forecasting_uses_production_bundle(
    tmp_path,
    monkeypatch,
):
    """Forecasting must regenerate scores from the production bundle."""

    feature_path = tmp_path / "visitor_features.csv"
    bundle, score_path, manifest_path = create_test_bundle(tmp_path)

    data = pd.DataFrame(
        {
            "visitorid": [1, 2],
            "total_events": [5, 10],
            "view_count": [3, 7],
            "addtocart_count": [1, 2],
            "unique_items": [2, 4],
            "activity_span_ms": [100, 200],
            "cart_to_view_ratio": [0.25, 0.25],
            "events_per_unique_item": [5 / 3, 2],
        }
    )
    data.to_csv(feature_path, index=False)

    monkeypatch.setattr(
        build_business_forecasts,
        "VISITOR_FEATURES_PATH",
        feature_path,
    )
    monkeypatch.setattr(
        build_business_forecasts,
        "PRODUCTION_VISITOR_SCORES_PATH",
        score_path,
    )
    monkeypatch.setattr(
        build_business_forecasts,
        "PRODUCTION_SCORE_MANIFEST_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        build_business_forecasts,
        "load_production_bundle",
        lambda: bundle,
    )

    scores = build_business_forecasts.ensure_visitor_scores_exist()

    assert score_path.exists()
    assert manifest_path.exists()
    assert scores["visitorid"].tolist() == [1, 2]
    assert np.allclose(
        scores["purchase_intent_score"],
        [0.30, 0.70],
    )


def test_forecasting_uses_production_threshold(monkeypatch):
    """Forecasting must read the threshold from production metadata."""

    bundle = {
        "metadata": {
            "best_threshold": 0.97,
        },
    }

    monkeypatch.setattr(
        build_business_forecasts,
        "load_production_bundle",
        lambda: bundle,
    )

    assert build_business_forecasts.get_champion_threshold() == 0.97
