import numpy as np
import pandas as pd

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.forecasting import build_business_forecasts


class FakeModel:
    def predict_proba(self, features):
        scores = np.linspace(0.30, 0.70, len(features))
        return np.column_stack([1 - scores, scores])


def test_forecasting_uses_production_bundle(tmp_path, monkeypatch):
    feature_path = tmp_path / "visitor_features.csv"
    score_path = tmp_path / "final_champion_visitor_scores.csv"

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

    bundle = {
        "model": FakeModel(),
        "metadata": {"best_threshold": 0.97},
        "generation": "final_champion",
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

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
        "load_production_bundle",
        lambda: bundle,
    )

    scores = build_business_forecasts.ensure_visitor_scores_exist()

    assert score_path.exists()
    assert scores["visitorid"].tolist() == [1, 2]
    assert np.allclose(
        scores["purchase_intent_score"],
        [0.30, 0.70],
    )


def test_forecasting_uses_production_threshold(monkeypatch):
    bundle = {
        "metadata": {"best_threshold": 0.97},
    }

    monkeypatch.setattr(
        build_business_forecasts,
        "load_production_bundle",
        lambda: bundle,
    )

    assert build_business_forecasts.get_champion_threshold() == 0.97
