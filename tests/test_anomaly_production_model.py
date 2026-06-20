# test_anomaly_production_model.py
# Confirm that anomaly scoring uses the shared production model bundle.

import numpy as np
import pandas as pd

from src.anomaly import build_anomaly_signals
from src.data.feature_engineering import MODEL_FEATURE_COLUMNS


class FakeProbabilityModel:
    """Small model replacement used only for a fast scoring test."""

    def predict_proba(self, features):
        probabilities = np.linspace(0.20, 0.80, len(features))

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


def test_anomaly_scores_use_shared_production_bundle(
    tmp_path,
    monkeypatch,
):
    """Anomaly scoring must use the model and features from the resolver."""

    output_path = tmp_path / "final_champion_visitor_scores.csv"

    monkeypatch.setattr(
        build_anomaly_signals,
        "PRODUCTION_VISITOR_SCORES_PATH",
        output_path,
    )

    bundle = {
        "model": FakeProbabilityModel(),
        "metadata": {},
        "generation": "final_champion",
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    monkeypatch.setattr(
        build_anomaly_signals,
        "load_production_bundle",
        lambda: bundle,
    )

    visitor_features = pd.DataFrame(
        {
            "visitorid": [101, 202],
            "total_events": [10, 20],
            "view_count": [5, 12],
            "addtocart_count": [1, 4],
            "unique_items": [4, 8],
            "activity_span_ms": [1000, 3000],
            "cart_to_view_ratio": [1 / 6, 4 / 13],
            "events_per_unique_item": [2, 20 / 9],
        }
    )

    scores = (
        build_anomaly_signals.create_champion_visitor_scores(
            visitor_features
        )
    )

    assert output_path.exists()
    assert scores["visitorid"].tolist() == [101, 202]
    assert np.allclose(
        scores["purchase_intent_score"],
        [0.20, 0.80],
    )


def test_anomaly_final_score_filename_is_explicit():
    """Anomaly scoring must not reuse old generic score filenames."""

    assert (
        build_anomaly_signals.PRODUCTION_VISITOR_SCORES_PATH.name
        == "final_champion_visitor_scores.csv"
    )
