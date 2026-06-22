# test_anomaly_production_model.py
# Confirm that anomaly scoring uses the shared production model bundle
# and writes traceable final-champion scores.

import json

import numpy as np
import pandas as pd

from src.anomaly import build_anomaly_signals
from src.data.feature_engineering import MODEL_FEATURE_COLUMNS


class FakeProbabilityModel:
    """Small model replacement used for fast scoring tests."""

    def predict_proba(self, features):
        probabilities = np.linspace(0.20, 0.80, len(features))

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


def create_test_bundle(tmp_path):
    """Create a complete temporary model and metadata pair."""

    model_path = tmp_path / "final_champion_model.joblib"
    metadata_path = tmp_path / "final_champion_metadata.json"
    score_path = tmp_path / "final_champion_visitor_scores.csv"
    manifest_path = tmp_path / "final_champion_score_manifest.json"

    # The score exporter only needs a stable file for hash validation.
    model_path.write_bytes(b"temporary-anomaly-model")

    metadata = {
        "final_model_name": "Test Final Champion",
        "best_threshold": 0.50,
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    bundle = {
        "model": FakeProbabilityModel(),
        "metadata": metadata,
        "generation": "final_champion",
        "model_path": model_path,
        "metadata_path": metadata_path,
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    return bundle, score_path, manifest_path


def test_anomaly_scores_use_shared_production_bundle(
    tmp_path,
    monkeypatch,
):
    """Anomaly scoring must use the validated production bundle."""

    bundle, score_path, manifest_path = create_test_bundle(tmp_path)

    monkeypatch.setattr(
        build_anomaly_signals,
        "PRODUCTION_VISITOR_SCORES_PATH",
        score_path,
    )
    monkeypatch.setattr(
        build_anomaly_signals,
        "PRODUCTION_SCORE_MANIFEST_PATH",
        manifest_path,
    )
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

    assert score_path.exists()
    assert manifest_path.exists()
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
