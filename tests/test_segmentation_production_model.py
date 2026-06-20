import numpy as np
import pandas as pd

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.segmentation import create_visitor_segments


class FakeModel:
    def predict_proba(self, features):
        scores = np.array([0.10, 0.98])
        return np.column_stack([1 - scores, scores])


def test_segmentation_uses_production_bundle():
    data = pd.DataFrame(
        {
            "visitorid": [1, 2],
            "converted": [0, 1],
            "total_events": [5, 10],
            "view_count": [3, 7],
            "addtocart_count": [1, 2],
            "unique_items": [2, 4],
            "activity_span_ms": [100, 200],
            "cart_to_view_ratio": [0.25, 0.25],
            "events_per_unique_item": [5 / 3, 2],
        }
    )

    bundle = {
        "model": FakeModel(),
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    scored = create_visitor_segments.score_visitors(data, bundle)

    assert np.allclose(
        scored["purchase_intent_score"],
        [0.10, 0.98],
    )
    assert scored["intent_segment"].tolist() == [
        "Cold Visitor",
        "High Intent",
    ]


def test_segmentation_rejects_missing_production_feature():
    data = pd.DataFrame(
        {
            "visitorid": [1],
            "total_events": [5],
        }
    )

    bundle = {
        "model": FakeModel(),
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    try:
        create_visitor_segments.create_features(data, bundle)
    except ValueError as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("Expected missing feature validation error.")
