# test_feature_parity.py
# These tests confirm that training and Streamlit use identical feature logic.
#
# Business reason:
#   One visitor must receive the same feature values regardless of whether
#   they are scored offline, manually in Streamlit, or through batch scoring.

import pandas as pd
from pandas.testing import assert_frame_equal

from app.app_utils import (
    build_model_input,
    prepare_batch_model_input,
)
from src.data.feature_engineering import (
    build_single_visitor_features,
    prepare_model_features,
)


def test_single_prediction_matches_shared_feature_engineering():
    """Confirm that Streamlit and the shared feature module return the same row."""

    shared_features = build_single_visitor_features(
        total_events=10,
        view_count=4,
        addtocart_count=2,
        unique_items=3,
        activity_span_ms=5000,
    )

    app_features = build_model_input(
        total_events=10,
        view_count=4,
        addtocart_count=2,
        unique_items=3,
        activity_span_ms=5000,
    )

    assert_frame_equal(
        app_features.reset_index(drop=True),
        shared_features.reset_index(drop=True),
        check_dtype=False,
    )


def test_batch_prediction_matches_shared_feature_engineering():
    """Confirm that batch scoring uses the same formulas and feature order."""

    visitor_data = pd.DataFrame(
        {
            "total_events": [10, 20],
            "view_count": [4, 0],
            "addtocart_count": [2, 1],
            "unique_items": [3, 0],
            "activity_span_ms": [5000, 9000],
        }
    )

    shared_features = prepare_model_features(visitor_data)
    app_features = prepare_batch_model_input(visitor_data)

    assert_frame_equal(
        app_features.reset_index(drop=True),
        shared_features.reset_index(drop=True),
        check_dtype=False,
    )


def test_zero_denominator_logic_matches_training_formula():
    """Confirm that app scoring preserves the approved plus-one formulas."""

    features = build_model_input(
        total_events=5,
        view_count=0,
        addtocart_count=2,
        unique_items=0,
        activity_span_ms=1000,
    )

    assert features.loc[0, "cart_to_view_ratio"] == 2.0
    assert features.loc[0, "events_per_unique_item"] == 5.0
