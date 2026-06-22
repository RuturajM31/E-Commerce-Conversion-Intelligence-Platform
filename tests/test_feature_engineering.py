# test_feature_engineering.py
# These tests protect the shared feature formulas used by training and prediction.
#
# Main goal:
#   The same visitor behaviour must produce the same model features everywhere.

import pandas as pd
import pytest

from src.data.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    build_single_visitor_features,
    prepare_model_features,
)


def test_single_visitor_uses_training_formulas():
    """Confirm that one visitor uses the approved +1 ratio formulas."""

    features = build_single_visitor_features(
        total_events=10,
        view_count=4,
        addtocart_count=2,
        unique_items=3,
        activity_span_ms=5000,
    )

    assert features.loc[0, "cart_to_view_ratio"] == pytest.approx(0.4)
    assert features.loc[0, "events_per_unique_item"] == pytest.approx(2.5)


def test_zero_denominators_are_handled_safely():
    """Confirm that zero views and zero unique items do not cause errors."""

    features = build_single_visitor_features(
        total_events=5,
        view_count=0,
        addtocart_count=2,
        unique_items=0,
        activity_span_ms=1000,
    )

    assert features.loc[0, "cart_to_view_ratio"] == pytest.approx(2.0)
    assert features.loc[0, "events_per_unique_item"] == pytest.approx(5.0)


def test_batch_features_use_the_approved_column_order():
    """Confirm that batch output matches the model feature schema."""

    batch_data = pd.DataFrame(
        {
            "total_events": [10, 20],
            "view_count": [4, 9],
            "addtocart_count": [2, 1],
            "unique_items": [3, 4],
            "activity_span_ms": [5000, 9000],
        }
    )

    features = prepare_model_features(batch_data)

    assert features.columns.tolist() == MODEL_FEATURE_COLUMNS
    assert features.loc[0, "cart_to_view_ratio"] == pytest.approx(0.4)
    assert features.loc[1, "events_per_unique_item"] == pytest.approx(4.0)


def test_missing_base_column_raises_clear_error():
    """Confirm that missing inputs produce a useful validation message."""

    incomplete_data = pd.DataFrame(
        {
            "total_events": [10],
            "view_count": [4],
            "addtocart_count": [2],
            "activity_span_ms": [5000],
        }
    )

    with pytest.raises(ValueError, match="unique_items"):
        prepare_model_features(incomplete_data)


def test_non_numeric_value_raises_clear_error():
    """Confirm that invalid text values are rejected before scoring."""

    invalid_data = pd.DataFrame(
        {
            "total_events": ["not-a-number"],
            "view_count": [4],
            "addtocart_count": [2],
            "unique_items": [3],
            "activity_span_ms": [5000],
        }
    )

    with pytest.raises(ValueError, match="total_events"):
        prepare_model_features(invalid_data)


def test_negative_value_is_rejected():
    """Confirm that impossible negative visitor counts are rejected."""

    invalid_data = pd.DataFrame(
        {
            "total_events": [10],
            "view_count": [-1],
            "addtocart_count": [2],
            "unique_items": [3],
            "activity_span_ms": [5000],
        }
    )

    with pytest.raises(ValueError, match="view_count"):
        prepare_model_features(invalid_data)
