# feature_engineering.py
# This file contains the feature formulas shared by training and prediction.
#
# Why this file exists:
#   The training pipeline and Streamlit app previously calculated the same
#   ratio features in different ways. That could produce different model
#   scores for the same visitor.
#
# Input:
#   Visitor-level base behaviour columns.
#
# Output:
#   The exact feature columns and order expected by the current model.

from __future__ import annotations

from typing import Iterable

import pandas as pd


# These columns must already exist before derived features are calculated.
BASE_FEATURE_COLUMNS = [
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
]


# These columns are calculated from the base visitor behaviour columns.
DERIVED_FEATURE_COLUMNS = [
    "cart_to_view_ratio",
    "events_per_unique_item",
]


# The model expects these columns in this exact order.
MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


def validate_base_feature_columns(data: pd.DataFrame) -> None:
    """Validate the base columns required for feature engineering."""

    missing_columns = [
        column
        for column in BASE_FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            "Cannot build model features because required columns are missing: "
            f"{missing_text}"
        )


def convert_base_features_to_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the required base feature columns into numeric values."""

    numeric_data = data.copy()

    for column in BASE_FEATURE_COLUMNS:
        numeric_data[column] = pd.to_numeric(
            numeric_data[column],
            errors="coerce",
        )

    invalid_columns = [
        column
        for column in BASE_FEATURE_COLUMNS
        if numeric_data[column].isna().any()
    ]

    if invalid_columns:
        invalid_text = ", ".join(invalid_columns)

        raise ValueError(
            "Some feature values are missing or are not numeric: "
            f"{invalid_text}"
        )

    return numeric_data


def validate_non_negative_features(
    data: pd.DataFrame,
    columns: Iterable[str] = BASE_FEATURE_COLUMNS,
) -> None:
    """Reject negative visitor behaviour values."""

    negative_columns = [
        column
        for column in columns
        if (data[column] < 0).any()
    ]

    if negative_columns:
        negative_text = ", ".join(negative_columns)

        raise ValueError(
            "Visitor behaviour values cannot be negative: "
            f"{negative_text}"
        )


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the shared ratio features used by training and prediction."""

    validate_base_feature_columns(data)

    feature_data = convert_base_features_to_numeric(data)

    validate_non_negative_features(feature_data)

    # Keep the current training formulas for model compatibility.
    # Adding 1 prevents division by zero and ensures that training,
    # single prediction, and batch prediction use identical logic.

    # Cart-to-view ratio shows how strongly viewing behaviour turns into
    # cart activity. A higher value can indicate stronger buying interest.
    feature_data["cart_to_view_ratio"] = (
        feature_data["addtocart_count"]
        / (feature_data["view_count"] + 1)
    )

    # Events per unique item shows repeat interaction with the same products.
    # A higher value can indicate focused or repeated product interest.
    feature_data["events_per_unique_item"] = (
        feature_data["total_events"]
        / (feature_data["unique_items"] + 1)
    )

    return feature_data


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated model features in the approved column order."""

    feature_data = add_derived_features(data)

    return feature_data[MODEL_FEATURE_COLUMNS]


def build_single_visitor_features(
    total_events: float,
    view_count: float,
    addtocart_count: float,
    unique_items: float,
    activity_span_ms: float,
) -> pd.DataFrame:
    """Create one validated visitor feature row for model scoring."""

    visitor_data = pd.DataFrame(
        [
            {
                "total_events": total_events,
                "view_count": view_count,
                "addtocart_count": addtocart_count,
                "unique_items": unique_items,
                "activity_span_ms": activity_span_ms,
            }
        ]
    )

    return prepare_model_features(visitor_data)
