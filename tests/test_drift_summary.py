"""Tests for compact Evidently drift and alert extraction."""

from __future__ import annotations

import pytest

from src.monitoring.drift_summary import (
    build_drift_monitoring_summary,
    extract_column_psi,
)


def make_report(
    column_values: dict[str, float],
) -> dict:
    """Create a small Evidently-style PSI report dictionary.

    Input:
        Column names and their test PSI values.

    Output:
        A dictionary matching the report structure used by the
        production extraction logic.
    """

    metrics = []

    for column, value in column_values.items():
        metrics.append(
            {
                "config": {
                    "column": column,
                    "method": "psi",
                    "threshold": 0.10,
                },
                "value": value,
            }
        )

    return {
        "metrics": metrics,
        "tests": [],
    }


def test_extract_column_psi_preserves_feature_order() -> None:
    """Output must follow the canonical model feature order."""

    report = make_report(
        {
            "feature_b": 0.02,
            "feature_a": 0.01,
        }
    )

    result = extract_column_psi(
        report=report,
        expected_columns=[
            "feature_a",
            "feature_b",
        ],
        threshold=0.10,
    )

    assert list(result) == [
        "feature_a",
        "feature_b",
    ]

    assert (
        result["feature_a"]["drift_detected"]
        is False
    )


def test_missing_psi_metric_raises_clear_error() -> None:
    """A missing required feature must not be silently ignored."""

    report = make_report(
        {
            "feature_a": 0.01,
        }
    )

    with pytest.raises(
        ValueError,
        match="feature_b",
    ):
        extract_column_psi(
            report=report,
            expected_columns=[
                "feature_a",
                "feature_b",
            ],
            threshold=0.10,
        )


def test_healthy_alert_when_no_drift_is_detected() -> None:
    """Low feature and prediction PSI should be healthy."""

    result = build_drift_monitoring_summary(
        feature_report=make_report(
            {
                "feature_a": 0.01,
                "feature_b": 0.02,
            }
        ),
        prediction_report=make_report(
            {
                "purchase_intent_score": 0.03,
            }
        ),
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        score_column="purchase_intent_score",
        column_threshold=0.10,
        dataset_drift_share=0.50,
    )

    assert (
        result["feature_drift"][
            "drifted_feature_count"
        ]
        == 0
    )

    assert (
        result["prediction_drift"][
            "drift_detected"
        ]
        is False
    )

    assert (
        result["alert"]["level"]
        == "healthy"
    )


def test_warning_alert_for_partial_feature_drift() -> None:
    """Some feature drift below the dataset share should warn."""

    result = build_drift_monitoring_summary(
        feature_report=make_report(
            {
                "feature_a": 0.15,
                "feature_b": 0.02,
                "feature_c": 0.03,
            }
        ),
        prediction_report=make_report(
            {
                "purchase_intent_score": 0.04,
            }
        ),
        feature_columns=[
            "feature_a",
            "feature_b",
            "feature_c",
        ],
        score_column="purchase_intent_score",
        column_threshold=0.10,
        dataset_drift_share=0.50,
    )

    assert (
        result["feature_drift"][
            "drifted_feature_count"
        ]
        == 1
    )

    assert (
        result["feature_drift"][
            "dataset_drift_detected"
        ]
        is False
    )

    assert (
        result["alert"]["level"]
        == "warning"
    )


def test_critical_alert_for_dataset_drift() -> None:
    """Drift in at least half the features should be critical."""

    result = build_drift_monitoring_summary(
        feature_report=make_report(
            {
                "feature_a": 0.15,
                "feature_b": 0.20,
            }
        ),
        prediction_report=make_report(
            {
                "purchase_intent_score": 0.12,
            }
        ),
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        score_column="purchase_intent_score",
        column_threshold=0.10,
        dataset_drift_share=0.50,
    )

    assert (
        result["feature_drift"][
            "dataset_drift_detected"
        ]
        is True
    )

    assert (
        result["prediction_drift"][
            "drift_detected"
        ]
        is True
    )

    assert (
        result["alert"]["level"]
        == "critical"
    )
