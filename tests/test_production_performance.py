# test_production_performance.py
# These tests protect truthful delayed-label production reporting.
#
# Business reason:
#   Performance must remain hidden without matured outcomes and must be
#   calculated only from valid prediction-and-label records.

import json

import pytest

from src.monitoring.production_performance import (
    build_production_performance_snapshot,
    calculate_metric_block,
    validate_matured_outcome_row,
    write_production_performance_snapshot,
)


def matured_outcome_row(
    *,
    prediction_id="pred_001",
    score=0.90,
    predicted=1,
    actual=1,
    model_generation="champion",
    model_hash="model-hash-a",
):
    """Return one valid matured production outcome."""

    return {
        "ledger_schema_version": "1.0",
        "prediction_id": prediction_id,
        "visitorid": "123",
        "scoring_time": "2015-09-18T00:00:00+00:00",
        "outcome_window_days": "14",
        "outcome_window_end": "2015-10-02T00:00:00+00:00",
        "purchase_intent_score": str(score),
        "production_threshold": "0.50",
        "predicted_conversion": str(predicted),
        "model_name": "Test Model",
        "model_generation": model_generation,
        "model_hash": model_hash,
        "metadata_hash": "metadata-hash",
        "feature_schema": '["feature_a"]',
        "feature_schema_hash": "feature-schema-hash",
        "score_source": "test_scores.csv",
        "outcome_observed_at": "2015-10-02T00:00:00+00:00",
        "actual_conversion": str(actual),
        "label_source": "test_labels",
        "label_recorded_at": "2015-10-03T00:00:00+00:00",
    }


def representative_matured_rows():
    """Return two outcome classes across two model versions."""

    return [
        matured_outcome_row(
            prediction_id="pred_001",
            score=0.90,
            predicted=1,
            actual=1,
            model_hash="model-hash-a",
        ),
        matured_outcome_row(
            prediction_id="pred_002",
            score=0.70,
            predicted=1,
            actual=0,
            model_hash="model-hash-a",
        ),
        matured_outcome_row(
            prediction_id="pred_003",
            score=0.40,
            predicted=0,
            actual=0,
            model_hash="model-hash-b",
        ),
        matured_outcome_row(
            prediction_id="pred_004",
            score=0.60,
            predicted=1,
            actual=1,
            model_hash="model-hash-b",
        ),
    ]


def test_empty_snapshot_hides_performance_metrics():
    """Do not make performance claims when matured labels are unavailable."""

    snapshot = build_production_performance_snapshot(
        [],
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    assert snapshot["status"] == "labels_unavailable"
    assert snapshot["metrics_available"] is False
    assert snapshot["matured_prediction_volume"] == 0
    assert snapshot["overall_metrics"] is None
    assert snapshot["by_model_version"] == []


def test_metric_block_calculates_expected_values():
    """Confirm classification, calibration, and volume calculations."""

    metrics = calculate_metric_block(
        representative_matured_rows()
    )

    assert metrics["matured_prediction_volume"] == 4
    assert metrics["actual_positive_volume"] == 2
    assert metrics["predicted_positive_volume"] == 3
    assert metrics["actual_conversion_rate"] == pytest.approx(0.50)
    assert metrics["mean_prediction_score"] == pytest.approx(0.65)
    assert metrics["calibration_brier_score"] == pytest.approx(
        0.205
    )
    assert metrics["calibration_mean_score_gap"] == pytest.approx(
        0.15
    )
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(0.8)
    assert metrics["pr_auc"] is not None
    assert metrics["pr_auc_status"] == "available"


def test_single_outcome_class_hides_pr_auc():
    """Avoid publishing misleading PR-AUC from a one-class cohort."""

    rows = [
        matured_outcome_row(
            prediction_id="pred_001",
            score=0.20,
            predicted=0,
            actual=0,
        ),
        matured_outcome_row(
            prediction_id="pred_002",
            score=0.30,
            predicted=0,
            actual=0,
        ),
    ]

    metrics = calculate_metric_block(rows)

    assert metrics["pr_auc"] is None
    assert (
        metrics["pr_auc_status"]
        == "unavailable_single_outcome_class"
    )


def test_snapshot_groups_metrics_by_model_version():
    """Keep separate evidence for different deployed model hashes."""

    snapshot = build_production_performance_snapshot(
        representative_matured_rows(),
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    grouped_metrics = snapshot["by_model_version"]

    assert snapshot["metrics_available"] is True
    assert snapshot["matured_prediction_volume"] == 4
    assert len(grouped_metrics) == 2

    grouped_volumes = sorted(
        group["metrics"]["matured_prediction_volume"]
        for group in grouped_metrics
    )

    assert grouped_volumes == [2, 2]


def test_invalid_probability_is_rejected():
    """Reject impossible production scores before calculating metrics."""

    invalid_row = matured_outcome_row(score=1.20)

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_matured_outcome_row(invalid_row)


def test_missing_matured_column_is_rejected():
    """Reject rows that do not follow the joined outcome contract."""

    invalid_row = matured_outcome_row()
    invalid_row.pop("label_source")

    with pytest.raises(
        ValueError,
        match="approved schema",
    ):
        validate_matured_outcome_row(invalid_row)


def test_snapshot_is_written_as_valid_json(tmp_path):
    """Persist the monitoring snapshot without a partial output file."""

    output_path = tmp_path / "production_performance.json"

    snapshot = build_production_performance_snapshot(
        representative_matured_rows(),
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    created_path = write_production_performance_snapshot(
        snapshot,
        path=output_path,
    )

    saved_snapshot = json.loads(
        created_path.read_text(encoding="utf-8")
    )

    assert created_path == output_path
    assert saved_snapshot["status"] == "available"
    assert saved_snapshot["matured_prediction_volume"] == 4
    assert not output_path.with_suffix(".json.tmp").exists()
