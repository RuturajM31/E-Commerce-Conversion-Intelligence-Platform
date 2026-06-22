# test_delayed_label_evaluation_runner.py
# These tests protect the complete delayed-label monitoring workflow.
#
# Business reason:
#   The operational runner must rebuild trustworthy evaluation outputs from
#   the current ledger and delayed-label files without duplicate accumulation.
#
# Protected behavior:
#   - Matured labels produce joined outcomes and performance metrics.
#   - Missing labels produce no false performance claims.
#   - Late-arriving labels appear safely after rerunning the workflow.
#   - Unknown and premature labels remain visible in rejection evidence.

import csv
import json

from src.monitoring.delayed_labels import (
    append_delayed_label_row,
    build_delayed_label_row,
)
from src.monitoring.prediction_ledger import (
    append_prediction_ledger_row,
    build_prediction_ledger_row,
)
from src.monitoring.run_delayed_label_evaluation import (
    run_delayed_label_evaluation,
)


def valid_prediction():
    """Return one production prediction with a 14-day outcome window."""

    return build_prediction_ledger_row(
        visitorid=12345,
        scoring_time="2015-09-18T02:59:47.788000+00:00",
        purchase_intent_score=0.991,
        production_threshold=0.98,
        model_name="Tuned Random Forest",
        model_generation="final_champion",
        model_hash="model-sha256-example",
        metadata_hash="metadata-sha256-example",
        feature_columns=[
            "total_events",
            "view_count",
            "addtocart_count",
            "unique_items",
            "activity_span_ms",
            "cart_to_view_ratio",
            "events_per_unique_item",
        ],
        score_source="final_champion_visitor_scores.csv",
    )


def valid_matured_label(prediction):
    """Return one conversion outcome observed after full maturity."""

    return build_delayed_label_row(
        prediction_id=prediction["prediction_id"],
        outcome_observed_at=prediction["outcome_window_end"],
        converted=1,
        label_source="retailrocket_future_events",
        label_recorded_at="2015-10-03T09:00:00+00:00",
    )


def workflow_paths(tmp_path):
    """Return isolated paths for one complete evaluation run."""

    return {
        "ledger_path": tmp_path / "prediction_ledger.csv",
        "label_path": tmp_path / "delayed_labels.csv",
        "matured_outcomes_path": tmp_path / "matured_outcomes.csv",
        "validation_report_path": tmp_path / "validation_report.json",
        "performance_snapshot_path": tmp_path / "performance.json",
    }


def read_json(path):
    """Read one generated JSON monitoring artifact."""

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_csv_rows(path):
    """Read generated CSV rows as dictionaries."""

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def test_runner_creates_all_outputs_for_matured_label(tmp_path):
    """Accept a matured label and create all monitoring artifacts."""

    paths = workflow_paths(tmp_path)
    prediction = valid_prediction()
    label = valid_matured_label(prediction)

    append_prediction_ledger_row(
        prediction,
        path=paths["ledger_path"],
    )
    append_delayed_label_row(
        label,
        path=paths["label_path"],
    )

    summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    matured_rows = read_csv_rows(
        paths["matured_outcomes_path"]
    )
    validation = read_json(
        paths["validation_report_path"]
    )
    performance = read_json(
        paths["performance_snapshot_path"]
    )

    assert summary["ledger_rows_read"] == 1
    assert summary["label_rows_read"] == 1
    assert summary["matured_rows_written"] == 1
    assert summary["metrics_available"] is True

    assert len(matured_rows) == 1
    assert (
        matured_rows[0]["prediction_id"]
        == prediction["prediction_id"]
    )
    assert matured_rows[0]["actual_conversion"] == "1"

    assert validation["accepted_matured_rows"] == 1
    assert validation["rejections"] == []

    assert performance["status"] == "available"
    assert performance["matured_prediction_volume"] == 1


def test_runner_hides_metrics_when_labels_are_missing(tmp_path):
    """Write honest empty outputs when no future outcomes exist."""

    paths = workflow_paths(tmp_path)
    prediction = valid_prediction()

    append_prediction_ledger_row(
        prediction,
        path=paths["ledger_path"],
    )

    summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    matured_rows = read_csv_rows(
        paths["matured_outcomes_path"]
    )
    performance = read_json(
        paths["performance_snapshot_path"]
    )

    assert summary["ledger_rows_read"] == 1
    assert summary["label_rows_read"] == 0
    assert summary["matured_rows_written"] == 0
    assert summary["metrics_available"] is False

    assert matured_rows == []
    assert performance["status"] == "labels_unavailable"
    assert performance["overall_metrics"] is None


def test_rerun_includes_late_arriving_label_without_duplicates(
    tmp_path,
):
    """Rebuild outputs safely when a valid label arrives later."""

    paths = workflow_paths(tmp_path)
    prediction = valid_prediction()

    append_prediction_ledger_row(
        prediction,
        path=paths["ledger_path"],
    )

    first_summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    assert first_summary["matured_rows_written"] == 0

    append_delayed_label_row(
        valid_matured_label(prediction),
        path=paths["label_path"],
    )

    second_summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-22T12:00:00+00:00",
    )
    third_summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-23T12:00:00+00:00",
    )

    matured_rows = read_csv_rows(
        paths["matured_outcomes_path"]
    )

    assert second_summary["matured_rows_written"] == 1
    assert third_summary["matured_rows_written"] == 1
    assert len(matured_rows) == 1


def test_runner_records_unknown_and_premature_rejections(tmp_path):
    """Keep invalid outcomes out of metrics while preserving evidence."""

    paths = workflow_paths(tmp_path)
    prediction = valid_prediction()

    append_prediction_ledger_row(
        prediction,
        path=paths["ledger_path"],
    )

    premature_label = build_delayed_label_row(
        prediction_id=prediction["prediction_id"],
        outcome_observed_at="2015-09-25T09:00:00+00:00",
        converted=1,
        label_source="premature_test",
        label_recorded_at="2015-09-25T10:00:00+00:00",
    )

    unknown_label = build_delayed_label_row(
        prediction_id="pred_unknown123",
        outcome_observed_at="2015-10-03T09:00:00+00:00",
        converted=0,
        label_source="unknown_test",
        label_recorded_at="2015-10-03T10:00:00+00:00",
    )

    append_delayed_label_row(
        premature_label,
        path=paths["label_path"],
    )
    append_delayed_label_row(
        unknown_label,
        path=paths["label_path"],
    )

    summary = run_delayed_label_evaluation(
        **paths,
        created_at_utc="2026-06-21T12:00:00+00:00",
    )

    validation = read_json(
        paths["validation_report_path"]
    )
    performance = read_json(
        paths["performance_snapshot_path"]
    )

    rejection_reasons = {
        rejection["reason"]
        for rejection in validation["rejections"]
    }

    assert summary["matured_rows_written"] == 0
    assert summary["metrics_available"] is False

    assert validation["rejected_premature_labels"] == 1
    assert (
        validation["rejected_unknown_prediction_ids"]
        == 1
    )
    assert rejection_reasons == {
        "premature_label",
        "unknown_prediction_id",
    }

    assert performance["status"] == "labels_unavailable"
