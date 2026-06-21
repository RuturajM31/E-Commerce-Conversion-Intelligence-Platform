# test_prediction_ledger.py
# These tests protect the production prediction-ledger contract.
#
# Business reason:
#   A future conversion label must join to exactly the prediction that created
#   it, using stable IDs, trustworthy timestamps, and fixed model provenance.

import csv

import pytest

from src.monitoring.prediction_ledger import (
    PREDICTION_LEDGER_COLUMNS,
    append_prediction_ledger_row,
    build_prediction_ledger_row,
    initialise_prediction_ledger,
    read_prediction_ledger,
)


def valid_prediction_details():
    """Return one valid prediction used across the focused ledger tests."""

    return {
        "visitorid": 12345,
        "scoring_time": "2015-09-18T02:59:47.788000+00:00",
        "purchase_intent_score": 0.991,
        "production_threshold": 0.98,
        "model_name": "Tuned Random Forest",
        "model_generation": "final_champion",
        "model_hash": "model-sha256-example",
        "metadata_hash": "metadata-sha256-example",
        "feature_columns": [
            "total_events",
            "view_count",
            "addtocart_count",
            "unique_items",
            "activity_span_ms",
            "cart_to_view_ratio",
            "events_per_unique_item",
        ],
        "score_source": "final_champion_visitor_scores.csv",
    }


def test_ledger_row_uses_approved_schema_and_outcome_window():
    """Confirm the ledger schema and 14-day maturity timestamp."""

    row = build_prediction_ledger_row(
        **valid_prediction_details()
    )

    assert list(row) == PREDICTION_LEDGER_COLUMNS
    assert row["prediction_id"].startswith("pred_")
    assert row["predicted_conversion"] == 1
    assert row["outcome_window_days"] == 14
    assert (
        row["outcome_window_end"]
        == "2015-10-02T02:59:47.788000+00:00"
    )


def test_same_prediction_creates_same_prediction_id():
    """Confirm repeated processing remains idempotent."""

    details = valid_prediction_details()

    first = build_prediction_ledger_row(**details)
    second = build_prediction_ledger_row(**details)

    assert first["prediction_id"] == second["prediction_id"]


def test_changed_prediction_creates_different_prediction_id():
    """Confirm a changed score is treated as a different scored record."""

    first_details = valid_prediction_details()
    second_details = valid_prediction_details()
    second_details["purchase_intent_score"] = 0.75

    first = build_prediction_ledger_row(**first_details)
    second = build_prediction_ledger_row(**second_details)

    assert first["prediction_id"] != second["prediction_id"]


def test_timestamp_requires_timezone():
    """Reject timestamps that cannot be interpreted safely in production."""

    details = valid_prediction_details()
    details["scoring_time"] = "2015-09-18T02:59:47.788000"

    with pytest.raises(
        ValueError,
        match="timezone",
    ):
        build_prediction_ledger_row(**details)


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("purchase_intent_score", 1.1),
        ("purchase_intent_score", -0.1),
        ("production_threshold", 1.1),
        ("production_threshold", -0.1),
    ],
)
def test_probabilities_must_be_between_zero_and_one(
    field_name,
    invalid_value,
):
    """Reject impossible scores and thresholds."""

    details = valid_prediction_details()
    details[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        build_prediction_ledger_row(**details)


def test_duplicate_feature_names_are_rejected():
    """Protect the ordered feature-schema provenance."""

    details = valid_prediction_details()
    details["feature_columns"] = [
        "total_events",
        "total_events",
    ]

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_prediction_ledger_row(**details)


def test_initialise_prediction_ledger_writes_only_the_header(tmp_path):
    """Create an empty ledger file with the approved column order."""

    ledger_path = tmp_path / "prediction_ledger.csv"

    created_path = initialise_prediction_ledger(
        path=ledger_path
    )

    with created_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.reader(csv_file))

    assert created_path == ledger_path
    assert rows == [PREDICTION_LEDGER_COLUMNS]

def test_append_writes_once_and_skips_exact_duplicate(tmp_path):
    """Ensure retrying the same prediction does not create duplicate rows."""

    ledger_path = tmp_path / "prediction_ledger.csv"
    row = build_prediction_ledger_row(
        **valid_prediction_details()
    )

    first_result = append_prediction_ledger_row(
        row,
        path=ledger_path,
    )
    second_result = append_prediction_ledger_row(
        row,
        path=ledger_path,
    )

    stored_rows = read_prediction_ledger(
        path=ledger_path
    )

    assert first_result == "written"
    assert second_result == "duplicate"
    assert len(stored_rows) == 1
    assert stored_rows[0]["prediction_id"] == row["prediction_id"]


def test_same_prediction_id_with_changed_data_is_rejected(tmp_path):
    """Prevent silent corruption when an existing ID has different values."""

    ledger_path = tmp_path / "prediction_ledger.csv"
    row = build_prediction_ledger_row(
        **valid_prediction_details()
    )

    append_prediction_ledger_row(
        row,
        path=ledger_path,
    )

    conflicting_row = row.copy()
    conflicting_row["model_name"] = "Unexpected Different Model"

    with pytest.raises(
        RuntimeError,
        match="Conflicting ledger row",
    ):
        append_prediction_ledger_row(
            conflicting_row,
            path=ledger_path,
        )


def test_append_rejects_incomplete_schema(tmp_path):
    """Reject rows that do not contain the complete ledger contract."""

    ledger_path = tmp_path / "prediction_ledger.csv"
    incomplete_row = {
        "prediction_id": "pred_incomplete",
    }

    with pytest.raises(
        ValueError,
        match="approved schema",
    ):
        append_prediction_ledger_row(
            incomplete_row,
            path=ledger_path,
        )
