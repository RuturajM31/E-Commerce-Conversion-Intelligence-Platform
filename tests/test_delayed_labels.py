# test_delayed_labels.py
# These tests protect the delayed conversion-label contract.
#
# Business reason:
#   Production performance must use one trustworthy final outcome per
#   prediction, with valid timestamps and no silent duplicate conflicts.

import csv

import pytest

from src.monitoring.delayed_labels import (
    DELAYED_LABEL_COLUMNS,
    MATURED_OUTCOME_COLUMNS,
    append_delayed_label_row,
    build_delayed_label_row,
    initialise_delayed_label_input,
    read_delayed_labels,
    validate_and_join_matured_outcomes,
)
from src.monitoring.prediction_ledger import (
    build_prediction_ledger_row,
)


def valid_label_details():
    """Return one valid delayed outcome used by multiple tests."""

    return {
        "prediction_id": "pred_85a0c6332a18b5093cd1d137",
        "outcome_observed_at": "2015-10-02T02:59:47.788000+00:00",
        "converted": 1,
        "label_source": "retailrocket_future_events",
        "label_recorded_at": "2015-10-03T09:00:00+00:00",
    }


def test_build_label_uses_approved_schema():
    """Confirm the label row has the fixed columns and canonical values."""

    row = build_delayed_label_row(
        **valid_label_details()
    )

    assert list(row) == DELAYED_LABEL_COLUMNS
    assert row["label_schema_version"] == "1.0"
    assert row["converted"] == 1
    assert row["prediction_id"].startswith("pred_")


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 2, "yes", "", None],
)
def test_converted_must_be_binary(invalid_value):
    """Reject outcomes other than zero or one."""

    details = valid_label_details()
    details["converted"] = invalid_value

    with pytest.raises(
        ValueError,
        match="either 0 or 1",
    ):
        build_delayed_label_row(**details)


def test_prediction_id_requires_expected_prefix():
    """Reject labels that cannot identify a production prediction."""

    details = valid_label_details()
    details["prediction_id"] = "unknown_prediction"

    with pytest.raises(
        ValueError,
        match="pred_",
    ):
        build_delayed_label_row(**details)


def test_recorded_time_cannot_precede_observed_time():
    """Reject labels recorded before the outcome was actually observed."""

    details = valid_label_details()
    details["label_recorded_at"] = (
        "2015-10-01T09:00:00+00:00"
    )

    with pytest.raises(
        ValueError,
        match="cannot be earlier",
    ):
        build_delayed_label_row(**details)


def test_initialise_label_file_writes_only_header(tmp_path):
    """Create an empty controlled input file with the correct schema."""

    label_path = tmp_path / "delayed_labels.csv"

    created_path = initialise_delayed_label_input(
        path=label_path
    )

    with created_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [DELAYED_LABEL_COLUMNS]


def test_append_writes_once_and_skips_exact_duplicate(tmp_path):
    """Ensure repeated ingestion does not duplicate the same outcome."""

    label_path = tmp_path / "delayed_labels.csv"
    row = build_delayed_label_row(
        **valid_label_details()
    )

    first_result = append_delayed_label_row(
        row,
        path=label_path,
    )
    second_result = append_delayed_label_row(
        row,
        path=label_path,
    )

    stored_rows = read_delayed_labels(
        path=label_path
    )

    assert first_result == "written"
    assert second_result == "duplicate"
    assert len(stored_rows) == 1


def test_conflicting_outcome_for_same_prediction_is_rejected(
    tmp_path,
):
    """Prevent one prediction receiving two different final outcomes."""

    label_path = tmp_path / "delayed_labels.csv"

    first_row = build_delayed_label_row(
        **valid_label_details()
    )
    append_delayed_label_row(
        first_row,
        path=label_path,
    )

    conflicting_details = valid_label_details()
    conflicting_details["converted"] = 0

    conflicting_row = build_delayed_label_row(
        **conflicting_details
    )

    with pytest.raises(
        RuntimeError,
        match="Conflicting delayed label",
    ):
        append_delayed_label_row(
            conflicting_row,
            path=label_path,
        )


def test_read_rejects_unexpected_csv_header(tmp_path):
    """Reject files that do not follow the controlled label schema."""

    label_path = tmp_path / "delayed_labels.csv"
    label_path.write_text(
        "prediction_id,converted\npred_example,1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unexpected schema",
    ):
        read_delayed_labels(
            path=label_path
        )

def valid_prediction_ledger_row():
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


def valid_matured_label(ledger_row, converted=1):
    """Return one outcome observed after the full prediction window."""

    return build_delayed_label_row(
        prediction_id=ledger_row["prediction_id"],
        outcome_observed_at=ledger_row["outcome_window_end"],
        converted=converted,
        label_source="retailrocket_future_events",
        label_recorded_at="2015-10-03T09:00:00+00:00",
    )


def test_matured_label_joins_to_exactly_one_prediction():
    """Accept a known label after the complete outcome window."""

    ledger_row = valid_prediction_ledger_row()
    label_row = valid_matured_label(ledger_row)

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[ledger_row],
        label_rows=[label_row],
    )

    assert len(matured_rows) == 1
    assert list(matured_rows[0]) == MATURED_OUTCOME_COLUMNS
    assert matured_rows[0]["prediction_id"] == ledger_row["prediction_id"]
    assert matured_rows[0]["actual_conversion"] == "1"
    assert report["accepted_matured_rows"] == 1


def test_premature_label_is_rejected():
    """Do not evaluate a prediction before its 14-day window ends."""

    ledger_row = valid_prediction_ledger_row()

    premature_label = build_delayed_label_row(
        prediction_id=ledger_row["prediction_id"],
        outcome_observed_at="2015-09-25T09:00:00+00:00",
        converted=1,
        label_source="premature_test",
        label_recorded_at="2015-09-25T10:00:00+00:00",
    )

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[ledger_row],
        label_rows=[premature_label],
    )

    assert matured_rows == []
    assert report["rejected_premature_labels"] == 1


def test_unknown_prediction_id_is_rejected():
    """Reject labels that cannot be traced to a ledger prediction."""

    unknown_label = build_delayed_label_row(
        prediction_id="pred_unknown123",
        outcome_observed_at="2015-10-03T09:00:00+00:00",
        converted=0,
        label_source="unknown_test",
        label_recorded_at="2015-10-03T10:00:00+00:00",
    )

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[],
        label_rows=[unknown_label],
    )

    assert matured_rows == []
    assert report["rejected_unknown_prediction_ids"] == 1


def test_exact_duplicate_labels_are_counted_once():
    """Accept one outcome while recording harmless ingestion retries."""

    ledger_row = valid_prediction_ledger_row()
    label_row = valid_matured_label(ledger_row)

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[ledger_row],
        label_rows=[label_row, label_row.copy()],
    )

    assert len(matured_rows) == 1
    assert report["duplicate_label_rows"] == 1
    assert report["accepted_matured_rows"] == 1


def test_conflicting_labels_for_one_prediction_are_rejected():
    """Reject one prediction receiving both converted and not converted."""

    ledger_row = valid_prediction_ledger_row()

    positive_label = valid_matured_label(
        ledger_row,
        converted=1,
    )
    negative_label = valid_matured_label(
        ledger_row,
        converted=0,
    )

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[ledger_row],
        label_rows=[positive_label, negative_label],
    )

    assert matured_rows == []
    assert report["rejected_conflicting_prediction_ids"] == 1


def test_duplicate_prediction_ids_in_ledger_are_rejected():
    """Protect the one-row-per-prediction ledger grain."""

    ledger_row = valid_prediction_ledger_row()

    with pytest.raises(
        ValueError,
        match="duplicate prediction_id",
    ):
        validate_and_join_matured_outcomes(
            ledger_rows=[ledger_row, ledger_row.copy()],
            label_rows=[],
        )


def test_invalid_label_row_is_reported_not_joined():
    """Keep malformed label input out of production evaluation."""

    ledger_row = valid_prediction_ledger_row()

    invalid_label = {
        "prediction_id": ledger_row["prediction_id"],
    }

    matured_rows, report = validate_and_join_matured_outcomes(
        ledger_rows=[ledger_row],
        label_rows=[invalid_label],
    )

    assert matured_rows == []
    assert report["rejected_invalid_rows"] == 1
    assert report["rejections"][0]["reason"] == "invalid_label"

