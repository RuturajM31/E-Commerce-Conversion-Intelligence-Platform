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
    append_delayed_label_row,
    build_delayed_label_row,
    initialise_delayed_label_input,
    read_delayed_labels,
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
