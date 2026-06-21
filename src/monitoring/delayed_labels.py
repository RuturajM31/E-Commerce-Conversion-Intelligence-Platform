# delayed_labels.py
# This module stores delayed conversion outcomes safely.
#
# Simple purpose:
#   A prediction is created now, but its true conversion result may only be
#   known after the full outcome window has ended.
#
# Input:
#   Prediction ID, final conversion result, observation timestamp,
#   recording timestamp, and label source.
#
# Output:
#   A validated delayed-label CSV that can later be joined one-to-one
#   with the production prediction ledger.

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.config.paths import DELAYED_LABEL_INPUT_PATH
from src.monitoring.prediction_ledger import (
    normalise_utc_timestamp,
    require_text,
)


DELAYED_LABEL_SCHEMA_VERSION = "1.0"


# Fixed column order prevents different label producers using different schemas.
DELAYED_LABEL_COLUMNS = [
    "label_schema_version",
    "prediction_id",
    "outcome_observed_at",
    "converted",
    "label_source",
    "label_recorded_at",
]


def validate_binary_outcome(value: Any) -> int:
    """Return a valid binary conversion outcome: zero or one."""

    # Boolean values are converted explicitly because bool is a subclass of int.
    if isinstance(value, bool):
        return int(value)

    text = str(value).strip()

    if text not in {"0", "1"}:
        raise ValueError(
            "converted must be either 0 or 1."
        )

    return int(text)


def build_delayed_label_row(
    *,
    prediction_id: Any,
    outcome_observed_at: Any,
    converted: Any,
    label_source: Any,
    label_recorded_at: Any,
) -> Dict[str, Any]:
    """Build one validated delayed-label record."""

    prediction_id_text = require_text(
        prediction_id,
        "prediction_id",
    )

    if not prediction_id_text.startswith("pred_"):
        raise ValueError(
            "prediction_id must use the expected 'pred_' prefix."
        )

    observed_time = normalise_utc_timestamp(
        outcome_observed_at
    )
    recorded_time = normalise_utc_timestamp(
        label_recorded_at
    )

    # A label cannot be recorded before its outcome was observed.
    if datetime.fromisoformat(recorded_time) < datetime.fromisoformat(
        observed_time
    ):
        raise ValueError(
            "label_recorded_at cannot be earlier than outcome_observed_at."
        )

    return {
        "label_schema_version": DELAYED_LABEL_SCHEMA_VERSION,
        "prediction_id": prediction_id_text,
        "outcome_observed_at": observed_time,
        "converted": validate_binary_outcome(converted),
        "label_source": require_text(
            label_source,
            "label_source",
        ),
        "label_recorded_at": recorded_time,
    }


def serialise_delayed_label_row(
    row: Dict[str, Any],
) -> Dict[str, str]:
    """Validate and convert a delayed-label row into CSV-safe values."""

    missing_columns = [
        column
        for column in DELAYED_LABEL_COLUMNS
        if column not in row
    ]

    extra_columns = [
        column
        for column in row
        if column not in DELAYED_LABEL_COLUMNS
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            "Delayed-label row does not match the approved schema. "
            f"Missing: {missing_columns}. Extra: {extra_columns}."
        )

    if row["label_schema_version"] != DELAYED_LABEL_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported delayed-label schema version."
        )

    # Rebuild the row so externally supplied dictionaries receive the same
    # timestamp, binary-value, and text validation as normal application input.
    validated_row = build_delayed_label_row(
        prediction_id=row["prediction_id"],
        outcome_observed_at=row["outcome_observed_at"],
        converted=row["converted"],
        label_source=row["label_source"],
        label_recorded_at=row["label_recorded_at"],
    )

    return {
        column: str(validated_row[column])
        for column in DELAYED_LABEL_COLUMNS
    }


def initialise_delayed_label_input(
    path: Path = DELAYED_LABEL_INPUT_PATH,
) -> Path:
    """Create an empty delayed-label CSV with the approved header."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        return path

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=DELAYED_LABEL_COLUMNS,
        )
        writer.writeheader()

    return path


def read_delayed_labels(
    path: Path = DELAYED_LABEL_INPUT_PATH,
) -> list[Dict[str, str]]:
    """Read delayed labels and verify the stored CSV contract."""

    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != DELAYED_LABEL_COLUMNS:
            raise ValueError(
                "Existing delayed-label file has an unexpected schema."
            )

        rows = []

        for source_row in reader:
            rows.append(
                serialise_delayed_label_row(source_row)
            )

    return rows


def append_delayed_label_row(
    row: Dict[str, Any],
    path: Path = DELAYED_LABEL_INPUT_PATH,
) -> str:
    """Append one outcome or safely ignore an exact duplicate."""

    serialised_row = serialise_delayed_label_row(row)

    initialise_delayed_label_input(path)

    existing_rows = read_delayed_labels(path)

    matching_rows = [
        existing_row
        for existing_row in existing_rows
        if existing_row["prediction_id"]
        == serialised_row["prediction_id"]
    ]

    if matching_rows:
        # One prediction can have only one final conversion outcome.
        if any(
            existing_row != serialised_row
            for existing_row in matching_rows
        ):
            raise RuntimeError(
                "Conflicting delayed label found for prediction_id "
                f"{serialised_row['prediction_id']}."
            )

        return "duplicate"

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=DELAYED_LABEL_COLUMNS,
        )
        writer.writerow(serialised_row)

    return "written"
