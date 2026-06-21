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
    PREDICTION_LEDGER_COLUMNS,
    normalise_utc_timestamp,
    require_text,
    serialise_prediction_ledger_row,
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


# The final joined output keeps all prediction provenance and adds the
# trustworthy observed outcome used for production performance reporting.
MATURED_OUTCOME_COLUMNS = PREDICTION_LEDGER_COLUMNS + [
    "outcome_observed_at",
    "actual_conversion",
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



def validate_and_join_matured_outcomes(
    ledger_rows: list[Dict[str, Any]],
    label_rows: list[Dict[str, Any]],
) -> tuple[list[Dict[str, str]], Dict[str, Any]]:
    """Join only known labels whose complete outcome window has matured."""

    report: Dict[str, Any] = {
        "total_ledger_rows": len(ledger_rows),
        "total_label_rows": len(label_rows),
        "accepted_matured_rows": 0,
        "duplicate_label_rows": 0,
        "rejected_invalid_rows": 0,
        "rejected_unknown_prediction_ids": 0,
        "rejected_premature_labels": 0,
        "rejected_conflicting_prediction_ids": 0,
        "rejections": [],
    }

    # Validate the stored ledger schema and require one row per prediction ID.
    ledger_by_id: Dict[str, Dict[str, str]] = {}

    for ledger_row in ledger_rows:
        serialised_ledger = serialise_prediction_ledger_row(
            ledger_row
        )
        prediction_id = serialised_ledger["prediction_id"]

        if prediction_id in ledger_by_id:
            raise ValueError(
                "Prediction ledger contains duplicate prediction_id "
                f"{prediction_id}."
            )

        ledger_by_id[prediction_id] = serialised_ledger

    # Validate labels first, then group them by prediction ID.
    labels_by_id: Dict[str, list[Dict[str, str]]] = {}

    for label_row in label_rows:
        try:
            serialised_label = serialise_delayed_label_row(
                label_row
            )
        except (TypeError, ValueError) as error:
            report["rejected_invalid_rows"] += 1
            report["rejections"].append(
                {
                    "prediction_id": str(
                        label_row.get(
                            "prediction_id",
                            "",
                        )
                    ),
                    "reason": "invalid_label",
                    "detail": str(error),
                }
            )
            continue

        prediction_id = serialised_label["prediction_id"]

        labels_by_id.setdefault(
            prediction_id,
            [],
        ).append(serialised_label)

    matured_rows: list[Dict[str, str]] = []

    for prediction_id, grouped_labels in labels_by_id.items():
        # Convert each label row into canonical JSON so exact duplicates and
        # true conflicts can be distinguished reliably.
        canonical_labels = {
            str(sorted(label.items()))
            for label in grouped_labels
        }

        if len(canonical_labels) > 1:
            report[
                "rejected_conflicting_prediction_ids"
            ] += 1
            report["rejections"].append(
                {
                    "prediction_id": prediction_id,
                    "reason": "conflicting_labels",
                    "detail": (
                        "Multiple different labels exist for one prediction."
                    ),
                }
            )
            continue

        # Exact retries are harmless. Keep one row and report the extras.
        report["duplicate_label_rows"] += (
            len(grouped_labels) - 1
        )
        label = grouped_labels[0]

        ledger_row = ledger_by_id.get(prediction_id)

        if ledger_row is None:
            report[
                "rejected_unknown_prediction_ids"
            ] += 1
            report["rejections"].append(
                {
                    "prediction_id": prediction_id,
                    "reason": "unknown_prediction_id",
                    "detail": (
                        "No matching prediction exists in the ledger."
                    ),
                }
            )
            continue

        outcome_window_end = datetime.fromisoformat(
            ledger_row["outcome_window_end"]
        )
        outcome_observed_at = datetime.fromisoformat(
            label["outcome_observed_at"]
        )

        # The final outcome cannot be trusted until the full configured
        # conversion window has finished.
        if outcome_observed_at < outcome_window_end:
            report["rejected_premature_labels"] += 1
            report["rejections"].append(
                {
                    "prediction_id": prediction_id,
                    "reason": "premature_label",
                    "detail": (
                        "Outcome was observed before the prediction "
                        "window ended."
                    ),
                }
            )
            continue

        joined_row = {
            **ledger_row,
            "outcome_observed_at": label[
                "outcome_observed_at"
            ],
            "actual_conversion": label["converted"],
            "label_source": label["label_source"],
            "label_recorded_at": label[
                "label_recorded_at"
            ],
        }

        matured_rows.append(
            {
                column: joined_row[column]
                for column in MATURED_OUTCOME_COLUMNS
            }
        )

    report["accepted_matured_rows"] = len(
        matured_rows
    )

    return matured_rows, report


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
