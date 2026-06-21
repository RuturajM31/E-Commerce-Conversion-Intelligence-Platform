# prediction_ledger.py
# This module defines the production prediction ledger.
#
# Simple purpose:
#   Save enough information about every prediction so that a future
#   conversion outcome can be joined back safely and evaluated.
#
# Input:
#   Visitor ID, score, threshold, scoring time, model provenance,
#   feature schema, and score source.
#
# Output:
#   One validated ledger row with a stable prediction ID and
#   a clearly defined outcome-maturity timestamp.

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from src.config.paths import PREDICTION_LEDGER_PATH


LEDGER_SCHEMA_VERSION = "1.0"
DEFAULT_OUTCOME_WINDOW_DAYS = 14


# The CSV column order is fixed so every producer writes the same schema.
PREDICTION_LEDGER_COLUMNS = [
    "ledger_schema_version",
    "prediction_id",
    "visitorid",
    "scoring_time",
    "outcome_window_days",
    "outcome_window_end",
    "purchase_intent_score",
    "production_threshold",
    "predicted_conversion",
    "model_name",
    "model_generation",
    "model_hash",
    "metadata_hash",
    "feature_schema",
    "feature_schema_hash",
    "score_source",
]


def normalise_utc_timestamp(value: Any) -> str:
    """Return one timezone-aware timestamp in canonical UTC format."""

    text = str(value).strip().replace("Z", "+00:00")

    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid ISO timestamp: {value}"
        ) from error

    if timestamp.tzinfo is None:
        raise ValueError(
            "Timestamp must include a timezone offset."
        )

    timestamp_utc = timestamp.astimezone(timezone.utc)

    return timestamp_utc.isoformat(timespec="microseconds")


def validate_probability(value: Any, field_name: str) -> float:
    """Convert and validate a probability between zero and one."""

    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be numeric."
        ) from error

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return probability


def canonical_feature_schema(
    feature_columns: Sequence[str],
) -> str:
    """Store model feature names and order as compact JSON."""

    columns = [
        str(column).strip()
        for column in feature_columns
    ]

    if not columns or any(not column for column in columns):
        raise ValueError(
            "feature_columns must contain valid feature names."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "feature_columns cannot contain duplicates."
        )

    return json.dumps(
        columns,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    """Return a SHA-256 hash for one canonical text value."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def require_text(value: Any, field_name: str) -> str:
    """Reject missing provenance values."""

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return text


def build_prediction_id(
    *,
    visitorid: Any,
    scoring_time: str,
    purchase_intent_score: float,
    production_threshold: float,
    model_hash: str,
    metadata_hash: str,
    feature_schema_hash: str,
    score_source: str,
) -> str:
    """Create a reproducible ID for one exact scored record."""

    identity_payload = {
        "visitorid": str(visitorid).strip(),
        "scoring_time": scoring_time,
        "purchase_intent_score": format(
            purchase_intent_score,
            ".12g",
        ),
        "production_threshold": format(
            production_threshold,
            ".12g",
        ),
        "model_hash": model_hash,
        "metadata_hash": metadata_hash,
        "feature_schema_hash": feature_schema_hash,
        "score_source": score_source,
    }

    canonical_payload = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    # The prefix makes the identifier easy to recognise in logs and reports.
    return "pred_" + sha256_text(canonical_payload)[:24]


def build_prediction_ledger_row(
    *,
    visitorid: Any,
    scoring_time: Any,
    purchase_intent_score: Any,
    production_threshold: Any,
    model_name: Any,
    model_generation: Any,
    model_hash: Any,
    metadata_hash: Any,
    feature_columns: Sequence[str],
    score_source: Any,
    outcome_window_days: int = DEFAULT_OUTCOME_WINDOW_DAYS,
) -> Dict[str, Any]:
    """Build one validated prediction-ledger record."""

    visitor_text = require_text(
        visitorid,
        "visitorid",
    )

    scoring_time_utc = normalise_utc_timestamp(
        scoring_time
    )

    score = validate_probability(
        purchase_intent_score,
        "purchase_intent_score",
    )

    threshold = validate_probability(
        production_threshold,
        "production_threshold",
    )

    if outcome_window_days <= 0:
        raise ValueError(
            "outcome_window_days must be greater than zero."
        )

    model_name_text = require_text(
        model_name,
        "model_name",
    )
    generation_text = require_text(
        model_generation,
        "model_generation",
    )
    model_hash_text = require_text(
        model_hash,
        "model_hash",
    )
    metadata_hash_text = require_text(
        metadata_hash,
        "metadata_hash",
    )
    score_source_text = require_text(
        score_source,
        "score_source",
    )

    feature_schema = canonical_feature_schema(
        feature_columns
    )
    feature_schema_hash = sha256_text(
        feature_schema
    )

    scoring_datetime = datetime.fromisoformat(
        scoring_time_utc
    )
    outcome_window_end = (
        scoring_datetime
        + timedelta(days=outcome_window_days)
    ).isoformat(timespec="microseconds")

    prediction_id = build_prediction_id(
        visitorid=visitor_text,
        scoring_time=scoring_time_utc,
        purchase_intent_score=score,
        production_threshold=threshold,
        model_hash=model_hash_text,
        metadata_hash=metadata_hash_text,
        feature_schema_hash=feature_schema_hash,
        score_source=score_source_text,
    )

    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "prediction_id": prediction_id,
        "visitorid": visitor_text,
        "scoring_time": scoring_time_utc,
        "outcome_window_days": outcome_window_days,
        "outcome_window_end": outcome_window_end,
        "purchase_intent_score": score,
        "production_threshold": threshold,
        "predicted_conversion": int(score >= threshold),
        "model_name": model_name_text,
        "model_generation": generation_text,
        "model_hash": model_hash_text,
        "metadata_hash": metadata_hash_text,
        "feature_schema": feature_schema,
        "feature_schema_hash": feature_schema_hash,
        "score_source": score_source_text,
    }


def initialise_prediction_ledger(
    path: Path = PREDICTION_LEDGER_PATH,
) -> Path:
    """Create an empty ledger CSV with the approved schema if needed."""

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
            fieldnames=PREDICTION_LEDGER_COLUMNS,
        )
        writer.writeheader()

    return path
