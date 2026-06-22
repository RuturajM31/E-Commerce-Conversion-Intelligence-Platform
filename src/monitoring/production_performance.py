# production_performance.py
# This module calculates production model performance only from predictions
# whose future conversion outcome windows have fully matured.
#
# Simple purpose:
#   No labels = no performance claims.
#   Matured labels = calculate truthful metrics and group them by model version.
#
# Input:
#   Rows created by validate_and_join_matured_outcomes().
#
# Output:
#   A machine-readable JSON snapshot for monitoring and later dashboards.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.paths import PRODUCTION_PERFORMANCE_PATH
from src.monitoring.delayed_labels import MATURED_OUTCOME_COLUMNS


SNAPSHOT_SCHEMA_VERSION = "1.0"


def validate_matured_outcome_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate one matured prediction-and-outcome record."""

    missing_columns = [
        column
        for column in MATURED_OUTCOME_COLUMNS
        if column not in row
    ]

    extra_columns = [
        column
        for column in row
        if column not in MATURED_OUTCOME_COLUMNS
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            "Matured outcome row does not match the approved schema. "
            f"Missing: {missing_columns}. Extra: {extra_columns}."
        )

    score = float(row["purchase_intent_score"])
    threshold = float(row["production_threshold"])
    predicted = int(row["predicted_conversion"])
    actual = int(row["actual_conversion"])

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            "purchase_intent_score must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "production_threshold must be between 0 and 1."
        )

    if predicted not in {0, 1}:
        raise ValueError(
            "predicted_conversion must be either 0 or 1."
        )

    if actual not in {0, 1}:
        raise ValueError(
            "actual_conversion must be either 0 or 1."
        )

    validated_row = dict(row)
    validated_row["purchase_intent_score"] = score
    validated_row["production_threshold"] = threshold
    validated_row["predicted_conversion"] = predicted
    validated_row["actual_conversion"] = actual

    return validated_row


def calculate_metric_block(
    matured_rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate classification and calibration metrics for one cohort."""

    if not matured_rows:
        raise ValueError(
            "At least one matured outcome row is required."
        )

    validated_rows = [
        validate_matured_outcome_row(row)
        for row in matured_rows
    ]

    actual_values = [
        row["actual_conversion"]
        for row in validated_rows
    ]
    predicted_values = [
        row["predicted_conversion"]
        for row in validated_rows
    ]
    scores = [
        row["purchase_intent_score"]
        for row in validated_rows
    ]

    volume = len(validated_rows)
    actual_positive_volume = sum(actual_values)
    predicted_positive_volume = sum(predicted_values)

    conversion_rate = actual_positive_volume / volume
    mean_prediction_score = sum(scores) / volume

    # Brier score measures probability calibration.
    # Lower is better, with zero representing perfect probability predictions.
    calibration_brier_score = brier_score_loss(
        actual_values,
        scores,
    )

    # PR-AUC requires both outcome classes to be present.
    # Returning None is more truthful than manufacturing a misleading value.
    if len(set(actual_values)) < 2:
        pr_auc = None
        pr_auc_status = "unavailable_single_outcome_class"
    else:
        pr_auc = float(
            average_precision_score(
                actual_values,
                scores,
            )
        )
        pr_auc_status = "available"

    return {
        "matured_prediction_volume": volume,
        "actual_positive_volume": actual_positive_volume,
        "predicted_positive_volume": predicted_positive_volume,
        "actual_conversion_rate": float(conversion_rate),
        "mean_prediction_score": float(mean_prediction_score),
        "calibration_brier_score": float(
            calibration_brier_score
        ),
        "calibration_mean_score_gap": float(
            mean_prediction_score - conversion_rate
        ),
        "pr_auc": pr_auc,
        "pr_auc_status": pr_auc_status,
        "precision": float(
            precision_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                actual_values,
                predicted_values,
                zero_division=0,
            )
        ),
    }


def build_model_version_id(
    row: Dict[str, Any],
) -> str:
    """Create a readable model-version identifier for grouping."""

    generation = str(row["model_generation"]).strip()
    model_hash = str(row["model_hash"]).strip()

    if not generation or not model_hash:
        raise ValueError(
            "Model generation and model hash are required."
        )

    return f"{generation}:{model_hash[:12]}"


def build_production_performance_snapshot(
    matured_rows: list[Dict[str, Any]],
    *,
    created_at_utc: str | None = None,
) -> Dict[str, Any]:
    """Build a truthful overall and model-version performance snapshot."""

    created_timestamp = (
        created_at_utc
        if created_at_utc is not None
        else datetime.now(timezone.utc).isoformat()
    )

    if not matured_rows:
        return {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_at_utc": created_timestamp,
            "status": "labels_unavailable",
            "metrics_available": False,
            "message": (
                "No matured production outcomes are available. "
                "Performance metrics are intentionally hidden."
            ),
            "matured_prediction_volume": 0,
            "overall_metrics": None,
            "by_model_version": [],
        }

    validated_rows = [
        validate_matured_outcome_row(row)
        for row in matured_rows
    ]

    grouped_rows: Dict[
        str,
        list[Dict[str, Any]],
    ] = {}

    for row in validated_rows:
        model_version_id = build_model_version_id(row)

        grouped_rows.setdefault(
            model_version_id,
            [],
        ).append(row)

    model_version_metrics = []

    for model_version_id in sorted(grouped_rows):
        version_rows = grouped_rows[model_version_id]
        first_row = version_rows[0]

        model_version_metrics.append(
            {
                "model_version_id": model_version_id,
                "model_name": first_row["model_name"],
                "model_generation": first_row[
                    "model_generation"
                ],
                "model_hash": first_row["model_hash"],
                "metadata_hash": first_row["metadata_hash"],
                "metrics": calculate_metric_block(
                    version_rows
                ),
            }
        )

    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at_utc": created_timestamp,
        "status": "available",
        "metrics_available": True,
        "message": (
            "Metrics use only predictions with matured outcome windows."
        ),
        "matured_prediction_volume": len(
            validated_rows
        ),
        "overall_metrics": calculate_metric_block(
            validated_rows
        ),
        "by_model_version": model_version_metrics,
    }


def write_production_performance_snapshot(
    snapshot: Dict[str, Any],
    path: Path = PRODUCTION_PERFORMANCE_PATH,
) -> Path:
    """Write the performance snapshot atomically as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path
