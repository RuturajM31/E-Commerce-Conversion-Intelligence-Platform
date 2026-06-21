"""Prepare production-monitoring evidence for ML Visual Intelligence.

Why this file exists:
    Monitoring evidence is spread across score exports, operational prediction
    logs, delayed-label outputs, a cached metrics snapshot, and MLflow lineage.
    This module reads those sources without pretending that missing labels are
    real zero-conversion outcomes.

Supported visuals:
    - MLV-J04 Delayed-Label Maturity Funnel
    - MLV-J08 Monitoring Freshness and Data-Coverage Card

Truthfulness rule:
    No matured production labels means no production-performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCORE_PATHS = [
    Path("data/processed/final_champion_visitor_scores.csv"),
    Path("data/processed/champion_visitor_scores.csv"),
    Path("data/processed/visitor_scores.csv"),
]
PREDICTION_LOG_PATH = Path(
    "monitoring/prediction_logs/prediction_log.csv"
)
LEDGER_PATHS = [
    Path("monitoring/prediction_logs/production_prediction_ledger.csv"),
    Path("monitoring/prediction_logs/prediction_ledger.csv"),
]
DELAYED_LABEL_PATHS = [
    Path("monitoring/delayed_labels/delayed_labels.csv"),
    Path("monitoring/delayed_labels/delayed_label_input.csv"),
]
MATURED_OUTCOMES_PATH = Path(
    "monitoring/delayed_labels/matured_prediction_outcomes.csv"
)
VALIDATION_REPORT_PATH = Path(
    "monitoring/delayed_labels/delayed_label_validation_report.json"
)
PERFORMANCE_SNAPSHOT_PATH = Path(
    "monitoring/delayed_labels/production_performance_snapshot.json"
)
METRICS_SNAPSHOT_PATH = Path(
    "monitoring/metrics_cache/ecommerce_metrics_snapshot.json"
)
LINEAGE_PATH = Path(
    "models/metadata/mlflow_champion_lineage.json"
)
EVIDENTLY_SUMMARY_PATH = Path(
    "monitoring/snapshots/evidently_monitoring_summary.json"
)

FRESH_HOURS = 24.0
AGING_HOURS = 72.0


@dataclass
class MonitoringVisualBundle:
    """Container holding all evidence required by J04 and J08."""

    funnel: pd.DataFrame
    rejection_summary: pd.DataFrame
    source_status: pd.DataFrame
    kpis: dict[str, Any]
    warnings: list[str]
    now_utc: pd.Timestamp


def first_existing(
    root: Path,
    candidates: Iterable[Path],
) -> Path | None:
    """Return the first existing project-relative path."""

    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path

    return None


def read_json(path: Path | None) -> dict[str, Any]:
    """Read a JSON object or return an empty dictionary."""

    if path is None or not path.exists():
        return {}

    try:
        content = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return {}

    return content if isinstance(content, dict) else {}


def read_csv(path: Path | None) -> pd.DataFrame:
    """Read a CSV source or return an empty DataFrame."""

    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse one timestamp as UTC without raising."""

    if value in (None, "", "None", "nan", "NaN"):
        return pd.NaT

    try:
        return pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )
    except Exception:
        return pd.NaT


def file_timestamp(path: Path | None) -> pd.Timestamp | pd.NaT:
    """Return the file modification time as a UTC timestamp."""

    if path is None or not path.exists():
        return pd.NaT

    return pd.Timestamp(
        datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
    )


def nested_get(
    content: dict[str, Any],
    dotted_key: str,
) -> Any:
    """Read one dotted key from a nested dictionary."""

    current: Any = content

    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)

    return current


def first_timestamp(
    content: dict[str, Any],
    keys: Iterable[str],
) -> pd.Timestamp | pd.NaT:
    """Return the first parseable timestamp from candidate JSON keys."""

    for key in keys:
        value = nested_get(content, key)
        timestamp = parse_timestamp(value)

        if pd.notna(timestamp):
            return timestamp

    unix_candidates = [
        "generated_at_unix",
        "snapshot_generated_at_unix",
        "metrics.ecommerce_snapshot_generated_at_unix",
    ]

    for key in unix_candidates:
        value = nested_get(content, key)

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if numeric > 0:
            return pd.to_datetime(
                numeric,
                unit="s",
                utc=True,
            )

    return pd.NaT


def latest_csv_timestamp(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Timestamp | pd.NaT:
    """Return the newest valid timestamp from likely CSV columns."""

    for column in candidates:
        if column not in frame.columns:
            continue

        parsed = pd.to_datetime(
            frame[column],
            utc=True,
            errors="coerce",
        )

        if parsed.notna().any():
            return parsed.max()

    return pd.NaT


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a monitoring count into a non-negative integer."""

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default

    return max(number, 0)


def safe_bool(value: Any) -> bool:
    """Convert monitoring JSON values into a truthful Boolean."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "available",
    }


def freshness_status(
    timestamp: pd.Timestamp | pd.NaT,
    now_utc: pd.Timestamp,
) -> tuple[float | None, str]:
    """Classify source freshness using transparent thresholds."""

    if pd.isna(timestamp):
        return None, "Missing"

    age_hours = max(
        0.0,
        (now_utc - timestamp).total_seconds() / 3600.0,
    )

    if age_hours <= FRESH_HOURS:
        return age_hours, "Fresh"

    if age_hours <= AGING_HOURS:
        return age_hours, "Aging"

    return age_hours, "Stale"


def build_rejection_summary(
    validation_report: dict[str, Any],
) -> pd.DataFrame:
    """Create delayed-label rejection evidence from approved report fields."""

    mappings = [
        (
            "Invalid labels",
            ["rejected_invalid_rows", "invalid_label_rows"],
        ),
        (
            "Unknown predictions",
            [
                "rejected_unknown_prediction_ids",
                "unknown_prediction_rows",
            ],
        ),
        (
            "Premature labels",
            [
                "rejected_premature_labels",
                "premature_label_rows",
            ],
        ),
        (
            "Conflicting labels",
            [
                "rejected_conflicting_prediction_ids",
                "conflicting_label_rows",
            ],
        ),
        (
            "Exact duplicates",
            ["duplicate_label_rows"],
        ),
    ]

    records: list[dict[str, Any]] = []

    for reason, keys in mappings:
        count = 0

        for key in keys:
            if key in validation_report:
                count = safe_int(
                    validation_report.get(key)
                )
                break

        records.append(
            {
                "reason": reason,
                "count": count,
            }
        )

    return pd.DataFrame(records)


def build_funnel(
    *,
    score_rows: int,
    prediction_rows: int,
    matured_rows: int,
    label_rows: int,
    evaluable_rows: int,
    source_notes: dict[str, str],
) -> pd.DataFrame:
    """Create the actual-count delayed-label maturity funnel."""

    stages = [
        ("Scored population", score_rows),
        ("Production predictions logged", prediction_rows),
        ("Outcome window matured", matured_rows),
        ("Labels received", label_rows),
        ("Evaluable outcomes", evaluable_rows),
    ]

    records: list[dict[str, Any]] = []
    previous_count: int | None = None

    for stage, count in stages:
        retention = (
            count / previous_count
            if previous_count not in (None, 0)
            else np.nan
        )
        records.append(
            {
                "stage": stage,
                "count": int(max(count, 0)),
                "retention_from_prior": retention,
                "source_note": source_notes.get(
                    stage,
                    "",
                ),
            }
        )
        previous_count = count

    return pd.DataFrame(records)


def build_source_status(
    *,
    root: Path,
    now_utc: pd.Timestamp,
    paths: dict[str, Path | None],
    explicit_timestamps: dict[str, pd.Timestamp | pd.NaT],
) -> pd.DataFrame:
    """Create file presence and freshness evidence for J08."""

    records: list[dict[str, Any]] = []

    for source_name, path in paths.items():
        exists = bool(path and path.exists())
        timestamp = explicit_timestamps.get(
            source_name,
            pd.NaT,
        )

        if pd.isna(timestamp):
            timestamp = file_timestamp(path)

        age_hours, status = freshness_status(
            timestamp,
            now_utc,
        )

        records.append(
            {
                "source": source_name,
                "path": (
                    str(path.relative_to(root))
                    if exists and path is not None
                    else (
                        str(path)
                        if path is not None
                        else "not found"
                    )
                ),
                "exists": exists,
                "latest_timestamp": (
                    timestamp.isoformat()
                    if pd.notna(timestamp)
                    else None
                ),
                "age_hours": age_hours,
                "freshness": status,
            }
        )

    return pd.DataFrame(records)


def build_monitoring_visual_bundle(
    project_root: str | Path = ".",
    *,
    now_utc: pd.Timestamp | None = None,
) -> MonitoringVisualBundle:
    """Load real project sources and build J04/J08 evidence."""

    root = Path(project_root)
    now = (
        pd.Timestamp.now(tz="UTC")
        if now_utc is None
        else pd.Timestamp(now_utc)
    )

    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    score_path = first_existing(root, SCORE_PATHS)
    prediction_log_path = root / PREDICTION_LOG_PATH
    ledger_path = first_existing(root, LEDGER_PATHS)
    delayed_label_path = first_existing(
        root,
        DELAYED_LABEL_PATHS,
    )
    matured_path = root / MATURED_OUTCOMES_PATH
    validation_path = root / VALIDATION_REPORT_PATH
    performance_path = root / PERFORMANCE_SNAPSHOT_PATH
    metrics_path = root / METRICS_SNAPSHOT_PATH
    lineage_path = root / LINEAGE_PATH
    evidently_path = root / EVIDENTLY_SUMMARY_PATH

    score_frame = read_csv(score_path)
    prediction_log = read_csv(
        prediction_log_path
        if prediction_log_path.exists()
        else None
    )
    ledger = read_csv(ledger_path)
    delayed_labels = read_csv(delayed_label_path)
    matured_outcomes = read_csv(
        matured_path if matured_path.exists() else None
    )

    validation_report = read_json(
        validation_path
        if validation_path.exists()
        else None
    )
    performance_snapshot = read_json(
        performance_path
        if performance_path.exists()
        else None
    )
    metrics_snapshot = read_json(
        metrics_path
        if metrics_path.exists()
        else None
    )
    lineage = read_json(
        lineage_path
        if lineage_path.exists()
        else None
    )
    evidently_summary = read_json(
        evidently_path
        if evidently_path.exists()
        else None
    )

    score_rows = len(score_frame)
    ledger_rows = len(ledger)
    prediction_rows = (
        ledger_rows
        if ledger_rows > 0
        else len(prediction_log)
    )

    matured_rows = 0

    if not ledger.empty and "outcome_window_end" in ledger.columns:
        outcome_end = pd.to_datetime(
            ledger["outcome_window_end"],
            utc=True,
            errors="coerce",
        )
        matured_rows = int(
            (outcome_end <= now).sum()
        )
    else:
        matured_rows = safe_int(
            validation_report.get(
                "matured_prediction_rows",
                validation_report.get(
                    "eligible_matured_rows",
                    validation_report.get(
                        "accepted_matured_rows",
                        len(matured_outcomes),
                    ),
                ),
            )
        )

    label_rows = safe_int(
        validation_report.get(
            "total_label_rows",
            len(delayed_labels),
        )
    )
    evaluable_rows = safe_int(
        validation_report.get(
            "accepted_matured_rows",
            len(matured_outcomes),
        )
    )

    rejection_summary = build_rejection_summary(
        validation_report
    )

    source_notes = {
        "Scored population": (
            str(score_path.relative_to(root))
            if score_path is not None
            else "score export unavailable"
        ),
        "Production predictions logged": (
            str(ledger_path.relative_to(root))
            if ledger_rows > 0 and ledger_path is not None
            else str(PREDICTION_LOG_PATH)
        ),
        "Outcome window matured": (
            "calculated from outcome_window_end"
            if not ledger.empty
            and "outcome_window_end" in ledger.columns
            else "delayed-label validation evidence"
        ),
        "Labels received": str(
            delayed_label_path.relative_to(root)
        )
        if delayed_label_path is not None
        else "no delayed-label input found",
        "Evaluable outcomes": str(
            MATURED_OUTCOMES_PATH
        ),
    }

    funnel = build_funnel(
        score_rows=score_rows,
        prediction_rows=prediction_rows,
        matured_rows=matured_rows,
        label_rows=label_rows,
        evaluable_rows=evaluable_rows,
        source_notes=source_notes,
    )

    snapshot_timestamp = first_timestamp(
        metrics_snapshot,
        [
            "generated_at_utc",
            "generated_at",
            "snapshot_generated_at_utc",
            "created_at_utc",
        ],
    )
    prediction_timestamp = latest_csv_timestamp(
        prediction_log,
        [
            "timestamp_utc",
            "scoring_time",
            "timestamp",
            "created_at_utc",
        ],
    )
    validation_timestamp = first_timestamp(
        validation_report,
        [
            "generated_at_utc",
            "generated_at",
            "validation_time",
            "evaluation_time",
        ],
    )
    performance_timestamp = first_timestamp(
        performance_snapshot,
        [
            "generated_at_utc",
            "generated_at",
            "evaluation_time",
        ],
    )
    evidently_timestamp = first_timestamp(
        evidently_summary,
        ["generated_at_utc", "generated_at"],
    )

    paths = {
        "Metrics snapshot": (
            metrics_path if metrics_path.exists() else None
        ),
        "Prediction log": (
            prediction_log_path
            if prediction_log_path.exists()
            else None
        ),
        "Prediction ledger": ledger_path,
        "Delayed-label validation": (
            validation_path
            if validation_path.exists()
            else None
        ),
        "Performance snapshot": (
            performance_path
            if performance_path.exists()
            else None
        ),
        "Champion lineage": (
            lineage_path
            if lineage_path.exists()
            else None
        ),
        "Evidently summary": (
            evidently_path
            if evidently_path.exists()
            else None
        ),
    }
    explicit_timestamps = {
        "Metrics snapshot": snapshot_timestamp,
        "Prediction log": prediction_timestamp,
        "Delayed-label validation": validation_timestamp,
        "Performance snapshot": performance_timestamp,
        "Evidently summary": evidently_timestamp,
    }
    source_status = build_source_status(
        root=root,
        now_utc=now,
        paths=paths,
        explicit_timestamps=explicit_timestamps,
    )

    outcome_metrics_available = safe_bool(
        performance_snapshot.get(
            "metrics_available",
            nested_get(
                metrics_snapshot,
                "metrics.ecommerce_outcome_labels_available",
            ),
        )
    )

    coverage_denominator = (
        ledger_rows
        if ledger_rows > 0
        else prediction_rows
    )
    coverage_rate = (
        evaluable_rows / coverage_denominator
        if coverage_denominator > 0
        else 0.0
    )

    warnings: list[str] = []

    if evaluable_rows == 0:
        warnings.append(
            "No matured evaluable production outcomes are available."
        )

    if not outcome_metrics_available:
        warnings.append(
            "Production performance metrics are unavailable by design."
        )

    if not funnel["count"].is_monotonic_decreasing:
        warnings.append(
            "Funnel stages use different operational source scopes."
        )

    kpis = {
        "score_rows": score_rows,
        "prediction_rows": prediction_rows,
        "ledger_rows": ledger_rows,
        "matured_rows": matured_rows,
        "label_rows": label_rows,
        "evaluable_rows": evaluable_rows,
        "label_coverage_rate": coverage_rate,
        "outcome_metrics_available": outcome_metrics_available,
        "snapshot_timestamp": (
            snapshot_timestamp.isoformat()
            if pd.notna(snapshot_timestamp)
            else None
        ),
        "prediction_timestamp": (
            prediction_timestamp.isoformat()
            if pd.notna(prediction_timestamp)
            else None
        ),
        "registered_model_name": lineage.get(
            "registered_model_name",
            "unknown",
        ),
        "registered_model_version": lineage.get(
            "registered_model_version",
            "unknown",
        ),
        "registered_model_alias": lineage.get(
            "registered_model_alias",
            "unknown",
        ),
        "available_sources": int(
            source_status["exists"].sum()
        ),
        "total_sources": len(source_status),
        "fresh_sources": int(
            (source_status["freshness"] == "Fresh").sum()
        ),
    }

    return MonitoringVisualBundle(
        funnel=funnel,
        rejection_summary=rejection_summary,
        source_status=source_status,
        kpis=kpis,
        warnings=warnings,
        now_utc=now,
    )
