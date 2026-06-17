
"""
Build one small monitoring snapshot JSON for Prometheus/Grafana.

Run locally from the project root:

    python3 -m src.monitoring.build_monitoring_snapshot

Output:

    monitoring/metrics_cache/ecommerce_metrics_snapshot.json

Why:
Prometheus should not scan 500MB CSV files every 15 seconds.
This script scans heavy files once and saves ready-to-serve KPI numbers.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(".").resolve()
SNAPSHOT_DIR = ROOT / "monitoring/metrics_cache"
SNAPSHOT_PATH = SNAPSHOT_DIR / "ecommerce_metrics_snapshot.json"

VISITOR_FEATURES = ROOT / "data/processed/visitor_features.csv"
SCORE_FILES = [
    ROOT / "data/processed/final_champion_visitor_scores.csv",
    ROOT / "data/processed/champion_visitor_scores.csv",
    ROOT / "data/processed/visitor_scores.csv",
    ROOT / "data/processed/visitor_scores_sample.csv",
]
ANOMALY_SCORES = ROOT / "data/processed/visitor_anomaly_scores.csv"
FINAL_SUMMARY = ROOT / "reports/tables/final_true_champion_summary.csv"
FINAL_METADATA = ROOT / "models/metadata/final_champion_metadata.json"
FORECAST_FUTURE = ROOT / "reports/tables/business_forecast_future.csv"
DAILY_KPIS = ROOT / "reports/tables/daily_business_kpis.csv"
ANOMALY_SUMMARY = ROOT / "reports/tables/anomaly_summary.csv"
MVD_COVERAGE = ROOT / "reports/tables/mvd_method_coverage_matrix.csv"
PREDICTION_LOG = ROOT / "monitoring/prediction_logs/prediction_log.csv"
BATCH_LOG = ROOT / "monitoring/prediction_logs/batch_scoring_log.csv"


def sf(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan", "NaN", "None"):
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def si(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_small_csv(path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                rows.append(row)
                if limit is not None and index + 1 >= limit:
                    break
    except Exception:
        return []
    return rows


def header(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return next(csv.reader(file), [])
    except Exception:
        return []


def find_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def first_score_file() -> Optional[Path]:
    for path in SCORE_FILES:
        if path.exists():
            return path
    return None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def file_age(path: Path) -> float:
    if not path.exists():
        return -1.0
    return time.time() - path.stat().st_mtime


def set_metric(snapshot: Dict[str, Any], name: str, value: Any) -> None:
    snapshot["metrics"][name] = sf(value)


def add_file_metrics(snapshot: Dict[str, Any]) -> None:
    files = {
        "visitor_features": VISITOR_FEATURES,
        "visitor_scores": first_score_file() or SCORE_FILES[0],
        "anomaly_scores": ANOMALY_SCORES,
        "final_champion_summary": FINAL_SUMMARY,
        "final_champion_metadata": FINAL_METADATA,
        "forecast_future": FORECAST_FUTURE,
        "daily_business_kpis": DAILY_KPIS,
        "anomaly_summary": ANOMALY_SUMMARY,
        "mvd_coverage": MVD_COVERAGE,
        "prediction_log": PREDICTION_LOG,
        "batch_log": BATCH_LOG,
    }
    snapshot["labeled_metrics"]["ecommerce_file_available"] = {k: 1 if v.exists() else 0 for k, v in files.items()}
    snapshot["labeled_metrics"]["ecommerce_file_age_seconds"] = {k: file_age(v) for k, v in files.items()}


def add_model_metrics(snapshot: Dict[str, Any]) -> float:
    threshold = 0.5
    metadata = read_json(FINAL_METADATA)
    rows = read_small_csv(FINAL_SUMMARY, limit=1)
    row = rows[0] if rows else {}
    columns = list(row.keys())

    mapping = {
        "ecommerce_model_pr_auc": ["pr_auc", "best_pr_auc"],
        "ecommerce_model_roc_auc": ["roc_auc", "best_roc_auc"],
        "ecommerce_model_precision": ["best_precision", "precision"],
        "ecommerce_model_recall": ["best_recall", "recall"],
        "ecommerce_model_f1_score": ["best_f1_score", "f1_score", "f1"],
        "ecommerce_model_business_score": ["business_score", "best_business_score"],
        "ecommerce_model_threshold": ["best_threshold", "threshold", "final_threshold"],
    }
    for metric_name, candidates in mapping.items():
        col = find_col(columns, candidates)
        value = sf(row.get(col), 0.0) if col else 0.0
        set_metric(snapshot, metric_name, value)
        if metric_name == "ecommerce_model_threshold" and value > 0:
            threshold = value

    for key in ["best_threshold", "final_threshold", "threshold"]:
        if key in metadata:
            threshold = sf(metadata.get(key), threshold)
    set_metric(snapshot, "ecommerce_model_threshold", threshold)

    model_name = metadata.get("model_name") or metadata.get("champion_model") or metadata.get("final_model_name") or "unknown"
    snapshot["labeled_metrics"]["ecommerce_model_info"] = {str(model_name): 1}
    set_metric(snapshot, "ecommerce_model_drift_detected", 0)
    return threshold


def add_visitor_metrics(snapshot: Dict[str, Any]) -> None:
    cols = header(VISITOR_FEATURES)
    if not cols:
        return

    visitor_col = find_col(cols, ["visitorid", "visitor_id"])
    target_col = find_col(cols, ["converted", "target"])
    feature_cols = {
        "total_events": find_col(cols, ["total_events", "event_count"]),
        "view_count": find_col(cols, ["view_count", "views"]),
        "addtocart_count": find_col(cols, ["addtocart_count", "add_to_cart_count"]),
        "unique_items": find_col(cols, ["unique_items", "unique_item_count"]),
        "activity_span_ms": find_col(cols, ["activity_span_ms"]),
    }

    row_count = 0
    converted = 0
    missing = 0
    cells = 0
    seen = set()
    duplicates = 0
    totals = {name: 0.0 for name, col in feature_cols.items() if col}

    with VISITOR_FEATURES.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_count += 1
            cells += len(cols)
            for c in cols:
                if row.get(c) in ("", None, "nan", "NaN", "None"):
                    missing += 1
            if visitor_col:
                visitor = row.get(visitor_col, "")
                if visitor in seen:
                    duplicates += 1
                else:
                    seen.add(visitor)
            if target_col:
                converted += si(row.get(target_col), 0)
            for name, col in feature_cols.items():
                if col:
                    totals[name] = totals.get(name, 0.0) + sf(row.get(col), 0.0)

    set_metric(snapshot, "ecommerce_total_visitors", row_count)
    set_metric(snapshot, "ecommerce_converted_visitors", converted)
    set_metric(snapshot, "ecommerce_conversion_rate", converted / max(row_count, 1))
    set_metric(snapshot, "ecommerce_duplicate_visitor_count", duplicates)
    set_metric(snapshot, "ecommerce_duplicate_visitor_rate", duplicates / max(row_count, 1))
    set_metric(snapshot, "ecommerce_missing_value_count", missing)
    set_metric(snapshot, "ecommerce_missing_value_rate", missing / max(cells, 1))
    for name, value in totals.items():
        set_metric(snapshot, f"ecommerce_{name}_total", value)
        set_metric(snapshot, f"ecommerce_{name}_average", value / max(row_count, 1))


def add_score_metrics(snapshot: Dict[str, Any], threshold: float) -> None:
    path = first_score_file()
    if not path:
        return
    cols = header(path)
    if not cols:
        return
    score_col = find_col(cols, ["purchase_intent_score", "score", "prediction_score", "probability"])
    visitor_col = find_col(cols, ["visitorid", "visitor_id"])
    if not score_col:
        snapshot["warnings"].append(f"Score column not found in {path}")
        return

    scores: List[float] = []
    seen = set()
    duplicates = 0
    invalid = 0

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            score = sf(row.get(score_col), 0.0)
            scores.append(score)
            if score < 0 or score > 1:
                invalid += 1
            if visitor_col:
                visitor = row.get(visitor_col, "")
                if visitor in seen:
                    duplicates += 1
                else:
                    seen.add(visitor)

    total = len(scores)
    avg = sum(scores) / max(total, 1)
    high = sum(1 for s in scores if s >= threshold)

    segments = {
        "high_intent": high,
        "strong_intent": sum(1 for s in scores if 0.80 <= s < threshold),
        "warm_intent": sum(1 for s in scores if 0.50 <= s < 0.80),
        "low_intent": sum(1 for s in scores if 0.20 <= s < 0.50),
        "cold": sum(1 for s in scores if s < 0.20),
    }

    set_metric(snapshot, "ecommerce_score_rows", total)
    set_metric(snapshot, "ecommerce_high_intent_visitors_total", high)
    set_metric(snapshot, "ecommerce_high_intent_rate", high / max(total, 1))
    set_metric(snapshot, "ecommerce_invalid_score_count", invalid)
    set_metric(snapshot, "ecommerce_invalid_score_rate", invalid / max(total, 1))
    set_metric(snapshot, "ecommerce_prediction_score_average", avg)
    set_metric(snapshot, "ecommerce_prediction_score", avg)
    set_metric(snapshot, "ecommerce_prediction_score_median", percentile(scores, 0.50))
    set_metric(snapshot, "ecommerce_prediction_score_p90", percentile(scores, 0.90))
    set_metric(snapshot, "ecommerce_prediction_score_p95", percentile(scores, 0.95))
    set_metric(snapshot, "ecommerce_prediction_score_max", max(scores) if scores else 0)
    set_metric(snapshot, "ecommerce_score_duplicate_visitor_count", duplicates)
    snapshot["labeled_metrics"]["ecommerce_visitor_segment_count"] = segments


def add_anomaly_metrics(snapshot: Dict[str, Any]) -> None:
    rows = read_small_csv(ANOMALY_SUMMARY)
    for row in rows:
        cols = list(row.keys())
        metric_col = find_col(cols, ["metric", "name"])
        value_col = find_col(cols, ["value", "metric_value"])
        if metric_col and value_col:
            name = str(row.get(metric_col, "")).lower()
            value = sf(row.get(value_col), 0.0)
            if "rate" in name:
                set_metric(snapshot, "ecommerce_anomaly_rate", value)
            if "count" in name or "anomal" in name:
                set_metric(snapshot, "ecommerce_anomaly_count", value)

    if "ecommerce_anomaly_rate" in snapshot["metrics"] and "ecommerce_anomaly_count" in snapshot["metrics"]:
        return

    cols = header(ANOMALY_SCORES)
    anomaly_col = find_col(cols, ["final_anomaly", "is_anomaly", "anomaly_flag"])
    if not anomaly_col:
        return

    total = 0
    count = 0
    with ANOMALY_SCORES.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            if si(row.get(anomaly_col), 0) == 1:
                count += 1
    set_metric(snapshot, "ecommerce_anomaly_count", count)
    set_metric(snapshot, "ecommerce_anomaly_rate", count / max(total, 1))


def add_forecast_metrics(snapshot: Dict[str, Any]) -> None:
    rows = read_small_csv(FORECAST_FUTURE)
    if not rows:
        return
    cols = list(rows[0].keys())
    target_col = find_col(cols, ["target_name", "metric", "kpi"])
    value_col = find_col(cols, ["predicted_value", "forecast", "yhat"])
    if not target_col or not value_col:
        return
    totals: Dict[str, float] = {}
    for row in rows:
        target = str(row.get(target_col, "unknown"))
        totals[target] = totals.get(target, 0.0) + sf(row.get(value_col), 0.0)
    set_metric(snapshot, "ecommerce_forecast_rows", len(rows))
    snapshot["labeled_metrics"]["ecommerce_forecast_predicted_total"] = totals


def add_daily_kpis(snapshot: Dict[str, Any]) -> None:
    rows = read_small_csv(DAILY_KPIS)
    if not rows:
        return
    latest = rows[-1]
    values = {}
    for key, value in latest.items():
        if key.lower() in {"date", "timestamp", "day"}:
            continue
        try:
            values[key] = float(value)
        except Exception:
            continue
    snapshot["labeled_metrics"]["ecommerce_daily_kpi_latest"] = values


def add_logs(snapshot: Dict[str, Any]) -> None:
    preds = read_small_csv(PREDICTION_LOG)
    batches = read_small_csv(BATCH_LOG)
    set_metric(snapshot, "ecommerce_predictions_total", len(preds))
    set_metric(snapshot, "ecommerce_batch_scoring_rows_total", len(batches))

    errors = 0
    for row in preds:
        cols = list(row.keys())
        status_col = find_col(cols, ["status", "result"])
        error_col = find_col(cols, ["error", "error_message"])
        if status_col and str(row.get(status_col, "")).lower() in {"error", "failed", "failure"}:
            errors += 1
        elif error_col and str(row.get(error_col, "")).strip():
            errors += 1
    set_metric(snapshot, "ecommerce_prediction_errors_total", errors)


def add_mvd(snapshot: Dict[str, Any]) -> None:
    rows = read_small_csv(MVD_COVERAGE)
    if not rows:
        return
    covered = sum(
        1 for row in rows
        if any(marker in " ".join(str(v).lower() for v in row.values()) for marker in ["yes", "implemented", "covered"])
    )
    set_metric(snapshot, "ecommerce_mvd_coverage_rows", len(rows))
    set_metric(snapshot, "ecommerce_mvd_implemented_or_covered_rows", covered)


def build_snapshot() -> Dict[str, Any]:
    started = time.time()
    snapshot = {
        "generated_at_unix": started,
        "generated_at_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {},
        "labeled_metrics": {},
        "warnings": [],
    }
    add_file_metrics(snapshot)
    threshold = add_model_metrics(snapshot)
    add_visitor_metrics(snapshot)
    add_score_metrics(snapshot, threshold)
    add_anomaly_metrics(snapshot)
    add_forecast_metrics(snapshot)
    add_daily_kpis(snapshot)
    add_logs(snapshot)
    add_mvd(snapshot)
    set_metric(snapshot, "ecommerce_snapshot_build_duration_seconds", time.time() - started)
    set_metric(snapshot, "ecommerce_snapshot_available", 1)
    return snapshot


def main() -> None:
    print("Building monitoring snapshot...")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot()
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Snapshot saved: {SNAPSHOT_PATH}")
    print(f"Metrics: {len(snapshot['metrics'])}")
    print(f"Labeled metric groups: {len(snapshot['labeled_metrics'])}")
    print(f"Warnings: {len(snapshot['warnings'])}")
    for warning in snapshot["warnings"]:
        print(f"- {warning}")


if __name__ == "__main__":
    main()
