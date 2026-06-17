
"""
Fast metrics exporter for Prometheus.

This exporter reads only:
    monitoring/metrics_cache/ecommerce_metrics_snapshot.json

It does not scan heavy CSV files during Prometheus scraping.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
PORT = int(os.getenv("METRICS_EXPORTER_PORT", "8000"))
SNAPSHOT_PATH = PROJECT_ROOT / "monitoring/metrics_cache/ecommerce_metrics_snapshot.json"


def sf(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_snapshot() -> Dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        return {
            "generated_at_unix": 0,
            "metrics": {},
            "labeled_metrics": {},
            "warnings": [f"Snapshot missing: {SNAPSHOT_PATH}"],
        }
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "generated_at_unix": 0,
            "metrics": {},
            "labeled_metrics": {},
            "warnings": [f"Snapshot read error: {exc}"],
        }


def metric(name: str, value: Any, help_text: str, labels: Optional[Dict[str, str]] = None) -> str:
    value = sf(value)
    if labels:
        label_text = ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in labels.items())
        line = f"{name}{{{label_text}}} {value}"
    else:
        line = f"{name} {value}"
    return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n{line}\n"


def label_name_for(metric_name: str) -> str:
    if metric_name in {"ecommerce_file_available", "ecommerce_file_age_seconds"}:
        return "file"
    if metric_name == "ecommerce_visitor_segment_count":
        return "segment"
    if metric_name == "ecommerce_forecast_predicted_total":
        return "target_name"
    if metric_name == "ecommerce_daily_kpi_latest":
        return "metric"
    if metric_name == "ecommerce_model_info":
        return "model_name"
    return "label"


def build_metrics() -> str:
    snapshot = load_snapshot()
    parts = []

    generated_at = sf(snapshot.get("generated_at_unix"), 0)
    snapshot_age = time.time() - generated_at if generated_at > 0 else -1

    parts.append(metric("ecommerce_metrics_exporter_up", 1, "Metrics exporter health."))
    parts.append(metric("ecommerce_snapshot_available", 1 if generated_at > 0 else 0, "Monitoring snapshot availability."))
    parts.append(metric("ecommerce_snapshot_generated_at_unix", generated_at, "Snapshot generation timestamp."))
    parts.append(metric("ecommerce_snapshot_age_seconds", snapshot_age, "Snapshot age in seconds."))

    for name, value in snapshot.get("metrics", {}).items():
        parts.append(metric(name, value, f"Snapshot metric: {name}."))

    for metric_name, values in snapshot.get("labeled_metrics", {}).items():
        if not isinstance(values, dict):
            continue
        label_key = label_name_for(metric_name)
        for label_value, value in values.items():
            parts.append(metric(metric_name, value, f"Labeled snapshot metric: {metric_name}.", {label_key: str(label_value)}))

    warnings = snapshot.get("warnings", [])
    parts.append(metric("ecommerce_snapshot_warning_count", len(warnings) if isinstance(warnings, list) else 0, "Snapshot warning count."))
    return "\n".join(parts) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, b"ok\n", "text/plain")
            return
        if self.path != "/metrics":
            self._send(404, b"Use /metrics or /health\n", "text/plain")
            return
        self._send(200, build_metrics().encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[metrics-exporter] {fmt % args}")


def main() -> None:
    print(f"Starting fast metrics exporter on port {PORT}")
    print(f"Snapshot path: {SNAPSHOT_PATH}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
