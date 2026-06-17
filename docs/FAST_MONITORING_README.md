# Fast Monitoring Solution

This package fixes slow Docker/Grafana monitoring.

## New flow

```text
Heavy CSV files are scanned once
        ↓
monitoring/metrics_cache/ecommerce_metrics_snapshot.json
        ↓
metrics_exporter.py reads only this small JSON
        ↓
Prometheus/Grafana run fast
```

## Must run before Docker

```bash
python3 -m src.monitoring.build_monitoring_snapshot
```

Then start Docker:

```bash
docker compose up -d --build
```
