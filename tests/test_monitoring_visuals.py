"""Tests for production-monitoring visuals J04 and J08."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualization.monitoring_visual_data import (
    MonitoringVisualBundle,
    build_funnel,
    freshness_status,
)
from src.visualization.monitoring_visuals import (
    delayed_label_retention_text,
    generate_monitoring_visual_package,
)


def build_bundle() -> MonitoringVisualBundle:
    """Create deterministic monitoring evidence with zero matured labels."""

    now = pd.Timestamp("2026-06-21T12:00:00Z")
    funnel = build_funnel(
        score_rows=78_998,
        prediction_rows=3,
        matured_rows=0,
        label_rows=0,
        evaluable_rows=0,
        source_notes={},
    )
    rejection_summary = pd.DataFrame(
        {
            "reason": [
                "Invalid labels",
                "Unknown predictions",
                "Premature labels",
                "Conflicting labels",
                "Exact duplicates",
            ],
            "count": [0, 0, 0, 0, 0],
        }
    )
    source_status = pd.DataFrame(
        {
            "source": [
                "Metrics snapshot",
                "Prediction log",
                "Prediction ledger",
                "Delayed-label validation",
                "Performance snapshot",
                "Champion lineage",
                "Evidently summary",
            ],
            "path": ["example"] * 7,
            "exists": [True, True, False, False, False, True, True],
            "latest_timestamp": [
                "2026-06-21T10:00:00+00:00",
                "2026-06-20T10:00:00+00:00",
                None,
                None,
                None,
                "2026-06-21T09:00:00+00:00",
                "2026-06-21T08:00:00+00:00",
            ],
            "age_hours": [2.0, 26.0, None, None, None, 3.0, 4.0],
            "freshness": [
                "Fresh",
                "Aging",
                "Missing",
                "Missing",
                "Missing",
                "Fresh",
                "Fresh",
            ],
        }
    )
    kpis = {
        "score_rows": 78_998,
        "prediction_rows": 3,
        "ledger_rows": 0,
        "matured_rows": 0,
        "label_rows": 0,
        "evaluable_rows": 0,
        "label_coverage_rate": 0.0,
        "outcome_metrics_available": False,
        "snapshot_timestamp": "2026-06-21T10:00:00+00:00",
        "prediction_timestamp": "2026-06-20T10:00:00+00:00",
        "registered_model_name": "ecommerce-conversion-champion",
        "registered_model_version": "3",
        "registered_model_alias": "champion",
        "available_sources": 4,
        "total_sources": 7,
        "fresh_sources": 3,
    }

    return MonitoringVisualBundle(
        funnel=funnel,
        rejection_summary=rejection_summary,
        source_status=source_status,
        kpis=kpis,
        warnings=[
            "No matured evaluable production outcomes are available."
        ],
        now_utc=now,
    )


def test_funnel_preserves_actual_zero_outcomes() -> None:
    """The funnel must keep zero labels and zero evaluable outcomes."""

    funnel = build_bundle().funnel.set_index("stage")

    assert funnel.loc["Scored population", "count"] == 78_998
    assert funnel.loc["Labels received", "count"] == 0
    assert funnel.loc["Evaluable outcomes", "count"] == 0


def test_freshness_thresholds_are_explicit() -> None:
    """Freshness should follow 24-hour and 72-hour thresholds."""

    now = pd.Timestamp("2026-06-21T12:00:00Z")

    assert freshness_status(
        now - pd.Timedelta(hours=10),
        now,
    )[1] == "Fresh"
    assert freshness_status(
        now - pd.Timedelta(hours=48),
        now,
    )[1] == "Aging"
    assert freshness_status(
        now - pd.Timedelta(hours=100),
        now,
    )[1] == "Stale"
    assert freshness_status(pd.NaT, now)[1] == "Missing"


def test_generate_monitoring_visual_package(
    tmp_path: Path,
) -> None:
    """The package should create two clean visuals and status evidence."""

    manifest = generate_monitoring_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    assert set(manifest["qa"]) == {
        "MLV-J04",
        "MLV-J08",
    }

    for qa_result in manifest["qa"].values():
        assert qa_result["passed"] is True
        assert qa_result["width_px"] >= 1_600
        assert qa_result["height_px"] >= 850

    for relative_path in manifest["artifacts"].values():
        output_file = tmp_path / relative_path
        assert output_file.exists()
        assert output_file.stat().st_size > 10_000

    support = manifest["supporting_files"]

    for key in (
        "funnel",
        "rejections",
        "source_status",
        "insights",
        "status",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    status_text = (
        tmp_path / support["status"]
    ).read_text(encoding="utf-8")

    assert "No alert, drift, conversion" in status_text
    assert manifest["blocked_visuals"] == [
        "MLV-J03",
        "MLV-J06",
        "MLV-J07",
    ]



def test_zero_after_zero_uses_na_retention_copy() -> None:
    """Later zero stages must not be labelled as a starting population."""

    funnel = build_bundle().funnel.reset_index(drop=True)
    assert delayed_label_retention_text(funnel, 0) == "Starting population"
    assert delayed_label_retention_text(funnel, 3) == (
        "N/A - prior stage has zero records"
    )
    assert delayed_label_retention_text(funnel, 4) == (
        "N/A - prior stage has zero records"
    )
