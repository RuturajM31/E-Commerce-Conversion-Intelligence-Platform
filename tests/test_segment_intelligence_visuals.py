"""Tests for MLV-G02 segment intelligence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.segment_intelligence_data import (
    SegmentIntelligenceBundle,
    build_score_summary,
    prepare_segment_summary,
    prepare_visitor_scores,
    resolve_segment_order,
)
from src.visualization.segment_intelligence_visuals import (
    generate_segment_intelligence_visual_package,
)


def build_sources() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create deterministic five-segment score sources."""

    rng = np.random.default_rng(42)
    definitions = [
        ("High Intent", 0.92, 120),
        ("Strong Intent", 0.78, 250),
        ("Warm Intent", 0.61, 320),
        ("Low Intent", 0.32, 500),
        ("Very Low Intent", 0.12, 260),
    ]

    rows = []
    visitor_id = 1

    for segment, centre, count in definitions:
        scores = np.clip(
            rng.normal(
                centre,
                0.035,
                count,
            ),
            0.0,
            1.0,
        )

        for score in scores:
            rows.append(
                {
                    "visitorid": visitor_id,
                    "purchase_intent_score": score,
                    "intent_segment": segment,
                }
            )
            visitor_id += 1

    visitor_scores = pd.DataFrame(rows)
    summary = (
        visitor_scores
        .groupby("intent_segment")
        .agg(
            visitors=("visitorid", "nunique"),
            avg_score=(
                "purchase_intent_score",
                "mean",
            ),
        )
        .reset_index()
    )
    summary["visitor_share"] = (
        summary["visitors"]
        / summary["visitors"].sum()
    )
    summary["actual_converters"] = np.nan
    summary["conversion_rate"] = np.nan

    return visitor_scores, summary


def build_bundle() -> SegmentIntelligenceBundle:
    """Create a complete synthetic segment bundle."""

    raw_scores, raw_summary = build_sources()
    scores = prepare_visitor_scores(
        raw_scores
    )
    provided = prepare_segment_summary(
        raw_summary
    )
    summary = build_score_summary(
        scores,
        provided,
    )
    order = resolve_segment_order(
        summary["intent_segment"].tolist()
    )

    return SegmentIntelligenceBundle(
        visitor_scores=scores,
        segment_summary=summary,
        segment_order=order,
        source_rows=len(scores),
        outcome_columns_available=False,
    )


def test_score_summary_reconciles() -> None:
    """Counts and shares should reconcile to the source."""

    bundle = build_bundle()

    assert (
        bundle.segment_summary["visitors"].sum()
        == bundle.source_rows
    )
    assert bundle.segment_summary[
        "visitor_share"
    ].sum() == pytest.approx(1.0)
    assert bundle.segment_summary[
        "outcomes_available"
    ].sum() == 0


def test_segment_order_is_business_order() -> None:
    """Segments should follow targeting priority."""

    assert build_bundle().segment_order == [
        "High Intent",
        "Strong Intent",
        "Warm Intent",
        "Low Intent",
        "Very Low Intent",
    ]


def test_duplicate_visitors_are_rejected() -> None:
    """Each visitor should have one current score row."""

    scores, _ = build_sources()
    broken = pd.concat(
        [scores, scores.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate visitor",
    ):
        prepare_visitor_scores(broken)


def test_generate_segment_package(
    tmp_path: Path,
) -> None:
    """The package should create G02 and conditional evidence."""

    manifest = (
        generate_segment_intelligence_visual_package(
            project_root=tmp_path,
            bundle=build_bundle(),
        )
    )

    assert set(manifest["qa"]) == {
        "MLV-G02"
    }
    qa = manifest["qa"]["MLV-G02"]
    assert qa["passed"] is True
    assert qa["width_px"] >= 1_600
    assert qa["height_px"] >= 850

    visual = (
        tmp_path
        / manifest["artifacts"]["MLV-G02"]
    )
    assert visual.exists()
    assert visual.stat().st_size > 10_000

    for key in (
        "segment_summary",
        "insights",
        "conditional_status",
        "manifest",
    ):
        assert (
            tmp_path
            / manifest["supporting_files"][key]
        ).exists()

    status_text = (
        tmp_path
        / manifest["supporting_files"][
            "conditional_status"
        ]
    ).read_text(encoding="utf-8")

    assert (
        "No conversion or error result is fabricated"
        in status_text
    )
    assert set(
        manifest["conditional_visuals"]
    ) == {
        "MLV-G01",
        "MLV-G03",
        "MLV-G04",
    }
