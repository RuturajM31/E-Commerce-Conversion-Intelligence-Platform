"""Tests for MLV-H01, MLV-H02, and MLV-H04."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.feature_health_data import (
    FEATURE_COLUMNS,
    FeatureHealthBundle,
    build_clustered_order,
    build_correlation_matrix,
    build_distribution_summary,
    build_validity_summary,
    prepare_feature_frame,
)
from src.visualization.feature_health_visuals import (
    generate_feature_health_visual_package,
)


def build_feature_frame(
    row_count: int = 600,
) -> pd.DataFrame:
    """Create deterministic visitor feature evidence."""

    rng = np.random.default_rng(42)

    views = rng.lognormal(
        mean=2.0,
        sigma=0.9,
        size=row_count,
    )
    carts = rng.poisson(
        lam=np.maximum(
            views * 0.08,
            0.05,
        )
    )
    unique_items = np.maximum(
        1,
        np.round(
            views
            * rng.uniform(
                0.25,
                0.75,
                row_count,
            )
        ),
    )
    total_events = (
        views
        + carts
        + rng.poisson(
            1.2,
            row_count,
        )
    )

    frame = pd.DataFrame(
        {
            "total_events": total_events,
            "view_count": views,
            "addtocart_count": carts,
            "unique_items": unique_items,
            "activity_span_ms": rng.lognormal(
                mean=16.0,
                sigma=1.0,
                size=row_count,
            ),
            "cart_to_view_ratio": (
                carts
                / np.maximum(views, 1)
            ),
            "events_per_unique_item": (
                total_events
                / unique_items
            ),
        }
    )

    return frame[FEATURE_COLUMNS]


def build_bundle() -> FeatureHealthBundle:
    """Create a complete synthetic feature-health bundle."""

    frame = build_feature_frame()
    correlation = build_correlation_matrix(
        frame
    )

    return FeatureHealthBundle(
        feature_frame=frame,
        distribution_summary=build_distribution_summary(
            frame
        ),
        correlation_matrix=correlation,
        clustered_order=build_clustered_order(
            correlation
        ),
        validity_summary=build_validity_summary(
            frame
        ),
        source_rows=len(frame),
    )


def test_prepare_feature_frame_keeps_canonical_order() -> None:
    """Feature preparation should preserve the controlled order."""

    frame = build_feature_frame()
    prepared = prepare_feature_frame(frame)

    assert prepared.columns.tolist() == FEATURE_COLUMNS
    assert len(prepared) == len(frame)


def test_validity_summary_detects_rule_violations() -> None:
    """Missing, negative, and ratio-above-one values must be counted."""

    frame = build_feature_frame(20)
    frame.loc[0, "view_count"] = np.nan
    frame.loc[1, "total_events"] = -2
    frame.loc[2, "cart_to_view_ratio"] = 1.5

    summary = (
        build_validity_summary(frame)
        .set_index("feature")
    )

    assert summary.loc[
        "view_count",
        "missing_count",
    ] == 1
    assert summary.loc[
        "total_events",
        "negative_count",
    ] == 1
    assert summary.loc[
        "cart_to_view_ratio",
        "ratio_above_one_count",
    ] == 1


def test_clustered_order_contains_every_feature() -> None:
    """Correlation ordering should contain each feature exactly once."""

    frame = build_feature_frame()
    correlation = build_correlation_matrix(
        frame
    )
    order = build_clustered_order(
        correlation
    )

    assert set(order) == set(FEATURE_COLUMNS)
    assert len(order) == len(FEATURE_COLUMNS)


def test_generate_feature_health_visual_package(
    tmp_path: Path,
) -> None:
    """The complete package should create three clean visuals."""

    manifest = generate_feature_health_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    assert set(manifest["qa"]) == {
        "MLV-H01",
        "MLV-H02",
        "MLV-H04",
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
        "distribution_summary",
        "correlation_matrix",
        "validity_summary",
        "insights",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    insight_text = (
        tmp_path / support["insights"]
    ).read_text(encoding="utf-8")

    assert "do not assume correlation means" in insight_text
    assert "Zero values must be interpreted" in insight_text
