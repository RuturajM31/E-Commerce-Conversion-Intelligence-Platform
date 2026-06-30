"""Regression tests for Package 1 visual and component quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.ui.components import render_evidence_chart, render_interpretation
from app.ui.design_system import product_css
from app.ui.ml_validation_visuals import (
    build_dbscan_grid_figure,
    build_kmeans_quality_figure,
    build_lof_score_figure,
    build_projection_figure,
)


def test_interpretation_html_is_not_rendered_as_markdown_code(monkeypatch) -> None:
    """The rendered component must not expose raw HTML tags."""

    calls: list[tuple[str, bool]] = []

    def capture_markdown(body: str, unsafe_allow_html: bool = False) -> None:
        calls.append((body, unsafe_allow_html))

    monkeypatch.setattr("app.ui.components.st.markdown", capture_markdown)

    render_interpretation(
        what_it_shows="A",
        how_to_read="B",
        actual_finding="C",
        conclusion="D",
        recommended_action="E",
        limitation="F",
    )

    assert len(calls) == 1
    body, unsafe = calls[0]
    assert unsafe is True
    assert body.startswith('<section class="eci-interpretation"')
    assert body.count('<article class="eci-interpretation-item">') == 6
    assert "\n        <div" not in body


def test_render_evidence_chart_enforces_chart_interpretation_and_source(monkeypatch) -> None:
    """A major chart must always render explanation and provenance together."""

    calls: list[str] = []
    monkeypatch.setattr(
        "app.ui.components.st.plotly_chart",
        lambda *args, **kwargs: calls.append("chart"),
    )
    monkeypatch.setattr(
        "app.ui.components.render_interpretation",
        lambda **kwargs: calls.append("interpretation"),
    )
    monkeypatch.setattr(
        "app.ui.components.render_source_note",
        lambda **kwargs: calls.append("source"),
    )

    rendered = render_evidence_chart(
        figure=object(),
        key="test_chart",
        what_it_shows="Shows",
        how_to_read="Read",
        actual_finding="Finding",
        conclusion="Conclusion",
        recommended_action="Action",
        limitation="Limitation",
        source="source.csv",
        evidence_type="offline evidence",
        refreshed_at="2026-06-22",
    )

    assert rendered is True
    assert calls == ["chart", "interpretation", "source"]


def test_shared_css_has_stable_responsive_grids() -> None:
    """Product CSS should prevent awkward KPI and explanation wrapping."""

    css = product_css()
    assert ".eci-kpi-grid" in css
    assert "grid-template-columns: repeat(4" in css
    assert ".eci-detail-grid" in css
    assert "grid-template-columns: repeat(3" in css
    assert ".eci-interpretation" in css


def sample_projection(rows: int = 420) -> pd.DataFrame:
    """Create deterministic projection evidence for figure tests."""

    rng = np.random.default_rng(42)
    clusters = np.arange(rows) % 14

    return pd.DataFrame(
        {
            "PC1": rng.normal(size=rows),
            "PC2": rng.normal(size=rows),
            "PC3": rng.normal(size=rows),
            "kmeans_cluster": clusters % 5,
            "dbscan_cluster": np.where(np.arange(rows) % 17 == 0, -1, clusters),
            "dbscan_point_type": np.where(
                np.arange(rows) % 17 == 0,
                "Noise",
                np.where(np.arange(rows) % 7 == 0, "Border", "Core"),
            ),
            "lof_severity": np.where(
                np.arange(rows) % 29 == 0,
                "Critical",
                "Normal",
            ),
            "lof_score": np.where(
                np.arange(rows) % 29 == 0,
                250.0,
                rng.uniform(1.0, 3.0, size=rows),
            ),
            "lof_status": np.where(
                np.arange(rows) % 29 == 0,
                "Outlier",
                "Normal",
            ),
        }
    )


def test_dbscan_projection_keeps_legend_readable() -> None:
    """Many raw DBSCAN clusters should collapse into a usable legend."""

    figure = build_projection_figure(
        sample_projection(),
        color_column="dbscan_cluster",
        dimension="3D",
        symbol_column="dbscan_point_type",
    )

    assert figure is not None
    assert len(figure.data) <= 10
    assert figure.layout.height == 760


def test_kmeans_projection_marks_centroids() -> None:
    """K-Means projection should expose one prominent centroid trace."""

    figure = build_projection_figure(
        sample_projection(),
        color_column="kmeans_cluster",
        dimension="3D",
    )

    assert figure is not None
    centroid = [trace for trace in figure.data if trace.name == "Cluster centroids"]
    assert len(centroid) == 1
    assert centroid[0].marker.size >= 10


def test_kmeans_quality_highlights_selected_candidate() -> None:
    """The selected same-sample candidate should be explicitly annotated."""

    summary = pd.DataFrame(
        {
            "k": list(range(1, 11)),
            "inertia": np.linspace(2500, 900, 10),
            "silhouette_score": [np.nan, 0.61, 0.76, 0.72, 0.86, 0.75, 0.69, 0.66, 0.63, 0.60],
        }
    )

    figure = build_kmeans_quality_figure(summary, selected_k=5)
    assert figure is not None
    annotation_text = [annotation.text for annotation in figure.layout.annotations]
    assert any("k=5" in str(text) for text in annotation_text)


def test_lof_distribution_uses_log_readability_and_threshold() -> None:
    """Extreme LOF scores should stay readable and show the active threshold."""

    figure = build_lof_score_figure(sample_projection(), threshold=2.5)
    assert figure is not None
    assert "log scale" in str(figure.layout.xaxis.title.text).lower()
    assert any("threshold" in str(annotation.text).lower() for annotation in figure.layout.annotations)


def test_dbscan_grid_has_two_diagnostics_and_selected_marker() -> None:
    """The complete 3x3 grid should show quality, noise, and selected settings."""

    rows = []
    for eps in (0.5, 1.0, 2.0):
        for min_samples in (3, 5, 10):
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "silhouette_score_no_noise": 0.40 + eps / 10,
                    "noise_rate": min(0.8, 0.35 / eps + min_samples / 100),
                    "cluster_count": max(2, int(12 / eps) - min_samples // 3),
                }
            )
    figure = build_dbscan_grid_figure(
        pd.DataFrame(rows),
        selected_eps=1.0,
        selected_min_samples=5,
    )
    assert figure is not None
    assert len(figure.data) == 4
    assert any(trace.name == "Selected parameters" for trace in figure.data)


def test_package_01_page_uses_shared_product_components() -> None:
    """The page should use the stable shared layout and evidence renderer."""

    text = Path("app/pages/8_ML_Validation_Evidence.py").read_text(encoding="utf-8")
    assert "render_kpi_grid(" in text
    assert "render_evidence_chart(" in text
    assert "render_detail_cards(" in text
    assert "kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)" not in text
