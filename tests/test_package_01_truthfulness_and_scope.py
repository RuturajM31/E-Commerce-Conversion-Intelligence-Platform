"""Truthfulness, scope, and evidence-contract tests for Package 1."""

from __future__ import annotations

import ast
from pathlib import Path

from src.visualization.build_ml_validation_evidence import (
    DBSCAN_EPS_GRID,
    DBSCAN_MIN_SAMPLES_GRID,
    KMEANS_DECISION_CANDIDATES,
    KMEANS_GRID,
)


PAGE = Path("app/pages/8_ML_Validation_Evidence.py")
BUILDER = Path("src/visualization/build_ml_validation_evidence.py")


def test_champion_metrics_prefer_the_controlling_comparison_row() -> None:
    """Top metrics must not silently come from a conflicting summary artifact."""

    source = PAGE.read_text(encoding="utf-8")
    assert 'tables.get("final_true_champion_comparison"' in source
    assert "champion_row = select_model_row(final_comparison, champion_name)" in source
    assert "metric_row = champion_row if not champion_row.empty else summary_row" in source
    assert '"Controlling comparison row"' in source
    assert '"Explicit alternate-summary fallback"' in source
    assert "Metrics from the same champion row shown" not in source
    assert "Production threshold" not in source
    assert "production threshold" not in source
    assert "Active threshold" in source


def test_same_sample_and_historical_evidence_are_explicitly_separated() -> None:
    """Historical course tables must not drive same-sample method selection."""

    source = PAGE.read_text(encoding="utf-8")
    required_phrases = {
        "Same-sample K-Means k=1-10 grid",
        "Same-sample DBSCAN 3x3 grid",
        "Historical K-Means course summary",
        "Historical DBSCAN course summary",
        "Historical LOF summary",
        "not mixed into the same-sample",
    }
    assert required_phrases.issubset(set(phrase for phrase in required_phrases if phrase in source))


def test_builder_constants_match_the_agreed_assignment_scope() -> None:
    """The controlling calculation grids must exactly match the agreed scope."""

    assert KMEANS_GRID == tuple(range(1, 11))
    assert KMEANS_DECISION_CANDIDATES == (3, 5, 7)
    assert DBSCAN_EPS_GRID == (0.5, 1.0, 2.0)
    assert DBSCAN_MIN_SAMPLES_GRID == (3, 5, 10)


def test_page_contains_every_agreed_package_01_visual_family() -> None:
    """PCA, K-Means, DBSCAN, and LOF scope must be complete in code."""

    source = PAGE.read_text(encoding="utf-8")
    required_builders = {
        "build_pca_density_figure",
        "build_scree_figure",
        "build_loading_heatmap",
        "build_parallel_profile_figure",
        "build_cluster_value_figure",
        "build_dbscan_k_distance_figure",
        "build_dbscan_grid_figure",
        "build_lof_deviation_figure",
        "build_lof_feature_figure",
    }
    for builder in required_builders:
        assert builder in source

    required_evidence_tables = {
        "ml_validation_cluster_business_summary.csv",
        "ml_validation_dbscan_k_distance.csv",
        "ml_validation_lof_feature_profile.csv",
        "ml_validation_lof_investigation.csv",
    }
    builder_source = BUILDER.read_text(encoding="utf-8")
    for table in required_evidence_tables:
        assert table in builder_source


def test_page_and_builder_are_valid_python() -> None:
    """The final correction must remain syntactically valid."""

    ast.parse(PAGE.read_text(encoding="utf-8"))
    ast.parse(BUILDER.read_text(encoding="utf-8"))
