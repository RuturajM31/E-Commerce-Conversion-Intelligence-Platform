"""Static coverage test for Package 1 charts and explanations."""

from __future__ import annotations

import ast
from pathlib import Path


PAGE = Path("app/pages/8_ML_Validation_Evidence.py")
EXPECTED_KEYS = {
    "mlv_model_comparison",
    "mlv_coverage",
    "mlv_pca_projection",
    "mlv_pca_density",
    "mlv_pca_scree",
    "mlv_pca_loadings",
    "mlv_kmeans_projection",
    "mlv_kmeans_quality",
    "mlv_kmeans_profile",
    "mlv_kmeans_parallel",
    "mlv_kmeans_value",
    "mlv_dbscan_projection",
    "mlv_dbscan_k_distance",
    "mlv_dbscan_grid",
    "mlv_lof_projection",
    "mlv_lof_distribution",
    "mlv_lof_deviation",
    "mlv_lof_selected_features",
    "mlv_outlier_comparison",
}
REQUIRED_EVIDENCE_ARGUMENTS = {
    "figure",
    "key",
    "what_it_shows",
    "how_to_read",
    "actual_finding",
    "conclusion",
    "recommended_action",
    "limitation",
    "source",
    "evidence_type",
    "refreshed_at",
}


def test_every_major_chart_uses_the_governed_evidence_component() -> None:
    """No page chart may bypass interpretation or provenance."""

    source = PAGE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls: list[ast.Call] = []
    keys: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "render_evidence_chart":
            continue

        calls.append(node)
        keyword_names = {item.arg for item in node.keywords if item.arg}
        assert REQUIRED_EVIDENCE_ARGUMENTS.issubset(keyword_names)

        key_node = next(
            item.value for item in node.keywords if item.arg == "key"
        )
        assert isinstance(key_node, ast.Constant)
        keys.add(str(key_node.value))

    assert len(calls) == len(EXPECTED_KEYS)
    assert keys == EXPECTED_KEYS
    assert "st.plotly_chart(" not in source
