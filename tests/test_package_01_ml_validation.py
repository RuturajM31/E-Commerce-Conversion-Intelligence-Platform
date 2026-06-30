"""Focused calculation and figure tests for Package 1."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from app.ui.design_system import (
    CHART_CONFIG,
    CLUSTER_PALETTE,
    COLORS,
    product_css,
)
from app.ui.ml_validation_visuals import (
    build_cluster_profile_heatmap,
    build_cluster_value_figure,
    build_dbscan_grid_figure,
    build_dbscan_k_distance_figure,
    build_kmeans_quality_figure,
    build_loading_heatmap,
    build_lof_deviation_figure,
    build_lof_feature_figure,
    build_lof_score_figure,
    build_parallel_profile_figure,
    build_pca_density_figure,
    build_projection_figure,
    build_scree_figure,
)
import src.visualization.build_ml_validation_evidence as evidence_builder
from src.visualization.build_ml_validation_evidence import (
    DBSCAN_EPS_GRID,
    DBSCAN_MIN_SAMPLES_GRID,
    KMEANS_GRID,
    build_outputs,
    evaluate_dbscan_grid,
    evaluate_kmeans_grid,
    prepare_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_PAGE = PROJECT_ROOT / "app/pages/8_ML_Validation_Evidence.py"
OLD_PAGE = PROJECT_ROOT / "app/pages/8_MVD_Coverage_Proof.py"


def synthetic_visitors(rows: int = 180) -> pd.DataFrame:
    """Create deterministic visitor-level evidence with three behaviours."""

    rng = np.random.default_rng(42)
    group = np.arange(rows) % 3

    total_events = np.where(
        group == 0,
        rng.poisson(5, rows),
        np.where(group == 1, rng.poisson(24, rows), rng.poisson(65, rows)),
    ) + 1
    view_share = np.where(group == 0, 0.92, np.where(group == 1, 0.72, 0.48))
    cart_share = np.where(group == 0, 0.02, np.where(group == 1, 0.12, 0.31))

    view_count = np.minimum(
        total_events,
        np.maximum(0, np.rint(total_events * view_share).astype(int)),
    )
    addtocart_count = np.minimum(
        total_events,
        np.maximum(0, np.rint(total_events * cart_share).astype(int)),
    )

    data = pd.DataFrame(
        {
            "visitorid": np.arange(10_000, 10_000 + rows),
            "total_events": total_events,
            "view_count": view_count,
            "addtocart_count": addtocart_count,
            "unique_items": np.maximum(
                1,
                np.rint(total_events * np.where(group == 2, 0.55, 0.32)),
            ).astype(int),
            "activity_span_ms": (
                np.where(group == 0, 8, np.where(group == 1, 45, 160))
                * 60_000
                + rng.integers(0, 20_000, rows)
            ),
            "purchase_intent_score": np.clip(
                np.where(group == 0, 0.08, np.where(group == 1, 0.36, 0.77))
                + rng.normal(0, 0.03, rows),
                0,
                1,
            ),
            "purchased": np.where(group == 2, np.arange(rows) % 4 == 0, False),
            "visitor_segment": np.where(
                group == 0,
                "Browser",
                np.where(group == 1, "Engaged", "High intent"),
            ),
        }
    )
    return data


@pytest.fixture(scope="module")
def coherent_outputs() -> dict[str, object]:
    """Build the complete deterministic evidence once for the test module."""

    with threadpool_limits(limits=1):
        return build_outputs(synthetic_visitors(120))


def test_shared_design_tokens_are_complete() -> None:
    """The shared product system must expose stable semantic tokens."""

    required_colors = {
        "app_bg",
        "surface_1",
        "surface_2",
        "border",
        "text",
        "text_muted",
        "brand",
        "positive",
        "warning",
        "critical",
    }
    assert required_colors.issubset(COLORS)
    assert len(CLUSTER_PALETTE) >= 8
    assert CHART_CONFIG["displaylogo"] is False
    css = product_css()
    assert "eci-page-hero" in css
    assert "eci-interpretation" in css
    assert "eci-detail-grid" in css


def test_new_page_name_replaces_old_user_facing_name() -> None:
    """The professional page identity must replace the old MVD label."""

    assert NEW_PAGE.is_file()
    assert not OLD_PAGE.exists()

    source = NEW_PAGE.read_text(encoding="utf-8")
    assert "Machine Learning Validation & Evidence" in source
    assert "MVD Coverage Proof" not in source
    ast.parse(source)


def test_feature_preparation_is_stable_and_finite() -> None:
    """Derived unsupervised features should be finite and deterministic."""

    prepared = prepare_feature_frame(synthetic_visitors(30))

    assert prepared.shape == (30, 7)
    assert np.isfinite(prepared.to_numpy()).all()
    assert prepared.columns.tolist() == [
        "Log total events",
        "Log views",
        "Log add-to-cart",
        "Log unique items",
        "Log activity hours",
        "View share",
        "Cart share",
    ]


def test_same_sample_parameter_grids_are_complete() -> None:
    """K-Means and DBSCAN must use the exact promised grids."""

    features = prepare_feature_frame(synthetic_visitors(180))
    scaled = (features - features.mean()) / features.std(ddof=0)
    scaled_values = scaled.fillna(0).to_numpy()

    with threadpool_limits(limits=1):
        kmeans_grid = evaluate_kmeans_grid(scaled_values)
    assert kmeans_grid["k"].tolist() == list(KMEANS_GRID)
    assert len(kmeans_grid) == 10
    assert set(kmeans_grid.loc[kmeans_grid["candidate_scope"] == "required_3_5_7", "k"]) == {3, 5, 7}
    assert kmeans_grid["sample_rows"].nunique() == 1

    with threadpool_limits(limits=1):
        dbscan_grid = evaluate_dbscan_grid(scaled_values)
    assert len(dbscan_grid) == 9
    assert set(dbscan_grid["eps"].tolist()) == set(DBSCAN_EPS_GRID)
    assert set(dbscan_grid["min_samples"].tolist()) == set(DBSCAN_MIN_SAMPLES_GRID)
    assert set(
        zip(dbscan_grid["eps"], dbscan_grid["min_samples"], strict=True)
    ) == {
        (eps, min_samples)
        for eps in DBSCAN_EPS_GRID
        for min_samples in DBSCAN_MIN_SAMPLES_GRID
    }
    assert dbscan_grid["sample_rows"].nunique() == 1


def test_build_outputs_uses_one_coherent_sample_and_complete_evidence(
    coherent_outputs: dict[str, object],
) -> None:
    """Every Package 1 method must be calculated from the same visitor rows."""

    sample = synthetic_visitors(120)
    outputs = coherent_outputs

    expected = {
        "projection",
        "variance",
        "loadings",
        "cluster_profile",
        "cluster_business",
        "kmeans_grid",
        "dbscan_grid",
        "dbscan_k_distance",
        "lof_feature_profile",
        "lof_investigation",
        "method_summary",
        "parameters",
        "features",
    }
    assert expected.issubset(outputs)

    assert len(outputs["projection"]) == len(sample)
    assert len(outputs["kmeans_grid"]) == 10
    assert len(outputs["dbscan_grid"]) == 9
    assert outputs["kmeans_grid"]["sample_rows"].eq(len(sample)).all()
    assert outputs["dbscan_grid"]["sample_rows"].eq(len(sample)).all()
    assert outputs["method_summary"]["sample_rows"].eq(len(sample)).all()
    assert outputs["kmeans_grid"]["selected"].sum() == 1
    assert outputs["dbscan_grid"]["selected"].sum() == 1
    assert not outputs["cluster_business"].empty
    assert not outputs["lof_feature_profile"].empty
    assert not outputs["lof_investigation"].empty

    projection_columns = set(outputs["projection"].columns)
    assert {"PC1", "PC2", "PC3"}.issubset(projection_columns)
    assert {
        "kmeans_cluster",
        "dbscan_cluster",
        "dbscan_point_type",
        "lof_score",
        "lof_outlier",
        "lof_severity",
    }.issubset(projection_columns)


def test_save_outputs_writes_every_governed_contract(
    coherent_outputs: dict[str, object],
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The CLI save step must write all tables and a complete manifest."""

    table_dir = tmp_path / "reports/tables"
    metadata_dir = tmp_path / "reports/metadata"
    source_path = tmp_path / "data/processed/visitor_features.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("visitorid,total_events\n1,1\n", encoding="utf-8")

    paths = {
        "TABLE_DIR": table_dir,
        "METADATA_DIR": metadata_dir,
        "FEATURES_PATH": source_path,
        "PROJECTION_PATH": table_dir / "ml_validation_projection_sample.csv",
        "PCA_VARIANCE_PATH": table_dir / "ml_validation_pca_variance.csv",
        "PCA_LOADINGS_PATH": table_dir / "ml_validation_pca_loadings.csv",
        "CLUSTER_PROFILE_PATH": table_dir / "ml_validation_cluster_profile.csv",
        "CLUSTER_BUSINESS_PATH": table_dir / "ml_validation_cluster_business_summary.csv",
        "KMEANS_GRID_PATH": table_dir / "ml_validation_kmeans_grid.csv",
        "DBSCAN_GRID_PATH": table_dir / "ml_validation_dbscan_grid.csv",
        "DBSCAN_K_DISTANCE_PATH": table_dir / "ml_validation_dbscan_k_distance.csv",
        "LOF_FEATURE_PROFILE_PATH": table_dir / "ml_validation_lof_feature_profile.csv",
        "LOF_INVESTIGATION_PATH": table_dir / "ml_validation_lof_investigation.csv",
        "METHOD_SUMMARY_PATH": table_dir / "ml_validation_method_summary.csv",
        "MANIFEST_PATH": metadata_dir / "ml_validation_manifest.json",
    }
    for name, value in paths.items():
        monkeypatch.setattr(evidence_builder, name, value)

    evidence_builder.save_outputs(
        coherent_outputs,
        source_rows=120,
        sample_rows=120,
    )

    expected_files = [
        value
        for name, value in paths.items()
        if name.endswith("_PATH") and name not in {"FEATURES_PATH", "LOF_SUMMARY_PATH"}
    ]
    assert all(path.exists() for path in expected_files)

    import json

    manifest = json.loads(paths["MANIFEST_PATH"].read_text(encoding="utf-8"))
    assert manifest["sample_rows"] == 120
    assert manifest["outputs"]["pca_variance"].endswith(
        "ml_validation_pca_variance.csv"
    )
    assert manifest["outputs"]["pca_loadings"].endswith(
        "ml_validation_pca_loadings.csv"
    )


def test_all_package_01_figures_build_from_verified_tables(
    coherent_outputs: dict[str, object],
) -> None:
    """Every agreed Package 1 visual must build from precomputed evidence."""

    outputs = coherent_outputs
    projection = outputs["projection"]
    variance = outputs["variance"]
    loadings = outputs["loadings"]
    profile = outputs["cluster_profile"]
    business = outputs["cluster_business"]
    kmeans_grid = outputs["kmeans_grid"]
    dbscan_grid = outputs["dbscan_grid"]
    k_distance = outputs["dbscan_k_distance"]
    lof_profile = outputs["lof_feature_profile"]
    investigation = outputs["lof_investigation"]
    params = outputs["parameters"]

    projection_3d = build_projection_figure(
        projection,
        color_column="kmeans_cluster",
        dimension="3D",
    )
    projection_2d = build_projection_figure(
        projection,
        color_column="kmeans_cluster",
        dimension="2D",
    )
    dbscan_projection = build_projection_figure(
        projection,
        color_column="dbscan_cluster",
        dimension="3D",
        symbol_column="dbscan_point_type",
    )
    lof_projection = build_projection_figure(
        projection,
        color_column="lof_severity",
        dimension="3D",
        size_column="lof_score",
    )

    assert projection_3d is not None
    assert projection_3d.layout.height == 760
    assert projection_2d is not None
    assert projection_2d.layout.height == 640
    assert dbscan_projection is not None
    assert lof_projection is not None
    assert any(trace.name == "Cluster centroids" for trace in projection_3d.data)

    figures = [
        build_scree_figure(variance),
        build_loading_heatmap(loadings),
        build_pca_density_figure(projection),
        build_kmeans_quality_figure(kmeans_grid, selected_k=params["kmeans_k"]),
        build_cluster_profile_heatmap(profile),
        build_parallel_profile_figure(profile),
        build_cluster_value_figure(
            business,
            value_column="purchase_rate",
            value_label="Observed purchase rate",
            value_is_rate=True,
        ),
        build_dbscan_k_distance_figure(
            k_distance,
            selected_eps=params["dbscan_eps"],
        ),
        build_dbscan_grid_figure(
            dbscan_grid,
            selected_eps=params["dbscan_eps"],
            selected_min_samples=params["dbscan_min_samples"],
        ),
        build_lof_score_figure(
            projection,
            threshold=params["lof_score_threshold"],
        ),
        build_lof_deviation_figure(lof_profile),
        build_lof_feature_figure(investigation.iloc[0]),
    ]

    assert all(figure is not None for figure in figures)
