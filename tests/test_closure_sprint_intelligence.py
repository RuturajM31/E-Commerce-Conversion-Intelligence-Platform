"""Focused tests for the coordinated Packages 3-8 closure sprint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.ui.architecture_intelligence import (
    build_business_flow_figure,
    build_ml_lifecycle_figure,
    component_inventory,
)
from app.ui.campaign_intelligence import (
    apply_campaign_filters,
    batch_diagnostics,
    build_score_histogram,
    build_threshold_chart,
    build_threshold_scenarios,
)
from app.ui.customer_segmentation_journey import (
    build_cluster_profile_figure,
    build_journey_funnel_figure,
    build_pca_figure,
    build_quality_figure,
    build_tier_funnel_figure,
    cluster_personas,
    journey_funnel,
)
from app.ui.model_decision_intelligence import (
    build_business_scenarios,
    build_confusion_matrix_figure,
    build_generalisation_figure,
    build_model_comparison_figure,
    build_precision_recall_frontier,
    build_stability_figure,
    build_threshold_tradeoff,
)
from app.ui.operations_intelligence import (
    anomaly_action,
    build_drift_figure,
    build_forecast_outlook_figure,
    build_residual_figure,
    forecast_error_metrics,
    parse_value_drift,
    prepare_forecast_target,
)
from app.ui.visitor_similarity_explainability import (
    build_feature_comparison_figure,
    build_neighbor_score_figure,
    get_neighbours,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
VISUALS = ROOT / "reports" / "visuals" / "ml_visual_intelligence"


def test_batch_filters_and_scenarios_are_reconciled() -> None:
    """Campaign controls must cap capacity and use saved threshold evidence."""

    scored = pd.DataFrame(
        {
            "visitorid": ["a", "b", "c", "d"],
            "purchase_intent_score": [0.90, 0.70, 0.40, 0.20],
            "intent_segment": ["High", "Strong", "Warm", "Low"],
            "total_events": [10, 8, 4, 2],
            "view_count": [7, 6, 3, 2],
            "addtocart_count": [2, 1, 0, 0],
            "unique_items": [5, 4, 3, 2],
            "activity_span_ms": [100, 80, 40, 20],
        }
    )

    diagnostics = batch_diagnostics(scored)
    assert diagnostics["rows"] == 4
    assert diagnostics["missing_cells"] == 0

    campaign = apply_campaign_filters(
        scored,
        threshold=0.50,
        capacity=1,
        intent_tiers=["High", "Strong"],
    )
    assert campaign["visitorid"].tolist() == ["a"]

    thresholds = pd.read_csv(TABLES / "final_true_champion_thresholds.csv")
    scenarios = build_threshold_scenarios(
        scored,
        thresholds,
        [0.50, 0.80],
        positive_rows=10,
    )
    assert len(scenarios) == 2
    assert build_score_histogram(scored, 0.50) is not None
    assert build_threshold_chart(scenarios) is not None


def test_model_decision_figures_use_saved_evidence() -> None:
    """Model decision charts must build from the approved report tables."""

    comparison = pd.read_csv(TABLES / "final_true_champion_comparison.csv")
    holdout = pd.read_csv(TABLES / "final_true_champion_holdout.csv")
    thresholds = pd.read_csv(TABLES / "final_true_champion_thresholds.csv")
    stability = pd.read_csv(TABLES / "final_true_champion_stability.csv")
    confusion = pd.read_csv(
        VISUALS / "threshold_decisions" / "confusion_matrix_counts.csv"
    )
    generalisation = pd.read_csv(
        VISUALS
        / "champion_selection"
        / "champion_generalisation_metrics.csv"
    )

    assert build_model_comparison_figure(comparison) is not None
    assert build_precision_recall_frontier(thresholds) is not None
    assert build_threshold_tradeoff(thresholds) is not None
    assert build_confusion_matrix_figure(confusion) is not None
    assert build_generalisation_figure(generalisation) is not None
    assert build_stability_figure(stability) is not None

    scenarios = build_business_scenarios(
        thresholds,
        holdout_rows=int(holdout.iloc[0]["rows"]),
        positive_rows=int(holdout.iloc[0]["positive_rows"]),
        contact_cost=1.0,
        buyer_value=50.0,
    )
    assert len(scenarios) == len(thresholds)
    assert "Scenario ROI" in scenarios.columns


def test_neighbor_search_and_comparison_are_deterministic() -> None:
    """KNN evidence must return exact visitors and requested neighbour count."""

    feature_names = ("total_events", "view_count")
    data = pd.DataFrame(
        {
            "visitorid": ["a", "b", "c"],
            "total_events": [1, 2, 10],
            "view_count": [1, 2, 9],
            "purchase_intent_score": [0.1, 0.2, 0.9],
        }
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[list(feature_names)])
    index = NearestNeighbors().fit(scaled)
    resource = {
        "data": data,
        "scaled": scaled,
        "index": index,
        "feature_names": feature_names,
    }

    selected, neighbours, comparison = get_neighbours(
        resource,
        visitor_id="a",
        neighbor_count=2,
    )

    assert selected["visitorid"] == "a"
    assert len(neighbours) == 2
    assert len(comparison) == 2
    assert build_neighbor_score_figure(neighbours) is not None
    assert build_feature_comparison_figure(comparison) is not None


def test_segmentation_and_journey_evidence_builds() -> None:
    """Segment maps, profiles, quality, and journey funnels must build."""

    projection = pd.read_csv(TABLES / "ml_validation_projection_sample.csv")
    summary = pd.read_csv(TABLES / "ml_validation_cluster_business_summary.csv")
    profile = pd.read_csv(TABLES / "ml_validation_cluster_profile.csv")
    kmeans = pd.read_csv(TABLES / "ml_validation_kmeans_grid.csv")

    clusters = sorted(
        projection["kmeans_cluster"].astype(int).unique().tolist()
    )
    figure, filtered = build_pca_figure(
        projection,
        clusters=clusters,
        anomaly_only=False,
        three_d=False,
    )

    assert figure is not None
    assert len(filtered) == len(projection)
    assert len(cluster_personas(summary)) == len(summary)
    assert build_cluster_profile_figure(profile, clusters) is not None
    assert build_quality_figure(kmeans) is not None

    journey = pd.DataFrame(
        {
            "visitorid": ["a", "b", "c"],
            "Viewed": [True, True, True],
            "Added to cart": [True, False, True],
            "Converted": [True, False, False],
            "Behavior tier": ["Cart", "Browse", "Cart"],
            "view_count": [2, 3, 4],
            "addtocart_count": [1, 0, 2],
        }
    )
    funnel = journey_funnel(journey)
    assert int(funnel.iloc[-1]["Rows"]) == 1
    assert build_journey_funnel_figure(funnel) is not None
    assert build_tier_funnel_figure(journey) is not None


def test_forecast_anomaly_and_drift_helpers_build() -> None:
    """Operations helpers must use saved forecast and Evidently evidence."""

    history = pd.read_csv(TABLES / "business_forecast_history_with_predictions.csv")
    future = pd.read_csv(TABLES / "business_forecast_future.csv")
    target = str(history.iloc[0]["target_name"])

    filtered_history, filtered_future = prepare_forecast_target(
        history,
        future,
        target,
        horizon=10,
    )
    assert len(filtered_future) <= 10
    assert build_forecast_outlook_figure(
        filtered_history,
        filtered_future,
        target=target,
        scenario_factor=1.0,
    ) is not None
    assert build_residual_figure(filtered_history) is not None
    assert forecast_error_metrics(filtered_history)["rows"] >= 0

    anomalies = pd.read_csv(TABLES / "top_anomalous_visitors.csv")
    assert isinstance(anomaly_action(anomalies.iloc[0]), str)

    report = pd.read_json(
        ROOT / "reports" / "evidently" / "latest" / "feature_drift_report.json",
        typ="series",
    ).to_dict()
    drift = parse_value_drift(report)
    assert len(drift) == 7
    assert build_drift_figure(drift) is not None


def test_architecture_inventory_and_flows_build() -> None:
    """Architecture inventory and relationship maps must be exportable."""

    inventory = component_inventory()
    assert len(inventory) >= 10
    assert {"Layer", "Component", "Status", "Evidence"}.issubset(
        inventory.columns
    )
    assert build_business_flow_figure() is not None
    assert build_ml_lifecycle_figure() is not None


def test_all_pages_call_their_closure_extensions() -> None:
    """Packages 3-8 must remain wired into the Streamlit navigation pages."""

    required_calls = {
        "app/pages/1_Visitor_Intent_Predictor.py": "render_visitor_similarity_explainability()",
        "app/pages/2_Batch_Scoring.py": "render_batch_campaign_intelligence(",
        "app/pages/3_Model_Benchmark_Selection.py": "render_model_decision_intelligence()",
        "app/pages/4_Business_KPI_Forecasting.py": "render_forecast_decision_intelligence()",
        "app/pages/5_Anomaly_Outlier.py": "render_anomaly_investigation_intelligence()",
        "app/pages/6_Monitoring_Drift_Health.py": "render_monitoring_health_intelligence()",
        "app/pages/7_MLOps_Architecture.py": "render_architecture_governance_intelligence()",
        "app/pages/9_Customer_Segmentation_Journey.py": "render_segmentation_and_journey_page()",
    }

    for relative, call in required_calls.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert call in text


def test_matrix_still_contains_all_204_unique_rows() -> None:
    """The authoritative matrix must retain every unique requirement ID."""

    matrix = pd.read_csv(
        ROOT
        / "docs"
        / "streamlit"
        / "STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.csv"
    )
    assert len(matrix) == 204
    assert matrix["ID"].nunique() == 204
