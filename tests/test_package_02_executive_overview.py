"""Focused tests for Package 2 Executive Overview Intelligence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ui.executive_intelligence import (
    FORECAST_CANDIDATES,
    anomaly_summary,
    available_composition_views,
    build_composition_figure,
    build_forecast_figure,
    build_funnel_table,
    build_selected_campaign_audience,
    calculate_scenario,
    controlling_holdout_row,
    decision_summary,
    executive_brief_table,
    readiness_table,
)


PAGE = Path("app/Executive_Overview.py")


def test_scenario_formulas_are_reconciled() -> None:
    """Campaign economics must use transparent, tested formulas."""

    result = calculate_scenario(
        available_targets=250,
        target_size=100,
        precision=0.30,
        baseline_rate=0.01,
        contact_cost=2.0,
        buyer_value=50.0,
    )

    assert result["target_size"] == 100
    assert result["expected_buyers"] == 30
    assert result["baseline_buyers"] == 1
    assert result["incremental_buyers"] == 29
    assert result["campaign_cost"] == 200
    assert result["expected_value"] == 1500
    assert result["net_value"] == 1300
    assert result["roi"] == 6.5


def test_scenario_target_cannot_exceed_eligible_audience() -> None:
    """The simulator must not invent more targets than are available."""

    result = calculate_scenario(
        available_targets=25,
        target_size=100,
        precision=0.20,
        baseline_rate=0.01,
        contact_cost=1.0,
        buyer_value=10.0,
    )

    assert result["target_size"] == 25


def test_funnel_has_required_stages_and_nonnegative_counts() -> None:
    """The Executive funnel must contain the five agreed stages."""

    funnel = build_funnel_table(
        source_visitors=1000,
        holdout_rows=500,
        threshold_eligible=50,
        target_size=25,
        expected_buyers=7.5,
    )

    assert funnel["Stage"].tolist() == [
        "Source visitors",
        "Validated holdout",
        "Threshold eligible",
        "Campaign target",
        "Expected buyers",
    ]
    assert (funnel["Visitors"] >= 0).all()


def test_controlling_holdout_uses_first_approved_row() -> None:
    """Executive metrics must come from one controlling holdout row."""

    holdout = pd.DataFrame(
        [
            {
                "model_name": "Tuned XGBoost",
                "threshold": 0.98,
                "rows": 1000,
                "positive_rows": 10,
                "positive_rate": 0.01,
                "predicted_positive_rows": 20,
                "predicted_positive_rate": 0.02,
                "pr_auc": 0.20,
                "roc_auc": 0.85,
                "precision": 0.30,
                "recall": 0.60,
                "f1_score": 0.40,
            }
        ]
    )

    result = controlling_holdout_row(
        holdout,
        fallback={
            "model_name": "Fallback",
            "threshold": 0,
            "rows": 0,
            "positive_rows": 0,
            "positive_rate": 0,
            "predicted_positive_rows": 0,
            "predicted_positive_rate": 0,
            "pr_auc": 0,
            "roc_auc": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        },
    )

    assert result["model_name"] == "Tuned XGBoost"
    assert result["threshold"] == 0.98
    assert result["precision"] == 0.30



def test_selected_composition_uses_only_ranked_campaign_rows() -> None:
    """Composition must never fall back to the complete projection sample."""

    scores = pd.DataFrame(
        {
            "visitorid": [1, 2, 3, 4, 5],
            "purchase_intent_score": [0.91, 0.99, 0.70, 0.95, 0.40],
            "predicted_conversion": [1, 1, 1, 1, 0],
            "production_threshold": [0.70] * 5,
        }
    )
    projection = pd.DataFrame(
        {
            "visitorid": [1, 2, 3, 99],
            "kmeans_cluster": [0, 1, 0, 7],
            "lof_status": ["Normal", "Outlier", "Normal", "Normal"],
        }
    )

    selected, message = build_selected_campaign_audience(
        projection,
        scores,
        target_size=2,
    )

    assert selected["visitorid"].tolist() == ["2", "4"]
    assert len(selected) == 2
    assert "2 of 2 requested" in message
    assert "Intent tier" in available_composition_views(selected)

    _, counts = build_composition_figure(selected, "Intent tier")
    assert int(counts["Visitors"].sum()) == 2
    assert int(counts["Visitors"].sum()) != len(projection)

    _, segment_counts = build_composition_figure(selected, "Segment")
    missing_label = "Segment not covered by projection evidence"
    assert int(segment_counts["Visitors"].sum()) == 2
    assert int(
        segment_counts.loc[
            segment_counts["Category"].eq(missing_label),
            "Visitors",
        ].iloc[0]
    ) == 1


def test_unsupported_audience_evidence_returns_honest_empty_state() -> None:
    """Missing score columns must not create a fake zero-visitor category."""

    selected, message = build_selected_campaign_audience(
        pd.DataFrame({"visitorid": [1], "kmeans_cluster": [0]}),
        pd.DataFrame({"visitorid": [1]}),
        target_size=1,
    )

    assert selected.empty
    assert available_composition_views(selected) == []
    assert "required ID or score columns" in message


def test_business_forecast_future_schema_uses_best_conversion_model() -> None:
    """The real multi-target forecast schema must produce one honest line."""

    assert any(
        path.name == "business_forecast_future.csv"
        for path in FORECAST_CANDIDATES
    )

    forecast = pd.DataFrame(
        {
            "date": [
                "2026-07-01",
                "2026-07-02",
                "2026-07-01",
                "2026-07-02",
                "2026-07-01",
                "2026-07-02",
            ],
            "target_name": [
                "converted_visitor_count",
                "converted_visitor_count",
                "converted_visitor_count",
                "converted_visitor_count",
                "event_volume",
                "event_volume",
            ],
            "model_name": [
                "Prophet",
                "Prophet",
                "ARIMA",
                "ARIMA",
                "Prophet",
                "Prophet",
            ],
            "predicted_value": [10.0, 12.0, 30.0, 40.0, 100.0, 120.0],
            "is_best_model": [True, True, False, False, True, True],
        }
    )

    figure, message = build_forecast_figure(forecast)

    assert figure is not None
    assert len(figure.data) == 1
    assert list(figure.data[0].y) == [10.0, 12.0]
    assert "Converted Visitor Count" in message
    assert "Prophet" in message


def test_delayed_label_readiness_accepts_real_implementation_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A delayed-label source or report artifact must mark Labels ready."""

    import app.ui.executive_intelligence as executive_module

    label_source = tmp_path / "delayed_labels.py"
    label_source.write_text("# delayed-label implementation", encoding="utf-8")
    monkeypatch.setattr(
        executive_module,
        "MONITORING_CANDIDATES",
        {"Labels": (label_source,)},
    )

    readiness = executive_module.readiness_table()

    assert readiness.to_dict("records") == [
        {
            "Area": "Labels",
            "Status": "Ready",
            "Evidence": str(label_source),
        }
    ]

def test_anomaly_summary_reconciles_flags() -> None:
    """Anomaly KPIs must reconcile with the evidence rows."""

    projection = pd.DataFrame(
        {
            "lof_outlier": [0, 1, 0, 1],
            "lof_score": [1.0, 2.5, 1.1, 4.0],
            "kmeans_cluster": [0, 1, 0, 1],
        }
    )

    summary = anomaly_summary(projection)

    assert summary["count"] == 2
    assert summary["rate"] == 0.5
    assert summary["key_segment"] == "1"


def test_decision_summary_contains_finding_action_and_limitation() -> None:
    """The executive decision summary must remain complete."""

    summary = decision_summary(
        target_size=100,
        expected_buyers=25,
        incremental_value=1000,
        roi=2.5,
        anomaly_rate=0.04,
        forecast_available=False,
    )

    assert summary["finding"]
    assert summary["action"]
    assert summary["limitation"]
    assert "forecast" in summary["limitation"].lower()


def test_exportable_brief_has_evidence_type() -> None:
    """Exported rows must distinguish validated evidence and assumptions."""

    brief = executive_brief_table(
        holdout={
            "model_name": "Model",
            "threshold": 0.9,
            "pr_auc": 0.2,
            "roc_auc": 0.8,
            "precision": 0.3,
            "recall": 0.4,
        },
        scenario={
            "target_size": 10,
            "expected_buyers": 3,
            "campaign_cost": 10,
            "expected_value": 30,
            "net_value": 20,
            "roi": 2,
        },
        anomaly={
            "count": 1,
            "rate": 0.1,
        },
        readiness=readiness_table(),
        source_timestamp="2026-06-22 12:00 UTC",
    )

    assert "Evidence type" in brief.columns
    assert "Scenario" in set(brief["Evidence type"])


def test_executive_sidebar_has_polished_owner_and_evidence_cards() -> None:
    """The sidebar must retain the portfolio owner and governed evidence."""

    text = PAGE.read_text(encoding="utf-8")

    required_sidebar_content = [
        "render_executive_sidebar(",
        "Ruturaj Mokashi",
        "Data Analyst",
        "eci-sidebar-brand-card",
        "eci-sidebar-owner-card",
        "eci-sidebar-snapshot-card",
        "eci-sidebar-evidence-card",
        "Approved evidence loaded",
    ]

    for required_text in required_sidebar_content:
        assert required_text in text

    assert "escape_text(model_name)" in text
    assert "escape_text(champion_track)" in text
    assert "escape_text(evidence_timestamp)" in text


def test_sidebar_html_stays_in_one_streamlit_markdown_block() -> None:
    """Blank lines must not turn later sidebar cards into visible HTML code."""

    text = PAGE.read_text(encoding="utf-8")

    assert "sidebar_html = sidebar_html.replace(" in text
    assert '"\\n\\n            <section"' in text
    assert '"\\n            <section"' in text
    assert "render_sidebar_html(sidebar_html)" in text


def test_page_maps_all_twelve_exec_requirements() -> None:
    """All controlling Executive Overview IDs must remain explicit."""

    text = PAGE.read_text(encoding="utf-8")

    for number in range(1, 13):
        assert f"SVE-EXEC-{number:02d}" in text


def test_page_has_required_visual_and_explanation_coverage() -> None:
    """Every major Package 2 visual must have an interpretation."""

    text = PAGE.read_text(encoding="utf-8")

    chart_count = text.count("st.plotly_chart")
    interpretation_count = text.count("render_interpretation(")

    assert chart_count >= 5
    assert interpretation_count >= chart_count


def test_page_has_scenario_links_and_export() -> None:
    """Scenario controls, investigation links, and export must be present."""

    text = PAGE.read_text(encoding="utf-8")

    assert "st.slider(" in text
    assert text.count("st.number_input(") >= 2
    assert text.count("st.page_link(") >= 5
    assert "show_table_with_download(" in text
    assert "build_selected_campaign_audience(" in text
    assert "build_composition_figure(" in text
    assert "selected_audience," in text


def test_executive_page_does_not_load_production_model_bundle() -> None:
    """The dashboard must render from evidence without loading the model."""

    text = PAGE.read_text(encoding="utf-8")

    forbidden_calls = [
        "get_champion_metrics(",
        "get_champion_model_name(",
        "get_best_threshold(",
        "get_champion_track(",
        "load_active_production_bundle(",
        "load_champion_metadata(",
    ]

    for forbidden in forbidden_calls:
        assert forbidden not in text

    assert "final_true_champion_holdout.csv" in text
    assert "champion_track" in text


def test_shared_component_calls_match_package_1_api() -> None:
    """Package 2 must use the existing shared component signatures."""

    import ast

    tree = ast.parse(PAGE.read_text(encoding="utf-8"))

    expected = {
        "render_source_note": {
            "source",
            "evidence_type",
            "refreshed_at",
            "extra",
        },
        "render_empty_state": {
            "title",
            "message",
            "next_action",
        },
        "show_table_with_download": {
            "label",
            "data",
            "file_name",
            "metadata",
            "description",
        },
    }

    found = {name: 0 for name in expected}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        function_name = node.func.id

        if function_name not in expected:
            continue

        found[function_name] += 1

        keywords = {
            keyword.arg
            for keyword in node.keywords
            if keyword.arg
        }

        assert expected[function_name].issubset(keywords)

    assert found["render_source_note"] == 3
    assert found["render_empty_state"] == 4
    assert found["show_table_with_download"] == 1


def test_all_executive_chart_builders_return_figures() -> None:
    """All Package 2 chart builders must match the shared Plotly API."""

    from app.ui.executive_intelligence import (
        build_anomaly_figure,
        build_composition_figure,
        build_efficiency_figure,
        build_forecast_figure,
        build_funnel_figure,
        build_funnel_table,
        build_readiness_figure,
    )

    funnel = build_funnel_table(
        source_visitors=1000,
        holdout_rows=200,
        threshold_eligible=40,
        target_size=20,
        expected_buyers=5,
    )

    assert build_funnel_figure(funnel) is not None

    assert build_efficiency_figure(
        source_visitors=1000,
        target_size=20,
        expected_buyers=5,
        baseline_rate=0.01,
    ) is not None

    projection = pd.DataFrame(
        {
            "kmeans_cluster": [0, 1, 0, 1],
            "lof_outlier": [0, 1, 0, 1],
            "lof_score": [1.0, 2.5, 1.1, 3.0],
        }
    )

    composition, counts = build_composition_figure(
        projection,
        "Segment",
    )

    assert composition is not None
    assert int(counts["Visitors"].sum()) == 4
    assert build_anomaly_figure(projection) is not None

    forecast = pd.DataFrame(
        {
            "date": [
                "2026-06-01",
                "2026-06-02",
            ],
            "forecast": [10.0, 12.0],
            "lower": [8.0, 9.0],
            "upper": [12.0, 15.0],
        }
    )

    forecast_figure, message = build_forecast_figure(forecast)

    assert forecast_figure is not None
    assert "2" in message

    readiness = pd.DataFrame(
        {
            "Area": [
                "Application",
                "Alerts",
            ],
            "Status": [
                "Ready",
                "Evidence unavailable",
            ],
            "Evidence": [
                "app",
                "none",
            ],
        }
    )

    assert build_readiness_figure(readiness) is not None
