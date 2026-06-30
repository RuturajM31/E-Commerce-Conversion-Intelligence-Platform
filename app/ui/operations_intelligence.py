"""Forecast, anomaly, and monitoring intelligence extensions.

These renderers use existing project artifacts and do not retrain models. They
add scenario controls, diagnostics, investigation detail, provenance, and
exports while preserving the current pages and approved dark theme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui.closure_core import (
    add_metadata_columns,
    compact_count,
    export_metadata,
    file_timestamp,
    finish_figure,
    percent,
    read_csv_if_exists,
    read_json_if_exists,
    safe_divide,
    source_hash,
)
from app.ui.components import (
    inject_product_css,
    render_detail_cards,
    render_empty_state,
    render_evidence_chart,
    render_kpi_grid,
    render_section_header,
    render_source_note,
    show_table_with_download,
)


FORECAST_HISTORY_PATH = Path("reports/tables/business_forecast_history_with_predictions.csv")
FORECAST_FUTURE_PATH = Path("reports/tables/business_forecast_future.csv")
FORECAST_COMPARISON_PATH = Path("reports/tables/business_forecast_comparison.csv")
DAILY_KPI_PATH = Path("reports/tables/daily_business_kpis.csv")

ANOMALY_VISITOR_PATH = Path("reports/tables/top_anomalous_visitors.csv")
ANOMALY_SEGMENT_PATH = Path("reports/tables/anomaly_segment_summary.csv")
ANOMALY_RULE_PATH = Path("reports/tables/anomaly_rule_summary.csv")
ANOMALY_SUMMARY_PATH = Path("reports/tables/anomaly_summary.csv")

FEATURE_DRIFT_PATH = Path("reports/evidently/latest/feature_drift_report.json")
PREDICTION_DRIFT_PATH = Path("reports/evidently/latest/prediction_drift_report.json")
MONITORING_STATUS_PATH = Path(
    "reports/visuals/ml_visual_intelligence/monitoring/monitoring_source_status.csv"
)
DELAYED_FUNNEL_PATH = Path(
    "reports/visuals/ml_visual_intelligence/monitoring/delayed_label_funnel.csv"
)
DELAYED_REJECTIONS_PATH = Path(
    "reports/visuals/ml_visual_intelligence/monitoring/delayed_label_rejections.csv"
)
MODEL_METADATA_PATH = Path("models/metadata/final_champion_metadata.json")
SCORE_MANIFEST_PATH = Path("models/metadata/final_champion_score_manifest.json")
MLFLOW_LINEAGE_PATH = Path("models/metadata/mlflow_champion_lineage.json")
REGISTRY_PATH = Path(
    "reports/visuals/ml_visual_intelligence/experiment_tracking/"
    "verified_ecommerce_registry_versions.csv"
)


def _normalise_forecast_frame(
    frame: pd.DataFrame,
    *,
    default_target: str = "forecast",
) -> pd.DataFrame:
    """Return a forecast frame with one stable canonical schema.

    Historical project artifacts used several names for observed and predicted
    values. This normaliser preserves genuine data, maps recognised aliases,
    and creates optional columns as missing values instead of raising a
    KeyError.
    """

    output = frame.copy()

    aliases = {
        "date": (
            "timestamp",
            "ds",
            "forecast_date",
        ),
        "target_name": (
            "target",
            "metric",
            "kpi_name",
        ),
        "actual_value": (
            "actual",
            "observed_value",
            "y",
        ),
        "predicted_value": (
            "prediction",
            "forecast_value",
            "yhat",
        ),
        "model_name": (
            "model",
            "forecast_model",
        ),
    }

    for canonical, candidates in aliases.items():
        if canonical in output.columns:
            continue

        for candidate in candidates:
            if candidate in output.columns:
                output[canonical] = output[candidate]
                break

    if "date" not in output.columns:
        output["date"] = pd.NaT

    if "target_name" not in output.columns:
        output["target_name"] = default_target

    if "actual_value" not in output.columns:
        output["actual_value"] = np.nan

    if "predicted_value" not in output.columns:
        output["predicted_value"] = np.nan

    if "model_name" not in output.columns:
        output["model_name"] = "Saved forecast model"

    if "is_best_model" not in output.columns:
        output["is_best_model"] = True

    output["date"] = pd.to_datetime(
        output["date"],
        errors="coerce",
    )
    output["actual_value"] = pd.to_numeric(
        output["actual_value"],
        errors="coerce",
    )
    output["predicted_value"] = pd.to_numeric(
        output["predicted_value"],
        errors="coerce",
    )

    return output

def prepare_forecast_target(
    history: pd.DataFrame,
    future: pd.DataFrame,
    target: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter one KPI and horizon after normalising forecast schemas."""

    history_normalised = _normalise_forecast_frame(
        history,
        default_target=target,
    )
    future_normalised = _normalise_forecast_frame(
        future,
        default_target=target,
    )

    history_target = history_normalised.loc[
        history_normalised["target_name"]
        .astype(str)
        .eq(str(target))
    ].copy()
    future_target = future_normalised.loc[
        future_normalised["target_name"]
        .astype(str)
        .eq(str(target))
    ].copy()

    for frame_name, frame in (
        ("history", history_target),
        ("future", future_target),
    ):
        best_mask = (
            frame["is_best_model"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

        if not best_mask.any():
            continue

        if frame_name == "history":
            history_target = frame.loc[best_mask].copy()
        else:
            future_target = frame.loc[best_mask].copy()

    history_target = (
        history_target
        .dropna(subset=["date"])
        .sort_values("date")
    )
    future_target = (
        future_target
        .dropna(subset=["date"])
        .sort_values("date")
        .head(max(int(horizon), 0))
    )

    return history_target, future_target

def empirical_forecast_band(history: pd.DataFrame) -> float:
    """Return a transparent empirical absolute-error band."""

    frame = _normalise_forecast_frame(history)
    valid = frame.dropna(
        subset=["actual_value", "predicted_value"]
    )

    if valid.empty:
        return 0.0

    residual = (
        valid["actual_value"]
        - valid["predicted_value"]
    )

    return float(
        residual.abs().quantile(0.80)
    )

def build_forecast_outlook_figure(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    target: str,
    scenario_factor: float,
) -> go.Figure:
    """Build actual, backtest, and future traces from canonical columns."""

    history_frame = _normalise_forecast_frame(
        history,
        default_target=target,
    )
    future_frame = _normalise_forecast_frame(
        future,
        default_target=target,
    )

    figure = go.Figure()

    history_actual = history_frame.dropna(
        subset=["actual_value"]
    )
    history_predicted = history_frame.dropna(
        subset=["predicted_value"]
    )
    future_valid = future_frame.dropna(
        subset=["predicted_value"]
    ).copy()

    future_valid["scenario_value"] = (
        future_valid["predicted_value"]
        * float(scenario_factor)
    )

    band = empirical_forecast_band(history_frame)
    future_valid["lower"] = (
        future_valid["scenario_value"] - band
    ).clip(lower=0)
    future_valid["upper"] = (
        future_valid["scenario_value"] + band
    )

    figure.add_trace(
        go.Scatter(
            x=history_actual["date"],
            y=history_actual["actual_value"],
            mode="lines",
            name="Historical actual",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history_predicted["date"],
            y=history_predicted["predicted_value"],
            mode="lines",
            name="Backtest prediction",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_valid["date"],
            y=future_valid["scenario_value"],
            mode="lines+markers",
            name="Future scenario",
        )
    )

    if not future_valid.empty:
        figure.add_trace(
            go.Scatter(
                x=(
                    list(future_valid["date"])
                    + list(future_valid["date"][::-1])
                ),
                y=(
                    list(future_valid["upper"])
                    + list(future_valid["lower"][::-1])
                ),
                fill="toself",
                fillcolor="rgba(56,189,248,0.14)",
                line={"color": "rgba(255,255,255,0)"},
                name="Empirical 80% error band",
                hoverinfo="skip",
            )
        )

    figure.update_xaxes(title="Date")
    figure.update_yaxes(
        title=target.replace("_", " ").title()
    )

    return finish_figure(
        figure,
        title=f"{target.replace('_', ' ').title()} outlook",
        subtitle=(
            "Historical actuals, backtest predictions, future scenario, "
            "and empirical error band are separated."
        ),
        height=570,
        legend_title="Evidence",
    )

def build_residual_figure(history: pd.DataFrame) -> go.Figure:
    """Show historical forecast residuals without assuming optional columns."""

    frame = _normalise_forecast_frame(history)
    frame = frame.dropna(
        subset=["actual_value", "predicted_value"]
    ).copy()
    frame["Residual"] = (
        frame["actual_value"]
        - frame["predicted_value"]
    )

    figure = px.scatter(
        frame,
        x="date",
        y="Residual",
        color=(
            "model_name"
            if "model_name" in frame.columns
            else None
        ),
        hover_data={
            "actual_value": True,
            "predicted_value": True,
        },
    )
    figure.add_hline(y=0, line_dash="dash")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(
        title="Actual minus predicted"
    )

    return finish_figure(
        figure,
        title="Backtest residuals over time",
        subtitle=(
            "Large positive values are under-forecasts and large "
            "negative values are over-forecasts."
        ),
        height=470,
        legend_title="Model",
    )

def forecast_error_metrics(
    history: pd.DataFrame,
) -> dict[str, float]:
    """Calculate error metrics after canonical forecast normalisation."""

    frame = _normalise_forecast_frame(history)
    frame = frame.dropna(
        subset=["actual_value", "predicted_value"]
    ).copy()

    if frame.empty:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "rows": 0,
        }

    residual = (
        frame["actual_value"]
        - frame["predicted_value"]
    )
    non_zero = (
        frame["actual_value"].abs() > 1e-12
    )
    mape = (
        (
            residual[non_zero].abs()
            / frame.loc[
                non_zero,
                "actual_value",
            ].abs()
        ).mean()
        if non_zero.any()
        else 0.0
    )

    return {
        "mae": float(residual.abs().mean()),
        "rmse": float(
            np.sqrt((residual ** 2).mean())
        ),
        "mape": float(mape),
        "rows": int(len(frame)),
    }

def render_forecast_decision_intelligence() -> None:
    """Render the final forecast scenario and diagnostics layer."""

    inject_product_css()

    render_section_header(
        "Planning intelligence",
        "Choose a KPI, horizon, scenario, and inspect backtest error",
        "Scenario scaling is explicit and never overwrites the saved forecast artifact.",
    )

    history = read_csv_if_exists(str(FORECAST_HISTORY_PATH))
    future = read_csv_if_exists(str(FORECAST_FUTURE_PATH))
    comparison = read_csv_if_exists(str(FORECAST_COMPARISON_PATH))

    if history.empty or future.empty:
        render_empty_state(
            "Forecast evidence unavailable",
            "Historical prediction or future forecast tables are missing.",
            "Regenerate the approved business forecast artifacts.",
        )
        return

    targets = sorted(set(history["target_name"]).intersection(future["target_name"]))
    if not targets:
        render_empty_state(
            "No common forecast KPI",
            "Historical and future tables do not share a supported target name.",
            "Reconcile the forecast artifact schema.",
        )
        return

    control_1, control_2, control_3 = st.columns(3)
    with control_1:
        target = st.selectbox("Forecast KPI", targets, key="closure_forecast_target")
    with control_2:
        max_horizon = max(int((future["target_name"] == target).sum()), 1)
        horizon = st.slider(
            "Future periods",
            min_value=1,
            max_value=max_horizon,
            value=min(28, max_horizon),
            key="closure_forecast_horizon",
        )
    with control_3:
        scenario = st.selectbox(
            "Business scenario",
            ["Conservative", "Baseline", "Optimistic"],
            index=1,
            key="closure_forecast_scenario",
        )

    factor = {"Conservative": 0.90, "Baseline": 1.00, "Optimistic": 1.10}[scenario]
    history_target, future_target = prepare_forecast_target(
        history,
        future,
        target,
        horizon,
    )
    metrics = forecast_error_metrics(history_target)
    band = empirical_forecast_band(history_target)

    baseline_sum = float(pd.to_numeric(future_target["predicted_value"], errors="coerce").sum())
    scenario_sum = baseline_sum * factor

    render_kpi_grid(
        [
            ("Backtest rows", compact_count(metrics["rows"]), "Saved actual-versus-predicted periods"),
            ("MAE", f"{metrics['mae']:,.2f}", "Average absolute forecast error"),
            ("RMSE", f"{metrics['rmse']:,.2f}", "Larger errors receive more weight"),
            ("Scenario total", f"{scenario_sum:,.1f}", f"{scenario} factor {factor:.0%} across {horizon} periods"),
        ]
    )

    outlook = build_forecast_outlook_figure(
        history_target,
        future_target,
        target=target,
        scenario_factor=factor,
    )
    render_evidence_chart(
        figure=outlook,
        key="closure_forecast_outlook",
        what_it_shows="Historical actuals, backtest predictions, and a selected future planning scenario.",
        how_to_read="The future line begins after the historical boundary; the shaded band uses the 80th percentile absolute backtest error.",
        actual_finding=f"The {scenario.lower()} {horizon}-period total is {scenario_sum:,.1f}; the empirical error band is ±{band:,.1f} per period.",
        conclusion="Capacity planning should use both the central scenario and historical error range.",
        recommended_action="Plan to the baseline and hold contingency for the conservative range when error widens.",
        limitation="The scenario factor is an assumption and the empirical band is not a formal probabilistic interval.",
        source=f"{FORECAST_HISTORY_PATH}; {FORECAST_FUTURE_PATH}",
        evidence_type="Saved historical backtest plus explicit future scenario",
        refreshed_at=file_timestamp(FORECAST_FUTURE_PATH),
    )

    residual = build_residual_figure(history_target)
    worst_row = history_target.dropna(subset=["actual_value", "predicted_value"]).assign(
        absolute_error=lambda frame: (frame["actual_value"] - frame["predicted_value"]).abs()
    ).sort_values("absolute_error", ascending=False).head(1)
    worst_text = (
        f"The largest saved absolute error is {float(worst_row.iloc[0]['absolute_error']):,.1f}."
        if not worst_row.empty
        else "No usable residual rows are available."
    )
    render_evidence_chart(
        figure=residual,
        key="closure_forecast_residuals",
        what_it_shows="Backtest error for each historical period.",
        how_to_read="Points above zero were under-forecast; points below zero were over-forecast.",
        actual_finding=worst_text,
        conclusion="Residual concentration and outliers reveal whether operational risk is occasional or systematic.",
        recommended_action="Investigate the largest error dates before relying on the same horizon for staffing or campaign timing.",
        limitation="Residuals explain past backtests and cannot guarantee future error size.",
        source=str(FORECAST_HISTORY_PATH),
        evidence_type="Historical forecast backtest diagnostics",
        refreshed_at=file_timestamp(FORECAST_HISTORY_PATH),
    )

    scenario_export = future_target.copy()
    scenario_export["scenario"] = scenario
    scenario_export["scenario_factor"] = factor
    scenario_export["scenario_value"] = pd.to_numeric(
        scenario_export["predicted_value"], errors="coerce"
    ) * factor
    scenario_export["empirical_error_band"] = band
    scenario_export["scenario_lower"] = (
        scenario_export["scenario_value"] - band
    ).clip(lower=0)
    scenario_export["scenario_upper"] = scenario_export["scenario_value"] + band
    scenario_export = add_metadata_columns(
        scenario_export,
        export_metadata(
            source=str(FORECAST_FUTURE_PATH),
            evidence_type="Filtered forecast with explicit scenario assumptions",
            filters={"target": target, "horizon": horizon, "scenario": scenario},
        ),
    )
    st.download_button(
        "Download filtered forecast scenario",
        data=scenario_export.to_csv(index=False),
        file_name="filtered_forecast_scenario.csv",
        mime="text/csv",
        use_container_width=True,
        key="closure_forecast_export",
    )

    if not comparison.empty:
        show_table_with_download(
            label="forecast model comparison",
            data=comparison.loc[comparison["target_name"].eq(target)],
            file_name="forecast_model_comparison.csv",
            metadata=export_metadata(
                source=str(FORECAST_COMPARISON_PATH),
                evidence_type="Saved forecast model backtest comparison",
                filters={"target": target},
            ),
        )

    render_source_note(
        source="No approved feature-decomposition artifact",
        evidence_type="Conditional forecast explanation",
        refreshed_at=file_timestamp(FORECAST_FUTURE_PATH),
        extra="Forecast driver decomposition remains conditional and is not inferred from correlation alone.",
    )


def anomaly_action(row: pd.Series) -> str:
    """Map verified anomaly and intent evidence to a review action."""

    flagged = bool(row.get("final_anomaly_flag", False))
    risk = float(row.get("anomaly_risk_score", 0) or 0)
    intent = float(row.get("purchase_intent_score", 0) or 0)

    if flagged and risk >= 0.8:
        return "Suppress automated activation and investigate first."
    if flagged and intent >= 0.7:
        return "Manual review before using a high-value campaign treatment."
    if flagged:
        return "Monitor or review; do not treat the flag as proof of fraud."
    if intent >= 0.7:
        return "Campaign-safe high-intent candidate under current evidence."
    return "No immediate anomaly action; retain normal campaign rules."


def render_anomaly_investigation_intelligence() -> None:
    """Render selected-visitor anomaly evidence and governed export."""

    inject_product_css()

    render_section_header(
        "Investigation intelligence",
        "Filter anomalies, inspect evidence, compare baselines, and assign review action",
        "Anomaly flags identify unusual behaviour for review; they do not prove fraud or data error.",
    )

    visitors = read_csv_if_exists(str(ANOMALY_VISITOR_PATH))
    segments = read_csv_if_exists(str(ANOMALY_SEGMENT_PATH))
    rules = read_csv_if_exists(str(ANOMALY_RULE_PATH))

    if visitors.empty:
        render_empty_state(
            "Anomaly visitor evidence unavailable",
            "The ranked anomaly investigation table is missing.",
            "Regenerate anomaly outputs before investigation.",
        )
        return

    risk_bands = sorted(visitors["anomaly_risk_band"].dropna().astype(str).unique().tolist())
    intent_segments = sorted(visitors["purchase_intent_segment"].dropna().astype(str).unique().tolist())

    control_1, control_2, control_3 = st.columns(3)
    with control_1:
        selected_risk = st.multiselect(
            "Risk bands",
            risk_bands,
            default=risk_bands,
            key="closure_anomaly_risk_filter",
        )
    with control_2:
        selected_intent = st.multiselect(
            "Intent segments",
            intent_segments,
            default=intent_segments,
            key="closure_anomaly_intent_filter",
        )
    with control_3:
        flagged_only = st.checkbox(
            "Final anomaly flags only",
            value=True,
            key="closure_anomaly_flag_filter",
        )

    filtered = visitors.loc[
        visitors["anomaly_risk_band"].astype(str).isin(selected_risk)
        & visitors["purchase_intent_segment"].astype(str).isin(selected_intent)
    ].copy()
    if flagged_only:
        filtered = filtered.loc[
            filtered["final_anomaly_flag"].fillna(False).astype(bool)
        ]

    filtered["recommended_review_action"] = filtered.apply(anomaly_action, axis=1)
    flagged_rate = safe_divide(
        visitors["final_anomaly_flag"].fillna(False).astype(bool).sum(),
        len(visitors),
    )

    render_kpi_grid(
        [
            ("Visible investigation rows", compact_count(len(filtered)), "Rows after current filters"),
            ("Source flagged rate", percent(flagged_rate), "Final flags in the ranked source table"),
            ("Risk bands", compact_count(len(selected_risk)), "Selected severity groups"),
            ("Intent segments", compact_count(len(selected_intent)), "Selected campaign-intent groups"),
        ]
    )

    if filtered.empty:
        render_empty_state(
            "No anomalies match the current filters",
            "The selected risk, intent, and final-flag controls returned no rows.",
            "Broaden one or more filters.",
        )
        return

    selected_id = st.selectbox(
        "Inspect anomalous visitor",
        filtered["visitorid"].astype(str).tolist(),
        key="closure_anomaly_visitor",
    )
    selected = filtered.loc[filtered["visitorid"].astype(str).eq(selected_id)].iloc[0]
    segment_name = str(selected.get("purchase_intent_segment", "Unavailable"))
    segment_row = segments.loc[
        segments["purchase_intent_segment"].astype(str).eq(segment_name)
    ] if not segments.empty and "purchase_intent_segment" in segments.columns else pd.DataFrame()

    segment_anomaly_rate = (
        float(segment_row.iloc[0].get("anomaly_rate", 0))
        if not segment_row.empty
        else 0.0
    )
    render_detail_cards(
        [
            ("Visitor", selected_id, f"Intent score {float(selected.get('purchase_intent_score', 0)):.1%}"),
            ("Anomaly risk", f"{float(selected.get('anomaly_risk_score', 0)):.1%}", str(selected.get("anomaly_risk_band", "Unavailable"))),
            ("Segment baseline", percent(segment_anomaly_rate), f"Anomaly rate for {segment_name}"),
            ("Recommended review", "Rules-based", anomaly_action(selected)),
        ]
    )

    rule_columns = [column for column in filtered.columns if column.startswith("rule_")]
    rule_rows = []
    for column in rule_columns:
        rule_rows.append(
            {
                "Rule": column.replace("rule_", "").replace("_", " ").title(),
                "Triggered": bool(selected.get(column, False)),
            }
        )
    rule_frame = pd.DataFrame(rule_rows)
    if not rule_frame.empty:
        rule_figure = px.bar(
            rule_frame,
            x="Rule",
            y=rule_frame["Triggered"].astype(int),
            color="Triggered",
            text="Triggered",
        )
        rule_figure.update_yaxes(range=[0, 1.05], tickvals=[0, 1], ticktext=["No", "Yes"])
        rule_figure = finish_figure(
            rule_figure,
            title="Selected visitor rule evidence",
            subtitle="Triggered rules explain the rule-based part of the anomaly decision.",
            height=460,
            legend_title="Triggered",
        )
        triggered = rule_frame.loc[rule_frame["Triggered"], "Rule"].tolist()
        render_evidence_chart(
            figure=rule_figure,
            key="closure_anomaly_rules",
            what_it_shows="Which saved anomaly rules triggered for the selected visitor.",
            how_to_read="A Yes bar means the visitor crossed that documented rule threshold.",
            actual_finding=(
                f"{len(triggered)} rules triggered: {', '.join(triggered) if triggered else 'none'}."
            ),
            conclusion="Rule evidence makes the final flag reviewable instead of a hidden binary label.",
            recommended_action=anomaly_action(selected),
            limitation="Rules identify extremes relative to the saved population and do not prove malicious behaviour.",
            source=str(ANOMALY_VISITOR_PATH),
            evidence_type="Saved visitor-level anomaly rules",
            refreshed_at=file_timestamp(ANOMALY_VISITOR_PATH),
        )

    comparison_columns = [
        column
        for column in (
            "total_events",
            "view_count",
            "addtocart_count",
            "unique_items",
            "activity_span_ms",
            "cart_to_view_ratio",
            "events_per_unique_item",
        )
        if column in filtered.columns
    ]
    normal_population = visitors.loc[
        ~visitors["final_anomaly_flag"].fillna(False).astype(bool)
    ]
    comparison = pd.DataFrame(
        {
            "Feature": comparison_columns,
            "Selected visitor": [float(selected[column]) for column in comparison_columns],
            "Normal median": [
                float(pd.to_numeric(normal_population[column], errors="coerce").median())
                for column in comparison_columns
            ],
        }
    )
    comparison_melted = comparison.melt(
        id_vars="Feature",
        var_name="Population",
        value_name="Value",
    )
    comparison_figure = px.bar(
        comparison_melted,
        x="Feature",
        y="Value",
        color="Population",
        barmode="group",
    )
    comparison_figure = finish_figure(
        comparison_figure,
        title="Selected visitor versus normal baseline",
        subtitle="Raw feature values are compared with the median non-flagged visitor.",
        height=500,
        legend_title="Evidence",
    )
    render_evidence_chart(
        figure=comparison_figure,
        key="closure_anomaly_baseline",
        what_it_shows="Selected visitor behaviour against the normal-population median.",
        how_to_read="Large gaps identify the strongest behavioural extremes for investigation.",
        actual_finding=f"The selected visitor has anomaly risk {float(selected.get('anomaly_risk_score', 0)):.1%}.",
        conclusion="Baseline comparison identifies what is unusual without claiming why it happened.",
        recommended_action=anomaly_action(selected),
        limitation="Raw scales differ across features; use the chart for directional investigation rather than feature ranking.",
        source=str(ANOMALY_VISITOR_PATH),
        evidence_type="Saved anomaly table and normal-population baseline",
        refreshed_at=file_timestamp(ANOMALY_VISITOR_PATH),
    )

    export_columns = [
        column
        for column in (
            "visitorid",
            "purchase_intent_score",
            "purchase_intent_segment",
            "anomaly_risk_score",
            "anomaly_risk_band",
            "final_anomaly_flag",
            "rule_anomaly_count",
            "business_interpretation",
            "recommended_review_action",
        )
        if column in filtered.columns
    ] + rule_columns
    export = add_metadata_columns(
        filtered[export_columns],
        export_metadata(
            source=str(ANOMALY_VISITOR_PATH),
            evidence_type="Filtered anomaly investigation evidence",
            filters={
                "risk_bands": selected_risk,
                "intent_segments": selected_intent,
                "flagged_only": flagged_only,
            },
        ),
    )
    st.download_button(
        "Download filtered anomaly investigation table",
        data=export.to_csv(index=False),
        file_name="filtered_anomaly_investigation.csv",
        mime="text/csv",
        use_container_width=True,
        key="closure_anomaly_export",
    )

    render_source_note(
        source="No approved anomaly snapshot history or event timeline artifact",
        evidence_type="Conditional anomaly time evidence",
        refreshed_at=file_timestamp(ANOMALY_VISITOR_PATH),
        extra="Anomaly volume over time and exact visitor event timelines are not claimed.",
    )


def parse_value_drift(report: dict[str, Any]) -> pd.DataFrame:
    """Extract Evidently ValueDrift metrics from the saved JSON report."""

    rows = []
    for metric in report.get("metrics", []):
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        value = metric.get("value")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        threshold = float(config.get("threshold", 0.1))
        rows.append(
            {
                "Feature": config.get("column", "Unknown"),
                "Drift score": score,
                "Threshold": threshold,
                "Alert state": "Drift" if score >= threshold else "Stable",
                "Method": config.get("method", "Unknown"),
            }
        )
    return pd.DataFrame(rows)


def build_drift_figure(drift: pd.DataFrame) -> go.Figure:
    """Build a current feature and prediction drift comparison."""

    figure = px.bar(
        drift.sort_values("Drift score", ascending=True),
        x="Drift score",
        y="Feature",
        orientation="h",
        color="Alert state",
        text="Drift score",
    )
    figure.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    if not drift.empty:
        figure.add_vline(
            x=float(drift["Threshold"].max()),
            line_dash="dash",
            annotation_text="Alert threshold",
        )
    return finish_figure(
        figure,
        title="Current feature and prediction drift",
        subtitle="Saved Evidently PSI scores are compared with the configured alert threshold.",
        height=max(430, 48 * len(drift)),
        legend_title="State",
    )


def render_monitoring_health_intelligence() -> None:
    """Render drift, delayed-label, lineage, and operational health evidence."""

    inject_product_css()

    render_section_header(
        "Operational intelligence",
        "Current drift, delayed labels, lineage, alerts, and evidence freshness",
        "Snapshot evidence is kept separate from genuine matured production performance.",
    )

    feature_report = read_json_if_exists(str(FEATURE_DRIFT_PATH))
    prediction_report = read_json_if_exists(str(PREDICTION_DRIFT_PATH))
    feature_drift = parse_value_drift(feature_report)
    prediction_drift = parse_value_drift(prediction_report)
    drift = pd.concat([feature_drift, prediction_drift], ignore_index=True)

    source_status = read_csv_if_exists(str(MONITORING_STATUS_PATH))
    delayed_funnel = read_csv_if_exists(str(DELAYED_FUNNEL_PATH))
    delayed_rejections = read_csv_if_exists(str(DELAYED_REJECTIONS_PATH))
    registry = read_csv_if_exists(str(REGISTRY_PATH))
    metadata = read_json_if_exists(str(MODEL_METADATA_PATH))
    score_manifest = read_json_if_exists(str(SCORE_MANIFEST_PATH))
    lineage = read_json_if_exists(str(MLFLOW_LINEAGE_PATH))

    drifted = int(drift["Alert state"].eq("Drift").sum()) if not drift.empty else 0
    ready_sources = int(source_status["exists"].astype(str).str.lower().isin(["true", "1"]).sum()) if not source_status.empty and "exists" in source_status.columns else 0
    matured_count = int(delayed_funnel.iloc[-1]["count"]) if not delayed_funnel.empty and "count" in delayed_funnel.columns else 0

    render_kpi_grid(
        [
            ("Drift alerts", compact_count(drifted), "Saved Evidently features above threshold"),
            ("Monitoring sources", f"{ready_sources}/{len(source_status)}" if len(source_status) else "0/0", "Existing files in the source-status audit"),
            ("Evaluated matured cohort", compact_count(matured_count), "Final delayed-label funnel stage"),
            ("Verified registry rows", compact_count(len(registry)), "Genuine ecommerce registry evidence only"),
        ]
    )

    if drift.empty:
        render_empty_state(
            "Drift JSON evidence unavailable",
            "No compatible ValueDrift metrics were found in the saved Evidently reports.",
            "Regenerate the latest feature and prediction drift reports.",
        )
    else:
        drift_figure = build_drift_figure(drift)
        highest = drift.sort_values("Drift score", ascending=False).iloc[0]
        render_evidence_chart(
            figure=drift_figure,
            key="closure_monitoring_drift",
            what_it_shows="Current feature and prediction PSI drift scores against the configured threshold.",
            how_to_read="Bars crossing the dashed line require investigation; lower bars are stable under the saved rule.",
            actual_finding=f"The highest current drift score is {highest['Drift score']:.4f} for {highest['Feature']}; {drifted} alerts are active.",
            conclusion="Current saved drift evidence is healthy only when every required feature remains below threshold.",
            recommended_action="Regenerate reports on the scheduled cadence and investigate any feature crossing the alert line.",
            limitation="This is the latest snapshot, not a drift-history time series.",
            source=f"{FEATURE_DRIFT_PATH}; {PREDICTION_DRIFT_PATH}",
            evidence_type="Latest Evidently feature and prediction drift snapshot",
            refreshed_at=max(file_timestamp(FEATURE_DRIFT_PATH), file_timestamp(PREDICTION_DRIFT_PATH)),
        )
        show_table_with_download(
            label="current drift evidence",
            data=drift,
            file_name="current_drift_evidence.csv",
            metadata=export_metadata(
                source=f"{FEATURE_DRIFT_PATH}; {PREDICTION_DRIFT_PATH}",
                evidence_type="Latest Evidently drift snapshot",
            ),
        )

    if not delayed_funnel.empty:
        funnel_figure = go.Figure(
            go.Funnel(
                y=delayed_funnel["stage"],
                x=delayed_funnel["count"],
                textinfo="value+percent initial",
            )
        )
        funnel_figure = finish_figure(
            funnel_figure,
            title="Delayed-label maturity funnel",
            subtitle="Prediction ledger rows narrow to valid matured-label evaluation evidence.",
            height=500,
        )
        render_evidence_chart(
            figure=funnel_figure,
            key="closure_delayed_label_funnel",
            what_it_shows="Ledger rows progressing through eligibility, label receipt, valid joins, and evaluated cohort.",
            how_to_read="Every lower stage must be a valid subset of the prior stage.",
            actual_finding=f"The final evaluated matured cohort contains {matured_count:,} rows.",
            conclusion="Production performance is valid only after labels mature and identity joins pass quality checks.",
            recommended_action="Keep production metrics blocked when the evaluated cohort is empty or join conflicts exist.",
            limitation="A non-empty funnel does not by itself prove the sample is large enough for stable performance estimates.",
            source=str(DELAYED_FUNNEL_PATH),
            evidence_type="Delayed-label evaluation pipeline evidence",
            refreshed_at=file_timestamp(DELAYED_FUNNEL_PATH),
        )

    if not delayed_rejections.empty:
        show_table_with_download(
            label="delayed-label rejection reasons",
            data=delayed_rejections,
            file_name="delayed_label_rejections.csv",
            metadata=export_metadata(
                source=str(DELAYED_REJECTIONS_PATH),
                evidence_type="Delayed-label data-quality evidence",
            ),
        )

    lineage_cards = [
        (
            "Model artifact",
            str(metadata.get("final_model_name") or metadata.get("model_name") or "Unavailable"),
            f"Hash {source_hash(Path('models/trained/final_champion_model.joblib'))}",
        ),
        (
            "Model metadata",
            str(metadata.get("environment", "Unavailable")),
            f"Hash {source_hash(MODEL_METADATA_PATH)}",
        ),
        (
            "Score manifest",
            str(score_manifest.get("model_generation", "Unavailable")),
            f"Rows {score_manifest.get('row_count', 'Unavailable')} · Hash {source_hash(SCORE_MANIFEST_PATH)}",
        ),
        (
            "MLflow lineage",
            str(lineage.get("run_id", lineage.get("model_name", "Unavailable"))),
            f"Hash {source_hash(MLFLOW_LINEAGE_PATH)}",
        ),
    ]
    render_detail_cards(lineage_cards)

    if not source_status.empty:
        show_table_with_download(
            label="monitoring source health",
            data=source_status,
            file_name="monitoring_source_health.csv",
            metadata=export_metadata(
                source=str(MONITORING_STATUS_PATH),
                evidence_type="Monitoring source presence and freshness",
            ),
        )

    if not registry.empty:
        show_table_with_download(
            label="verified ecommerce registry evidence",
            data=registry,
            file_name="verified_ecommerce_registry_evidence.csv",
            metadata=export_metadata(
                source=str(REGISTRY_PATH),
                evidence_type="Verified ecommerce MLflow registry evidence",
            ),
        )

    current_state = "Healthy snapshot" if drifted == 0 else "Drift investigation required"
    render_detail_cards(
        [
            ("Current health", current_state, f"{drifted} saved drift alerts"),
            ("Required action", "Regenerate and review", "Use the existing monitoring runner and alert rules on schedule"),
            ("Production limitation", "Matured labels only", "Do not relabel offline holdout metrics as live production performance"),
        ]
    )

    render_source_note(
        source="No multi-snapshot drift-history table was found",
        evidence_type="Conditional monitoring history",
        refreshed_at=file_timestamp(FEATURE_DRIFT_PATH),
        extra="Scoring volume history, feature drift history, and prediction drift history remain conditional until scheduled snapshots are stored longitudinally.",
    )
