"""Campaign decision extension for the Batch Scoring page.

The existing Batch Scoring flow remains intact. This module adds the missing
campaign-capacity, threshold-comparison, filtering, row-detail, and governed
export layers required for the final portfolio closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    render_active_filter_summary,
    safe_divide,
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


THRESHOLD_PATH = Path("reports/tables/final_true_champion_thresholds.csv")
HOLDOUT_PATH = Path("reports/tables/final_true_champion_holdout.csv")
ANOMALY_PATH = Path("reports/tables/top_anomalous_visitors.csv")


def batch_diagnostics(scored: pd.DataFrame) -> dict[str, int]:
    """Return transparent row and quality diagnostics for one scored batch."""

    duplicate_ids = 0
    if "visitorid" in scored.columns:
        duplicate_ids = int(scored["visitorid"].astype(str).duplicated().sum())

    required = [
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "purchase_intent_score",
    ]
    present = [column for column in required if column in scored.columns]
    missing_cells = int(scored[present].isna().sum().sum()) if present else 0

    return {
        "rows": int(len(scored)),
        "duplicate_ids": duplicate_ids,
        "missing_cells": missing_cells,
        "required_columns_present": len(present),
        "required_columns_expected": len(required),
    }


def nearest_threshold_row(
    threshold_table: pd.DataFrame,
    threshold: float,
) -> pd.Series | None:
    """Return the saved threshold row closest to a scenario threshold."""

    if threshold_table.empty or "threshold" not in threshold_table.columns:
        return None

    table = threshold_table.copy()
    table["threshold"] = pd.to_numeric(table["threshold"], errors="coerce")
    table = table.dropna(subset=["threshold"])

    if table.empty:
        return None

    index = (table["threshold"] - float(threshold)).abs().idxmin()
    return table.loc[index]


def build_threshold_scenarios(
    scored: pd.DataFrame,
    threshold_table: pd.DataFrame,
    thresholds: list[float],
    positive_rows: int,
) -> pd.DataFrame:
    """Build actual audience counts plus validated threshold evidence."""

    rows: list[dict[str, float]] = []
    scores = pd.to_numeric(
        scored.get("purchase_intent_score", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()

    for threshold in sorted(set(float(value) for value in thresholds)):
        selected_count = int((scores >= threshold).sum())
        saved = nearest_threshold_row(threshold_table, threshold)

        precision = float(saved.get("precision", np.nan)) if saved is not None else np.nan
        recall = float(saved.get("recall", np.nan)) if saved is not None else np.nan
        f1_score = float(saved.get("f1_score", np.nan)) if saved is not None else np.nan

        expected_buyers = (
            selected_count * precision
            if not pd.isna(precision)
            else np.nan
        )
        captured_buyers = (
            positive_rows * recall
            if not pd.isna(recall)
            else np.nan
        )

        rows.append(
            {
                "threshold": threshold,
                "selected_visitors": selected_count,
                "selected_share": safe_divide(selected_count, len(scored)),
                "validated_precision": precision,
                "validated_recall": recall,
                "validated_f1": f1_score,
                "expected_buyers": expected_buyers,
                "validated_buyers_captured": captured_buyers,
            }
        )

    return pd.DataFrame(rows)


def apply_campaign_filters(
    scored: pd.DataFrame,
    *,
    threshold: float,
    capacity: int,
    intent_tiers: list[str],
) -> pd.DataFrame:
    """Apply scenario filters and return the highest-priority campaign rows."""

    output = scored.copy()
    output["purchase_intent_score"] = pd.to_numeric(
        output["purchase_intent_score"],
        errors="coerce",
    )
    output = output.dropna(subset=["purchase_intent_score"])
    output = output.loc[output["purchase_intent_score"] >= float(threshold)]

    if intent_tiers and "intent_segment" in output.columns:
        output = output.loc[output["intent_segment"].isin(intent_tiers)]

    output = output.sort_values(
        "purchase_intent_score",
        ascending=False,
    ).head(max(int(capacity), 0))
    output = output.reset_index(drop=True)
    output["scenario_rank"] = output.index + 1
    output["scenario_threshold"] = float(threshold)
    return output


def enrich_campaign_rows(scored: pd.DataFrame) -> pd.DataFrame:
    """Add available anomaly evidence without requiring a successful join."""

    output = scored.copy()
    anomaly = read_csv_if_exists(str(ANOMALY_PATH))

    if (
        anomaly.empty
        or "visitorid" not in output.columns
        or "visitorid" not in anomaly.columns
    ):
        output["anomaly_risk_band"] = "Unavailable"
        output["final_anomaly_flag"] = False
        return output

    columns = [
        column
        for column in (
            "visitorid",
            "anomaly_risk_score",
            "anomaly_risk_band",
            "final_anomaly_flag",
            "business_interpretation",
        )
        if column in anomaly.columns
    ]

    lookup = anomaly[columns].drop_duplicates("visitorid")
    output["visitorid"] = output["visitorid"].astype(str)
    lookup["visitorid"] = lookup["visitorid"].astype(str)

    output = output.merge(
        lookup,
        on="visitorid",
        how="left",
        validate="many_to_one",
    )

    if "anomaly_risk_band" not in output.columns:
        output["anomaly_risk_band"] = "Unavailable"
    else:
        output["anomaly_risk_band"] = output["anomaly_risk_band"].fillna(
            "No matching evidence"
        )

    if "final_anomaly_flag" not in output.columns:
        output["final_anomaly_flag"] = False
    else:
        output["final_anomaly_flag"] = output["final_anomaly_flag"].fillna(False)

    return output


def build_score_histogram(
    scored: pd.DataFrame,
    threshold: float,
) -> go.Figure:
    """Build an interactive score distribution with scenario threshold."""

    figure = px.histogram(
        scored,
        x="purchase_intent_score",
        nbins=35,
        color="intent_segment" if "intent_segment" in scored.columns else None,
        hover_data=["visitorid"] if "visitorid" in scored.columns else None,
    )
    figure.add_vline(
        x=float(threshold),
        line_dash="dash",
        line_width=2,
        annotation_text=f"Scenario threshold {threshold:.3f}",
        annotation_position="top right",
    )
    figure.update_xaxes(tickformat=".0%", title="Purchase-intent score")
    figure.update_yaxes(title="Visitors")
    return finish_figure(
        figure,
        title="Campaign score distribution",
        subtitle="Scenario threshold and intent tiers show how audience selection changes.",
        height=500,
        legend_title="Intent tier",
    )


def build_threshold_chart(scenarios: pd.DataFrame) -> go.Figure:
    """Build a dual-axis threshold and audience trade-off chart."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=scenarios["threshold"],
            y=scenarios["selected_share"],
            mode="lines+markers",
            name="Selected share",
            hovertemplate="Threshold %{x:.3f}<br>Selected %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scenarios["threshold"],
            y=scenarios["validated_precision"],
            mode="lines+markers",
            name="Validated precision",
            hovertemplate="Threshold %{x:.3f}<br>Precision %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scenarios["threshold"],
            y=scenarios["validated_recall"],
            mode="lines+markers",
            name="Validated recall",
            hovertemplate="Threshold %{x:.3f}<br>Recall %{y:.2%}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Threshold")
    figure.update_yaxes(title="Rate", tickformat=".0%")
    return finish_figure(
        figure,
        title="Threshold strategy comparison",
        subtitle="Actual batch audience share is paired with saved validation precision and recall.",
        height=500,
        legend_title="Metric",
    )


def strategy_comparison(
    scored: pd.DataFrame,
    *,
    production_threshold: float,
    scenario_threshold: float,
    capacity: int,
    precision: float,
) -> pd.DataFrame:
    """Compare three practical campaign-selection strategies."""

    score = pd.to_numeric(scored["purchase_intent_score"], errors="coerce")
    strategies = [
        (
            "Production threshold",
            score >= production_threshold,
            "Saved champion decision rule",
        ),
        (
            "Scenario threshold",
            score >= scenario_threshold,
            "User-controlled planning scenario",
        ),
        (
            "Capacity-constrained top N",
            pd.Series(False, index=scored.index),
            "Highest scores within campaign capacity",
        ),
    ]

    top_index = score.sort_values(ascending=False).head(capacity).index
    strategies[2][1].loc[top_index] = True

    rows = []
    for name, mask, note in strategies:
        selected = int(mask.fillna(False).sum())
        rows.append(
            {
                "Strategy": name,
                "Selected visitors": selected,
                "Selected share": safe_divide(selected, len(scored)),
                "Expected buyers at validated precision": selected * precision,
                "Evidence note": note,
            }
        )
    return pd.DataFrame(rows)


def render_batch_campaign_intelligence(
    *,
    production_threshold: float,
    validated_precision: float,
) -> None:
    """Render the final campaign-intelligence layer below Batch Scoring."""

    inject_product_css()

    render_section_header(
        "Campaign intelligence",
        "Control capacity, threshold, audience filters, and activation output",
        (
            "This section uses the latest scored batch in session state. "
            "Threshold changes are planning scenarios and never overwrite the saved production rule."
        ),
    )

    result: dict[str, Any] | None = st.session_state.get("latest_batch_scoring")
    if not result or "scored_data" not in result:
        render_empty_state(
            title="Score a batch to unlock campaign intelligence",
            message="No scored audience is available in the current session.",
            next_action="Upload or use the sample batch, then click Score Batch Audience.",
        )
        return

    scored = enrich_campaign_rows(result["scored_data"])
    diagnostics = batch_diagnostics(scored)
    holdout = read_csv_if_exists(str(HOLDOUT_PATH))
    thresholds = read_csv_if_exists(str(THRESHOLD_PATH))

    positive_rows = 0
    if not holdout.empty and "positive_rows" in holdout.columns:
        positive_value = pd.to_numeric(
            holdout.iloc[0]["positive_rows"],
            errors="coerce",
        )
        if not pd.isna(positive_value):
            positive_rows = int(positive_value)

    render_kpi_grid(
        [
            (
                "Rows available",
                compact_count(diagnostics["rows"]),
                "Scored visitors in the current session",
            ),
            (
                "Duplicate visitor IDs",
                compact_count(diagnostics["duplicate_ids"]),
                "Duplicates remain visible; exports are not silently deduplicated",
            ),
            (
                "Missing required cells",
                compact_count(diagnostics["missing_cells"]),
                "Checked after scoring across model and score columns",
            ),
            (
                "Required columns",
                f"{diagnostics['required_columns_present']}/{diagnostics['required_columns_expected']}",
                "Expected scoring and output contract",
            ),
        ]
    )

    min_score = float(pd.to_numeric(scored["purchase_intent_score"], errors="coerce").min())
    max_score = float(pd.to_numeric(scored["purchase_intent_score"], errors="coerce").max())

    control_1, control_2 = st.columns(2)
    with control_1:
        scenario_threshold = st.slider(
            "Planning threshold",
            min_value=max(0.0, min_score),
            max_value=min(1.0, max_score),
            value=min(max(float(production_threshold), min_score), max_score),
            step=0.005,
            help="This changes only the current planning view.",
            key="batch_scenario_threshold",
        )
        st.session_state.eci_threshold_override = float(scenario_threshold)

    with control_2:
        capacity = st.number_input(
            "Maximum campaign capacity",
            min_value=1,
            max_value=max(len(scored), 1),
            value=min(max(len(scored) // 4, 1), len(scored)),
            step=1,
            help="Only the highest-scoring eligible rows are kept within this limit.",
            key="batch_campaign_capacity",
        )

    intent_options = (
        scored["intent_segment"].dropna().astype(str).unique().tolist()
        if "intent_segment" in scored.columns
        else []
    )
    selected_intents = st.multiselect(
        "Intent tiers",
        options=intent_options,
        default=intent_options,
        key="batch_intent_filter",
    )
    st.session_state.eci_intent_tiers = selected_intents
    render_active_filter_summary()

    campaign = apply_campaign_filters(
        scored,
        threshold=scenario_threshold,
        capacity=int(capacity),
        intent_tiers=selected_intents,
    )

    selected_share = safe_divide(len(campaign), len(scored))
    expected_buyers = len(campaign) * float(validated_precision)
    anomaly_count = int(
        campaign.get("final_anomaly_flag", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )

    render_kpi_grid(
        [
            ("Campaign audience", compact_count(len(campaign)), "Eligible rows after threshold, tier, and capacity"),
            ("Audience share", percent(selected_share), "Share of the current scored batch"),
            ("Expected buyers", f"{expected_buyers:,.1f}", "Scenario count × validated precision"),
            ("Anomaly flags", compact_count(anomaly_count), "Matched anomaly evidence requiring review"),
        ]
    )

    histogram = build_score_histogram(scored, scenario_threshold)
    render_evidence_chart(
        figure=histogram,
        key="closure_batch_score_histogram",
        what_it_shows="The complete scored population by purchase-intent score and intent tier.",
        how_to_read="Bars to the right of the vertical line pass the selected planning threshold.",
        actual_finding=(
            f"{len(campaign):,} of {len(scored):,} visitors remain after threshold, tier, and capacity controls."
        ),
        conclusion="A controlled threshold and capacity prevent broad low-quality outreach.",
        recommended_action="Activate the highest-ranked eligible rows and keep anomaly-flagged rows in review.",
        limitation="Score distributions describe model output, not guaranteed purchase probability.",
        source="Current Batch Scoring session",
        evidence_type="Live local scoring plus saved validation evidence",
        refreshed_at=str(result.get("scored_at_utc", "Current session")),
    )

    scenario_values = sorted(
        set(
            [
                max(0.0, scenario_threshold - 0.10),
                max(0.0, scenario_threshold - 0.05),
                scenario_threshold,
                min(1.0, scenario_threshold + 0.05),
                min(1.0, scenario_threshold + 0.10),
                production_threshold,
            ]
        )
    )
    scenario_table = build_threshold_scenarios(
        scored,
        thresholds,
        scenario_values,
        positive_rows,
    )

    if not scenario_table.empty:
        threshold_chart = build_threshold_chart(scenario_table)
        render_evidence_chart(
            figure=threshold_chart,
            key="closure_batch_threshold_comparison",
            what_it_shows="How audience share, validated precision, and validated recall change across nearby thresholds.",
            how_to_read="Higher thresholds usually reduce audience size while increasing selectivity.",
            actual_finding=(
                f"At {scenario_threshold:.3f}, the current batch selects {len(campaign):,} visitors before any manual review."
            ),
            conclusion="Threshold choice is a business trade-off between campaign reach and target quality.",
            recommended_action="Use the saved production threshold by default and document any scenario override.",
            limitation="Precision and recall come from saved validation rows, while audience counts come from this batch.",
            source=str(THRESHOLD_PATH),
            evidence_type="Saved threshold validation plus current batch scores",
            refreshed_at=file_timestamp(THRESHOLD_PATH),
        )

    comparison = strategy_comparison(
        scored,
        production_threshold=production_threshold,
        scenario_threshold=scenario_threshold,
        capacity=int(capacity),
        precision=validated_precision,
    )
    show_table_with_download(
        label="campaign strategy comparison",
        data=comparison,
        file_name="campaign_strategy_comparison.csv",
        metadata=export_metadata(
            source="Current Batch Scoring session",
            evidence_type="Scenario comparison",
            threshold=scenario_threshold,
            filters={"intent_tiers": selected_intents, "capacity": int(capacity)},
        ),
        description="Compare the saved rule, the current threshold scenario, and a strict capacity limit.",
    )

    if campaign.empty:
        render_empty_state(
            title="No visitors match the current campaign rules",
            message="The threshold, tier filter, or capacity combination returned no rows.",
            next_action="Lower the planning threshold or restore additional intent tiers.",
        )
        return

    detail_columns = [
        column
        for column in (
            "scenario_rank",
            "visitorid",
            "purchase_intent_score",
            "intent_segment",
            "recommended_action",
            "anomaly_risk_band",
            "final_anomaly_flag",
            "total_events",
            "view_count",
            "addtocart_count",
            "unique_items",
            "activity_span_ms",
        )
        if column in campaign.columns
    ]

    selected_visitor = st.selectbox(
        "Inspect a selected campaign row",
        options=campaign["visitorid"].astype(str).tolist(),
        key="batch_selected_visitor",
    )
    selected = campaign.loc[
        campaign["visitorid"].astype(str).eq(str(selected_visitor))
    ].iloc[0]

    render_detail_cards(
        [
            ("Score", f"{float(selected['purchase_intent_score']):.1%}", "Model ranking signal"),
            ("Intent tier", str(selected.get("intent_segment", "Unavailable")), "Campaign treatment group"),
            ("Anomaly status", str(selected.get("anomaly_risk_band", "Unavailable")), "Review before activation when elevated"),
            ("Recommended action", str(selected.get("recommended_action", "Review")), "Rules-based next action"),
        ]
    )

    export = add_metadata_columns(
        campaign[detail_columns],
        export_metadata(
            source="Current Batch Scoring session",
            evidence_type="Campaign-ready selected audience",
            threshold=scenario_threshold,
            filters={"intent_tiers": selected_intents, "capacity": int(capacity)},
        ),
    )
    st.download_button(
        "Download campaign-ready selected audience",
        data=export.to_csv(index=False),
        file_name="campaign_ready_selected_audience.csv",
        mime="text/csv",
        use_container_width=True,
        key="closure_batch_campaign_export",
    )

    render_source_note(
        source="Current Batch Scoring session and saved threshold evidence",
        evidence_type="Live scoring, offline validation, and explicit planning scenario",
        refreshed_at=str(result.get("scored_at_utc", "Current session")),
        extra="Scenario overrides do not change the production model or saved threshold.",
    )
