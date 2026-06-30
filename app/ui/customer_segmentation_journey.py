"""Business segmentation and customer-journey intelligence.

This module separates clustering proof from business interpretation. It uses
Package 1 projection evidence for interactive segment exploration and the
leakage-safe historical training snapshots for an aggregate view-to-cart-to-
conversion journey. Unsupported UMAP, assignment confidence, segment movement,
and exact event-transition claims remain clearly conditional.
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
    safe_divide,
)
from app.ui.components import (
    render_detail_cards,
    render_empty_state,
    render_evidence_chart,
    render_kpi_grid,
    render_page_header,
    render_section_header,
    render_source_note,
    show_table_with_download,
)


PROJECTION_PATH = Path("reports/tables/ml_validation_projection_sample.csv")
CLUSTER_SUMMARY_PATH = Path("reports/tables/ml_validation_cluster_business_summary.csv")
CLUSTER_PROFILE_PATH = Path("reports/tables/ml_validation_cluster_profile.csv")
KMEANS_GRID_PATH = Path("reports/tables/ml_validation_kmeans_grid.csv")
DBSCAN_GRID_PATH = Path("reports/tables/ml_validation_dbscan_grid.csv")
METHOD_SUMMARY_PATH = Path("reports/tables/ml_validation_method_summary.csv")
TRAINING_PATH = Path("data/processed/visitor_training_snapshots.csv")


def cluster_personas(summary: pd.DataFrame) -> pd.DataFrame:
    """Create evidence-based cluster personas from actual summary columns."""

    if summary.empty:
        return pd.DataFrame()

    frame = summary.copy()
    numeric_columns = [
        "visitor_count",
        "purchase_rate",
        "avg_purchase_intent_score",
        "avg_total_events",
        "avg_views",
        "avg_add_to_cart",
        "avg_unique_items",
        "avg_activity_hours",
        "avg_cart_share",
        "avg_view_share",
        "visitor_share",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    size_values = frame.get("visitor_count", pd.Series(dtype=float)).dropna()
    engagement_values = frame.get("avg_total_events", pd.Series(dtype=float)).dropna()
    cart_values = frame.get("avg_add_to_cart", pd.Series(dtype=float)).dropna()

    size_median = float(size_values.median()) if not size_values.empty else 0.0
    engagement_median = (
        float(engagement_values.median())
        if not engagement_values.empty
        else 0.0
    )
    cart_median = float(cart_values.median()) if not cart_values.empty else 0.0

    rows = []
    for _, row in frame.iterrows():
        cluster = row.get("cluster", "Unknown")
        size_value = pd.to_numeric(row.get("visitor_count", 0), errors="coerce")
        engagement_value = pd.to_numeric(
            row.get("avg_total_events", 0),
            errors="coerce",
        )
        cart_value = pd.to_numeric(
            row.get("avg_add_to_cart", 0),
            errors="coerce",
        )

        size = 0.0 if pd.isna(size_value) else float(size_value)
        engagement = (
            0.0 if pd.isna(engagement_value) else float(engagement_value)
        )
        cart = 0.0 if pd.isna(cart_value) else float(cart_value)

        if engagement >= engagement_median and cart >= cart_median:
            persona = "High-engagement opportunity"
            action = "Prioritise strong-engagement messaging and measure incremental conversion."
        elif engagement >= engagement_median:
            persona = "Active explorer"
            action = "Use product education and low-friction reminders."
        elif cart >= cart_median:
            persona = "Cart-active uncertainty"
            action = "Investigate friction and test recovery messaging."
        else:
            persona = "Low-intensity browser"
            action = "Use low-cost nurture rather than immediate paid outreach."

        risk = (
            "Large audience; monitor cost at scale."
            if size >= size_median
            else "Smaller audience; avoid overgeneralising from limited volume."
        )

        rows.append(
            {
                "Cluster": str(cluster),
                "Persona": persona,
                "Visitors": int(size),
                "Average events": engagement,
                "Average carts": cart,
                "Opportunity": action,
                "Risk": risk,
            }
        )

    return pd.DataFrame(rows)


def build_pca_figure(
    projection: pd.DataFrame,
    *,
    clusters: list[int],
    anomaly_only: bool,
    three_d: bool,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build an interactive PCA map from the deterministic validation sample."""

    frame = projection.copy()
    frame["kmeans_cluster"] = pd.to_numeric(
        frame["kmeans_cluster"],
        errors="coerce",
    ).astype("Int64")
    frame = frame.loc[frame["kmeans_cluster"].isin(clusters)]

    if anomaly_only and "lof_outlier" in frame.columns:
        frame = frame.loc[
            pd.to_numeric(frame["lof_outlier"], errors="coerce")
            .fillna(0)
            .astype(bool)
        ]

    frame["Cluster"] = frame["kmeans_cluster"].astype(str)
    hover_columns = [
        column
        for column in (
            "visitorid",
            "total_events",
            "view_count",
            "addtocart_count",
            "unique_items",
            "lof_severity",
        )
        if column in frame.columns
    ]

    if three_d and "PC3" in frame.columns:
        figure = px.scatter_3d(
            frame,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Cluster",
            hover_data=hover_columns,
            opacity=0.72,
        )
        return (
            finish_figure(
                figure,
                title="Three-dimensional PCA cluster map",
                subtitle="The third component adds depth only for structural exploration.",
                height=650,
                legend_title="Cluster",
                three_d=True,
            ),
            frame,
        )

    figure = px.scatter(
        frame,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=hover_columns,
        opacity=0.72,
    )
    return (
        finish_figure(
            figure,
            title="PCA business-segment map",
            subtitle="Each point is a real sampled visitor coloured by the selected K-Means cluster.",
            height=590,
            legend_title="Cluster",
        ),
        frame,
    )


def build_cluster_profile_figure(profile: pd.DataFrame, clusters: list[int]) -> go.Figure:
    """Build a normalized feature-profile heatmap for selected clusters."""

    frame = profile.copy()
    frame["cluster"] = pd.to_numeric(frame["cluster"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["cluster"].isin(clusters)]
    pivot = frame.pivot(index="cluster", columns="feature", values="standardized_mean")

    figure = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[f"Cluster {value}" for value in pivot.index],
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Standardized mean"},
            hovertemplate="%{y}<br>%{x}<br>Standardized mean %{z:.2f}<extra></extra>",
        )
    )
    return finish_figure(
        figure,
        title="Normalized cluster feature profiles",
        subtitle="Positive cells are above the validation-sample mean and negative cells are below it.",
        height=470,
    )


def build_quality_figure(kmeans: pd.DataFrame) -> go.Figure:
    """Show K-Means silhouette and inertia evidence across candidate k values."""

    frame = kmeans.copy()
    for column in ("k", "inertia", "silhouette_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["k"],
            y=frame["silhouette_score"],
            mode="lines+markers",
            name="Silhouette score",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["k"],
            y=frame["inertia"],
            mode="lines+markers",
            name="Inertia",
            yaxis="y2",
        )
    )
    figure.update_layout(
        yaxis2={
            "title": "Inertia",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    )
    figure.update_xaxes(title="Number of clusters")
    figure.update_yaxes(title="Silhouette score")
    return finish_figure(
        figure,
        title="Cluster-quality evidence",
        subtitle="Silhouette separation and inertia compactness are reviewed together before selecting k.",
        height=500,
        legend_title="Metric",
    )


@st.cache_data(show_spinner=False)
def load_journey_snapshots() -> pd.DataFrame:
    """Load leakage-safe historical snapshots and derive aggregate journey stages."""

    if not TRAINING_PATH.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(TRAINING_PATH)
    required = {"visitorid", "view_count", "addtocart_count", "converted"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    frame["visitorid"] = frame["visitorid"].astype(str)
    for column in ("view_count", "addtocart_count", "converted"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame["Viewed"] = frame["view_count"] > 0
    frame["Added to cart"] = frame["addtocart_count"] > 0
    frame["Converted"] = frame["converted"] > 0

    frame["Behavior tier"] = np.select(
        [
            frame["addtocart_count"] >= 3,
            frame["addtocart_count"] >= 1,
            frame["view_count"] >= 5,
        ],
        [
            "Cart intensive",
            "Cart engaged",
            "Active browser",
        ],
        default="Light browser",
    )

    frame["Journey pattern"] = np.select(
        [
            frame["Converted"],
            frame["Added to cart"] & ~frame["Converted"],
            frame["Viewed"] & ~frame["Added to cart"],
        ],
        [
            "Viewed → cart → converted",
            "Viewed → cart → no conversion",
            "Viewed → no cart",
        ],
        default="No supported view stage",
    )

    if "snapshot_time" in frame.columns:
        frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce")

    return frame


def journey_funnel(frame: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate journey-stage counts and rates."""

    total = len(frame)
    viewed = int(frame["Viewed"].sum())
    carted = int(frame["Added to cart"].sum())
    converted = int(frame["Converted"].sum())

    output = pd.DataFrame(
        {
            "Stage": ["Eligible snapshots", "Viewed", "Added to cart", "Converted"],
            "Rows": [total, viewed, carted, converted],
        }
    )
    output["Share of eligible"] = output["Rows"].apply(lambda value: safe_divide(value, total))
    output["Retention from prior"] = [
        1.0,
        safe_divide(viewed, total),
        safe_divide(carted, viewed),
        safe_divide(converted, carted),
    ]
    return output


def build_journey_funnel_figure(funnel: pd.DataFrame) -> go.Figure:
    """Build the aggregate view-to-cart-to-conversion funnel."""

    figure = go.Figure(
        go.Funnel(
            y=funnel["Stage"],
            x=funnel["Rows"],
            textinfo="value+percent initial",
            hovertemplate="%{y}<br>Rows %{x:,.0f}<extra></extra>",
        )
    )
    return finish_figure(
        figure,
        title="Historical snapshot journey funnel",
        subtitle="Leakage-safe snapshots show view, cart, and future-conversion stages at snapshot grain.",
        height=500,
    )


def build_tier_funnel_figure(frame: pd.DataFrame) -> go.Figure:
    """Compare cart and conversion rates across behaviour tiers."""

    summary = (
        frame.groupby("Behavior tier", dropna=False)
        .agg(
            Rows=("visitorid", "size"),
            Cart_rate=("Added to cart", "mean"),
            Conversion_rate=("Converted", "mean"),
        )
        .reset_index()
    )
    melted = summary.melt(
        id_vars=["Behavior tier", "Rows"],
        value_vars=["Cart_rate", "Conversion_rate"],
        var_name="Metric",
        value_name="Rate",
    )
    figure = px.bar(
        melted,
        x="Behavior tier",
        y="Rate",
        color="Metric",
        barmode="group",
        hover_data={"Rows": True},
    )
    figure.update_yaxes(tickformat=".0%")
    return finish_figure(
        figure,
        title="Journey performance by behaviour tier",
        subtitle="Behaviour tiers use pre-outcome view and cart activity; they are not the production model's intent tiers.",
        height=500,
        legend_title="Rate",
    )


def render_segmentation_and_journey_page() -> None:
    """Render the new business segmentation and journey page."""

    render_page_header(
        eyebrow="Customer intelligence",
        title="Understand visitor groups and where journeys lose momentum.",
        subtitle=(
            "Interactive business segmentation uses the governed Package 1 validation sample. "
            "Journey analysis uses leakage-safe historical snapshots and keeps unsupported path claims conditional."
        ),
        badges=[
            ("Segmentation: validation evidence", "positive"),
            ("Journey: historical snapshots", "info"),
            ("No invented transitions", "warning"),
        ],
    )

    segment_tab, journey_tab = st.tabs(
        ["Business segmentation", "Journey and funnel"]
    )

    with segment_tab:
        projection = read_csv_if_exists(str(PROJECTION_PATH))
        summary = read_csv_if_exists(str(CLUSTER_SUMMARY_PATH))
        profile = read_csv_if_exists(str(CLUSTER_PROFILE_PATH))
        kmeans = read_csv_if_exists(str(KMEANS_GRID_PATH))
        dbscan = read_csv_if_exists(str(DBSCAN_GRID_PATH))
        method_summary = read_csv_if_exists(str(METHOD_SUMMARY_PATH))

        if projection.empty or summary.empty:
            render_empty_state(
                "Segmentation evidence unavailable",
                "The governed projection sample or cluster business summary is missing.",
                "Regenerate Package 1 ML validation evidence.",
            )
        else:
            cluster_options = sorted(
                pd.to_numeric(projection["kmeans_cluster"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            control_1, control_2 = st.columns(2)
            with control_1:
                selected_clusters = st.multiselect(
                    "Clusters",
                    options=cluster_options,
                    default=cluster_options,
                    key="customer_cluster_filter",
                )
                st.session_state.eci_cluster_filter = selected_clusters
            with control_2:
                map_mode = st.radio(
                    "PCA map",
                    ["2D", "3D"],
                    horizontal=True,
                    key="customer_pca_mode",
                )
                anomaly_only = st.checkbox(
                    "Show LOF outliers only",
                    value=False,
                    key="customer_anomaly_only",
                )
                st.session_state.eci_anomaly_only = anomaly_only

            if not selected_clusters:
                render_empty_state(
                    "No clusters selected",
                    "The cluster filter currently removes every segment.",
                    "Select at least one cluster.",
                )
            else:
                pca_figure, filtered_projection = build_pca_figure(
                    projection,
                    clusters=selected_clusters,
                    anomaly_only=anomaly_only,
                    three_d=map_mode == "3D",
                )
                render_kpi_grid(
                    [
                        ("Visible visitors", compact_count(len(filtered_projection)), "Rows in the current governed projection view"),
                        ("Selected clusters", compact_count(len(selected_clusters)), "K-Means groups included"),
                        ("LOF outliers", compact_count(pd.to_numeric(filtered_projection.get("lof_outlier", 0), errors="coerce").fillna(0).astype(bool).sum()), "Current filtered sample"),
                        ("Source sample", compact_count(len(projection)), "Deterministic Package 1 projection sample"),
                    ]
                )
                render_evidence_chart(
                    figure=pca_figure,
                    key="closure_customer_pca",
                    what_it_shows="Governed PCA projection of the selected visitor clusters with anomaly evidence in hover detail.",
                    how_to_read="Nearby points have similar compressed behaviour; colour indicates the selected K-Means assignment.",
                    actual_finding=f"{len(filtered_projection):,} visitors are visible across {len(selected_clusters)} selected clusters.",
                    conclusion="Clusters reveal distinct behavioural groups that can receive different campaign treatment.",
                    recommended_action="Use cluster personas and profile evidence together before assigning business actions.",
                    limitation="PCA is a compressed validation view and does not prove stable customer personas over time.",
                    source=str(PROJECTION_PATH),
                    evidence_type="Deterministic offline validation sample",
                    refreshed_at=file_timestamp(PROJECTION_PATH),
                )

                if not profile.empty:
                    profile_figure = build_cluster_profile_figure(profile, selected_clusters)
                    render_evidence_chart(
                        figure=profile_figure,
                        key="closure_cluster_profile",
                        what_it_shows="Standardized behavioural feature averages for the selected clusters.",
                        how_to_read="Warm cells are above the full-sample mean and cool cells are below it.",
                        actual_finding=f"The profile compares {profile['feature'].nunique()} governed features across selected clusters.",
                        conclusion="Cluster meaning comes from the full pattern of features, not the cluster number.",
                        recommended_action="Use the largest standardized differences to design distinct messaging and review rules.",
                        limitation="Standardized means hide within-cluster variation and should be paired with visitor-level review.",
                        source=str(CLUSTER_PROFILE_PATH),
                        evidence_type="Governed K-Means feature profile",
                        refreshed_at=file_timestamp(CLUSTER_PROFILE_PATH),
                    )

                personas = cluster_personas(summary)
                show_table_with_download(
                    label="cluster personas and business actions",
                    data=personas.loc[personas["Cluster"].astype(int).isin(selected_clusters)] if not personas.empty else personas,
                    file_name="cluster_personas_and_actions.csv",
                    metadata=export_metadata(
                        source=str(CLUSTER_SUMMARY_PATH),
                        evidence_type="Evidence-based cluster interpretation",
                        filters={"clusters": selected_clusters},
                    ),
                )

                selected_rows = filtered_projection.copy()
                visitor_query = st.text_input(
                    "Search visitor ID within selected clusters",
                    value="",
                    key="customer_visitor_search",
                )
                if visitor_query and "visitorid" in selected_rows.columns:
                    selected_rows = selected_rows.loc[
                        selected_rows["visitorid"].astype(str).str.contains(
                            visitor_query,
                            case=False,
                            regex=False,
                        )
                    ]
                display_columns = [
                    column
                    for column in (
                        "visitorid",
                        "kmeans_cluster",
                        "dbscan_cluster",
                        "lof_outlier",
                        "lof_severity",
                        "total_events",
                        "view_count",
                        "addtocart_count",
                        "unique_items",
                    )
                    if column in selected_rows.columns
                ]
                show_table_with_download(
                    label="filtered cluster visitor assignments",
                    data=selected_rows[display_columns].head(5000),
                    file_name="filtered_cluster_visitors.csv",
                    metadata=export_metadata(
                        source=str(PROJECTION_PATH),
                        evidence_type="Filtered validation visitor assignments",
                        filters={
                            "clusters": selected_clusters,
                            "anomaly_only": anomaly_only,
                            "visitor_query": visitor_query,
                        },
                    ),
                    description="The interactive table is capped at 5,000 rows; source totals remain visible above.",
                )

            if not kmeans.empty:
                quality_figure = build_quality_figure(kmeans)
                selected_k = int(pd.to_numeric(kmeans.loc[kmeans.get("selected", False).astype(bool), "k"], errors="coerce").iloc[0]) if "selected" in kmeans.columns and kmeans["selected"].astype(bool).any() else int(pd.to_numeric(kmeans["k"], errors="coerce").iloc[0])
                render_evidence_chart(
                    figure=quality_figure,
                    key="closure_cluster_quality",
                    what_it_shows="K-Means silhouette and inertia evidence across tested cluster counts.",
                    how_to_read="Higher silhouette and a useful inertia elbow support a practical cluster count.",
                    actual_finding=f"The governed Package 1 selection is k={selected_k}.",
                    conclusion="The selected segmentation is supported by quantitative evidence and business interpretability.",
                    recommended_action="Rebuild and compare this evidence before changing the production segmentation design.",
                    limitation="Cluster quality metrics do not guarantee stable business outcomes or causal personas.",
                    source=str(KMEANS_GRID_PATH),
                    evidence_type="Offline clustering model-selection evidence",
                    refreshed_at=file_timestamp(KMEANS_GRID_PATH),
                )

            if not method_summary.empty:
                show_table_with_download(
                    label="clustering method evidence",
                    data=method_summary,
                    file_name="clustering_method_evidence.csv",
                    metadata=export_metadata(
                        source=str(METHOD_SUMMARY_PATH),
                        evidence_type="PCA, K-Means, DBSCAN, and LOF validation summary",
                    ),
                )

            render_source_note(
                source="No approved UMAP, assignment-confidence, or comparable segment-snapshot artifact",
                evidence_type="Conditional segmentation enhancements",
                refreshed_at=file_timestamp(PROJECTION_PATH),
                extra="UMAP, assignment confidence, and segment movement are not claimed until those artifacts exist.",
            )

    with journey_tab:
        journey = load_journey_snapshots()
        if journey.empty:
            render_empty_state(
                "Journey evidence unavailable",
                "The leakage-safe visitor training snapshot table is missing or lacks required stage columns.",
                "Regenerate data/processed/visitor_training_snapshots.csv.",
            )
        else:
            filtered = journey.copy()
            if "snapshot_time" in filtered.columns and filtered["snapshot_time"].notna().any():
                minimum_date = filtered["snapshot_time"].min().date()
                maximum_date = filtered["snapshot_time"].max().date()
                date_range = st.date_input(
                    "Snapshot date range",
                    value=(minimum_date, maximum_date),
                    min_value=minimum_date,
                    max_value=maximum_date,
                    key="journey_date_range",
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = date_range
                    filtered = filtered.loc[
                        filtered["snapshot_time"].dt.date.between(start, end)
                    ]

            tiers = sorted(filtered["Behavior tier"].dropna().unique().tolist())
            selected_tiers = st.multiselect(
                "Behaviour tiers",
                options=tiers,
                default=tiers,
                key="journey_tier_filter",
            )
            filtered = filtered.loc[filtered["Behavior tier"].isin(selected_tiers)]

            funnel = journey_funnel(filtered)
            cart_loss = int(funnel.loc[funnel["Stage"].eq("Viewed"), "Rows"].iloc[0] - funnel.loc[funnel["Stage"].eq("Added to cart"), "Rows"].iloc[0])
            conversion_loss = int(funnel.loc[funnel["Stage"].eq("Added to cart"), "Rows"].iloc[0] - funnel.loc[funnel["Stage"].eq("Converted"), "Rows"].iloc[0])

            render_kpi_grid(
                [
                    ("Eligible snapshots", compact_count(len(filtered)), "One row per leakage-safe visitor snapshot"),
                    ("Cart rate", percent(filtered["Added to cart"].mean()), "Snapshots with at least one add-to-cart"),
                    ("Conversion rate", percent(filtered["Converted"].mean()), "Future transaction label within target window"),
                    ("Largest stage loss", compact_count(max(cart_loss, conversion_loss)), "Maximum absolute loss between supported stages"),
                ]
            )

            funnel_figure = build_journey_funnel_figure(funnel)
            render_evidence_chart(
                figure=funnel_figure,
                key="closure_journey_funnel",
                what_it_shows="Eligible historical snapshots progressing through view, add-to-cart, and future conversion stages.",
                how_to_read="Each lower stage is a subset of the supported prior behaviour stage.",
                actual_finding=f"The view-to-cart loss is {cart_loss:,} rows and the cart-to-conversion loss is {conversion_loss:,} rows.",
                conclusion="The largest supported loss identifies where recovery experiments may have the most volume.",
                recommended_action="Investigate the largest loss by behaviour tier before designing a broad intervention.",
                limitation="Rows are historical snapshots, not unique lifetime customers; repeated visitors may appear at multiple dates.",
                source=str(TRAINING_PATH),
                evidence_type="Leakage-safe historical snapshot funnel",
                refreshed_at=file_timestamp(TRAINING_PATH),
            )

            tier_figure = build_tier_funnel_figure(filtered)
            best_tier = (
                filtered.groupby("Behavior tier")["Converted"].mean().idxmax()
                if len(filtered)
                else "Unavailable"
            )
            render_evidence_chart(
                figure=tier_figure,
                key="closure_journey_tier_comparison",
                what_it_shows="Cart and conversion rates across pre-outcome behaviour tiers.",
                how_to_read="Higher bars indicate stronger stage progression within the selected historical filter.",
                actual_finding=f"The highest observed conversion rate belongs to {best_tier}.",
                conclusion="Journey performance differs materially by observed behaviour intensity.",
                recommended_action="Prioritise friction analysis for high-cart tiers and low-cost nurture for light browsers.",
                limitation="Behaviour tiers are descriptive rules and are not the production model's saved intent segments.",
                source=str(TRAINING_PATH),
                evidence_type="Historical behaviour-tier comparison",
                refreshed_at=file_timestamp(TRAINING_PATH),
            )

            path_summary = (
                filtered.groupby("Journey pattern", dropna=False)
                .agg(
                    Snapshots=("visitorid", "size"),
                    Unique_visitors=("visitorid", "nunique"),
                    Average_views=("view_count", "mean"),
                    Average_carts=("addtocart_count", "mean"),
                )
                .reset_index()
                .sort_values("Snapshots", ascending=False)
            )
            show_table_with_download(
                label="aggregated journey patterns",
                data=path_summary,
                file_name="aggregated_journey_patterns.csv",
                metadata=export_metadata(
                    source=str(TRAINING_PATH),
                    evidence_type="Snapshot-level aggregate journey patterns",
                    filters={"behavior_tiers": selected_tiers},
                ),
                description="Patterns are supported stage combinations, not reconstructed event-by-event transitions.",
            )

            visitor_id = st.text_input(
                "Optional exact visitor ID trace",
                value="",
                key="journey_visitor_id",
            )
            if visitor_id:
                trace = filtered.loc[
                    filtered["visitorid"].astype(str).eq(visitor_id)
                ].sort_values("snapshot_time" if "snapshot_time" in filtered.columns else "visitorid")
                if trace.empty:
                    st.info("No matching visitor appears in the current journey filter.")
                else:
                    columns = [
                        column
                        for column in (
                            "visitorid",
                            "snapshot_time",
                            "data_split",
                            "view_count",
                            "addtocart_count",
                            "Behavior tier",
                            "Journey pattern",
                            "converted",
                        )
                        if column in trace.columns
                    ]
                    st.dataframe(trace[columns], use_container_width=True, hide_index=True)

            render_source_note(
                source="No approved item-level path sequence or exact stage-duration artifact",
                evidence_type="Conditional journey details",
                refreshed_at=file_timestamp(TRAINING_PATH),
                extra="Exact transitions, item filters, and time-between-stage claims remain conditional until aggregated sequence evidence is generated.",
            )
