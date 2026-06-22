"""Interactive figures for the Machine Learning Validation & Evidence page.

The functions are intentionally data-frame driven. They accept the verified
precomputed tables created by ``build_ml_validation_evidence.py`` and never
invent missing analytical evidence.
"""

from __future__ import annotations

from typing import Iterable
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui.design_system import CLUSTER_PALETTE, COLORS, SEMANTIC_PALETTE
from app.ui.plotly_style import apply_product_layout


def first_existing_column(data: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column using case-insensitive comparison."""

    lookup = {column.lower(): column for column in data.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _projection_columns(data: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Resolve the first three PCA columns from a projection table."""

    pc1 = first_existing_column(data, ["PC1", "pc1", "pca_1", "component_1"])
    pc2 = first_existing_column(data, ["PC2", "pc2", "pca_2", "component_2"])
    pc3 = first_existing_column(data, ["PC3", "pc3", "pca_3", "component_3"])
    return pc1, pc2, pc3



def safe_value(value: object) -> str:
    """Format one hover value without exposing NaN noise."""

    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)



def build_projection_figure(
    data: pd.DataFrame,
    *,
    color_column: str,
    dimension: str,
    symbol_column: str | None = None,
    size_column: str | None = None,
    max_points: int = 6000,
) -> go.Figure | None:
    """Build an interactive 2D or 3D projection with readable legends.

    Dense point clouds are sampled deterministically for browser performance.
    DBSCAN clusters are consolidated to the largest groups so the legend stays
    usable, while K-Means views add clearly marked centroids. LOF marker sizes
    are log-scaled and capped to prevent one extreme score from collapsing the
    rest of the cloud.
    """

    if data.empty or color_column not in data.columns:
        return None

    pc1, pc2, pc3 = _projection_columns(data)
    if pc1 is None or pc2 is None:
        return None

    plot_data = data.copy()
    if len(plot_data) > max_points:
        plot_data = plot_data.sample(max_points, random_state=42)

    hover_candidates = [
        "visitorid",
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "purchase_intent_score",
        "purchased",
        "lof_score",
        "dbscan_point_type",
    ]
    hover_data = [
        column for column in hover_candidates if column in plot_data.columns
    ]

    active_size = None
    if size_column and size_column in plot_data.columns:
        numeric_size = pd.to_numeric(
            plot_data[size_column],
            errors="coerce",
        ).fillna(0)
        upper = float(numeric_size.quantile(0.99)) if len(numeric_size) else 0.0
        clipped = numeric_size.clip(upper=upper if upper > 0 else None)
        plot_data["_plot_size"] = np.log1p(clipped - clipped.min() + 0.05)
        active_size = "_plot_size"

    # DBSCAN can produce many cluster/symbol combinations. Build one trace per
    # visible cluster and vary marker symbol inside the trace instead.
    if color_column == "dbscan_cluster":
        raw_cluster = plot_data[color_column].astype(str)
        non_noise = raw_cluster[raw_cluster != "-1"]
        largest = list(non_noise.value_counts().head(8).index)

        def display_cluster(value: str) -> str:
            if value == "-1":
                return "Noise"
            if value in largest:
                return f"Cluster {value}"
            return "Other dense clusters"

        plot_data["_cluster_display"] = raw_cluster.map(display_cluster)

        order = [
            *(f"Cluster {value}" for value in largest),
            "Other dense clusters",
            "Noise",
        ]
        order = [
            value
            for value in order
            if value in set(plot_data["_cluster_display"])
        ]

        symbol_map = {
            "Core": "circle",
            "Border": "diamond",
            "Noise": "x",
        }

        figure = go.Figure()

        for index, group_name in enumerate(order):
            subset = plot_data.loc[
                plot_data["_cluster_display"] == group_name
            ].copy()

            if subset.empty:
                continue

            point_types = (
                subset[symbol_column].astype(str)
                if symbol_column and symbol_column in subset.columns
                else pd.Series(["Core"] * len(subset), index=subset.index)
            )
            symbols = [
                symbol_map.get(value, "circle")
                for value in point_types
            ]
            colour = (
                COLORS["neutral"]
                if group_name == "Noise"
                else CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
            )
            hover_text = []
            for _, row in subset.iterrows():
                parts = [
                    f"{pc1}: {safe_value(row.get(pc1))}",
                    f"{pc2}: {safe_value(row.get(pc2))}",
                ]
                if pc3 is not None:
                    parts.append(f"{pc3}: {safe_value(row.get(pc3))}")
                for column in hover_data:
                    parts.append(f"{column}: {safe_value(row.get(column))}")
                hover_text.append("<br>".join(parts))

            marker = {
                "color": colour,
                "opacity": 0.72 if group_name != "Noise" else 0.46,
                "line": {"width": 0},
                "symbol": symbols,
                "size": 5 if dimension == "3D" else 7,
            }

            if dimension == "3D" and pc3 is not None:
                figure.add_trace(
                    go.Scatter3d(
                        x=subset[pc1],
                        y=subset[pc2],
                        z=subset[pc3],
                        mode="markers",
                        name=group_name,
                        marker=marker,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
            else:
                figure.add_trace(
                    go.Scattergl(
                        x=subset[pc1],
                        y=subset[pc2],
                        mode="markers",
                        name=group_name,
                        marker=marker,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        return apply_product_layout(
            figure,
            height=760 if dimension == "3D" else 640,
            legend_title="DBSCAN region",
            three_d=dimension == "3D" and pc3 is not None,
        )

    plot_data[color_column] = plot_data[color_column].astype(str)

    colour_map = None
    if color_column == "lof_severity":
        colour_map = SEMANTIC_PALETTE

    kwargs = {
        "data_frame": plot_data,
        "x": pc1,
        "y": pc2,
        "color": color_column,
        "symbol": (
            symbol_column
            if symbol_column and symbol_column in plot_data.columns
            else None
        ),
        "size": active_size,
        "hover_data": hover_data,
        "color_discrete_sequence": CLUSTER_PALETTE,
        "color_discrete_map": colour_map,
        "opacity": 0.74,
    }

    three_d = dimension == "3D" and pc3 is not None

    if three_d:
        figure = px.scatter_3d(z=pc3, **kwargs)
        figure.update_traces(
            marker={"line": {"width": 0}, "sizemin": 3}
        )
    else:
        figure = px.scatter(**kwargs)
        marker_style = {"line": {"width": 0}}
        if active_size is None:
            marker_style["size"] = 7
        figure.update_traces(marker=marker_style)

    # Centroids make K-Means structure easier to read than colour alone.
    if color_column == "kmeans_cluster":
        centroid_columns = [pc1, pc2] + ([pc3] if three_d else [])
        centroids = (
            plot_data.groupby(color_column, observed=True)[centroid_columns]
            .mean()
            .reset_index()
        )
        labels = [
            f"Cluster {value}"
            for value in centroids[color_column].astype(str)
        ]

        if three_d:
            figure.add_trace(
                go.Scatter3d(
                    x=centroids[pc1],
                    y=centroids[pc2],
                    z=centroids[pc3],
                    mode="markers+text",
                    name="Cluster centroids",
                    text=labels,
                    textposition="top center",
                    marker={
                        "symbol": "diamond",
                        "size": 10,
                        "color": COLORS["text"],
                        "line": {
                            "color": COLORS["app_bg"],
                            "width": 2,
                        },
                    },
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=centroids[pc1],
                    y=centroids[pc2],
                    mode="markers+text",
                    name="Cluster centroids",
                    text=labels,
                    textposition="top center",
                    marker={
                        "symbol": "diamond",
                        "size": 13,
                        "color": COLORS["text"],
                        "line": {
                            "color": COLORS["app_bg"],
                            "width": 2,
                        },
                    },
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    return apply_product_layout(
        figure,
        height=760 if three_d else 640,
        legend_title=color_column,
        three_d=three_d,
    )

def build_scree_figure(variance: pd.DataFrame) -> go.Figure | None:
    """Build explained-variance bars with a cumulative variance line."""

    if variance.empty:
        return None

    component = first_existing_column(
        variance,
        ["component", "principal_component", "pca_component"],
    )
    ratio = first_existing_column(
        variance,
        ["explained_variance_ratio", "variance_ratio", "explained_variance"],
    )
    cumulative = first_existing_column(
        variance,
        ["cumulative_explained_variance", "cumulative_variance"],
    )

    if component is None or ratio is None:
        return None

    data = variance.copy()
    data[ratio] = pd.to_numeric(data[ratio], errors="coerce")

    if cumulative is None:
        data["_cumulative"] = data[ratio].fillna(0).cumsum()
        cumulative = "_cumulative"

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=data[component].astype(str),
            y=data[ratio],
            name="Explained variance",
            marker_color=COLORS["brand"],
            text=data[ratio],
            texttemplate="%{text:.1%}",
            textposition="outside",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=data[component].astype(str),
            y=data[cumulative],
            name="Cumulative variance",
            mode="lines+markers",
            line={"color": COLORS["positive"], "width": 3},
            marker={"size": 8},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Explained variance", tickformat=".0%", secondary_y=False)
    figure.update_yaxes(title_text="Cumulative variance", tickformat=".0%", range=[0, 1.08], secondary_y=True)
    return apply_product_layout(figure, height=520, legend_title="PCA evidence")


def build_loading_heatmap(loadings: pd.DataFrame) -> go.Figure | None:
    """Show how original features contribute to the first components."""

    if loadings.empty:
        return None

    feature_column = first_existing_column(loadings, ["feature", "feature_name"])
    component_columns = [
        column
        for column in loadings.columns
        if str(column).upper() in {"PC1", "PC2", "PC3"}
    ]

    if feature_column is None or not component_columns:
        return None

    matrix = loadings.set_index(feature_column)[component_columns]
    figure = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#FB7185"],
            [0.5, "#0F172A"],
            [1.0, "#38BDF8"],
        ],
        zmin=-1,
        zmax=1,
        labels={"x": "Principal component", "y": "Input feature", "color": "Loading"},
    )
    return apply_product_layout(figure, height=540)



def build_kmeans_quality_figure(
    summary: pd.DataFrame,
    *,
    selected_k: int | None = None,
) -> go.Figure | None:
    """Compare k=1..10 compactness and separation on one sample.

    ``selected_k`` marks the operational choice. When it is omitted, the
    strongest available silhouette candidate is highlighted.
    """

    if summary.empty:
        return None

    k_column = first_existing_column(
        summary,
        ["k", "n_clusters", "cluster_count"],
    )
    inertia_column = first_existing_column(summary, ["inertia"])
    silhouette_column = first_existing_column(
        summary,
        ["silhouette_score", "silhouette"],
    )

    if k_column is None or (
        inertia_column is None and silhouette_column is None
    ):
        return None

    data = summary.copy().sort_values(k_column)
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    best_k = selected_k
    if silhouette_column:
        silhouette_values = pd.to_numeric(
            data[silhouette_column],
            errors="coerce",
        )
        if not silhouette_values.dropna().empty:
            if best_k is None:
                best_index = silhouette_values.idxmax()
                best_k = int(data.loc[best_index, k_column])

            colours = [
                COLORS["positive"]
                if int(value) == int(best_k)
                else COLORS["brand"]
                for value in data[k_column]
            ]

            figure.add_trace(
                go.Bar(
                    x=data[k_column],
                    y=silhouette_values,
                    name="Silhouette",
                    marker_color=colours,
                    opacity=0.88,
                    text=silhouette_values,
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                ),
                secondary_y=True,
            )
            figure.update_yaxes(
                title_text="Silhouette score",
                range=[0, 1.05],
                secondary_y=True,
            )

    if inertia_column:
        inertia_values = pd.to_numeric(
            data[inertia_column],
            errors="coerce",
        )
        figure.add_trace(
            go.Scatter(
                x=data[k_column],
                y=inertia_values,
                mode="lines+markers",
                name="Inertia",
                line={
                    "color": COLORS["information"],
                    "width": 3,
                },
                marker={"size": 9},
            ),
            secondary_y=False,
        )
        figure.update_yaxes(
            title_text="Inertia",
            secondary_y=False,
        )

    if best_k is not None:
        figure.add_vline(
            x=best_k,
            line_width=1,
            line_dash="dot",
            line_color=COLORS["positive"],
        )
        figure.add_annotation(
            x=best_k,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"Best separation: k={best_k}",
            showarrow=False,
            font={"color": COLORS["positive"], "size": 12},
        )

    figure.update_xaxes(
        title_text="Number of clusters (k)",
        dtick=1,
    )
    return apply_product_layout(
        figure,
        height=560,
        legend_title="Quality measure",
    )

def build_cluster_profile_heatmap(profile: pd.DataFrame) -> go.Figure | None:
    """Show standardized behaviour profiles for each K-Means cluster."""

    if profile.empty:
        return None

    cluster_column = first_existing_column(profile, ["cluster", "kmeans_cluster"])
    feature_column = first_existing_column(profile, ["feature", "feature_name"])
    value_column = first_existing_column(
        profile,
        ["standardized_mean", "mean_z_score", "profile_value"],
    )

    if cluster_column is None or feature_column is None or value_column is None:
        return None

    matrix = profile.pivot_table(
        index=cluster_column,
        columns=feature_column,
        values=value_column,
        aggfunc="mean",
    )
    matrix.index = [f"Cluster {value}" for value in matrix.index]

    figure = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#FB7185"],
            [0.5, "#0F172A"],
            [1.0, "#34D399"],
        ],
        zmin=-2,
        zmax=2,
        labels={"x": "Behaviour feature", "y": "Customer cluster", "color": "Relative level"},
    )
    return apply_product_layout(figure, height=560)


def build_dbscan_parameter_figure(summary: pd.DataFrame) -> go.Figure | None:
    """Show how DBSCAN parameters affect quality, cluster count, and noise."""

    if summary.empty:
        return None

    eps_column = first_existing_column(summary, ["eps", "epsilon"])
    min_samples_column = first_existing_column(summary, ["min_samples"])
    silhouette_column = first_existing_column(summary, ["silhouette_score", "silhouette"])
    cluster_column = first_existing_column(summary, ["cluster_count", "n_clusters_found", "n_clusters"])
    noise_column = first_existing_column(summary, ["noise_rate", "noise_count"])

    if eps_column is None or min_samples_column is None:
        return None

    data = summary.copy()
    if silhouette_column is None:
        data["_quality"] = pd.to_numeric(data.get(cluster_column, 0), errors="coerce").fillna(0)
        silhouette_column = "_quality"
    if cluster_column is None:
        data["_cluster_count"] = 1
        cluster_column = "_cluster_count"

    hover_data = [column for column in [noise_column, cluster_column] if column]
    figure = px.scatter(
        data,
        x=eps_column,
        y=min_samples_column,
        color=silhouette_column,
        size=cluster_column,
        hover_data=hover_data,
        color_continuous_scale="Viridis",
        size_max=34,
    )
    figure.update_xaxes(title_text="Epsilon (neighbourhood radius)")
    figure.update_yaxes(title_text="Minimum neighbours", dtick=1)
    return apply_product_layout(figure, height=560)



def build_lof_score_figure(
    projection: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> go.Figure | None:
    """Compare normal and flagged LOF distributions on a readable log scale."""

    if projection.empty or "lof_score" not in projection.columns:
        return None

    data = projection.copy()
    status_column = (
        "lof_status"
        if "lof_status" in data.columns
        else "lof_outlier"
    )
    if status_column not in data.columns:
        data[status_column] = "Observation"

    data["lof_score"] = pd.to_numeric(
        data["lof_score"],
        errors="coerce",
    )
    data = data.loc[data["lof_score"] > 0].copy()
    if data.empty:
        return None

    data[status_column] = data[status_column].astype(str)
    data["_log_lof_score"] = np.log10(data["lof_score"])

    figure = px.histogram(
        data,
        x="_log_lof_score",
        color=status_column,
        nbins=48,
        barmode="overlay",
        histnorm="probability density",
        opacity=0.68,
        color_discrete_map={
            "Normal": COLORS["brand"],
            "Outlier": COLORS["critical"],
            "False": COLORS["brand"],
            "True": COLORS["critical"],
        },
    )

    minimum = max(float(data["_log_lof_score"].min()), 0.0)
    maximum = float(data["_log_lof_score"].max())
    tick_start = math.floor(minimum)
    tick_end = math.ceil(maximum)
    tick_values = list(range(tick_start, tick_end + 1))
    tick_text = [f"{10 ** value:,.0f}" for value in tick_values]

    figure.update_xaxes(
        title_text="LOF anomaly score (log scale)",
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
    )
    figure.update_yaxes(title_text="Relative density")
    figure.update_layout(barmode="overlay")

    if threshold is not None and threshold > 0:
        threshold_x = math.log10(threshold)
        figure.add_vline(
            x=threshold_x,
            line_width=2,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text=f"Flag threshold {threshold:.2f}",
            annotation_position="top right",
        )

    return apply_product_layout(
        figure,
        height=540,
        legend_title="LOF status",
    )

def build_outlier_comparison_figure(summary: pd.DataFrame) -> go.Figure | None:
    """Compare available outlier methods using one readable metric."""

    if summary.empty:
        return None

    method_column = first_existing_column(summary, ["method", "model", "algorithm"])
    rate_column = first_existing_column(summary, ["outlier_rate", "anomaly_rate"])
    count_column = first_existing_column(summary, ["outlier_count", "anomaly_count"])
    value_column = rate_column or count_column

    if method_column is None or value_column is None:
        return None

    data = summary.copy().sort_values(value_column, ascending=True)
    figure = px.bar(
        data,
        x=value_column,
        y=method_column,
        orientation="h",
        text=value_column,
        color=method_column,
        color_discrete_sequence=CLUSTER_PALETTE,
    )
    figure.update_traces(texttemplate="%{text:.2%}" if rate_column else "%{text:,.0f}", textposition="outside")
    figure.update_xaxes(title_text="Outlier rate" if rate_column else "Outlier count")
    figure.update_yaxes(title_text="Method")
    figure.update_layout(showlegend=False)
    return apply_product_layout(figure, height=480)



def build_pca_density_figure(
    projection: pd.DataFrame,
) -> go.Figure | None:
    """Show visitor density in the PC1/PC2 plane without hiding overlap."""

    if projection.empty:
        return None

    pc1, pc2, _ = _projection_columns(projection)
    if pc1 is None or pc2 is None:
        return None

    data = projection[[pc1, pc2]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    if data.empty:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Histogram2dContour(
            x=data[pc1],
            y=data[pc2],
            contours={"coloring": "heatmap", "showlabels": False},
            colorscale=[
                [0.0, "rgba(7,13,24,0.0)"],
                [0.35, "#17233A"],
                [0.70, "#38BDF8"],
                [1.0, "#FBBF24"],
            ],
            ncontours=20,
            colorbar={"title": "Visitor density"},
            hovertemplate=(
                f"{pc1}: %{{x:.2f}}<br>{pc2}: %{{y:.2f}}"
                "<br>Density: %{z}<extra></extra>"
            ),
        )
    )

    sample = data.sample(
        min(1_500, len(data)),
        random_state=42,
    )
    figure.add_trace(
        go.Scattergl(
            x=sample[pc1],
            y=sample[pc2],
            mode="markers",
            name="Deterministic point sample",
            marker={
                "size": 4,
                "color": COLORS["text"],
                "opacity": 0.18,
                "line": {"width": 0},
            },
            hoverinfo="skip",
        )
    )
    figure.update_xaxes(title_text=pc1)
    figure.update_yaxes(title_text=pc2)
    return apply_product_layout(
        figure,
        height=590,
        legend_title="Layer",
    )


def build_parallel_profile_figure(
    profile: pd.DataFrame,
) -> go.Figure | None:
    """Compare cluster profiles across all standardized behaviour features."""

    if profile.empty:
        return None

    cluster_column = first_existing_column(
        profile,
        ["cluster", "kmeans_cluster"],
    )
    feature_column = first_existing_column(
        profile,
        ["feature", "feature_name"],
    )
    value_column = first_existing_column(
        profile,
        ["standardized_mean", "mean_z_score", "profile_value"],
    )
    if not cluster_column or not feature_column or not value_column:
        return None

    matrix = profile.pivot_table(
        index=cluster_column,
        columns=feature_column,
        values=value_column,
        aggfunc="mean",
    ).sort_index()
    if matrix.empty:
        return None

    cluster_values = pd.to_numeric(
        pd.Series(matrix.index),
        errors="coerce",
    ).fillna(pd.Series(range(len(matrix)))).to_numpy()

    dimensions = []
    for feature in matrix.columns:
        values = pd.to_numeric(matrix[feature], errors="coerce").fillna(0)
        low = float(min(values.min(), -0.25))
        high = float(max(values.max(), 0.25))
        dimensions.append(
            {
                "label": str(feature),
                "values": values,
                "range": [low, high],
            }
        )

    figure = go.Figure(
        data=go.Parcoords(
            line={
                "color": cluster_values,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Cluster"},
            },
            dimensions=dimensions,
            labelfont={"color": COLORS["text_muted"], "size": 11},
            tickfont={"color": COLORS["text_subtle"], "size": 10},
        )
    )
    return apply_product_layout(
        figure,
        height=600,
        legend_title="Cluster",
    )


def build_cluster_value_figure(
    summary: pd.DataFrame,
    *,
    value_column: str,
    value_label: str,
    value_is_rate: bool,
) -> go.Figure | None:
    """Compare cluster population share with one verified value metric."""

    if (
        summary.empty
        or "cluster" not in summary.columns
        or "visitor_share" not in summary.columns
        or value_column not in summary.columns
    ):
        return None

    data = summary.copy().sort_values("cluster")
    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )
    data = data.loc[data[value_column].notna()].copy()
    if data.empty:
        return None

    labels = [f"Cluster {value}" for value in data["cluster"]]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=labels,
            y=data["visitor_share"],
            name="Visitor share",
            marker_color=COLORS["brand"],
            text=data["visitor_share"],
            texttemplate="%{text:.1%}",
            textposition="outside",
            opacity=0.88,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=data[value_column],
            name=value_label,
            mode="lines+markers+text",
            line={"color": COLORS["positive"], "width": 3},
            marker={"size": 10},
            text=data[value_column],
            texttemplate=("%{text:.1%}" if value_is_rate else "%{text:.3f}"),
            textposition="top center",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(
        title_text="Visitor share",
        tickformat=".0%",
        range=[0, max(float(data["visitor_share"].max()) * 1.25, 0.1)],
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text=value_label,
        tickformat=".1%" if value_is_rate else ".3f",
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Behaviour cluster")
    return apply_product_layout(
        figure,
        height=560,
        legend_title="Cluster evidence",
    )


def build_dbscan_k_distance_figure(
    data: pd.DataFrame,
    *,
    selected_eps: float,
) -> go.Figure | None:
    """Show the sorted neighbour distance curve and selected epsilon."""

    required = {"rank_percentile", "k_distance"}
    if data.empty or not required.issubset(data.columns):
        return None

    working = data.copy()
    working["rank_percentile"] = pd.to_numeric(
        working["rank_percentile"],
        errors="coerce",
    )
    working["k_distance"] = pd.to_numeric(
        working["k_distance"],
        errors="coerce",
    )
    working = working.dropna(subset=["rank_percentile", "k_distance"])
    if working.empty:
        return None

    figure = go.Figure(
        go.Scatter(
            x=working["rank_percentile"],
            y=working["k_distance"],
            mode="lines",
            name="Sorted neighbour distance",
            line={"color": COLORS["brand"], "width": 3},
            hovertemplate=(
                "Rank percentile: %{x:.1%}<br>"
                "Neighbour distance: %{y:.3f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=selected_eps,
        line_width=2,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Selected eps={selected_eps:g}",
        annotation_position="top left",
    )
    figure.update_xaxes(
        title_text="Sorted visitor rank",
        tickformat=".0%",
    )
    figure.update_yaxes(title_text="Distance to selected neighbour")
    return apply_product_layout(
        figure,
        height=520,
        legend_title="Density diagnostic",
    )


def build_dbscan_grid_figure(
    summary: pd.DataFrame,
    *,
    selected_eps: float,
    selected_min_samples: int,
) -> go.Figure | None:
    """Show same-sample DBSCAN quality and noise rate as two heatmaps."""

    required = {
        "eps",
        "min_samples",
        "silhouette_score_no_noise",
        "noise_rate",
        "cluster_count",
    }
    if summary.empty or not required.issubset(summary.columns):
        return None

    working = summary.copy()
    for column in required:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    eps_values = sorted(working["eps"].dropna().unique().tolist())
    min_values = sorted(
        working["min_samples"].dropna().astype(int).unique().tolist()
    )
    if not eps_values or not min_values:
        return None

    silhouette = working.pivot_table(
        index="min_samples",
        columns="eps",
        values="silhouette_score_no_noise",
    ).reindex(index=min_values, columns=eps_values)
    noise = working.pivot_table(
        index="min_samples",
        columns="eps",
        values="noise_rate",
    ).reindex(index=min_values, columns=eps_values)
    clusters = working.pivot_table(
        index="min_samples",
        columns="eps",
        values="cluster_count",
    ).reindex(index=min_values, columns=eps_values)

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Silhouette after excluding noise",
            "Noise rate",
        ),
        horizontal_spacing=0.13,
    )
    figure.add_trace(
        go.Heatmap(
            z=silhouette.values,
            x=eps_values,
            y=min_values,
            coloraxis="coloraxis",
            customdata=clusters.values,
            hovertemplate=(
                "eps=%{x}<br>min_samples=%{y}<br>"
                "Silhouette=%{z:.3f}<br>Clusters=%{customdata:.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=noise.values,
            x=eps_values,
            y=min_values,
            coloraxis="coloraxis2",
            customdata=clusters.values,
            hovertemplate=(
                "eps=%{x}<br>min_samples=%{y}<br>"
                "Noise=%{z:.1%}<br>Clusters=%{customdata:.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    for row_index, min_samples in enumerate(min_values):
        for col_index, eps in enumerate(eps_values):
            sil_value = silhouette.iloc[row_index, col_index]
            noise_value = noise.iloc[row_index, col_index]
            cluster_value = clusters.iloc[row_index, col_index]
            if pd.notna(sil_value):
                figure.add_annotation(
                    x=eps,
                    y=min_samples,
                    text=f"{sil_value:.3f}<br>{cluster_value:.0f} cl.",
                    showarrow=False,
                    font={"size": 11, "color": COLORS["text"]},
                    row=1,
                    col=1,
                )
            if pd.notna(noise_value):
                figure.add_annotation(
                    x=eps,
                    y=min_samples,
                    text=f"{noise_value:.1%}",
                    showarrow=False,
                    font={"size": 11, "color": COLORS["text"]},
                    row=1,
                    col=2,
                )

    for col in (1, 2):
        figure.add_trace(
            go.Scatter(
                x=[selected_eps],
                y=[selected_min_samples],
                mode="markers",
                name="Selected parameters" if col == 1 else None,
                showlegend=col == 1,
                marker={
                    "symbol": "square-open",
                    "size": 23,
                    "color": COLORS["warning"],
                    "line": {"width": 3},
                },
                hovertemplate=(
                    f"Selected eps={selected_eps:g}<br>"
                    f"min_samples={selected_min_samples}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )

    figure.update_xaxes(title_text="Epsilon", row=1, col=1)
    figure.update_xaxes(title_text="Epsilon", row=1, col=2)
    figure.update_yaxes(title_text="Minimum samples", row=1, col=1)
    figure.update_yaxes(title_text="Minimum samples", row=1, col=2)
    figure.update_layout(
        coloraxis={
            "colorscale": "Viridis",
            "colorbar": {"title": "Silhouette", "x": 0.46},
        },
        coloraxis2={
            "colorscale": [
                [0.0, COLORS["positive"]],
                [0.55, COLORS["warning"]],
                [1.0, COLORS["critical"]],
            ],
            "colorbar": {"title": "Noise rate", "tickformat": ".0%"},
        },
    )
    return apply_product_layout(
        figure,
        height=590,
        legend_title="Selection",
    )


def build_lof_deviation_figure(
    profile: pd.DataFrame,
) -> go.Figure | None:
    """Show which standardized behaviours differ most for LOF outliers."""

    required = {"feature", "standardized_gap"}
    if profile.empty or not required.issubset(profile.columns):
        return None

    data = profile.copy()
    data["standardized_gap"] = pd.to_numeric(
        data["standardized_gap"],
        errors="coerce",
    )
    data = data.dropna(subset=["standardized_gap"])
    data["absolute_gap"] = data["standardized_gap"].abs()
    data = data.sort_values("absolute_gap", ascending=True)
    colours = np.where(
        data["standardized_gap"] >= 0,
        COLORS["critical"],
        COLORS["information"],
    )

    figure = go.Figure(
        go.Bar(
            x=data["standardized_gap"],
            y=data["feature"],
            orientation="h",
            marker_color=colours,
            text=data["standardized_gap"],
            texttemplate="%{text:+.2f}σ",
            textposition="outside",
            hovertemplate=(
                "%{y}<br>Outlier minus normal mean: %{x:+.3f}σ"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    figure.update_xaxes(title_text="Outlier mean minus normal mean (standard deviations)")
    figure.update_yaxes(title_text="Behaviour feature")
    return apply_product_layout(figure, height=520)


def build_lof_feature_figure(
    selected_row: pd.Series,
) -> go.Figure | None:
    """Show one selected visitor's standardized feature deviations.

    These bars are not SHAP values and are not additive contributions. They
    simply show how far the selected visitor sits from the sample mean.
    """

    z_columns = [
        column
        for column in selected_row.index
        if str(column).startswith("z_")
    ]
    if not z_columns:
        return None

    labels = [
        str(column)[2:].replace("_", " ").title()
        for column in z_columns
    ]
    values = pd.to_numeric(
        selected_row[z_columns],
        errors="coerce",
    ).fillna(0)
    data = pd.DataFrame({"feature": labels, "z_score": values.values})
    data["absolute"] = data["z_score"].abs()
    data = data.sort_values("absolute", ascending=True)
    colours = np.where(
        data["z_score"] >= 0,
        COLORS["warning"],
        COLORS["information"],
    )

    figure = go.Figure(
        go.Bar(
            x=data["z_score"],
            y=data["feature"],
            orientation="h",
            marker_color=colours,
            text=data["z_score"],
            texttemplate="%{text:+.2f}σ",
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f}σ<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    figure.update_xaxes(title_text="Selected visitor deviation from sample mean")
    figure.update_yaxes(title_text="Behaviour feature")
    return apply_product_layout(figure, height=520)
