"""Render MLV-E01 to MLV-E06 explainability visual intelligence.

Why this file exists:
    The native XGBoost explanation bundle needs to be translated into clean,
    business-readable visuals with actual-number findings and honest limits.

Visuals created:
    - MLV-E01: Global SHAP Beeswarm
    - MLV-E02: Global Feature-Impact Ranking
    - MLV-E03: SHAP Dependence Plot
    - MLV-E04: Visitor-Level Waterfall
    - MLV-E05: Representative Visitor Decision Paths
    - MLV-E06: Feature Interaction Heatmap

Main output:
    Six QA-validated PNGs plus CSV evidence, insight Markdown, and a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.visualization.explainability_data import (
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    ExplainabilityBundle,
    build_explainability_bundle,
    sigmoid,
)
from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    VisualQAResult,
    add_footer,
    add_title_block,
    apply_ml_visual_style,
    create_figure,
    percent_formatter,
    reserve_chart_space,
    safe_filename,
    save_figure_with_qa,
    style_axes,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/explainability"
)


# ---------------------------------------------------------------------------
# Summary calculations
# ---------------------------------------------------------------------------


def build_feature_impact_summary(
    bundle: ExplainabilityBundle,
) -> pd.DataFrame:
    """Create global impact and direction evidence.

    Input:
        Explanation bundle.

    Output:
        One row per feature with mean absolute contribution, mean signed
        contribution, value/contribution correlation, and rank.

    Used next:
        E01, E02, E03, insight text, and evidence CSVs.
    """

    records: list[dict[str, Any]] = []

    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        feature_values = bundle.feature_frame[
            feature_name
        ].to_numpy(dtype=float)
        contributions = bundle.shap_values[
            :,
            feature_index,
        ].astype(float)

        # Correlation can be undefined for a constant feature.
        if (
            np.std(feature_values) == 0
            or np.std(contributions) == 0
        ):
            correlation = 0.0
        else:
            correlation = float(
                np.corrcoef(
                    feature_values,
                    contributions,
                )[0, 1]
            )

        records.append(
            {
                "feature": feature_name,
                "display_name": FEATURE_DISPLAY_NAMES[
                    feature_name
                ],
                "mean_abs_contribution": float(
                    np.mean(np.abs(contributions))
                ),
                "mean_signed_contribution": float(
                    np.mean(contributions)
                ),
                "value_contribution_correlation": correlation,
            }
        )

    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        "mean_abs_contribution",
        ascending=False,
    ).reset_index(drop=True)
    summary["impact_rank"] = (
        np.arange(len(summary)) + 1
    )

    return summary


def build_interaction_summary(
    bundle: ExplainabilityBundle,
) -> pd.DataFrame:
    """Create the mean absolute feature-interaction matrix."""

    interaction_strength = np.mean(
        np.abs(bundle.interaction_values),
        axis=0,
    )

    return pd.DataFrame(
        interaction_strength,
        index=[
            FEATURE_DISPLAY_NAMES[name]
            for name in FEATURE_COLUMNS
        ],
        columns=[
            FEATURE_DISPLAY_NAMES[name]
            for name in FEATURE_COLUMNS
        ],
    )


def select_dependence_pair(
    feature_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
) -> tuple[str, str]:
    """Select the leading feature and strongest interaction partner."""

    leading_feature = str(
        feature_summary.iloc[0]["feature"]
    )
    leading_display = FEATURE_DISPLAY_NAMES[
        leading_feature
    ]

    partner_strengths = (
        interaction_summary.loc[leading_display]
        .drop(index=leading_display)
        .sort_values(ascending=False)
    )

    if partner_strengths.empty:
        partner_display = FEATURE_DISPLAY_NAMES[
            FEATURE_COLUMNS[1]
        ]
    else:
        partner_display = str(
            partner_strengths.index[0]
        )

    reverse_lookup = {
        display: feature
        for feature, display in FEATURE_DISPLAY_NAMES.items()
    }

    return (
        leading_feature,
        reverse_lookup[partner_display],
    )


def representative_evidence(
    bundle: ExplainabilityBundle,
) -> pd.DataFrame:
    """Create one evidence row for each representative visitor."""

    records: list[dict[str, Any]] = []

    for segment, position in bundle.representative_positions.items():
        contributions = bundle.shap_values[position]
        top_index = int(
            np.argmax(np.abs(contributions))
        )

        records.append(
            {
                "segment": segment,
                "sample_position": int(position),
                "source_position": int(
                    bundle.source_positions[position]
                ),
                "visitor_id": str(
                    bundle.visitor_ids.iloc[position]
                ),
                "probability": float(
                    bundle.probabilities[position]
                ),
                "top_driver": FEATURE_DISPLAY_NAMES[
                    FEATURE_COLUMNS[top_index]
                ],
                "top_driver_contribution": float(
                    contributions[top_index]
                ),
            }
        )

    return pd.DataFrame(records)


def strongest_interaction_pair(
    interaction_summary: pd.DataFrame,
) -> tuple[str, str, float]:
    """Return the strongest off-diagonal interaction."""

    matrix = interaction_summary.to_numpy(dtype=float).copy()

    # Ignore diagonal main-effect cells for pair selection.
    np.fill_diagonal(matrix, -np.inf)

    row_index, column_index = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    return (
        str(interaction_summary.index[row_index]),
        str(interaction_summary.columns[column_index]),
        float(matrix[row_index, column_index]),
    )


# ---------------------------------------------------------------------------
# Shared render helpers
# ---------------------------------------------------------------------------


def _feature_colour_map() -> LinearSegmentedColormap:
    """Return a restrained low-to-high feature-value colour map."""

    return LinearSegmentedColormap.from_list(
        "feature_value",
        [
            COLORS["blue"],
            COLORS["grey_100"],
            COLORS["red"],
        ],
    )


def _add_probability_card(
    axis,
    *,
    x: float,
    y: float,
    label: str,
    value: str,
    note: str,
    accent: str,
) -> None:
    """Draw one compact probability or identity card."""

    card = FancyBboxPatch(
        (x, y),
        0.29,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.0,
    )
    axis.add_patch(card)

    axis.text(
        x + 0.02,
        y + 0.125,
        label.upper(),
        transform=axis.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["grey_500"],
        ha="left",
        va="top",
    )
    axis.text(
        x + 0.02,
        y + 0.075,
        value,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color=accent,
        ha="left",
        va="center",
    )
    axis.text(
        x + 0.02,
        y + 0.018,
        note,
        transform=axis.transAxes,
        fontsize=7.5,
        color=COLORS["grey_700"],
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# MLV-E01 Global SHAP Beeswarm
# ---------------------------------------------------------------------------


def create_global_beeswarm(
    bundle: ExplainabilityBundle,
    feature_summary: pd.DataFrame,
    output_path: Path,
) -> VisualQAResult:
    """Create a global contribution beeswarm."""

    title = "MLV-E01 | Global SHAP Beeswarm"
    subtitle = (
        "Native XGBoost contribution distribution across the deterministic "
        "visitor sample; colour represents each feature's relative value."
    )

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note=(
            "Sources: final champion model and visitor_features.csv | "
            f"Explanation sample: {bundle.sample_rows:,} visitors"
        ),
        interpretation_note=(
            "Positive values increase model score; negative values reduce it."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.20,
        right=0.91,
        bottom=0.13,
        top=0.82,
    )

    ordered_features = feature_summary[
        "feature"
    ].tolist()
    colour_map = _feature_colour_map()
    rng = np.random.default_rng(42)

    for y_position, feature_name in enumerate(
        ordered_features
    ):
        feature_index = FEATURE_COLUMNS.index(
            feature_name
        )
        values = bundle.feature_frame[
            feature_name
        ].to_numpy(dtype=float)
        contributions = bundle.shap_values[
            :,
            feature_index,
        ].astype(float)

        # Winsorised colour scaling prevents extreme values dominating colour.
        low, high = np.quantile(
            values,
            [0.02, 0.98],
        )

        if np.isclose(low, high):
            scaled_values = np.full(
                len(values),
                0.5,
            )
        else:
            scaled_values = np.clip(
                (values - low) / (high - low),
                0.0,
                1.0,
            )

        jitter = rng.normal(
            loc=0.0,
            scale=0.075,
            size=len(values),
        )

        ax.scatter(
            contributions,
            np.full(len(values), y_position) + jitter,
            c=scaled_values,
            cmap=colour_map,
            vmin=0.0,
            vmax=1.0,
            s=15,
            alpha=0.70,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(
        0.0,
        color=COLORS["grey_500"],
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_yticks(
        np.arange(len(ordered_features)),
        labels=[
            FEATURE_DISPLAY_NAMES[name]
            for name in ordered_features
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to model margin (log-odds)")
    ax.set_ylabel("")
    style_axes(ax)

    # Add one compact low-to-high feature-value guide.
    colour_axis = fig.add_axes(
        [0.925, 0.24, 0.015, 0.46]
    )
    scalar_map = mpl.cm.ScalarMappable(
        norm=Normalize(0, 1),
        cmap=colour_map,
    )
    colour_bar = fig.colorbar(
        scalar_map,
        cax=colour_axis,
    )
    colour_bar.set_ticks([0, 1])
    colour_bar.set_ticklabels(["Low", "High"])
    colour_bar.set_label(
        "Relative feature value",
        fontsize=9,
    )
    colour_bar.outline.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-E02 Global Feature-Impact Ranking
# ---------------------------------------------------------------------------


def create_global_impact_ranking(
    feature_summary: pd.DataFrame,
    output_path: Path,
) -> VisualQAResult:
    """Create an executive global feature-impact ranking."""

    title = "MLV-E02 | Global Feature-Impact Ranking"
    subtitle = (
        "Mean absolute native contribution ranks overall model influence; "
        "direction describes how higher feature values relate to score."
    )

    ordered = feature_summary.sort_values(
        "mean_abs_contribution",
        ascending=True,
    ).reset_index(drop=True)

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: native XGBoost contributions",
        interpretation_note=(
            "Impact measures model influence, not business causality."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.22,
        right=0.93,
        bottom=0.12,
        top=0.82,
    )

    colours = [
        COLORS["blue"]
        if index == len(ordered) - 1
        else COLORS["teal"]
        for index in range(len(ordered))
    ]

    bars = ax.barh(
        ordered["display_name"],
        ordered["mean_abs_contribution"],
        color=colours,
        height=0.58,
        alpha=0.92,
    )

    maximum = float(
        ordered["mean_abs_contribution"].max()
    )
    right_padding = maximum * 0.48

    for bar, row in zip(
        bars,
        ordered.itertuples(index=False),
    ):
        value = float(row.mean_abs_contribution)
        correlation = float(
            row.value_contribution_correlation
        )

        if correlation > 0.15:
            direction = "Higher values ↑ score"
        elif correlation < -0.15:
            direction = "Higher values ↓ score"
        else:
            direction = "Non-linear / mixed"

        ax.text(
            value + maximum * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["navy"],
        )
        ax.text(
            maximum + maximum * 0.10,
            bar.get_y() + bar.get_height() / 2,
            direction,
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLORS["grey_700"],
        )

    ax.set_xlim(
        0.0,
        maximum + right_padding,
    )
    ax.set_xlabel("Mean absolute contribution (log-odds)")
    ax.set_ylabel("")
    style_axes(ax)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-E03 SHAP Dependence Plot
# ---------------------------------------------------------------------------


def create_dependence_plot(
    bundle: ExplainabilityBundle,
    leading_feature: str,
    partner_feature: str,
    output_path: Path,
) -> VisualQAResult:
    """Create the leading feature dependence and interaction view.

    Input:
        Explanation bundle plus the selected leading and partner features.

    Output:
        QA-validated dependence PNG with outlier-safe axis formatting.

    Used next:
        Business users can read the model response without extreme values
        compressing the main visitor pattern.
    """

    leading_index = FEATURE_COLUMNS.index(
        leading_feature
    )

    # Raw feature values are preserved for evidence and readable tick labels.
    raw_x_values = bundle.feature_frame[
        leading_feature
    ].to_numpy(dtype=float)
    y_values = bundle.shap_values[
        :,
        leading_index,
    ].astype(float)
    raw_colour_values = bundle.feature_frame[
        partner_feature
    ].to_numpy(dtype=float)

    # Count-like and duration features are strongly right-skewed.
    # `log1p` keeps zero values valid and prevents a few extreme visitors from
    # compressing the main relationship into the left edge of the chart.
    log_scale_features = {
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
    }
    use_log_x = leading_feature in log_scale_features

    if use_log_x:
        # Model features are expected to be non-negative counts or durations.
        # Clipping protects the display if an unexpected negative value exists.
        x_values = np.log1p(
            np.clip(
                raw_x_values,
                a_min=0.0,
                a_max=None,
            )
        )
        x_axis_label = (
            f"{FEATURE_DISPLAY_NAMES[leading_feature]} "
            "(log scale; ticks show original values)"
        )
    else:
        x_values = raw_x_values.copy()
        x_axis_label = FEATURE_DISPLAY_NAMES[
            leading_feature
        ]

    # Convert activity span from milliseconds into days for a readable
    # business-facing colour scale. Other features keep their native unit.
    if partner_feature == "activity_span_ms":
        colour_values = (
            raw_colour_values / 86_400_000.0
        )
        colour_label = "Activity span (days)"
    else:
        colour_values = raw_colour_values.copy()
        colour_label = FEATURE_DISPLAY_NAMES[
            partner_feature
        ]

    title = "MLV-E03 | SHAP Dependence Plot"

    if use_log_x:
        subtitle = (
            f"{FEATURE_DISPLAY_NAMES[leading_feature]} uses a log display "
            f"so the main visitor pattern remains readable; colour shows "
            f"{colour_label.lower()}."
        )
    else:
        subtitle = (
            f"{FEATURE_DISPLAY_NAMES[leading_feature]} value versus "
            f"contribution; colour shows {colour_label.lower()}."
        )

    fig, ax = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note="Source: native XGBoost contributions",
        interpretation_note=(
            "The binned median line shows model response, not causal effect."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.10,
        right=0.91,
        bottom=0.13,
        top=0.82,
    )

    # Winsorise only the colour normalisation. All points remain visible.
    low, high = np.quantile(
        colour_values,
        [0.02, 0.98],
    )
    normaliser = Normalize(
        vmin=float(low),
        vmax=(
            float(high)
            if not np.isclose(low, high)
            else float(low + 1.0)
        ),
        clip=True,
    )
    colour_map = _feature_colour_map()

    scatter = ax.scatter(
        x_values,
        y_values,
        c=colour_values,
        cmap=colour_map,
        norm=normaliser,
        s=24,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )

    # Calculate the median response on the displayed x variable so the line
    # matches the same log or linear scale shown to the user.
    unique_count = len(np.unique(x_values))
    bin_count = min(
        14,
        max(4, unique_count),
    )

    if unique_count > 1:
        bins = pd.qcut(
            x_values,
            q=bin_count,
            duplicates="drop",
        )
        binned = pd.DataFrame(
            {
                "x_display": x_values,
                "contribution": y_values,
                "bin": bins,
            }
        )
        medians = (
            binned
            .groupby(
                "bin",
                observed=True,
            )
            .agg(
                feature_value=("x_display", "median"),
                contribution=("contribution", "median"),
            )
            .dropna()
        )

        ax.plot(
            medians["feature_value"],
            medians["contribution"],
            color=COLORS["navy"],
            linewidth=2.8,
            marker="o",
            markersize=4.5,
            label="Binned median response",
            zorder=5,
        )
        ax.legend(loc="upper right")

    ax.axhline(
        0.0,
        color=COLORS["grey_500"],
        linewidth=1.1,
        linestyle="--",
    )
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(
        "Contribution to model margin (log-odds)"
    )
    style_axes(ax)

    if use_log_x:
        # Keep transformed values hidden from the viewer. Tick labels convert
        # back into the original count or duration scale.
        ax.set_xlim(
            0.0,
            float(np.max(x_values)) * 1.03,
        )

        def _original_value_label(
            transformed_value: float,
            _: int,
        ) -> str:
            """Convert one log1p tick back into a compact original value."""

            original_value = max(
                0.0,
                float(np.expm1(transformed_value)),
            )

            if original_value >= 1_000_000:
                return f"{original_value / 1_000_000:.1f}M"

            if original_value >= 1_000:
                return f"{original_value / 1_000:.1f}K"

            if original_value >= 100:
                return f"{original_value:.0f}"

            if original_value >= 10:
                return f"{original_value:.0f}"

            return f"{original_value:.1f}"

        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(
                _original_value_label
            )
        )

    colour_axis = fig.add_axes(
        [0.925, 0.24, 0.015, 0.46]
    )
    colour_bar = fig.colorbar(
        scatter,
        cax=colour_axis,
    )
    colour_bar.set_label(
        colour_label,
        fontsize=9,
    )
    colour_bar.outline.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-E04 Visitor-Level Waterfall
# ---------------------------------------------------------------------------


def create_visitor_waterfall(
    bundle: ExplainabilityBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create one priority visitor waterfall explanation."""

    position = int(bundle.priority_position)
    contributions = bundle.shap_values[
        position
    ].astype(float)
    base_margin = float(
        bundle.base_values[position]
    )
    final_margin = float(
        base_margin + contributions.sum()
    )
    baseline_probability = float(
        sigmoid(base_margin)
    )
    final_probability = float(
        sigmoid(final_margin)
    )

    order = np.argsort(
        np.abs(contributions)
    )[::-1]
    ordered_contributions = contributions[order]
    ordered_names = [
        FEATURE_DISPLAY_NAMES[
            FEATURE_COLUMNS[index]
        ]
        for index in order
    ]

    title = "MLV-E04 | Visitor-Level Waterfall"
    subtitle = (
        "How each feature moved the highest-score sampled visitor from the "
        "model baseline margin to the final purchase-intent probability."
    )

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note=(
            "Source: native XGBoost contributions | "
            f"Visitor ID: {bundle.visitor_ids.iloc[position]}"
        ),
        interpretation_note=(
            "Local contributions describe this prediction only."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.22,
        right=0.95,
        bottom=0.14,
        top=0.78,
    )

    running_margin = base_margin
    starts: list[float] = []
    ends: list[float] = []

    for contribution in ordered_contributions:
        starts.append(running_margin)
        running_margin += float(contribution)
        ends.append(running_margin)

    y_positions = np.arange(
        len(ordered_names)
    )

    for y_position, start, end, contribution in zip(
        y_positions,
        starts,
        ends,
        ordered_contributions,
    ):
        left = min(start, end)
        width = abs(end - start)
        colour = (
            COLORS["red"]
            if contribution > 0
            else COLORS["blue"]
        )

        ax.barh(
            y_position,
            width,
            left=left,
            height=0.58,
            color=colour,
            alpha=0.90,
        )
        ax.plot(
            [end, end],
            [y_position - 0.29, y_position + 0.29],
            color=COLORS["grey_700"],
            linewidth=0.8,
        )
        ax.text(
            end,
            y_position,
            f" {contribution:+.3f}",
            ha="left" if contribution >= 0 else "right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["grey_900"],
        )

    ax.axvline(
        base_margin,
        color=COLORS["grey_500"],
        linestyle="--",
        linewidth=1.2,
        label="Baseline margin",
    )
    ax.axvline(
        final_margin,
        color=COLORS["navy"],
        linewidth=2.0,
        label="Final margin",
    )
    ax.set_yticks(
        y_positions,
        labels=ordered_names,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Model margin (log-odds)")
    ax.set_ylabel("")
    style_axes(ax)
    ax.legend(
        loc="lower right",
        ncol=2,
    )

    _add_probability_card(
        ax,
        x=0.02,
        y=1.03,
        label="Baseline probability",
        value=f"{baseline_probability:.1%}",
        note="Before visitor features",
        accent=COLORS["grey_700"],
    )
    _add_probability_card(
        ax,
        x=0.355,
        y=1.03,
        label="Final probability",
        value=f"{final_probability:.1%}",
        note="After all contributions",
        accent=COLORS["red"],
    )
    _add_probability_card(
        ax,
        x=0.69,
        y=1.03,
        label="Visitor",
        value=str(
            bundle.visitor_ids.iloc[position]
        )[:16],
        note="Highest score in explanation sample",
        accent=COLORS["navy"],
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-E05 Representative Visitor Decision Paths
# ---------------------------------------------------------------------------


def create_representative_paths(
    bundle: ExplainabilityBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create comparable low-, medium-, and high-intent decision paths.

    Input:
        Explanation bundle with deterministic representative visitors.

    Output:
        QA-validated three-panel local contribution comparison.

    Used next:
        Users can compare visitor paths on one shared scale without label
        collisions or misleading differences caused by separate axes.
    """

    title = "MLV-E05 | Representative Visitor Decision Paths"
    subtitle = (
        "Low-, medium-, and high-intent visitors use the same five features "
        "and one shared contribution scale for direct comparison."
    )

    # Collect representative contribution vectors before rendering.
    visitor_records: list[dict[str, Any]] = []

    for segment, position in (
        bundle.representative_positions.items()
    ):
        visitor_records.append(
            {
                "segment": segment,
                "position": int(position),
                "contributions": bundle.shap_values[
                    position
                ].astype(float),
            }
        )

    # Use one common feature order across all three panels.
    # Features are ranked by their maximum absolute contribution among the
    # three representative visitors, making the comparison honest and stable.
    contribution_matrix = np.vstack(
        [
            record["contributions"]
            for record in visitor_records
        ]
    )
    maximum_feature_impact = np.max(
        np.abs(contribution_matrix),
        axis=0,
    )
    shared_feature_indices = np.argsort(
        maximum_feature_impact
    )[-5:]
    shared_feature_indices = shared_feature_indices[
        np.argsort(
            maximum_feature_impact[
                shared_feature_indices
            ]
        )
    ]
    shared_feature_names = [
        FEATURE_DISPLAY_NAMES[
            FEATURE_COLUMNS[index]
        ]
        for index in shared_feature_indices
    ]

    # One symmetric x-range prevents visual exaggeration across panels.
    shared_values = contribution_matrix[
        :,
        shared_feature_indices,
    ]
    global_max = float(
        np.max(np.abs(shared_values))
    )
    x_limit = max(
        global_max * 1.22,
        0.05,
    )
    label_padding = x_limit * 0.025
    inside_threshold = x_limit * 0.18

    apply_ml_visual_style(DEFAULT_SPEC)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            DEFAULT_SPEC.width,
            DEFAULT_SPEC.height,
        ),
        facecolor=DEFAULT_SPEC.facecolor,
        sharex=True,
    )

    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Sources: final champion model and visitor_features.csv"
        ),
        interpretation_note=(
            "Examples are score quantiles, not matured outcome labels."
        ),
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.17,
        top=0.79,
        wspace=0.30,
    )

    for axis, record in zip(
        axes,
        visitor_records,
    ):
        segment = str(record["segment"])
        position = int(record["position"])
        top_values = record["contributions"][
            shared_feature_indices
        ]

        colours = [
            (
                COLORS["red"]
                if value > 0
                else COLORS["blue"]
            )
            for value in top_values
        ]

        y_positions = np.arange(
            len(shared_feature_names)
        )

        bars = axis.barh(
            y_positions,
            top_values,
            color=colours,
            height=0.58,
            alpha=0.90,
        )

        # Use the same feature labels and order in every panel.
        axis.set_yticks(
            y_positions,
            labels=shared_feature_names,
        )

        for bar, value in zip(
            bars,
            top_values,
        ):
            # Large bars hold white labels inside the bar.
            # Small bars use an outside label with fixed padding.
            if value >= 0:
                if abs(value) >= inside_threshold:
                    x_text = value - label_padding
                    horizontal_alignment = "right"
                    text_colour = COLORS["white"]
                else:
                    x_text = value + label_padding
                    horizontal_alignment = "left"
                    text_colour = COLORS["grey_900"]
            else:
                if abs(value) >= inside_threshold:
                    x_text = value + label_padding
                    horizontal_alignment = "left"
                    text_colour = COLORS["white"]
                else:
                    x_text = value - label_padding
                    horizontal_alignment = "right"
                    text_colour = COLORS["grey_900"]

            axis.text(
                x_text,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                ha=horizontal_alignment,
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_colour,
                clip_on=True,
            )

        axis.axvline(
            0.0,
            color=COLORS["grey_500"],
            linewidth=1.0,
        )
        axis.set_xlim(
            -x_limit,
            x_limit,
        )
        axis.set_title(
            (
                f"{segment}\n"
                f"Score {bundle.probabilities[position]:.1%}"
            ),
            fontsize=12,
            fontweight="bold",
            color=COLORS["navy"],
            pad=12,
        )
        axis.set_xlabel("Local contribution")
        axis.set_ylabel("")
        style_axes(axis)

        # Place the visitor identifier inside the reserved lower margin.
        axis.text(
            0.5,
            -0.12,
            (
                f"Visitor "
                f"{str(bundle.visitor_ids.iloc[position])[:18]}"
            ),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["grey_700"],
        )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-E06 Feature Interaction Heatmap
# ---------------------------------------------------------------------------


def create_interaction_heatmap(
    interaction_summary: pd.DataFrame,
    output_path: Path,
) -> VisualQAResult:
    """Create the global native interaction-strength heatmap."""

    title = "MLV-E06 | Feature Interaction Heatmap"
    subtitle = (
        "Mean absolute native XGBoost interaction contribution across the "
        "deterministic interaction sample."
    )

    values = interaction_summary.to_numpy(
        dtype=float
    )
    maximum = float(values.max())

    colour_map = LinearSegmentedColormap.from_list(
        "interaction_strength",
        [
            COLORS["grey_100"],
            "#D8E7F8",
            COLORS["teal"],
            COLORS["navy"],
        ],
    )

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: native XGBoost interaction contributions",
        interpretation_note=(
            "Diagonal cells are main effects; off-diagonal cells are pairs."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.20,
        right=0.91,
        bottom=0.20,
        top=0.80,
    )

    image = ax.imshow(
        values,
        cmap=colour_map,
        vmin=0.0,
        vmax=maximum if maximum > 0 else 1.0,
        aspect="auto",
    )

    labels = interaction_summary.index.tolist()
    ax.set_xticks(
        np.arange(len(labels)),
        labels=labels,
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(labels)),
        labels=labels,
    )

    # Annotate every cell because there are only seven model features.
    threshold = maximum * 0.56

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    COLORS["white"]
                    if value >= threshold
                    else COLORS["grey_900"]
                ),
                fontweight=(
                    "bold"
                    if row_index != column_index
                    and value >= threshold
                    else "normal"
                ),
            )

    ax.set_xticks(
        np.arange(-0.5, len(labels), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(labels), 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=COLORS["white"],
        linewidth=1.5,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colour_axis = fig.add_axes(
        [0.925, 0.24, 0.015, 0.46]
    )
    colour_bar = fig.colorbar(
        image,
        cax=colour_axis,
    )
    colour_bar.set_label(
        "Mean absolute interaction",
        fontsize=9,
    )
    colour_bar.outline.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# Insights and package generation
# ---------------------------------------------------------------------------


def build_explainability_insights(
    *,
    bundle: ExplainabilityBundle,
    feature_summary: pd.DataFrame,
    leading_feature: str,
    partner_feature: str,
    representatives: pd.DataFrame,
    interaction_summary: pd.DataFrame,
) -> str:
    """Create actual-number findings for E01 to E06."""

    top_row = feature_summary.iloc[0]
    top_correlation = float(
        top_row["value_contribution_correlation"]
    )

    if top_correlation > 0.15:
        direction_text = "higher values generally increase model score"
    elif top_correlation < -0.15:
        direction_text = "higher values generally reduce model score"
    else:
        direction_text = "the relationship is non-linear or mixed"

    priority = int(bundle.priority_position)
    priority_contributions = bundle.shap_values[
        priority
    ]
    priority_top_index = int(
        np.argmax(np.abs(priority_contributions))
    )
    priority_top_feature = FEATURE_DISPLAY_NAMES[
        FEATURE_COLUMNS[priority_top_index]
    ]
    priority_top_value = float(
        priority_contributions[priority_top_index]
    )

    first_feature, second_feature, interaction_value = (
        strongest_interaction_pair(interaction_summary)
    )

    representative_lines = "\n".join(
        (
            f"- **{row.segment}:** visitor `{row.visitor_id}`, "
            f"score **{row.probability:.1%}**, leading driver "
            f"**{row.top_driver}** ({row.top_driver_contribution:+.3f})."
        )
        for row in representatives.itertuples(index=False)
    )

    return f"""# Explainability Visual Intelligence — Actual-Number Findings

## MLV-E01 — Global SHAP Beeswarm

**What it shows:** The distribution and direction of native XGBoost contributions across **{bundle.sample_rows:,}** sampled visitors.

**Actual finding:** **{top_row['display_name']}** has the largest global mean absolute contribution at **{float(top_row['mean_abs_contribution']):.3f}** log-odds.

**Business conclusion:** This feature has the strongest influence on how the saved champion separates higher- and lower-intent visitors.

**Limitation:** Contributions describe model behaviour and do not prove causality.

## MLV-E02 — Global Feature-Impact Ranking

**Actual finding:** For **{top_row['display_name']}**, {direction_text}; its value/contribution correlation is **{top_correlation:+.3f}**.

**Recommended action:** Prioritise data-quality monitoring and business interpretation for the highest-impact features before expanding the model.

## MLV-E03 — SHAP Dependence Plot

**What it shows:** The model response to **{FEATURE_DISPLAY_NAMES[leading_feature]}**, coloured by **{FEATURE_DISPLAY_NAMES[partner_feature]}**.

**Business conclusion:** Read the binned median line as the model's learned response pattern, not as a causal conversion curve.

## MLV-E04 — Visitor-Level Waterfall

**Actual finding:** The highest-score sampled visitor has probability **{float(bundle.probabilities[priority]):.1%}**. Its strongest local driver is **{priority_top_feature}** with contribution **{priority_top_value:+.3f}** log-odds.

**Recommended action:** Use local explanations to support campaign-review and debugging, not as standalone automated decisions.

## MLV-E05 — Representative Visitor Decision Paths

{representative_lines}

**Business conclusion:** Different visitors can reach different intent scores through different behavioural paths, even when the same global features dominate overall.

## MLV-E06 — Feature Interaction Heatmap

**Actual finding:** The strongest off-diagonal pair is **{first_feature} × {second_feature}** with mean absolute interaction **{interaction_value:.3f}**.

**Recommended action:** Monitor both features together and consider interaction-aware business rules only after confirming stability across time.

## Shared limitation

The visitor feature source does not contain matured production outcomes. These visuals explain the saved model's scoring logic; they do not measure live campaign lift or prove that changing a feature will change conversion.
"""


def generate_explainability_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: ExplainabilityBundle | None = None,
) -> dict[str, Any]:
    """Generate E01 to E06 plus supporting evidence.

    Input:
        Project root, optional output directory, and optional prebuilt bundle.

    Output:
        Manifest containing all artifact paths and automated QA results.

    Used next:
        The runner generates real artifacts and MLflow logs the approved folder.
    """

    root = Path(project_root)
    relative_output = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    final_output_dir = (
        relative_output
        if relative_output.is_absolute()
        else root / relative_output
    )
    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanation_bundle = (
        bundle
        if bundle is not None
        else build_explainability_bundle(root)
    )

    feature_summary = build_feature_impact_summary(
        explanation_bundle
    )
    interaction_summary = build_interaction_summary(
        explanation_bundle
    )
    leading_feature, partner_feature = select_dependence_pair(
        feature_summary,
        interaction_summary,
    )
    representatives = representative_evidence(
        explanation_bundle
    )

    visual_names = {
        "MLV-E01": "MLV-E01 Global SHAP Beeswarm",
        "MLV-E02": "MLV-E02 Global Feature Impact Ranking",
        "MLV-E03": "MLV-E03 SHAP Dependence Plot",
        "MLV-E04": "MLV-E04 Visitor Level Waterfall",
        "MLV-E05": "MLV-E05 Representative Visitor Decision Paths",
        "MLV-E06": "MLV-E06 Feature Interaction Heatmap",
    }
    visual_paths = {
        visual_id: final_output_dir
        / f"{safe_filename(name)}.png"
        for visual_id, name in visual_names.items()
    }

    qa_results = {
        "MLV-E01": create_global_beeswarm(
            explanation_bundle,
            feature_summary,
            visual_paths["MLV-E01"],
        ),
        "MLV-E02": create_global_impact_ranking(
            feature_summary,
            visual_paths["MLV-E02"],
        ),
        "MLV-E03": create_dependence_plot(
            explanation_bundle,
            leading_feature,
            partner_feature,
            visual_paths["MLV-E03"],
        ),
        "MLV-E04": create_visitor_waterfall(
            explanation_bundle,
            visual_paths["MLV-E04"],
        ),
        "MLV-E05": create_representative_paths(
            explanation_bundle,
            visual_paths["MLV-E05"],
        ),
        "MLV-E06": create_interaction_heatmap(
            interaction_summary,
            visual_paths["MLV-E06"],
        ),
    }

    feature_summary_path = (
        final_output_dir / "global_feature_impact.csv"
    )
    feature_summary.to_csv(
        feature_summary_path,
        index=False,
    )

    representative_path = (
        final_output_dir / "representative_visitors.csv"
    )
    representatives.to_csv(
        representative_path,
        index=False,
    )

    interaction_path = (
        final_output_dir / "interaction_strength_matrix.csv"
    )
    interaction_summary.to_csv(interaction_path)

    insight_path = (
        final_output_dir / "explainability_insights.md"
    )
    insight_path.write_text(
        build_explainability_insights(
            bundle=explanation_bundle,
            feature_summary=feature_summary,
            leading_feature=leading_feature,
            partner_feature=partner_feature,
            representatives=representatives,
            interaction_summary=interaction_summary,
        ),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir / "explainability_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Explainability",
        "source_rows": explanation_bundle.source_rows,
        "sample_rows": explanation_bundle.sample_rows,
        "interaction_rows": int(
            len(explanation_bundle.interaction_sample_positions)
        ),
        "model_name": explanation_bundle.model_name,
        "features": FEATURE_COLUMNS,
        "missing_counts": explanation_bundle.missing_counts,
        "dependence_feature": leading_feature,
        "dependence_partner": partner_feature,
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "feature_impact": (
                str(feature_summary_path.relative_to(root))
                if feature_summary_path.is_relative_to(root)
                else str(feature_summary_path)
            ),
            "representatives": (
                str(representative_path.relative_to(root))
                if representative_path.is_relative_to(root)
                else str(representative_path)
            ),
            "interaction_matrix": (
                str(interaction_path.relative_to(root))
                if interaction_path.is_relative_to(root)
                else str(interaction_path)
            ),
            "insights": (
                str(insight_path.relative_to(root))
                if insight_path.is_relative_to(root)
                else str(insight_path)
            ),
        },
        "qa": {
            visual_id: {
                "passed": result.passed,
                "width_px": result.width_px,
                "height_px": result.height_px,
                "checks": result.checks,
            }
            for visual_id, result in qa_results.items()
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest["supporting_files"]["manifest"] = (
        str(manifest_path.relative_to(root))
        if manifest_path.is_relative_to(root)
        else str(manifest_path)
    )

    return manifest
