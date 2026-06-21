"""Render MLV-H01, MLV-H02, and MLV-H04 feature-health visuals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.visualization.feature_health_data import (
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    FeatureHealthBundle,
    build_feature_health_bundle,
)
from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    VisualQAResult,
    add_footer,
    add_title_block,
    apply_ml_visual_style,
    create_figure,
    reserve_chart_space,
    safe_filename,
    save_figure_with_qa,
    style_axes,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/feature_health"
)


def _format_compact_value(value: float) -> str:
    """Format feature values compactly for chart cards."""

    if not np.isfinite(value):
        return "n/a"

    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"

    return f"{value:.2f}"


def _distribution_display_values(
    feature_name: str,
    values: pd.Series,
) -> tuple[np.ndarray, str, bool]:
    """Return outlier-safe display values and scale metadata."""

    finite = values[
        np.isfinite(values)
    ].dropna().to_numpy(dtype=float)

    if feature_name in {
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "events_per_unique_item",
    }:
        return (
            np.log1p(
                np.clip(
                    finite,
                    a_min=0.0,
                    a_max=None,
                )
            ),
            "Log display",
            True,
        )

    return finite, "Original scale", False


def create_feature_distribution_profile(
    bundle: FeatureHealthBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create H01 seven-feature distribution profiles."""

    title = "MLV-H01 | Feature Distribution Profile"
    subtitle = (
        "Seven model features shown with outlier-safe density profiles, "
        "median and 95th-percentile markers, plus original-value summaries."
    )

    apply_ml_visual_style(DEFAULT_SPEC)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(
            DEFAULT_SPEC.width,
            DEFAULT_SPEC.height,
        ),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    axes_flat = axes.flatten()

    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: visitor_features.csv | "
            f"{bundle.source_rows:,} visitors"
        ),
        interpretation_note=(
            "Count and duration features use log display; cards retain original units."
        ),
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.12,
        top=0.78,
        hspace=0.64,
        wspace=0.30,
    )

    summary_lookup = (
        bundle.distribution_summary
        .set_index("feature")
    )

    for axis, feature_name in zip(
        axes_flat,
        FEATURE_COLUMNS,
    ):
        display_values, scale_label, is_log = (
            _distribution_display_values(
                feature_name,
                bundle.feature_frame[feature_name],
            )
        )

        if len(display_values) == 0:
            axis.text(
                0.5,
                0.5,
                "No finite values",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            continue

        display_upper = float(
            np.quantile(display_values, 0.995)
        )
        clipped = np.clip(
            display_values,
            None,
            display_upper,
        )

        axis.hist(
            clipped,
            bins=34,
            density=True,
            color=COLORS["blue"],
            alpha=0.78,
            edgecolor=COLORS["white"],
            linewidth=0.35,
        )

        row = summary_lookup.loc[feature_name]
        raw_median = float(row["median"])
        raw_p95 = float(row["p95"])

        median_display = (
            float(np.log1p(max(raw_median, 0.0)))
            if is_log
            else raw_median
        )
        p95_display = (
            float(np.log1p(max(raw_p95, 0.0)))
            if is_log
            else raw_p95
        )

        axis.axvline(
            median_display,
            color=COLORS["navy"],
            linewidth=2.0,
            label="Median",
        )
        axis.axvline(
            p95_display,
            color=COLORS["amber"],
            linewidth=1.8,
            linestyle="--",
            label="P95",
        )

        axis.set_title(
            FEATURE_DISPLAY_NAMES[feature_name],
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["navy"],
            pad=8,
        )
        axis.set_xlabel(scale_label)
        axis.set_ylabel("Density")
        style_axes(axis)

        axis.text(
            0.98,
            0.92,
            (
                f"Median {_format_compact_value(raw_median)}\n"
                f"P95 {_format_compact_value(raw_p95)}\n"
                f"Zero {float(row['zero_rate']):.1%}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color=COLORS["grey_700"],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": COLORS["white"],
                "edgecolor": COLORS["grey_200"],
                "linewidth": 0.8,
                "alpha": 0.94,
            },
        )

    guide_axis = axes_flat[-1]
    guide_axis.set_axis_off()
    guide_axis.text(
        0.05,
        0.88,
        "HOW TO READ",
        transform=guide_axis.transAxes,
        fontsize=9,
        fontweight="bold",
        color=COLORS["grey_500"],
        va="top",
    )
    guide_axis.text(
        0.05,
        0.68,
        (
            "• Blue bars show visitor density.\n"
            "• Navy line = median.\n"
            "• Amber dashed line = 95th percentile.\n"
            "• Cards retain original units.\n"
            "• P99.5 clipping affects layout only."
        ),
        transform=guide_axis.transAxes,
        fontsize=9,
        color=COLORS["grey_700"],
        va="top",
        linespacing=1.6,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_correlation_cluster_map(
    bundle: FeatureHealthBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create H02 clustered Spearman correlation map."""

    title = "MLV-H02 | Feature Correlation Cluster Map"
    subtitle = (
        "Spearman correlations reordered so related behavioural features sit "
        "together; every cell retains the actual coefficient."
    )

    ordered = bundle.correlation_matrix.loc[
        bundle.clustered_order,
        bundle.clustered_order,
    ]
    values = ordered.to_numpy(dtype=float)

    colour_map = LinearSegmentedColormap.from_list(
        "correlation",
        [
            COLORS["red"],
            COLORS["grey_100"],
            COLORS["blue"],
        ],
    )

    fig, ax = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: visitor_features.csv | Spearman correlation"
        ),
        interpretation_note=(
            "Correlation measures association, not causation."
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
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )

    labels = [
        FEATURE_DISPLAY_NAMES[name]
        for name in ordered.index
    ]
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

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = float(
                values[row_index, column_index]
            )
            ax.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight=(
                    "bold"
                    if abs(value) >= 0.65
                    and row_index != column_index
                    else "normal"
                ),
                color=(
                    COLORS["white"]
                    if abs(value) >= 0.62
                    else COLORS["grey_900"]
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
        linewidth=1.6,
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
    colour_bar.set_ticks(
        [-1, -0.5, 0, 0.5, 1]
    )
    colour_bar.set_label(
        "Spearman correlation",
        fontsize=9,
    )
    colour_bar.outline.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_validity_matrix(
    validity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the percentage matrix displayed by H04."""

    return (
        validity_summary
        .set_index("display_name")[
            [
                "valid_rate",
                "missing_rate",
                "non_finite_rate",
                "negative_rate",
                "zero_rate",
                "ratio_above_one_rate",
            ]
        ]
        .rename(
            columns={
                "valid_rate": "Valid",
                "missing_rate": "Missing",
                "non_finite_rate": "Non-finite",
                "negative_rate": "Negative",
                "zero_rate": "Zero",
                "ratio_above_one_rate": "Ratio > 1",
            }
        )
        * 100.0
    )


def create_missingness_validity_map(
    bundle: FeatureHealthBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create H04 missingness and validity diagnostic map."""

    title = "MLV-H04 | Missingness and Validity Map"
    subtitle = (
        "Feature-level quality checks across the processed visitor snapshot; "
        "every cell shows the actual percentage."
    )

    matrix = build_validity_matrix(
        bundle.validity_summary
    )
    values = matrix.to_numpy(dtype=float)

    risk_values = values.copy()
    risk_values[:, 0] = 100.0 - risk_values[:, 0]

    colour_map = LinearSegmentedColormap.from_list(
        "quality_risk",
        [
            "#E8F4EC",
            "#FFF3D6",
            COLORS["red"],
        ],
    )

    upper_risk = max(
        1.0,
        float(np.quantile(risk_values, 0.95)),
    )

    fig, ax = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: visitor_features.csv | "
            f"{bundle.source_rows:,} visitors"
        ),
        interpretation_note=(
            "Dark cells need more attention; zero can be valid behaviour."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.22,
        right=0.95,
        bottom=0.14,
        top=0.80,
    )

    ax.imshow(
        risk_values,
        cmap=colour_map,
        vmin=0.0,
        vmax=upper_risk,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(matrix.columns)),
        labels=matrix.columns,
    )
    ax.set_yticks(
        np.arange(len(matrix.index)),
        labels=matrix.index,
    )
    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=10,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(
                values[row_index, column_index]
            )
            risk = float(
                risk_values[row_index, column_index]
            )

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight=(
                    "bold"
                    if column_index == 0 or value > 0
                    else "normal"
                ),
                color=(
                    COLORS["white"]
                    if risk >= upper_risk * 0.58
                    else COLORS["grey_900"]
                ),
            )

    ax.set_xticks(
        np.arange(-0.5, matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, matrix.shape[0], 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=COLORS["white"],
        linewidth=1.8,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def strongest_correlation_pair(
    correlation_matrix: pd.DataFrame,
) -> tuple[str, str, float]:
    """Return the strongest absolute off-diagonal correlation."""

    matrix = correlation_matrix.to_numpy(
        dtype=float
    ).copy()
    np.fill_diagonal(matrix, np.nan)

    flat_position = int(
        np.nanargmax(np.abs(matrix))
    )
    row_index, column_index = np.unravel_index(
        flat_position,
        matrix.shape,
    )

    return (
        str(correlation_matrix.index[row_index]),
        str(correlation_matrix.columns[column_index]),
        float(matrix[row_index, column_index]),
    )


def build_feature_health_insights(
    bundle: FeatureHealthBundle,
) -> str:
    """Create actual-number findings for H01, H02, and H04."""

    distribution = (
        bundle.distribution_summary
        .set_index("feature")
    )

    first_feature, second_feature, coefficient = (
        strongest_correlation_pair(
            bundle.correlation_matrix
        )
    )

    highest_zero = (
        bundle.validity_summary
        .sort_values(
            "zero_rate",
            ascending=False,
        )
        .iloc[0]
    )
    total_invalid = int(
        bundle.validity_summary[
            [
                "missing_count",
                "non_finite_count",
                "negative_count",
                "ratio_above_one_count",
            ]
        ]
        .sum(axis=1)
        .sum()
    )

    return f"""# Data and Feature Health Visual Intelligence — Actual-Number Findings

## MLV-H01 — Feature Distribution Profile

**What it shows:** Outlier-safe density profiles for all seven model features across **{bundle.source_rows:,}** processed visitors.

**Actual finding:** Median total events is **{distribution.loc['total_events', 'median']:.2f}** and the 95th percentile is **{distribution.loc['total_events', 'p95']:.2f}**. Median add-to-cart events is **{distribution.loc['addtocart_count', 'median']:.2f}**.

**Business conclusion:** Strongly skewed behaviour features justify robust scaling, percentile monitoring, and careful treatment of extreme visitors.

**Limitation:** Count and duration panels use log display for readability, while cards preserve original units.

## MLV-H02 — Feature Correlation Cluster Map

**Actual finding:** The strongest absolute off-diagonal association is **{FEATURE_DISPLAY_NAMES[first_feature]} × {FEATURE_DISPLAY_NAMES[second_feature]}** with Spearman correlation **{coefficient:+.3f}**.

**Business conclusion:** Highly related features may carry overlapping information. Keep this visible during feature review, but do not assume correlation means one behaviour causes another.

## MLV-H04 — Missingness and Validity Map

**Actual finding:** The current feature snapshot contains **{total_invalid:,}** missing, non-finite, negative, or ratio-above-one rule violations. The highest zero rate is **{highest_zero['display_name']}** at **{float(highest_zero['zero_rate']):.2%}**.

**Business conclusion:** Zero values must be interpreted feature by feature. Zero add-to-cart activity can be valid behaviour rather than a defect.

**Recommended action:** Re-run these checks during every scoring and retraining pipeline and alert only on rule violations or meaningful distribution shifts.
"""


def generate_feature_health_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: FeatureHealthBundle | None = None,
) -> dict[str, Any]:
    """Generate H01, H02, and H04 plus supporting evidence."""

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

    feature_bundle = (
        bundle
        if bundle is not None
        else build_feature_health_bundle(root)
    )

    visual_names = {
        "MLV-H01": "MLV-H01 Feature Distribution Profile",
        "MLV-H02": "MLV-H02 Feature Correlation Cluster Map",
        "MLV-H04": "MLV-H04 Missingness and Validity Map",
    }
    visual_paths = {
        visual_id: final_output_dir
        / f"{safe_filename(name)}.png"
        for visual_id, name in visual_names.items()
    }

    qa_results = {
        "MLV-H01": create_feature_distribution_profile(
            feature_bundle,
            visual_paths["MLV-H01"],
        ),
        "MLV-H02": create_correlation_cluster_map(
            feature_bundle,
            visual_paths["MLV-H02"],
        ),
        "MLV-H04": create_missingness_validity_map(
            feature_bundle,
            visual_paths["MLV-H04"],
        ),
    }

    distribution_path = (
        final_output_dir / "feature_distribution_summary.csv"
    )
    feature_bundle.distribution_summary.to_csv(
        distribution_path,
        index=False,
    )

    correlation_path = (
        final_output_dir / "feature_correlation_matrix.csv"
    )
    feature_bundle.correlation_matrix.to_csv(
        correlation_path
    )

    validity_path = (
        final_output_dir / "feature_validity_summary.csv"
    )
    feature_bundle.validity_summary.to_csv(
        validity_path,
        index=False,
    )

    insight_path = (
        final_output_dir / "feature_health_insights.md"
    )
    insight_path.write_text(
        build_feature_health_insights(
            feature_bundle
        ),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir / "feature_health_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Data and feature health",
        "source_rows": feature_bundle.source_rows,
        "features": FEATURE_COLUMNS,
        "clustered_order": feature_bundle.clustered_order,
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "distribution_summary": (
                str(distribution_path.relative_to(root))
                if distribution_path.is_relative_to(root)
                else str(distribution_path)
            ),
            "correlation_matrix": (
                str(correlation_path.relative_to(root))
                if correlation_path.is_relative_to(root)
                else str(correlation_path)
            ),
            "validity_summary": (
                str(validity_path.relative_to(root))
                if validity_path.is_relative_to(root)
                else str(validity_path)
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
