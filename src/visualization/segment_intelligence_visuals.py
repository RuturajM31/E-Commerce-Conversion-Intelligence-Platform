"""Render MLV-G02 score distribution by segment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    VisualQAResult,
    add_footer,
    add_title_block,
    apply_ml_visual_style,
    safe_filename,
    save_figure_with_qa,
    style_axes,
)
from src.visualization.segment_intelligence_data import (
    SegmentIntelligenceBundle,
    build_segment_intelligence_bundle,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/segment_intelligence"
)

SEGMENT_COLOURS = {
    "High Intent": "#123B6D",
    "Strong Intent": "#1F6F8B",
    "Warm Intent": "#2A9D8F",
    "Low Intent": "#E9A23B",
    "Very Low Intent": "#D66A55",
}


def _segment_colour(
    segment: str,
    index: int,
) -> str:
    """Return a controlled segment colour."""

    fallback = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["amber"],
        COLORS["red"],
        COLORS["grey_500"],
    ]

    return SEGMENT_COLOURS.get(
        segment,
        fallback[index % len(fallback)],
    )


def _density_curve(
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Create a fast smoothed-histogram density curve."""

    histogram, edges = np.histogram(
        values,
        bins=90,
        range=(0.0, 1.0),
        density=True,
    )
    centres = (
        edges[:-1] + edges[1:]
    ) / 2.0

    # A short weighted kernel creates a calm ridge without external plotting
    # dependencies or expensive repeated KDE calls.
    kernel = np.array(
        [1, 2, 4, 7, 10, 7, 4, 2, 1],
        dtype=float,
    )
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(
        histogram,
        kernel,
        mode="same",
    )

    return np.interp(
        grid,
        centres,
        smoothed,
        left=0.0,
        right=0.0,
    )


def _add_summary_card(
    axis,
    *,
    y_position: float,
    segment: str,
    visitors: int,
    share: float,
    median: float,
    p90: float,
    colour: str,
) -> None:
    """Add one right-side segment evidence card."""

    card = FancyBboxPatch(
        (1.025, y_position - 0.34),
        0.30,
        0.64,
        transform=axis.get_yaxis_transform(),
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor=COLORS["white"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.0,
        clip_on=False,
    )
    axis.add_patch(card)

    axis.text(
        1.045,
        y_position + 0.16,
        segment,
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=colour,
        clip_on=False,
    )
    axis.text(
        1.045,
        y_position - 0.01,
        f"{visitors:,} visitors | {share:.1%}",
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["grey_700"],
        clip_on=False,
    )
    axis.text(
        1.045,
        y_position - 0.18,
        f"Median {median:.3f} | P90 {p90:.3f}",
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["grey_700"],
        clip_on=False,
    )


def create_score_distribution_by_segment(
    bundle: SegmentIntelligenceBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create G02 ridgeline score distributions by intent segment."""

    title = "MLV-G02 | Score Distribution by Segment"
    subtitle = (
        "Purchase-intent score density by business segment, with median, "
        "interquartile range, visitor share, and actual-number summaries."
    )

    apply_ml_visual_style(DEFAULT_SPEC)

    fig, axis = plt.subplots(
        figsize=(
            DEFAULT_SPEC.width,
            DEFAULT_SPEC.height,
        ),
        facecolor=DEFAULT_SPEC.facecolor,
    )

    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: visitor_scores.csv | "
            f"{bundle.source_rows:,} visitors"
        ),
        interpretation_note=(
            "Segments are score bands; matured conversion outcomes are unavailable."
        ),
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.72,
        bottom=0.14,
        top=0.80,
    )

    grid = np.linspace(0.0, 1.0, 500)
    summary_lookup = (
        bundle.segment_summary
        .set_index("intent_segment")
    )
    y_positions = np.arange(
        len(bundle.segment_order)
    )[::-1]

    for index, (segment, y_position) in enumerate(
        zip(
            bundle.segment_order,
            y_positions,
        )
    ):
        values = (
            bundle.visitor_scores.loc[
                bundle.visitor_scores[
                    "intent_segment"
                ] == segment,
                "purchase_intent_score",
            ]
            .to_numpy(dtype=float)
        )

        density = _density_curve(
            values,
            grid,
        )
        maximum = float(density.max())

        if maximum > 0:
            density = (
                density / maximum * 0.70
            )

        colour = _segment_colour(
            segment,
            index,
        )
        baseline = float(y_position)

        axis.fill_between(
            grid,
            baseline,
            baseline + density,
            color=colour,
            alpha=0.78,
            linewidth=0,
        )
        axis.plot(
            grid,
            baseline + density,
            color=colour,
            linewidth=1.8,
        )
        axis.hlines(
            baseline,
            xmin=0.0,
            xmax=1.0,
            color=COLORS["grey_300"],
            linewidth=0.8,
        )

        row = summary_lookup.loc[segment]
        median = float(row["median"])
        p25 = float(row["p25"])
        p75 = float(row["p75"])
        p90 = float(row["p90"])

        axis.plot(
            [p25, p75],
            [baseline + 0.10, baseline + 0.10],
            color=COLORS["navy"],
            linewidth=5.0,
            solid_capstyle="round",
            zorder=5,
        )
        axis.scatter(
            [median],
            [baseline + 0.10],
            s=74,
            marker="D",
            color=COLORS["white"],
            edgecolor=COLORS["navy"],
            linewidth=1.5,
            zorder=6,
        )

        _add_summary_card(
            axis,
            y_position=baseline + 0.22,
            segment=segment,
            visitors=int(row["visitors"]),
            share=float(row["visitor_share"]),
            median=median,
            p90=p90,
            colour=colour,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(
        -0.35,
        len(bundle.segment_order) - 0.02,
    )
    axis.set_yticks(
        y_positions + 0.22,
        labels=bundle.segment_order,
    )
    axis.set_xlabel(
        "Purchase-intent score"
    )
    axis.set_ylabel("")
    style_axes(axis)

    axis.text(
        1.025,
        len(bundle.segment_order) - 0.02,
        "SEGMENT EVIDENCE",
        transform=axis.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["grey_500"],
        clip_on=False,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_segment_insights(
    bundle: SegmentIntelligenceBundle,
) -> str:
    """Create actual-number findings and conditional status."""

    largest_segment = (
        bundle.segment_summary
        .sort_values(
            "visitors",
            ascending=False,
        )
        .iloc[0]
    )
    highest_segment = (
        bundle.segment_summary
        .sort_values(
            "median",
            ascending=False,
        )
        .iloc[0]
    )

    return f"""# Segment Intelligence — Actual-Number Findings

## MLV-G02 — Score Distribution by Segment

**What it shows:** Purchase-intent score distributions across **{bundle.source_rows:,}** visitors and **{len(bundle.segment_order)}** business segments.

**Actual finding:** **{highest_segment['intent_segment']}** has the highest median score at **{float(highest_segment['median']):.3f}**. The largest audience is **{largest_segment['intent_segment']}** with **{int(largest_segment['visitors']):,}** visitors, representing **{float(largest_segment['visitor_share']):.1%}** of the scored population.

**Business conclusion:** The score bands create a prioritisation ladder, but the largest audience is not automatically the right audience for expensive campaigns.

**Recommended action:** Use segment distributions for campaign sizing and targeting priority, then validate treatment effectiveness when matured outcomes become available.

**Limitation:** The current source has no actual converter counts or conversion rates. G02 explains score composition only; it does not prove observed conversion performance.

## Conditional segment visuals

- **MLV-G01 Segment-Level Performance Heatmap — CONDITIONAL:** requires actual outcomes by segment.
- **MLV-G03 Error Rate by Behaviour Cohort — CONDITIONAL:** requires row-level actual labels and prediction errors.
- **MLV-G04 Cohort Opportunity Matrix — CONDITIONAL:** requires validated conversion or business-value outcomes by cohort.
"""


def build_conditional_status() -> str:
    """Document the three conditional segment visuals."""

    return """# Segment and Cohort Conditional Status

| Visual ID | Status | Current blocker | Required next evidence |
|---|---|---|---|
| MLV-G01 | CONDITIONAL | `actual_converters` and `conversion_rate` are null for every segment | Matured row-level outcomes aggregated by `intent_segment` |
| MLV-G03 | CONDITIONAL | No actual labels are present in `visitor_scores.csv` | Visitor-level actual outcome and prediction/error flag |
| MLV-G04 | CONDITIONAL | Opportunity cannot be validated from scores alone | Cohort outcomes plus business value or campaign economics |

These are source-data boundaries, not charting failures. No conversion or error result is fabricated.
"""


def generate_segment_intelligence_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: SegmentIntelligenceBundle | None = None,
) -> dict[str, Any]:
    """Generate G02 plus evidence and conditional documentation."""

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

    segment_bundle = (
        bundle
        if bundle is not None
        else build_segment_intelligence_bundle(
            root
        )
    )

    visual_path = (
        final_output_dir
        / f"{safe_filename('MLV-G02 Score Distribution by Segment')}.png"
    )
    qa_result = (
        create_score_distribution_by_segment(
            segment_bundle,
            visual_path,
        )
    )

    summary_path = (
        final_output_dir
        / "segment_score_summary.csv"
    )
    segment_bundle.segment_summary.to_csv(
        summary_path,
        index=False,
    )

    insight_path = (
        final_output_dir
        / "segment_intelligence_insights.md"
    )
    insight_path.write_text(
        build_segment_insights(
            segment_bundle
        ),
        encoding="utf-8",
    )

    conditional_path = (
        final_output_dir
        / "segment_conditional_status.md"
    )
    conditional_path.write_text(
        build_conditional_status(),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir
        / "segment_intelligence_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Segment intelligence",
        "source_rows": segment_bundle.source_rows,
        "segment_order": segment_bundle.segment_order,
        "outcomes_available": (
            segment_bundle.outcome_columns_available
        ),
        "supported_visuals": ["MLV-G02"],
        "conditional_visuals": {
            "MLV-G01": (
                "Actual outcomes by segment unavailable."
            ),
            "MLV-G03": (
                "Row-level actual labels unavailable."
            ),
            "MLV-G04": (
                "Validated cohort outcomes and value unavailable."
            ),
        },
        "artifacts": {
            "MLV-G02": (
                str(visual_path.relative_to(root))
                if visual_path.is_relative_to(root)
                else str(visual_path)
            )
        },
        "supporting_files": {
            "segment_summary": (
                str(summary_path.relative_to(root))
                if summary_path.is_relative_to(root)
                else str(summary_path)
            ),
            "insights": (
                str(insight_path.relative_to(root))
                if insight_path.is_relative_to(root)
                else str(insight_path)
            ),
            "conditional_status": (
                str(conditional_path.relative_to(root))
                if conditional_path.is_relative_to(root)
                else str(conditional_path)
            ),
        },
        "qa": {
            "MLV-G02": {
                "passed": qa_result.passed,
                "width_px": qa_result.width_px,
                "height_px": qa_result.height_px,
                "checks": qa_result.checks,
            }
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
