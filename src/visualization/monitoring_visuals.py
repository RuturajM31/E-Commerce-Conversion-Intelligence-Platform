"""Render monitoring visual intelligence J04 and J08."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

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
from src.visualization.monitoring_visual_data import (
    MonitoringVisualBundle,
    build_monitoring_visual_bundle,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/monitoring"
)

STATUS_COLOURS = {
    "Fresh": COLORS["green"],
    "Aging": COLORS["amber"],
    "Stale": COLORS["red"],
    "Missing": COLORS["grey_500"],
}


def _compact_count(value: int) -> str:
    """Format an integer without losing its exact meaning."""

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _age_text(age_hours: float | None) -> str:
    """Format source age for executive cards."""

    if age_hours is None or not np.isfinite(age_hours):
        return "No timestamp"
    if age_hours < 1:
        return f"{age_hours * 60:.0f} min old"
    if age_hours < 48:
        return f"{age_hours:.1f} h old"
    return f"{age_hours / 24:.1f} days old"



def delayed_label_retention_text(
    funnel: pd.DataFrame,
    index: int,
) -> str:
    """Return honest retention copy for one funnel stage."""

    if index == 0:
        return "Starting population"

    prior_count = int(funnel.iloc[index - 1]["count"])
    current_count = int(funnel.iloc[index]["count"])

    if prior_count == 0:
        return "N/A - prior stage has zero records"

    return f"{current_count / prior_count:.1%} of prior stage"



def create_delayed_label_maturity_funnel(
    bundle: MonitoringVisualBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create J04 as equal-width maturity cards so zero stages stay visible."""

    title = "MLV-J04 | Delayed-Label Maturity Funnel"
    subtitle = (
        "Operational progression from scored visitors to evaluable outcomes; "
        "zero stages remain fully visible and are treated as a source-data boundary."
    )
    apply_ml_visual_style(DEFAULT_SPEC)
    fig = plt.figure(
        figsize=(DEFAULT_SPEC.width, DEFAULT_SPEC.height),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    axis = fig.add_axes([0.07, 0.18, 0.62, 0.60])
    panel = fig.add_axes([0.73, 0.18, 0.23, 0.60])
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Sources: score export, production logs, delayed-label validation",
        interpretation_note="No matured labels means no production-performance claim.",
    )

    funnel = bundle.funnel.reset_index(drop=True)
    colours = [
        COLORS["navy"], COLORS["blue"], COLORS["teal"],
        COLORS["amber"], COLORS["green"],
    ]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    card_height = 0.145
    card_gap = 0.025
    start_y = 0.82

    for index, row in funnel.iterrows():
        y = start_y - index * (card_height + card_gap)
        count = int(row["count"])
        retention_text = delayed_label_retention_text(funnel, index)
        card = FancyBboxPatch(
            (0.02, y),
            0.96,
            card_height,
            transform=axis.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=COLORS["white"],
            edgecolor=colours[index],
            linewidth=2.0,
        )
        axis.add_patch(card)
        axis.add_patch(
            FancyBboxPatch(
                (0.02, y),
                0.018,
                card_height,
                transform=axis.transAxes,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                facecolor=colours[index],
                edgecolor=colours[index],
                linewidth=0,
            )
        )
        axis.text(
            0.065,
            y + card_height * 0.68,
            str(row["stage"]),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color=COLORS["navy"],
        )
        axis.text(
            0.065,
            y + card_height * 0.30,
            retention_text,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=7.7,
            color=COLORS["grey_700"],
        )
        axis.text(
            0.94,
            y + card_height * 0.52,
            f"{count:,}",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=colours[index],
        )

        if index < len(funnel) - 1:
            axis.annotate(
                "",
                xy=(0.50, y - 0.008),
                xytext=(0.50, y - card_gap + 0.010),
                xycoords=axis.transAxes,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLORS["grey_300"],
                    "linewidth": 1.3,
                },
            )

    panel.set_axis_off()
    panel.text(
        0.0, 0.98, "DELAYED-LABEL EVIDENCE",
        transform=panel.transAxes,
        ha="left", va="top",
        fontsize=9, fontweight="bold",
        color=COLORS["grey_500"],
    )
    y_cursor = 0.86
    for row in bundle.rejection_summary.itertuples(index=False):
        panel.text(
            0.0, y_cursor, str(row.reason),
            transform=panel.transAxes,
            ha="left", va="center",
            fontsize=8.5, color=COLORS["grey_700"],
        )
        panel.text(
            0.96, y_cursor, f"{int(row.count):,}",
            transform=panel.transAxes,
            ha="right", va="center",
            fontsize=9, fontweight="bold",
            color=COLORS["red"] if int(row.count) > 0 else COLORS["grey_500"],
        )
        y_cursor -= 0.10
    panel.axhline(
        y_cursor + 0.03,
        xmin=0, xmax=1,
        color=COLORS["grey_200"],
        linewidth=1.0,
    )
    evaluable = int(bundle.kpis["evaluable_rows"])
    coverage = float(bundle.kpis["label_coverage_rate"])
    outcome_available = bool(bundle.kpis["outcome_metrics_available"])
    panel.text(
        0.0, y_cursor - 0.04, "Evaluable outcome coverage",
        transform=panel.transAxes,
        fontsize=8.5, color=COLORS["grey_700"], va="top",
    )
    panel.text(
        0.0, y_cursor - 0.13, f"{coverage:.2%}",
        transform=panel.transAxes,
        fontsize=21, fontweight="bold",
        color=COLORS["green"] if evaluable > 0 else COLORS["red"],
        va="top",
    )
    panel.text(
        0.0, y_cursor - 0.26,
        "Production metrics available" if outcome_available else "Production metrics unavailable",
        transform=panel.transAxes,
        fontsize=8.5, fontweight="bold",
        color=COLORS["green"] if outcome_available else COLORS["red"],
        va="top",
    )
    panel.text(
        0.0, y_cursor - 0.36,
        (
            "The pipeline is ready, but the source does not contain future "
            "matured conversion outcomes."
            if not outcome_available
            else "Performance is calculated only from accepted matured labels."
        ),
        transform=panel.transAxes,
        fontsize=8, color=COLORS["grey_700"],
        va="top", wrap=True,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def _draw_card(
    fig,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    detail: str,
    status: str,
) -> None:
    """Draw one monitoring executive card."""

    colour = STATUS_COLOURS.get(
        status,
        COLORS["blue"],
    )
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=COLORS["white"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.1,
    )
    fig.add_artist(card)
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            0.010,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.0,rounding_size=0.01",
            facecolor=colour,
            edgecolor=colour,
            linewidth=0,
        )
    )

    fig.text(
        x + 0.025,
        y + height - 0.045,
        title.upper(),
        fontsize=8,
        fontweight="bold",
        color=COLORS["grey_500"],
        ha="left",
        va="top",
    )
    fig.text(
        x + 0.025,
        y + height * 0.52,
        value,
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )
    fig.text(
        x + 0.025,
        y + 0.035,
        detail,
        fontsize=7.8,
        color=COLORS["grey_700"],
        ha="left",
        va="bottom",
    )
    fig.text(
        x + width - 0.020,
        y + height - 0.045,
        status,
        fontsize=7.8,
        fontweight="bold",
        color=colour,
        ha="right",
        va="top",
    )


def create_monitoring_freshness_coverage_card(
    bundle: MonitoringVisualBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create J08 freshness, lineage, and coverage executive card."""

    title = "MLV-J08 | Monitoring Freshness and Data-Coverage Card"
    subtitle = (
        "Executive view of snapshot recency, prediction evidence, delayed-label "
        "coverage, champion lineage, and monitoring-source availability."
    )

    apply_ml_visual_style(DEFAULT_SPEC)
    fig = plt.figure(
        figsize=(
            DEFAULT_SPEC.width,
            DEFAULT_SPEC.height,
        ),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note=(
            "Sources: metrics cache, prediction logs, delayed labels, lineage"
        ),
        interpretation_note=(
            "Fresh ≤24h | Aging ≤72h | Stale >72h | Missing = unavailable"
        ),
    )

    status_lookup = bundle.source_status.set_index("source")

    def source_value(source: str) -> tuple[str, str]:
        if source not in status_lookup.index:
            return "Missing", "No evidence"
        row = status_lookup.loc[source]
        return str(row["freshness"]), _age_text(row["age_hours"])

    snapshot_status, snapshot_age = source_value(
        "Metrics snapshot"
    )
    prediction_status, prediction_age = source_value(
        "Prediction log"
    )

    coverage_rate = float(
        bundle.kpis["label_coverage_rate"]
    )
    evaluable = int(bundle.kpis["evaluable_rows"])
    prediction_rows = int(
        bundle.kpis["prediction_rows"]
    )
    outcome_available = bool(
        bundle.kpis["outcome_metrics_available"]
    )
    available_sources = int(
        bundle.kpis["available_sources"]
    )
    total_sources = int(
        bundle.kpis["total_sources"]
    )
    fresh_sources = int(
        bundle.kpis["fresh_sources"]
    )

    cards = [
        (
            "Metrics snapshot",
            snapshot_age,
            "Cached Prometheus/Grafana monitoring evidence",
            snapshot_status,
        ),
        (
            "Prediction log",
            prediction_age,
            f"{prediction_rows:,} operational prediction records",
            prediction_status,
        ),
        (
            "Delayed-label coverage",
            f"{coverage_rate:.2%}",
            f"{evaluable:,} evaluable outcomes",
            "Fresh" if evaluable > 0 else "Missing",
        ),
        (
            "Outcome metrics",
            "Available" if outcome_available else "Unavailable",
            "No fallback to offline validation metrics",
            "Fresh" if outcome_available else "Missing",
        ),
        (
            "Champion lineage",
            (
                f"v{bundle.kpis['registered_model_version']} "
                f"[{bundle.kpis['registered_model_alias']}]"
            ),
            str(bundle.kpis["registered_model_name"]),
            "Fresh"
            if bundle.kpis["registered_model_version"] != "unknown"
            else "Missing",
        ),
        (
            "Evidence coverage",
            f"{available_sources}/{total_sources}",
            f"{fresh_sources} sources currently fresh",
            (
                "Fresh"
                if available_sources == total_sources
                else (
                    "Aging"
                    if available_sources >= total_sources / 2
                    else "Missing"
                )
            ),
        ),
    ]

    x_positions = [0.08, 0.375, 0.67]
    y_positions = [0.49, 0.22]
    card_width = 0.25
    card_height = 0.20

    for index, card_data in enumerate(cards):
        row_index = index // 3
        column_index = index % 3
        _draw_card(
            fig,
            x=x_positions[column_index],
            y=y_positions[row_index],
            width=card_width,
            height=card_height,
            title=card_data[0],
            value=card_data[1],
            detail=card_data[2],
            status=card_data[3],
        )

    # Compact evidence strip shows each monitored source without overcrowding.
    strip_axis = fig.add_axes([0.08, 0.10, 0.84, 0.065])
    strip_axis.set_axis_off()
    strip_axis.text(
        0.0,
        1.08,
        "SOURCE STATUS",
        transform=strip_axis.transAxes,
        fontsize=8,
        fontweight="bold",
        color=COLORS["grey_500"],
        va="bottom",
    )

    source_rows = bundle.source_status.reset_index(drop=True)
    spacing = 1.0 / max(len(source_rows), 1)

    for index, row in source_rows.iterrows():
        x_position = index * spacing
        status = str(row["freshness"])
        colour = STATUS_COLOURS.get(
            status,
            COLORS["grey_500"],
        )
        strip_axis.scatter(
            [x_position + 0.015],
            [0.55],
            s=45,
            color=colour,
            edgecolor=COLORS["white"],
            linewidth=0.8,
        )
        strip_axis.text(
            x_position + 0.035,
            0.55,
            str(row["source"]),
            fontsize=6.8,
            color=COLORS["grey_700"],
            va="center",
            ha="left",
        )

    strip_axis.set_xlim(0, 1.02)
    strip_axis.set_ylim(0, 1)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_monitoring_insights(
    bundle: MonitoringVisualBundle,
) -> str:
    """Create actual-number monitoring findings."""

    scored = int(bundle.kpis["score_rows"])
    predictions = int(bundle.kpis["prediction_rows"])
    matured = int(bundle.kpis["matured_rows"])
    labels = int(bundle.kpis["label_rows"])
    evaluable = int(bundle.kpis["evaluable_rows"])
    coverage = float(bundle.kpis["label_coverage_rate"])
    available = bool(
        bundle.kpis["outcome_metrics_available"]
    )

    return f"""# Production Monitoring Visual Intelligence — Actual-Number Findings

## MLV-J04 — Delayed-Label Maturity Funnel

**What it shows:** The current operational evidence contains **{scored:,}** scored visitors, **{predictions:,}** logged production predictions, **{matured:,}** outcome-window-matured predictions, **{labels:,}** received labels, and **{evaluable:,}** evaluable matured outcomes.

**Actual finding:** Evaluable delayed-label coverage is **{coverage:.2%}**.

**Business conclusion:** {'Production performance can be calculated from matured outcomes.' if available else 'Production performance cannot yet be claimed because no accepted matured labels are available.'}

**Recommended action:** Continue ingesting final conversion outcomes after the complete observation window and rerun the controlled delayed-label evaluation.

## MLV-J08 — Monitoring Freshness and Data-Coverage Card

**Actual finding:** **{bundle.kpis['available_sources']} of {bundle.kpis['total_sources']}** monitored evidence sources are present, and **{bundle.kpis['fresh_sources']}** are within the 24-hour freshness threshold.

**Champion lineage:** **{bundle.kpis['registered_model_name']}**, version **{bundle.kpis['registered_model_version']}**, alias **{bundle.kpis['registered_model_alias']}**.

**Limitation:** Fresh monitoring infrastructure does not create missing outcome labels. Freshness and performance availability are separate controls.
"""


def build_monitoring_status_document() -> str:
    """Document supported, conditional, and blocked production visuals."""

    return """# Production Monitoring Visual Status

| Visual ID | Status | Evidence / blocker |
|---|---|---|
| MLV-J01 | CONDITIONAL | Requires at least two comparable timestamped prediction-score windows |
| MLV-J02 | CONDITIONAL | Requires a structured per-feature drift-severity summary |
| MLV-J03 | BLOCKED | Requires matured production outcomes for multiple deployed versions |
| MLV-J04 | SUPPORTED | Current funnel can show scored, logged, matured, labelled, and evaluable counts honestly |
| MLV-J05 | CONDITIONAL | Requires genuine Alertmanager/webhook incident history |
| MLV-J06 | BLOCKED | Requires both a drift event boundary and matured production outcomes |
| MLV-J07 | BLOCKED | Requires matured labelled outcomes for multiple model versions |
| MLV-J08 | SUPPORTED | Current freshness, lineage, source-presence, and label-coverage evidence is sufficient |

No alert, drift, conversion, or production-performance result is fabricated.
"""


def generate_monitoring_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: MonitoringVisualBundle | None = None,
) -> dict[str, Any]:
    """Generate J04 and J08 plus monitoring evidence and status records."""

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
    final_output_dir.mkdir(parents=True, exist_ok=True)

    monitoring_bundle = (
        bundle
        if bundle is not None
        else build_monitoring_visual_bundle(root)
    )

    visual_names = {
        "MLV-J04": "MLV-J04 Delayed Label Maturity Funnel",
        "MLV-J08": "MLV-J08 Monitoring Freshness and Data Coverage Card",
    }
    visual_paths = {
        visual_id: final_output_dir
        / f"{safe_filename(name)}.png"
        for visual_id, name in visual_names.items()
    }

    qa_results = {
        "MLV-J04": create_delayed_label_maturity_funnel(
            monitoring_bundle,
            visual_paths["MLV-J04"],
        ),
        "MLV-J08": create_monitoring_freshness_coverage_card(
            monitoring_bundle,
            visual_paths["MLV-J08"],
        ),
    }

    funnel_path = final_output_dir / "delayed_label_funnel.csv"
    monitoring_bundle.funnel.to_csv(funnel_path, index=False)

    rejection_path = final_output_dir / "delayed_label_rejections.csv"
    monitoring_bundle.rejection_summary.to_csv(
        rejection_path,
        index=False,
    )

    source_status_path = final_output_dir / "monitoring_source_status.csv"
    monitoring_bundle.source_status.to_csv(
        source_status_path,
        index=False,
    )

    insight_path = final_output_dir / "monitoring_visual_insights.md"
    insight_path.write_text(
        build_monitoring_insights(monitoring_bundle),
        encoding="utf-8",
    )

    status_path = final_output_dir / "monitoring_visual_status.md"
    status_path.write_text(
        build_monitoring_status_document(),
        encoding="utf-8",
    )

    manifest_path = final_output_dir / "monitoring_visual_manifest.json"
    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Production monitoring",
        "generated_at_utc": monitoring_bundle.now_utc.isoformat(),
        "kpis": monitoring_bundle.kpis,
        "warnings": monitoring_bundle.warnings,
        "supported_visuals": ["MLV-J04", "MLV-J08"],
        "conditional_visuals": ["MLV-J01", "MLV-J02", "MLV-J05"],
        "blocked_visuals": ["MLV-J03", "MLV-J06", "MLV-J07"],
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "funnel": str(funnel_path.relative_to(root))
            if funnel_path.is_relative_to(root)
            else str(funnel_path),
            "rejections": str(rejection_path.relative_to(root))
            if rejection_path.is_relative_to(root)
            else str(rejection_path),
            "source_status": str(source_status_path.relative_to(root))
            if source_status_path.is_relative_to(root)
            else str(source_status_path),
            "insights": str(insight_path.relative_to(root))
            if insight_path.is_relative_to(root)
            else str(insight_path),
            "status": str(status_path.relative_to(root))
            if status_path.is_relative_to(root)
            else str(status_path),
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
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest["supporting_files"]["manifest"] = (
        str(manifest_path.relative_to(root))
        if manifest_path.is_relative_to(root)
        else str(manifest_path)
    )

    return manifest
