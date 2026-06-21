"""Create one truthful Experiment Tracking Readiness page.

The local MLflow database contains unrelated prompt/demo evaluation evidence.
The final portfolio therefore uses one consolidated readiness page instead of
four empty or misleading experiment charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from src.visualization.experiment_tracking_data import (
    ExperimentTrackingBundle,
    build_experiment_tracking_bundle,
)
from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    VisualQAResult,
    add_footer,
    add_title_block,
    apply_ml_visual_style,
    save_figure_with_qa,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/experiment_tracking"
)
READINESS_FILENAME = "experiment_tracking_readiness.png"


def _draw_metric_card(
    fig,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value: str,
    detail: str,
    accent: str,
) -> None:
    """Draw one executive evidence card."""

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
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
        )
    )
    fig.text(
        x + 0.025,
        y + height - 0.045,
        label.upper(),
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
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )
    fig.text(
        x + 0.025,
        y + 0.035,
        detail,
        fontsize=7.2,
        color=COLORS["grey_700"],
        ha="left",
        va="bottom",
        linespacing=1.25,
    )


def create_experiment_readiness_page(
    bundle: ExperimentTrackingBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create one consolidated experiment-tracking readiness visual."""

    title = "Experiment Tracking Readiness & Evidence Gate"
    subtitle = (
        "MLflow prompt/demo evidence was excluded. The five planned experiment "
        "visuals remain conditional until a clean ecommerce run history exists."
    )

    apply_ml_visual_style(DEFAULT_SPEC)
    fig = plt.figure(
        figsize=(DEFAULT_SPEC.width, DEFAULT_SPEC.height),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source audit: mlflow.db and saved ecommerce champion lineage",
        interpretation_note=(
            "Readiness status only; no prompt/demo metric is treated as ecommerce performance."
        ),
    )

    counts = bundle.counts
    cards = [
        (
            "Verified ecommerce runs",
            f"{int(counts.get('verified_ecommerce_runs', 0)):,}",
            "Exact champion or registered-\nmodel lineage only",
            COLORS["blue"],
        ),
        (
            "Excluded non-ecommerce runs",
            f"{int(counts.get('excluded_runs', 0)):,}",
            "Prompt, demo, and unrelated\nexperiment evidence",
            COLORS["red"],
        ),
        (
            "Verified registry versions",
            f"{int(counts.get('verified_registry_versions', 0)):,}",
            "Exact ecommerce registered\nmodel only",
            COLORS["teal"],
        ),
        (
            "Shared metric keys",
            f"{int(counts.get('shared_metric_keys', 0)):,}",
            "Comparable ecommerce metrics\nacross verified runs",
            COLORS["amber"],
        ),
    ]

    x_positions = [0.07, 0.295, 0.52, 0.745]

    for x, card in zip(x_positions, cards):
        _draw_metric_card(
            fig,
            x=x,
            y=0.58,
            width=0.19,
            height=0.20,
            label=card[0],
            value=card[1],
            detail=card[2],
            accent=card[3],
        )

    panel = FancyBboxPatch(
        (0.07, 0.24),
        0.86,
        0.25,
        transform=fig.transFigure,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=COLORS["grey_100"],
        edgecolor=COLORS["amber"],
        linewidth=1.8,
    )
    fig.add_artist(panel)

    fig.text(
        0.095,
        0.445,
        "FINAL STATUS: I01, I02, I03, I04, AND I05 ARE CONDITIONAL",
        fontsize=12,
        fontweight="bold",
        color=COLORS["amber"],
        ha="left",
        va="top",
    )
    fig.text(
        0.095,
        0.375,
        bundle.comparison_reason,
        fontsize=10,
        color=COLORS["navy"],
        ha="left",
        va="top",
        wrap=True,
    )
    fig.text(
        0.095,
        0.285,
        (
            "Activation plan: log clean ecommerce-only runs under one evaluation contract;\n"
            "retain exact validation/holdout metric keys and repeated hyperparameter trials;\n"
            "link model versions and aliases to those runs."
        ),
        fontsize=9,
        color=COLORS["grey_700"],
        ha="left",
        va="top",
        wrap=True,
    )

    fig.text(
        0.07,
        0.145,
        "Coverage retained in the controlling matrix",
        fontsize=9,
        fontweight="bold",
        color=COLORS["grey_500"],
        ha="left",
    )
    fig.text(
        0.07,
        0.108,
        (
            "I01 Parallel Coordinates  |  I02 Performance Timeline  |  "
            "I03 Hyperparameter Response Surface  |  I04 Run Comparison Matrix  |  "
            "I05 Champion-Challenger Evolution"
        ),
        fontsize=8.5,
        color=COLORS["grey_700"],
        ha="left",
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_experiment_insights(
    bundle: ExperimentTrackingBundle,
) -> str:
    """Create the final readiness decision record."""

    return f"""# Experiment Tracking Readiness - Final Decision

## Portfolio treatment

The four unsupported experiment charts are removed from the executive visual
review and replaced by one consolidated readiness page.

## Conditional coverage retained

- MLV-I01 Experiment Parallel Coordinates
- MLV-I02 Run Performance Timeline
- MLV-I03 Hyperparameter Response Surface
- MLV-I04 Run Comparison Matrix
- MLV-I05 Champion-Challenger Evolution

## Source audit

- Verified ecommerce runs: **{bundle.counts.get('verified_ecommerce_runs', 0)}**
- Excluded non-ecommerce runs: **{bundle.counts.get('excluded_runs', 0)}**
- Verified ecommerce registry versions: **{bundle.counts.get('verified_registry_versions', 0)}**
- Shared ecommerce metric keys: **{bundle.counts.get('shared_metric_keys', 0)}**

## Decision

{bundle.comparison_reason}

No prompt-quality metric, MLflow demo result, or unrelated registry model is
used as ecommerce model-performance evidence.
"""


def generate_experiment_tracking_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: ExperimentTrackingBundle | None = None,
) -> dict[str, Any]:
    """Generate one readiness visual and the final conditional manifest."""

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

    tracking_bundle = (
        bundle
        if bundle is not None
        else build_experiment_tracking_bundle(root)
    )

    readiness_path = final_output_dir / READINESS_FILENAME
    qa_result = create_experiment_readiness_page(
        tracking_bundle,
        readiness_path,
    )

    audit_path = final_output_dir / "experiment_source_audit.csv"
    pd.DataFrame(
        [
            {"measure": key, "value": value}
            for key, value in tracking_bundle.counts.items()
        ]
    ).to_csv(audit_path, index=False)

    registry_path = (
        final_output_dir
        / "verified_ecommerce_registry_versions.csv"
    )
    tracking_bundle.model_versions.to_csv(
        registry_path,
        index=False,
    )

    insights_path = (
        final_output_dir
        / "experiment_tracking_insights.md"
    )
    insights_path.write_text(
        build_experiment_insights(tracking_bundle),
        encoding="utf-8",
    )

    conditional_visuals = {
        "MLV-I01": tracking_bundle.comparison_reason,
        "MLV-I02": tracking_bundle.comparison_reason,
        "MLV-I03": (
            "Requires dense repeated numeric hyperparameter variation across clean ecommerce runs."
        ),
        "MLV-I04": tracking_bundle.comparison_reason,
        "MLV-I05": (
            "Requires a clean ecommerce-only champion-challenger registry history with comparable evaluation evidence."
        ),
    }

    manifest_path = (
        final_output_dir
        / "experiment_tracking_visual_manifest.json"
    )
    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Experiment tracking readiness",
        "final_evidence_decision": "single_readiness_page",
        "comparison_ready": False,
        "comparison_reason": tracking_bundle.comparison_reason,
        "counts": tracking_bundle.counts,
        "supported_visuals": [],
        "conditional_visuals": conditional_visuals,
        "readiness_artifact": (
            str(readiness_path.relative_to(root))
            if readiness_path.is_relative_to(root)
            else str(readiness_path)
        ),
        "supporting_files": {
            "source_audit": (
                str(audit_path.relative_to(root))
                if audit_path.is_relative_to(root)
                else str(audit_path)
            ),
            "registry_versions": (
                str(registry_path.relative_to(root))
                if registry_path.is_relative_to(root)
                else str(registry_path)
            ),
            "insights": (
                str(insights_path.relative_to(root))
                if insights_path.is_relative_to(root)
                else str(insights_path)
            ),
        },
        "qa": {
            "experiment_tracking_readiness": {
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
