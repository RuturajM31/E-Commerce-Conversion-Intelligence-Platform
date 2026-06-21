"""Render MLflow experiment-tracking visuals I01, I02, I04, and I05."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.visualization.experiment_tracking_data import (
    ExperimentTrackingBundle,
    build_experiment_tracking_bundle,
    metric_display_name,
)
from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    MODEL_PALETTE,
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
    "reports/visuals/ml_visual_intelligence/experiment_tracking"
)


def _champion_run_id(
    bundle: ExperimentTrackingBundle,
) -> str:
    """Return the current champion run ID from saved lineage."""

    return str(
        bundle.lineage.get("run_id", "")
    )


def _normalise_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Min-max normalise comparable metric columns."""

    normalised = pd.DataFrame(
        index=frame.index,
    )

    for column in columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(minimum, maximum):
            normalised[column] = np.where(
                values.notna(),
                0.5,
                np.nan,
            )
        else:
            normalised[column] = (
                values - minimum
            ) / (maximum - minimum)

    return normalised


def create_experiment_parallel_coordinates(
    bundle: ExperimentTrackingBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create I01 multi-metric parallel coordinates for comparable runs."""

    title = "MLV-I01 | Experiment Parallel Coordinates"
    subtitle = (
        "Comparable MLflow runs across exact shared metric keys; each axis is "
        "normalised independently while the footer preserves source context."
    )

    runs = bundle.comparable_runs.tail(12).copy()
    metrics = bundle.metric_columns[:5]
    normalised = _normalise_columns(
        runs,
        metrics,
    )
    champion_id = _champion_run_id(bundle)

    fig, axis = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: mlflow.db | "
            f"{len(runs)} comparable runs | "
            f"{len(metrics)} shared metrics"
        ),
        interpretation_note=(
            "Axis height is relative within each exact metric key, not a universal score."
        ),
    )
    # Reserve a dedicated right-side run key so long MLflow names never
    # extend beyond the figure canvas or cover the metric lines.
    reserve_chart_space(
        fig,
        left=0.10,
        right=0.72,
        bottom=0.17,
        top=0.80,
    )

    x_positions = np.arange(len(metrics))

    for index, row in runs.iterrows():
        values = normalised.loc[
            index,
            metrics,
        ].to_numpy(dtype=float)
        is_champion = str(row["run_id"]) == champion_id
        colour = (
            COLORS["amber"]
            if is_champion
            else MODEL_PALETTE[
                index % len(MODEL_PALETTE)
            ]
        )

        axis.plot(
            x_positions,
            values,
            color=colour,
            linewidth=(
                3.2
                if is_champion
                else 1.4
            ),
            alpha=(
                1.0
                if is_champion
                else 0.62
            ),
            marker="o",
            markersize=(
                6.5
                if is_champion
                else 4.0
            ),
            zorder=(
                5
                if is_champion
                else 2
            ),
        )

    for x_position in x_positions:
        axis.axvline(
            x_position,
            color=COLORS["grey_300"],
            linewidth=1.0,
            zorder=0,
        )

    axis.set_xticks(
        x_positions,
        labels=[
            metric_display_name(column)
            for column in metrics
        ],
        rotation=12,
        ha="right",
    )
    axis.set_ylim(-0.05, 1.05)
    axis.set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0],
        labels=[
            "Lowest",
            "25%",
            "50%",
            "75%",
            "Highest",
        ],
    )
    axis.set_ylabel("Relative position within metric")
    axis.set_xlabel("")
    style_axes(
        axis,
        show_x_grid=False,
        show_y_grid=True,
    )

    # Keep the legend compact and readable.
    legend_handles: list[Line2D] = []

    for index, row in runs.iterrows():
        run_name = str(row["run_name"])
        is_champion = str(row["run_id"]) == champion_id

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=(
                    COLORS["amber"]
                    if is_champion
                    else MODEL_PALETTE[
                        index % len(MODEL_PALETTE)
                    ]
                ),
                linewidth=(
                    3.0
                    if is_champion
                    else 1.5
                ),
                marker="o",
                label=(
                    f"{run_name} — champion"
                    if is_champion
                    else run_name
                ),
            )
        )

    axis.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.0,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.75,
        borderaxespad=0.0,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_run_performance_timeline(
    bundle: ExperimentTrackingBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create I02 performance timeline for the selected primary metric."""

    title = "MLV-I02 | Run Performance Timeline"
    metric_label = metric_display_name(
        bundle.primary_metric
    )
    subtitle = (
        f"MLflow run chronology using exact metric key "
        f"`{bundle.primary_metric.removeprefix('metric__')}`; "
        "the current champion is highlighted."
    )

    runs = bundle.comparable_runs.copy()
    runs[bundle.primary_metric] = pd.to_numeric(
        runs[bundle.primary_metric],
        errors="coerce",
    )
    runs = runs.dropna(
        subset=[
            "start_time",
            bundle.primary_metric,
        ]
    )

    if runs.empty:
        raise ValueError(
            "No dated MLflow runs contain the primary metric."
        )

    champion_id = _champion_run_id(bundle)

    fig, axis = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: mlflow.db | "
            f"{len(runs)} dated comparable runs"
        ),
        interpretation_note=(
            "Point labels use MLflow run names; metric split remains explicit in the key."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.10,
        right=0.96,
        bottom=0.20,
        top=0.80,
    )

    experiment_names = (
        runs["experiment_name"]
        .fillna("Unknown experiment")
        .astype(str)
        .unique()
        .tolist()
    )
    experiment_colours = {
        name: MODEL_PALETTE[
            index % len(MODEL_PALETTE)
        ]
        for index, name in enumerate(
            experiment_names
        )
    }

    for experiment_name, group in runs.groupby(
        runs["experiment_name"]
        .fillna("Unknown experiment")
    ):
        ordered = group.sort_values("start_time")
        axis.plot(
            ordered["start_time"],
            ordered[bundle.primary_metric],
            color=experiment_colours[
                str(experiment_name)
            ],
            linewidth=1.5,
            alpha=0.65,
            zorder=1,
        )
        axis.scatter(
            ordered["start_time"],
            ordered[bundle.primary_metric],
            color=experiment_colours[
                str(experiment_name)
            ],
            s=68,
            edgecolor=COLORS["white"],
            linewidth=1.0,
            label=str(experiment_name),
            zorder=3,
        )

    champion_rows = runs.loc[
        runs["run_id"].astype(str)
        == champion_id
    ]

    if not champion_rows.empty:
        champion_row = champion_rows.iloc[-1]
        axis.scatter(
            [champion_row["start_time"]],
            [champion_row[bundle.primary_metric]],
            marker="*",
            s=280,
            color=COLORS["amber"],
            edgecolor=COLORS["navy"],
            linewidth=1.3,
            label="Current champion",
            zorder=6,
        )

    y_range = (
        float(runs[bundle.primary_metric].max())
        - float(runs[bundle.primary_metric].min())
    )
    label_offset = (
        y_range * 0.035
        if y_range > 0
        else 0.01
    )

    for _, row in runs.iterrows():
        axis.text(
            row["start_time"],
            float(row[bundle.primary_metric])
            + label_offset,
            str(row["run_name"])[:24],
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["grey_700"],
            rotation=18,
        )

    axis.set_xlabel("Run start time")
    axis.set_ylabel(metric_label)
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b\n%H:%M",
            tz=mdates.UTC,
        )
    )
    style_axes(axis)
    axis.legend(
        loc="best",
        fontsize=8,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_run_comparison_matrix(
    bundle: ExperimentTrackingBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create I04 exact-value run comparison matrix."""

    title = "MLV-I04 | Run Comparison Matrix"
    subtitle = (
        "Comparable MLflow runs across shared metric keys; colour is "
        "column-relative and cell text preserves the exact logged value."
    )

    runs = bundle.comparable_runs.tail(12).copy()
    metrics = bundle.metric_columns[:5]
    champion_id = _champion_run_id(bundle)

    values = runs[metrics].apply(
        pd.to_numeric,
        errors="coerce",
    )
    normalised = _normalise_columns(
        values,
        metrics,
    )

    colour_map = LinearSegmentedColormap.from_list(
        "run_comparison",
        [
            COLORS["grey_100"],
            "#D7E8F8",
            COLORS["teal"],
            COLORS["navy"],
        ],
    )

    fig, axis = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: mlflow.db | "
            f"{len(runs)} comparable runs"
        ),
        interpretation_note=(
            "Compare values within a column; exact metric keys are not mixed."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.27,
        right=0.96,
        bottom=0.18,
        top=0.79,
    )

    image_values = normalised.to_numpy(
        dtype=float
    )
    axis.imshow(
        image_values,
        cmap=colour_map,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    run_labels = runs["run_name"].astype(
        str
    ).tolist()
    metric_labels = [
        metric_display_name(column)
        for column in metrics
    ]

    axis.set_xticks(
        np.arange(len(metrics)),
        labels=metric_labels,
        rotation=20,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(runs)),
        labels=run_labels,
    )
    axis.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=8,
    )

    for row_index in range(len(runs)):
        for column_index in range(
            len(metrics)
        ):
            raw_value = values.iloc[
                row_index,
                column_index,
            ]
            normalised_value = normalised.iloc[
                row_index,
                column_index,
            ]

            text = (
                "—"
                if pd.isna(raw_value)
                else f"{float(raw_value):.4f}"
            )

            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight=(
                    "bold"
                    if str(
                        runs.iloc[row_index][
                            "run_id"
                        ]
                    )
                    == champion_id
                    else "normal"
                ),
                color=(
                    COLORS["white"]
                    if pd.notna(normalised_value)
                    and float(
                        normalised_value
                    )
                    >= 0.58
                    else COLORS["grey_900"]
                ),
            )

    champion_positions = [
        index
        for index, run_id in enumerate(
            runs["run_id"].astype(str)
        )
        if run_id == champion_id
    ]

    for champion_position in champion_positions:
        axis.add_patch(
            Rectangle(
                (-0.5, champion_position - 0.5),
                width=len(metrics),
                height=1,
                fill=False,
                edgecolor=COLORS["amber"],
                linewidth=2.6,
                zorder=5,
            )
        )

    axis.set_xticks(
        np.arange(-0.5, len(metrics), 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(runs), 1),
        minor=True,
    )
    axis.grid(
        which="minor",
        color=COLORS["white"],
        linewidth=1.6,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in axis.spines.values():
        spine.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_champion_challenger_evolution(
    bundle: ExperimentTrackingBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create I05 registry version and alias evolution timeline."""

    title = "MLV-I05 | Champion–Challenger Evolution"
    subtitle = (
        "Registered model versions over time with current aliases and the "
        f"logged {metric_display_name(bundle.primary_metric)} value when available."
    )

    versions = bundle.model_versions.copy()

    if versions.empty:
        raise ValueError(
            "No MLflow registered model versions are available."
        )

    versions = versions.dropna(
        subset=["creation_time"]
    )

    if versions.empty:
        raise ValueError(
            "Registered model versions have no creation timestamps."
        )

    model_names = (
        versions["model_name"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    y_lookup = {
        model_name: index
        for index, model_name in enumerate(
            model_names
        )
    }

    champion_model = str(
        bundle.lineage.get(
            "registered_model_name",
            "",
        )
    )
    champion_version = str(
        bundle.lineage.get(
            "registered_model_version",
            "",
        )
    )
    champion_alias = str(
        bundle.lineage.get(
            "registered_model_alias",
            "champion",
        )
    )

    fig, axis = create_figure()
    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Source: mlflow.db model registry | "
            f"{len(versions)} versions across {len(model_names)} models"
        ),
        interpretation_note=(
            "Hollow circles indicate versions without the selected comparable metric."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.24,
        right=0.96,
        bottom=0.18,
        top=0.79,
    )

    for model_index, model_name in enumerate(
        model_names
    ):
        group = versions.loc[
            versions["model_name"]
            == model_name
        ].sort_values("creation_time")
        y_position = y_lookup[model_name]
        colour = MODEL_PALETTE[
            model_index % len(MODEL_PALETTE)
        ]

        axis.plot(
            group["creation_time"],
            np.full(
                len(group),
                y_position,
            ),
            color=colour,
            linewidth=2.0,
            alpha=0.55,
            zorder=1,
        )

        for _, row in group.iterrows():
            version_text = str(
                int(row["version"])
                if pd.notna(row["version"])
                else "?"
            )
            is_champion = (
                str(row["model_name"])
                == champion_model
                and version_text
                == champion_version
            )
            metric_value = pd.to_numeric(
                pd.Series(
                    [row[bundle.primary_metric]]
                ),
                errors="coerce",
            ).iloc[0]

            axis.scatter(
                [row["creation_time"]],
                [y_position],
                marker=(
                    "*"
                    if is_champion
                    else "o"
                ),
                s=(
                    300
                    if is_champion
                    else 105
                ),
                facecolor=(
                    COLORS["amber"]
                    if is_champion
                    else (
                        colour
                        if pd.notna(
                            metric_value
                        )
                        else COLORS["white"]
                    )
                ),
                edgecolor=(
                    COLORS["navy"]
                    if is_champion
                    else colour
                ),
                linewidth=1.5,
                zorder=5,
            )

            label = f"v{version_text}"
            alias = str(row.get("alias", "")).strip()

            if alias:
                label += f" [{alias}]"

            if pd.notna(metric_value):
                label += (
                    f"\n{float(metric_value):.3f}"
                )

            axis.text(
                row["creation_time"],
                y_position + 0.18,
                label,
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight=(
                    "bold"
                    if is_champion
                    else "normal"
                ),
                color=(
                    COLORS["navy"]
                    if is_champion
                    else COLORS["grey_700"]
                ),
            )

    axis.set_yticks(
        np.arange(len(model_names)),
        labels=model_names,
    )
    axis.set_xlabel("Registry version creation time")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b\n%H:%M",
            tz=mdates.UTC,
        )
    )
    axis.set_ylim(
        -0.5,
        len(model_names) - 0.25,
    )
    style_axes(
        axis,
        show_x_grid=True,
        show_y_grid=False,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLORS["amber"],
            markeredgecolor=COLORS["navy"],
            markersize=13,
            label=f"Current {champion_alias}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["white"],
            markeredgecolor=COLORS["grey_500"],
            markersize=8,
            label="Metric unavailable",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower right",
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_experiment_insights(
    bundle: ExperimentTrackingBundle,
) -> str:
    """Create actual-number findings and I03 conditional status."""

    runs = bundle.comparable_runs.copy()
    runs[bundle.primary_metric] = pd.to_numeric(
        runs[bundle.primary_metric],
        errors="coerce",
    )
    metric_runs = runs.dropna(
        subset=[bundle.primary_metric]
    )
    best_row = metric_runs.sort_values(
        bundle.primary_metric,
        ascending=False,
    ).iloc[0]
    champion_id = _champion_run_id(bundle)
    champion_rows = metric_runs.loc[
        metric_runs["run_id"].astype(str)
        == champion_id
    ]

    champion_text = (
        f"Current champion logged {float(champion_rows.iloc[-1][bundle.primary_metric]):.4f}."
        if not champion_rows.empty
        else "The current champion run does not contain this selected shared metric key."
    )

    dense_count = (
        int(
            bundle.parameter_density[
                "dense_candidate"
            ].sum()
        )
        if not bundle.parameter_density.empty
        else 0
    )

    return f"""# Experiment Tracking Visual Intelligence — Actual-Number Findings

## MLV-I01 — Experiment Parallel Coordinates

**What it shows:** **{bundle.counts['comparable_runs']}** comparable MLflow runs across **{len(bundle.metric_columns)}** exact shared metric keys.

**Business conclusion:** The visual reveals trade-offs between metrics without collapsing model selection into one number.

## MLV-I02 — Run Performance Timeline

**Actual finding:** The selected shared timeline metric is **{metric_display_name(bundle.primary_metric)}** using exact key `{bundle.primary_metric.removeprefix('metric__')}`. The highest logged value is **{float(best_row[bundle.primary_metric]):.4f}** from **{best_row['run_name']}**. {champion_text}

**Recommended action:** Keep validation and holdout metric keys explicitly separated in future MLflow runs.

## MLV-I04 — Run Comparison Matrix

**Actual finding:** The matrix compares only metrics recorded under the same MLflow key across at least two runs. Missing cells remain visibly blank rather than being imputed.

**Business conclusion:** This protects the comparison from silently mixing experiments with different evaluation contracts.

## MLV-I05 — Champion–Challenger Evolution

**Actual finding:** The registry contains **{bundle.counts['registered_models']}** registered models, **{bundle.counts['model_versions']}** versions, and **{bundle.counts['registered_model_aliases']}** aliases. The saved production lineage points to **{bundle.lineage.get('registered_model_name')}** version **{bundle.lineage.get('registered_model_version')}** with alias **{bundle.lineage.get('registered_model_alias')}**.

## MLV-I03 — Hyperparameter Response Surface

**Status:** {'READY' if bundle.i03_ready else 'CONDITIONAL'}.

**Observed evidence:** **{dense_count}** numeric parameters currently have at least three unique values across six or more comparable runs.

**Requirement:** A reliable response surface needs at least two sufficiently varied numeric parameters and enough comparable runs under one consistent evaluation contract. No sparse surface is fabricated.
"""


def build_i03_status(
    bundle: ExperimentTrackingBundle,
) -> str:
    """Create controlled evidence for conditional hyperparameter surface."""

    top_parameters = (
        bundle.parameter_density.head(12)
        if not bundle.parameter_density.empty
        else pd.DataFrame()
    )

    if top_parameters.empty:
        parameter_lines = (
            "No numeric parameter has reusable density evidence."
        )
    else:
        parameter_lines = "\n".join(
            (
                f"- `{row.parameter}`: "
                f"{int(row.non_null_runs)} runs, "
                f"{int(row.unique_values)} unique values, "
                f"dense candidate = {bool(row.dense_candidate)}"
            )
            for row in top_parameters.itertuples(
                index=False
            )
        )

    return f"""# MLV-I03 Hyperparameter Response Surface Status

**Status:** {'READY' if bundle.i03_ready else 'CONDITIONAL'}

A response surface is permitted only when at least two numeric parameters have
three or more unique values across six or more comparable runs, with at least
eight comparable runs overall.

{parameter_lines}

No hyperparameter-performance relationship is invented from sparse evidence.
"""


def generate_experiment_tracking_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: ExperimentTrackingBundle | None = None,
) -> dict[str, Any]:
    """Generate I01, I02, I04, and I05 plus evidence and I03 status."""

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

    tracking_bundle = (
        bundle
        if bundle is not None
        else build_experiment_tracking_bundle(root)
    )

    visual_names = {
        "MLV-I01": "MLV-I01 Experiment Parallel Coordinates",
        "MLV-I02": "MLV-I02 Run Performance Timeline",
        "MLV-I04": "MLV-I04 Run Comparison Matrix",
        "MLV-I05": "MLV-I05 Champion Challenger Evolution",
    }
    visual_paths = {
        visual_id: final_output_dir
        / f"{safe_filename(name)}.png"
        for visual_id, name in visual_names.items()
    }

    qa_results = {
        "MLV-I01": create_experiment_parallel_coordinates(
            tracking_bundle,
            visual_paths["MLV-I01"],
        ),
        "MLV-I02": create_run_performance_timeline(
            tracking_bundle,
            visual_paths["MLV-I02"],
        ),
        "MLV-I04": create_run_comparison_matrix(
            tracking_bundle,
            visual_paths["MLV-I04"],
        ),
        "MLV-I05": create_champion_challenger_evolution(
            tracking_bundle,
            visual_paths["MLV-I05"],
        ),
    }

    runs_path = (
        final_output_dir / "comparable_mlflow_runs.csv"
    )
    run_columns = [
        "run_id",
        "run_name",
        "experiment_name",
        "start_time",
        "status",
        *tracking_bundle.metric_columns,
    ]
    tracking_bundle.comparable_runs[
        run_columns
    ].to_csv(
        runs_path,
        index=False,
    )

    versions_path = (
        final_output_dir / "model_registry_versions.csv"
    )
    tracking_bundle.model_versions.to_csv(
        versions_path,
        index=False,
    )

    parameter_path = (
        final_output_dir / "hyperparameter_density.csv"
    )
    tracking_bundle.parameter_density.to_csv(
        parameter_path,
        index=False,
    )

    insight_path = (
        final_output_dir / "experiment_tracking_insights.md"
    )
    insight_path.write_text(
        build_experiment_insights(
            tracking_bundle
        ),
        encoding="utf-8",
    )

    i03_path = (
        final_output_dir / "i03_hyperparameter_surface_status.md"
    )
    i03_path.write_text(
        build_i03_status(tracking_bundle),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir
        / "experiment_tracking_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Experiment tracking",
        "counts": tracking_bundle.counts,
        "metric_columns": (
            tracking_bundle.metric_columns
        ),
        "primary_metric": (
            tracking_bundle.primary_metric
        ),
        "i03_ready": tracking_bundle.i03_ready,
        "supported_visuals": [
            "MLV-I01",
            "MLV-I02",
            "MLV-I04",
            "MLV-I05",
        ],
        "conditional_visuals": {
            "MLV-I03": (
                "Requires dense repeated numeric hyperparameter variation."
            )
        },
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "comparable_runs": (
                str(runs_path.relative_to(root))
                if runs_path.is_relative_to(root)
                else str(runs_path)
            ),
            "registry_versions": (
                str(versions_path.relative_to(root))
                if versions_path.is_relative_to(root)
                else str(versions_path)
            ),
            "parameter_density": (
                str(parameter_path.relative_to(root))
                if parameter_path.is_relative_to(root)
                else str(parameter_path)
            ),
            "insights": (
                str(insight_path.relative_to(root))
                if insight_path.is_relative_to(root)
                else str(insight_path)
            ),
            "i03_status": (
                str(i03_path.relative_to(root))
                if i03_path.is_relative_to(root)
                else str(i03_path)
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
