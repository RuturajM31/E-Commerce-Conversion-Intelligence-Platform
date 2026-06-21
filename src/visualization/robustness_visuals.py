"""Render MLV-F01 to MLV-F05 robustness and stability visuals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.visualization.ml_visual_style import (
    COLORS,
    MODEL_PALETTE,
    VisualQAResult,
    add_footer,
    add_title_block,
    create_figure,
    percent_formatter,
    reserve_chart_space,
    safe_filename,
    save_figure_with_qa,
    style_axes,
)
from src.visualization.robustness_data import (
    RobustnessBundle,
    build_robustness_bundle,
)


DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/robustness"
)


def build_robustness_heatmap_data(
    bundle: RobustnessBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw and column-normalised seed/metric matrices."""

    stability = bundle.stability.copy()
    seeds = sorted(
        stability["seed"].unique().tolist()
    )
    models = (
        bundle.stability_summary["model_name"]
        .astype(str)
        .tolist()
    )

    raw_columns: list[str] = []
    raw_data: dict[str, list[float]] = {}

    for metric, label in [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
    ]:
        for seed in seeds:
            column_name = f"{label}\nSeed {int(seed)}"
            raw_columns.append(column_name)

            seed_values = (
                stability.loc[
                    stability["seed"] == seed,
                    ["model_name", metric],
                ]
                .set_index("model_name")[metric]
                .reindex(models)
            )

            raw_data[column_name] = (
                seed_values.astype(float).tolist()
            )

    raw_matrix = pd.DataFrame(
        raw_data,
        index=models,
    )[raw_columns]

    # Colour compares models within each metric/seed column.
    normalised = raw_matrix.copy()

    for column in raw_matrix.columns:
        values = raw_matrix[column].astype(float)
        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(minimum, maximum):
            normalised[column] = 100.0
        else:
            normalised[column] = (
                (values - minimum)
                / (maximum - minimum)
                * 100.0
            )

    return raw_matrix, normalised


def create_robustness_heatmap(
    bundle: RobustnessBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create F01 column-normalised robustness heatmap."""

    title = "MLV-F01 | Model Robustness Heatmap"
    subtitle = (
        "Raw PR-AUC and ROC-AUC across seeds; colour ranks models within each "
        "seed/metric column while cell text preserves the actual value."
    )

    raw_matrix, normalised = (
        build_robustness_heatmap_data(bundle)
    )

    colour_map = LinearSegmentedColormap.from_list(
        "robustness_strength",
        [
            COLORS["grey_100"],
            "#D8E8FA",
            COLORS["teal"],
            COLORS["navy"],
        ],
    )

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note=(
            "Source: final_true_champion_stability.csv | "
            f"{bundle.seed_count} seeds per model"
        ),
        interpretation_note=(
            "Colour is relative within each column; use text for actual values."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.22,
        right=0.96,
        bottom=0.14,
        top=0.80,
    )

    image = ax.imshow(
        normalised.to_numpy(dtype=float),
        cmap=colour_map,
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(raw_matrix.columns)),
        labels=raw_matrix.columns,
    )
    ax.set_yticks(
        np.arange(len(raw_matrix.index)),
        labels=raw_matrix.index,
    )
    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=10,
    )

    for row_index in range(raw_matrix.shape[0]):
        for column_index in range(raw_matrix.shape[1]):
            raw_value = float(
                raw_matrix.iloc[row_index, column_index]
            )
            score = float(
                normalised.iloc[row_index, column_index]
            )

            ax.text(
                column_index,
                row_index,
                f"{raw_value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight=(
                    "bold"
                    if raw_matrix.index[row_index]
                    == bundle.champion_name
                    else "normal"
                ),
                color=(
                    COLORS["white"]
                    if score >= 58
                    else COLORS["grey_900"]
                ),
            )

    champion_position = list(
        raw_matrix.index
    ).index(bundle.champion_name)

    ax.add_patch(
        Rectangle(
            (-0.5, champion_position - 0.5),
            width=raw_matrix.shape[1],
            height=1,
            fill=False,
            edgecolor=COLORS["amber"],
            linewidth=2.8,
            zorder=5,
        )
    )
    ax.text(
        raw_matrix.shape[1] - 0.55,
        champion_position - 0.37,
        "CHAMPION",
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["amber"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": COLORS["white"],
            "edgecolor": COLORS["amber"],
            "linewidth": 1.0,
        },
        zorder=6,
    )

    ax.set_xticks(
        np.arange(-0.5, raw_matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, raw_matrix.shape[0], 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=COLORS["white"],
        linewidth=2.0,
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


def create_variability_distribution(
    bundle: RobustnessBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create F02 seed variability ranges and observations."""

    title = "MLV-F02 | Metric Variability Distribution"
    subtitle = (
        "Seed-level observations, min–max range, and mean for PR-AUC and "
        f"ROC-AUC; evidence is limited to {bundle.seed_count} seeds per model."
    )

    model_order = (
        bundle.stability_summary["model_name"]
        .astype(str)
        .tolist()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 7.6),
        facecolor="white",
    )
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: final_true_champion_stability.csv",
        interpretation_note=(
            "Shorter ranges indicate greater seed consistency."
        ),
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.97,
        bottom=0.13,
        top=0.78,
        wspace=0.24,
    )

    for axis, metric, label in [
        (axes[0], "pr_auc", "PR-AUC"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        for y_position, model_name in enumerate(
            model_order
        ):
            model_rows = bundle.stability.loc[
                bundle.stability["model_name"]
                == model_name
            ]
            values = model_rows[metric].to_numpy(
                dtype=float
            )
            minimum = float(values.min())
            maximum = float(values.max())
            mean = float(values.mean())
            colour = (
                COLORS["blue"]
                if model_name == bundle.champion_name
                else MODEL_PALETTE[
                    y_position % len(MODEL_PALETTE)
                ]
            )

            axis.plot(
                [minimum, maximum],
                [y_position, y_position],
                color=colour,
                linewidth=4.0,
                alpha=0.50,
                solid_capstyle="round",
            )
            axis.scatter(
                values,
                np.full(len(values), y_position),
                s=62,
                color=colour,
                edgecolor=COLORS["white"],
                linewidth=1.2,
                zorder=4,
            )
            axis.scatter(
                [mean],
                [y_position],
                s=135,
                marker="D",
                color=COLORS["navy"],
                edgecolor=COLORS["white"],
                linewidth=1.3,
                zorder=5,
            )
            axis.text(
                maximum,
                y_position + 0.23,
                f"Range {maximum - minimum:.3f}",
                ha="right",
                va="bottom",
                fontsize=8,
                color=COLORS["grey_700"],
            )

        axis.set_yticks(
            np.arange(len(model_order)),
            labels=model_order,
        )
        axis.invert_yaxis()
        axis.set_title(
            label,
            loc="left",
            fontsize=12,
            fontweight="bold",
            color=COLORS["navy"],
            pad=10,
        )
        axis.set_xlabel(label)
        axis.set_ylabel("")
        style_axes(axis)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_stability_profile(
    bundle: RobustnessBundle,
) -> pd.DataFrame:
    """Create transparent relative performance and consistency scores."""

    summary = bundle.stability_summary.copy()

    # Performance is relative to the best candidate mean.
    summary["pr_auc_performance_score"] = (
        summary["pr_auc_mean"]
        / summary["pr_auc_mean"].max()
        * 100.0
    )
    summary["roc_auc_performance_score"] = (
        summary["roc_auc_mean"]
        / summary["roc_auc_mean"].max()
        * 100.0
    )

    def consistency_score(
        series: pd.Series,
    ) -> pd.Series:
        """Map lowest coefficient of variation to 100 and highest to 0."""

        minimum = float(series.min())
        maximum = float(series.max())

        if np.isclose(minimum, maximum):
            return pd.Series(
                np.full(len(series), 100.0),
                index=series.index,
            )

        return (
            (maximum - series)
            / (maximum - minimum)
            * 100.0
        )

    summary["pr_auc_consistency_score"] = (
        consistency_score(summary["pr_auc_cv"])
    )
    summary["roc_auc_consistency_score"] = (
        consistency_score(summary["roc_auc_cv"])
    )

    return summary


def create_champion_stability_profile(
    bundle: RobustnessBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create F03 champion-versus-challenger stability profile."""

    title = "MLV-F03 | Champion Stability Profile"
    subtitle = (
        "Champion versus strongest challenger on relative mean performance "
        f"and seed consistency; scores are relative within the {bundle.seed_count}-seed candidate set."
    )

    profile = build_stability_profile(bundle)
    selected = profile.loc[
        profile["model_name"].isin(
            [
                bundle.champion_name,
                bundle.challenger_name,
            ]
        )
    ].set_index("model_name")

    dimensions = [
        (
            "pr_auc_performance_score",
            "Mean PR-AUC",
        ),
        (
            "pr_auc_consistency_score",
            "PR-AUC consistency",
        ),
        (
            "roc_auc_performance_score",
            "Mean ROC-AUC",
        ),
        (
            "roc_auc_consistency_score",
            "ROC-AUC consistency",
        ),
    ]

    fig, ax = create_figure()
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: final_true_champion_stability.csv",
        interpretation_note=(
            "100 means best in this candidate set; consistency uses coefficient of variation."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.24,
        right=0.94,
        bottom=0.16,
        top=0.78,
    )

    y_positions = np.arange(len(dimensions))

    champion_values = np.array(
        [
            float(
                selected.loc[
                    bundle.champion_name,
                    column,
                ]
            )
            for column, _ in dimensions
        ]
    )
    challenger_values = np.array(
        [
            float(
                selected.loc[
                    bundle.challenger_name,
                    column,
                ]
            )
            for column, _ in dimensions
        ]
    )

    for y_position, champion_value, challenger_value in zip(
        y_positions,
        champion_values,
        challenger_values,
    ):
        ax.plot(
            [champion_value, challenger_value],
            [y_position, y_position],
            color=COLORS["grey_300"],
            linewidth=3.0,
            zorder=1,
        )

    ax.scatter(
        champion_values,
        y_positions,
        s=140,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        linewidth=1.5,
        label=bundle.champion_name,
        zorder=3,
    )
    ax.scatter(
        challenger_values,
        y_positions,
        s=115,
        color=COLORS["teal"],
        edgecolor=COLORS["white"],
        linewidth=1.3,
        label=bundle.challenger_name,
        zorder=3,
    )

    for y_position, value in enumerate(
        champion_values
    ):
        ax.text(
            value,
            y_position - 0.18,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=COLORS["navy"],
        )

    for y_position, value in enumerate(
        challenger_values
    ):
        ax.text(
            value,
            y_position + 0.18,
            f"{value:.0f}",
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["grey_700"],
        )

    ax.set_yticks(
        y_positions,
        labels=[
            label
            for _, label in dimensions
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Relative score (0–100)")
    ax.set_ylabel("")
    style_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_sensitivity_tornado(
    bundle: RobustnessBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create F04 anomaly-removal sensitivity tornado."""

    title = "MLV-F04 | Sensitivity Tornado"
    subtitle = (
        "Metric change after removing anomaly-flagged validation visitors; "
        "negative bars indicate performance degradation."
    )

    pairs = bundle.sensitivity_pairs.sort_values(
        "pr_auc_delta"
    ).reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 7.6),
        facecolor="white",
    )
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: final_true_champion_sensitivity.csv",
        interpretation_note=(
            "Bars show percentage-point change from all validation visitors."
        ),
    )
    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        bottom=0.13,
        top=0.78,
        wspace=0.25,
    )

    for axis, metric, label in [
        (axes[0], "pr_auc_delta", "PR-AUC change"),
        (axes[1], "roc_auc_delta", "ROC-AUC change"),
    ]:
        y_positions = np.arange(len(pairs))
        delta_points = (
            pairs[metric].to_numpy(dtype=float)
            * 100.0
        )

        colours = [
            (
                COLORS["green"]
                if value >= 0
                else COLORS["red"]
            )
            for value in delta_points
        ]

        bars = axis.barh(
            y_positions,
            delta_points,
            color=colours,
            height=0.62,
            alpha=0.90,
        )

        for bar, value, model_name in zip(
            bars,
            delta_points,
            pairs["model_name"].astype(str),
        ):
            if model_name == bundle.champion_name:
                bar.set_edgecolor(COLORS["navy"])
                bar.set_linewidth(2.0)

            axis.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:+.2f} pp",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
                fontweight=(
                    "bold"
                    if model_name == bundle.champion_name
                    else "normal"
                ),
                color=COLORS["grey_900"],
            )

        axis.axvline(
            0.0,
            color=COLORS["grey_700"],
            linewidth=1.1,
        )
        axis.set_yticks(
            y_positions,
            labels=pairs["model_name"],
        )
        axis.set_title(
            label,
            loc="left",
            fontsize=12,
            fontweight="bold",
            color=COLORS["navy"],
            pad=10,
        )
        axis.set_xlabel("Percentage-point change")
        axis.set_ylabel("")
        style_axes(axis)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def _draw_waterfall_axis(
    axis,
    *,
    start: float,
    end: float,
    metric_label: str,
) -> None:
    """Draw one start–delta–end metric waterfall."""

    delta = end - start
    categories = [
        "All validation",
        "Remove anomalies",
        "Non-anomalous",
    ]

    axis.bar(
        0,
        start,
        color=COLORS["blue"],
        width=0.58,
    )
    axis.bar(
        1,
        abs(delta),
        bottom=min(start, end),
        color=(
            COLORS["green"]
            if delta >= 0
            else COLORS["red"]
        ),
        width=0.58,
    )
    axis.bar(
        2,
        end,
        color=COLORS["navy"],
        width=0.58,
    )

    axis.plot(
        [0.29, 0.71],
        [start, start],
        color=COLORS["grey_500"],
        linewidth=1.0,
        linestyle="--",
    )
    axis.plot(
        [1.29, 1.71],
        [end, end],
        color=COLORS["grey_500"],
        linewidth=1.0,
        linestyle="--",
    )

    relative_change = (
        delta / start
        if start != 0
        else 0.0
    )

    annotations = [
        f"{start:.3f}",
        f"{delta:+.3f}\n({relative_change:+.1%})",
        f"{end:.3f}",
    ]

    for x_position, text, value in zip(
        range(3),
        annotations,
        [start, max(start, end), end],
    ):
        axis.text(
            x_position,
            value + max(start, end) * 0.04,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["navy"],
        )

    axis.set_xticks(
        np.arange(3),
        labels=categories,
    )
    axis.set_title(
        metric_label,
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        pad=10,
    )
    axis.set_ylabel(metric_label)
    axis.set_ylim(
        0.0,
        max(start, end) * 1.24,
    )
    style_axes(
        axis,
        show_x_grid=False,
        show_y_grid=True,
    )


def create_performance_degradation_waterfall(
    bundle: RobustnessBundle,
    output_path: Path,
) -> VisualQAResult:
    """Create F05 champion anomaly-removal waterfall."""

    title = "MLV-F05 | Performance Degradation Waterfall"
    subtitle = (
        f"{bundle.champion_name} performance before and after removing "
        "anomaly-flagged validation visitors."
    )

    champion_row = bundle.sensitivity_pairs.loc[
        bundle.sensitivity_pairs["model_name"]
        == bundle.champion_name
    ]

    if champion_row.empty:
        raise ValueError(
            "Champion sensitivity pair is unavailable."
        )

    row = champion_row.iloc[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 7.6),
        facecolor="white",
    )
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(
        fig,
        source_note="Source: final_true_champion_sensitivity.csv",
        interpretation_note=(
            f"{int(row['removed_rows']):,} anomaly-flagged rows were removed."
        ),
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.16,
        top=0.78,
        wspace=0.25,
    )

    _draw_waterfall_axis(
        axes[0],
        start=float(row["all_pr_auc"]),
        end=float(row["non_anomalous_pr_auc"]),
        metric_label="PR-AUC",
    )
    _draw_waterfall_axis(
        axes[1],
        start=float(row["all_roc_auc"]),
        end=float(row["non_anomalous_roc_auc"]),
        metric_label="ROC-AUC",
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def build_robustness_insights(
    bundle: RobustnessBundle,
) -> str:
    """Create actual-number robustness findings."""

    champion_summary = (
        bundle.stability_summary.loc[
            bundle.stability_summary["model_name"]
            == bundle.champion_name
        ].iloc[0]
    )
    challenger_summary = (
        bundle.stability_summary.loc[
            bundle.stability_summary["model_name"]
            == bundle.challenger_name
        ].iloc[0]
    )
    champion_sensitivity = (
        bundle.sensitivity_pairs.loc[
            bundle.sensitivity_pairs["model_name"]
            == bundle.champion_name
        ].iloc[0]
    )

    return f"""# Robustness and Stability Visual Intelligence — Actual-Number Findings

## MLV-F01 — Model Robustness Heatmap

**Actual finding:** Across **{bundle.seed_count} seeds**, **{bundle.champion_name}** achieved mean PR-AUC **{champion_summary['pr_auc_mean']:.3f}** and mean ROC-AUC **{champion_summary['roc_auc_mean']:.3f}**.

**Business conclusion:** The champion remains competitive across the tested random seeds, but the evidence is limited to three runs per model.

## MLV-F02 — Metric Variability Distribution

**Actual finding:** Champion PR-AUC ranged from **{champion_summary['pr_auc_min']:.3f}** to **{champion_summary['pr_auc_max']:.3f}**; ROC-AUC ranged from **{champion_summary['roc_auc_min']:.3f}** to **{champion_summary['roc_auc_max']:.3f}**.

**Recommended action:** Keep seed-level variability visible during future retraining and expand the seed count before making statistical stability claims.

## MLV-F03 — Champion Stability Profile

**Actual finding:** The strongest challenger by mean PR-AUC is **{bundle.challenger_name}**, with mean PR-AUC **{challenger_summary['pr_auc_mean']:.3f}**.

**Limitation:** Profile scores are relative within the current candidate set and are not universal quality grades.

## MLV-F04 — Sensitivity Tornado

**Actual finding:** Removing anomaly-flagged visitors changed champion PR-AUC by **{champion_sensitivity['pr_auc_delta']:+.3f}** and ROC-AUC by **{champion_sensitivity['roc_auc_delta']:+.3f}**.

**Business conclusion:** A material negative delta means anomaly-heavy visitors contribute meaningfully to measured validation performance.

## MLV-F05 — Performance Degradation Waterfall

**Actual finding:** Champion PR-AUC moved from **{champion_sensitivity['all_pr_auc']:.3f}** to **{champion_sensitivity['non_anomalous_pr_auc']:.3f}** after removing **{int(champion_sensitivity['removed_rows']):,}** anomaly-flagged rows. ROC-AUC moved from **{champion_sensitivity['all_roc_auc']:.3f}** to **{champion_sensitivity['non_anomalous_roc_auc']:.3f}**.

**Recommended action:** Monitor anomaly composition alongside model metrics and avoid treating a single aggregate validation score as universally representative.
"""


def generate_robustness_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
    *,
    bundle: RobustnessBundle | None = None,
) -> dict[str, Any]:
    """Generate F01 to F05 plus supporting evidence."""

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

    robustness_bundle = (
        bundle
        if bundle is not None
        else build_robustness_bundle(root)
    )

    visual_names = {
        "MLV-F01": "MLV-F01 Model Robustness Heatmap",
        "MLV-F02": "MLV-F02 Metric Variability Distribution",
        "MLV-F03": "MLV-F03 Champion Stability Profile",
        "MLV-F04": "MLV-F04 Sensitivity Tornado",
        "MLV-F05": "MLV-F05 Performance Degradation Waterfall",
    }
    visual_paths = {
        visual_id: final_output_dir
        / f"{safe_filename(name)}.png"
        for visual_id, name in visual_names.items()
    }

    qa_results = {
        "MLV-F01": create_robustness_heatmap(
            robustness_bundle,
            visual_paths["MLV-F01"],
        ),
        "MLV-F02": create_variability_distribution(
            robustness_bundle,
            visual_paths["MLV-F02"],
        ),
        "MLV-F03": create_champion_stability_profile(
            robustness_bundle,
            visual_paths["MLV-F03"],
        ),
        "MLV-F04": create_sensitivity_tornado(
            robustness_bundle,
            visual_paths["MLV-F04"],
        ),
        "MLV-F05": create_performance_degradation_waterfall(
            robustness_bundle,
            visual_paths["MLV-F05"],
        ),
    }

    stability_path = (
        final_output_dir / "stability_summary.csv"
    )
    robustness_bundle.stability_summary.to_csv(
        stability_path,
        index=False,
    )

    sensitivity_path = (
        final_output_dir / "sensitivity_pairs.csv"
    )
    robustness_bundle.sensitivity_pairs.to_csv(
        sensitivity_path,
        index=False,
    )

    insight_path = (
        final_output_dir / "robustness_insights.md"
    )
    insight_path.write_text(
        build_robustness_insights(
            robustness_bundle
        ),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir / "robustness_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Robustness and stability",
        "champion_name": robustness_bundle.champion_name,
        "challenger_name": robustness_bundle.challenger_name,
        "seed_count": robustness_bundle.seed_count,
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "stability_summary": (
                str(stability_path.relative_to(root))
                if stability_path.is_relative_to(root)
                else str(stability_path)
            ),
            "sensitivity_pairs": (
                str(sensitivity_path.relative_to(root))
                if sensitivity_path.is_relative_to(root)
                else str(sensitivity_path)
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
