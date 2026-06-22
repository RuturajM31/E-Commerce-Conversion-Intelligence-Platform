"""Create the MLV-A01 to MLV-A04 champion-selection visual package.

Why this file exists:
    The project has several validated model-comparison tables and champion
    metadata files, but those results need to be turned into professional,
    reusable ML Visual Intelligence artifacts.

Visuals created:
    - MLV-A01: Model Performance Frontier
    - MLV-A02: Multi-Metric Model Ranking
    - MLV-A03: Validation-to-Holdout Generalisation Slopegraph
    - MLV-A04: Champion Scorecard / Identity Card

Main inputs:
    - reports/tables/final_true_champion_comparison.csv
    - models/metadata/final_champion_metadata.json
    - models/metadata/mlflow_champion_lineage.json

Main outputs:
    - Four QA-validated PNG files
    - One model-ranking CSV
    - One actual-number insight Markdown file
    - One JSON manifest describing sources and artifacts

Used next:
    The same calculation layer can support MLflow artifacts, Streamlit pages,
    reports, and presentation evidence without rebuilding the logic.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
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


# ---------------------------------------------------------------------------
# Repository-relative source and output paths
# ---------------------------------------------------------------------------

COMPARISON_PATH = Path(
    "reports/tables/final_true_champion_comparison.csv"
)
CHAMPION_METADATA_PATH = Path(
    "models/metadata/final_champion_metadata.json"
)
MLFLOW_LINEAGE_PATH = Path(
    "models/metadata/mlflow_champion_lineage.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/champion_selection"
)

# Metrics shown in the executive ranking matrix.
# They all reward larger values, so one normalisation method is sufficient.
RANKING_METRICS = [
    ("business_score", "Business score"),
    ("pr_auc", "PR-AUC"),
    ("best_f1_score", "F1"),
    ("best_precision", "Precision"),
    ("best_recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
]


# ---------------------------------------------------------------------------
# Source loading and preparation
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON dictionary.

    Input:
        Absolute path to a champion metadata or lineage file.

    Output:
        Parsed dictionary.

    Used next:
        Scorecard metrics and MLflow identity are read from this dictionary.
    """

    # Stop with a clear message when the required evidence is missing.
    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    # Load the existing file without modifying it.
    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    # Champion metadata must be a dictionary for named-field access.
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return content


def parse_mapping(value: Any) -> dict[str, Any]:
    """Convert a stored dictionary or dictionary-like string into a mapping.

    Input:
        A real dictionary or text such as "{'pr_auc': 0.15}".

    Output:
        A standard Python dictionary.

    Used next:
        Validation and final-holdout metric blocks are accessed consistently.
    """

    # Return dictionaries directly because no parsing is required.
    if isinstance(value, dict):
        return value

    # Empty or missing values become an empty mapping.
    if value is None or (
        isinstance(value, float) and np.isnan(value)
    ):
        return {}

    # CSV summaries may store dictionaries as Python-style text.
    if isinstance(value, str):
        parsed = ast.literal_eval(value)

        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Expected a dictionary or dictionary-like string.")


def load_champion_sources(
    project_root: str | Path = ".",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load the verified champion visual sources.

    Input:
        Project root. Relative paths keep the project portable.

    Output:
        Comparison DataFrame, champion metadata, and MLflow lineage.

    Used next:
        `generate_champion_visual_package()` prepares and renders all visuals.
    """

    # Resolve all repository-relative inputs against one project root.
    root = Path(project_root)
    comparison_file = root / COMPARISON_PATH
    metadata_file = root / CHAMPION_METADATA_PATH
    lineage_file = root / MLFLOW_LINEAGE_PATH

    # The comparison table is required for A01, A02, and A03.
    if not comparison_file.exists():
        raise FileNotFoundError(
            f"Required comparison source not found: {comparison_file}"
        )

    comparison = pd.read_csv(comparison_file)

    # Load champion identity and registered-model evidence for A04.
    metadata = read_json(metadata_file)
    lineage = read_json(lineage_file)

    return comparison, metadata, lineage


def validate_comparison_schema(comparison: pd.DataFrame) -> None:
    """Check that the verified comparison table still has required columns.

    Input:
        Raw `final_true_champion_comparison.csv` DataFrame.

    Output:
        No value. A clear error is raised when required columns are missing.

    Used next:
        Prevents charts from silently using incomplete or renamed data.
    """

    required_columns = {
        "evaluation_split",
        "model_name",
        "model_family",
        "deployable",
        "pr_auc",
        "roc_auc",
        "best_threshold",
        "best_precision",
        "best_recall",
        "best_f1_score",
        "business_score",
    }

    # Compare the required contract with the current source schema.
    missing_columns = sorted(
        required_columns.difference(comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "Champion comparison is missing required columns: "
            + ", ".join(missing_columns)
        )


def choose_comparison_split(comparison: pd.DataFrame) -> str:
    """Choose the common split used for fair model-to-model comparison.

    Input:
        Full comparison table containing one or more evaluation splits.

    Output:
        Preferred split name, normally `validation`.

    Used next:
        A01 and A02 compare all models on the same split.
    """

    # Normalise split names only for selection; source values are preserved.
    split_lookup = {
        str(value).strip().lower(): str(value)
        for value in comparison["evaluation_split"].dropna().unique()
    }

    # Validation is the model-selection split and is therefore preferred.
    for preferred in ("validation", "val"):
        if preferred in split_lookup:
            return split_lookup[preferred]

    # Fall back to the most common split when validation is unavailable.
    split_counts = comparison["evaluation_split"].value_counts()

    if split_counts.empty:
        raise ValueError("No evaluation split is available for comparison.")

    return str(split_counts.index[0])


def prepare_model_comparison(
    comparison: pd.DataFrame,
    champion_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Prepare one clean same-split model comparison table.

    Input:
        Raw comparison rows and champion metadata.

    Output:
        Sorted validation-level model table with champion indicator.

    Used next:
        A01, A02, the ranking CSV, and the insight report use this table.
    """

    # Confirm the source contract before any chart calculation begins.
    validate_comparison_schema(comparison)

    # Select one common split so model metrics remain directly comparable.
    selected_split = choose_comparison_split(comparison)

    model_table = comparison.loc[
        comparison["evaluation_split"] == selected_split
    ].copy()

    # Convert metric columns into numeric values and fail on invalid data.
    metric_columns = [
        column_name
        for column_name, _ in RANKING_METRICS
    ] + ["best_threshold"]

    for column in metric_columns:
        model_table[column] = pd.to_numeric(
            model_table[column],
            errors="raise",
        )

    # Use the saved champion identity rather than inferring it from ranking.
    champion_name = str(
        champion_metadata.get("final_model_name", "")
    ).strip()

    if not champion_name:
        raise ValueError(
            "Champion metadata does not contain `final_model_name`."
        )

    model_table["is_champion"] = (
        model_table["model_name"].astype(str) == champion_name
    )

    # Keep the strongest business score first for executive readability.
    model_table = model_table.sort_values(
        ["business_score", "pr_auc", "best_f1_score"],
        ascending=False,
    ).reset_index(drop=True)

    # Add a transparent one-based decision rank.
    model_table["validation_rank"] = (
        model_table["business_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return model_table


def normalise_ranking_metrics(
    model_table: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise ranking metrics to a comparable 0–100 scale.

    Input:
        Same-split model comparison table.

    Output:
        DataFrame indexed by model and containing normalised metric scores.

    Used next:
        A02 uses this matrix for the multi-metric executive ranking view.
    """

    normalised = pd.DataFrame(
        index=model_table["model_name"].astype(str)
    )

    for metric_column, _ in RANKING_METRICS:
        # Read the validated metric series for one ranking dimension.
        values = model_table[metric_column].astype(float)

        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(minimum, maximum):
            # Equal models receive the same neutral full score.
            scaled = pd.Series(
                np.full(len(values), 100.0),
                index=model_table.index,
            )
        else:
            # Convert the observed range into a clear 0–100 score.
            scaled = (
                (values - minimum)
                / (maximum - minimum)
                * 100.0
            )

        normalised[metric_column] = scaled.to_numpy()

    return normalised


def find_pareto_frontier(
    model_table: pd.DataFrame,
) -> pd.DataFrame:
    """Find models not dominated on both precision and recall.

    Input:
        Same-split model comparison table.

    Output:
        Pareto-efficient rows sorted by recall.

    Used next:
        A01 connects the strongest precision/recall trade-off models.
    """

    candidate_rows: list[pd.Series] = []

    for _, current_row in model_table.iterrows():
        current_precision = float(current_row["best_precision"])
        current_recall = float(current_row["best_recall"])

        # Another model dominates when it is at least as good on both
        # dimensions and strictly better on at least one dimension.
        dominated = (
            (
                model_table["best_precision"]
                >= current_precision
            )
            & (
                model_table["best_recall"]
                >= current_recall
            )
            & (
                (
                    model_table["best_precision"]
                    > current_precision
                )
                | (
                    model_table["best_recall"]
                    > current_recall
                )
            )
        ).any()

        if not dominated:
            candidate_rows.append(current_row)

    frontier = pd.DataFrame(candidate_rows)

    if frontier.empty:
        return model_table.iloc[0:0].copy()

    return frontier.sort_values("best_recall")


# ---------------------------------------------------------------------------
# Shared insight calculations
# ---------------------------------------------------------------------------


def build_generalisation_table(
    comparison: pd.DataFrame,
    champion_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Build validation-versus-holdout PR-AUC pairs.

    Input:
        Full comparison table and champion metadata.

    Output:
        One row per model with validation and holdout PR-AUC.

    Used next:
        A03 and the actual-number insight report use the delta values.
    """

    # Map common split labels into two business-readable stages.
    split_names = (
        comparison["evaluation_split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    working = comparison.copy()
    working["_split_normalised"] = split_names

    validation_mask = working["_split_normalised"].isin(
        {"validation", "val"}
    )
    holdout_mask = working["_split_normalised"].isin(
        {
            "final_holdout",
            "holdout",
            "test",
            "final_test",
        }
    )

    validation = working.loc[
        validation_mask,
        ["model_name", "pr_auc"],
    ].rename(columns={"pr_auc": "validation_pr_auc"})

    holdout = working.loc[
        holdout_mask,
        ["model_name", "pr_auc"],
    ].rename(columns={"pr_auc": "holdout_pr_auc"})

    # Keep only directly comparable models found on both splits.
    paired = validation.merge(
        holdout,
        on="model_name",
        how="inner",
    )

    # The saved metadata always contains the true champion holdout evidence.
    champion_name = str(
        champion_metadata["final_model_name"]
    )
    validation_metrics = parse_mapping(
        champion_metadata.get("validation_metrics", {})
    )
    holdout_metrics = parse_mapping(
        champion_metadata.get("final_holdout_metrics", {})
    )

    champion_pair = pd.DataFrame(
        {
            "model_name": [champion_name],
            "validation_pr_auc": [
                float(validation_metrics["pr_auc"])
            ],
            "holdout_pr_auc": [
                float(holdout_metrics["pr_auc"])
            ],
        }
    )

    # Add or replace the champion row with the authoritative metadata values.
    paired = paired.loc[
        paired["model_name"] != champion_name
    ]
    paired = pd.concat(
        [paired, champion_pair],
        ignore_index=True,
    )

    paired["pr_auc_delta"] = (
        paired["holdout_pr_auc"]
        - paired["validation_pr_auc"]
    )
    paired["is_champion"] = (
        paired["model_name"] == champion_name
    )

    return paired.sort_values(
        ["is_champion", "holdout_pr_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_champion_metrics(
    champion_metadata: dict[str, Any],
) -> dict[str, float | int | str]:
    """Create a clean champion scorecard metric dictionary.

    Input:
        Champion metadata JSON content.

    Output:
        Model identity, threshold, final metrics, and holdout volume.

    Used next:
        A04 and the Markdown insight report display these exact values.
    """

    holdout = parse_mapping(
        champion_metadata.get("final_holdout_metrics", {})
    )

    required_holdout_fields = {
        "rows",
        "positive_rows",
        "predicted_positive_rows",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1_score",
    }

    missing = sorted(
        required_holdout_fields.difference(holdout)
    )

    if missing:
        raise ValueError(
            "Final holdout metrics are missing fields: "
            + ", ".join(missing)
        )

    return {
        "model_name": str(
            champion_metadata["final_model_name"]
        ),
        "model_family": str(
            champion_metadata.get("model_family", "Unknown")
        ),
        "threshold": float(
            champion_metadata["best_threshold"]
        ),
        "pr_auc": float(holdout["pr_auc"]),
        "roc_auc": float(holdout["roc_auc"]),
        "precision": float(holdout["precision"]),
        "recall": float(holdout["recall"]),
        "f1_score": float(holdout["f1_score"]),
        "holdout_rows": int(holdout["rows"]),
        "positive_rows": int(holdout["positive_rows"]),
        "predicted_positive_rows": int(
            holdout["predicted_positive_rows"]
        ),
    }


# ---------------------------------------------------------------------------
# Visual renderers
# ---------------------------------------------------------------------------


def create_model_performance_frontier(
    model_table: pd.DataFrame,
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-A01 Model Performance Frontier.

    Input:
        Validation-level model metrics.

    Output:
        QA-validated PNG showing precision/recall trade-offs.

    Used next:
        Executives can see which models offer efficient targeting trade-offs.
    """

    title = "MLV-A01 | Model Performance Frontier"
    subtitle = (
        "Validation precision versus recall at each model's selected "
        "operating threshold; marker size reflects PR-AUC."
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
            "Source: final_true_champion_comparison.csv | "
            "Split: validation"
        ),
        interpretation_note=(
            "Upper-right is stronger; teal line marks the "
            "precision–recall frontier."
        ),
    )

    # Keep a dedicated model key on the right.
    # This prevents long model names from overlapping inside the plot.
    reserve_chart_space(
        fig,
        left=0.09,
        right=0.72,
        bottom=0.14,
        top=0.82,
    )
    key_ax = fig.add_axes([0.75, 0.15, 0.22, 0.66])
    key_ax.axis("off")

    # Scale marker sizes by PR-AUC while keeping nearby models separable.
    # The earlier, larger bubbles caused close candidates to cover each other.
    pr_auc = model_table["pr_auc"].astype(float)
    pr_range = float(pr_auc.max() - pr_auc.min())

    if np.isclose(pr_range, 0):
        marker_sizes = np.full(len(model_table), 360.0)
    else:
        marker_sizes = (
            240.0
            + (
                (pr_auc - pr_auc.min())
                / pr_range
                * 360.0
            )
        )

    # Plot challengers first so the champion remains visually dominant.
    challenger_mask = ~model_table["is_champion"]

    ax.scatter(
        model_table.loc[challenger_mask, "best_recall"],
        model_table.loc[challenger_mask, "best_precision"],
        s=marker_sizes[challenger_mask],
        c=COLORS["grey_500"],
        alpha=0.78,
        edgecolors=COLORS["white"],
        linewidths=1.8,
        zorder=3,
    )

    champion_rows = model_table.loc[
        model_table["is_champion"]
    ]

    ax.scatter(
        champion_rows["best_recall"],
        champion_rows["best_precision"],
        s=marker_sizes[model_table["is_champion"]],
        c=COLORS["blue"],
        alpha=0.98,
        edgecolors=COLORS["navy"],
        linewidths=2.2,
        zorder=5,
    )

    # Connect only the non-dominated trade-off models.
    frontier = find_pareto_frontier(model_table)

    if len(frontier) >= 2:
        ax.plot(
            frontier["best_recall"],
            frontier["best_precision"],
            color=COLORS["teal"],
            linewidth=2.4,
            alpha=0.9,
            zorder=2,
        )

    # Place numbered badges beside the true data points.
    # The badges use deterministic offsets and short connector lines, so
    # nearby candidates remain distinct without moving the actual metrics.
    badge_offsets = [
        (0, 17),
        (-22, 15),
        (22, -16),
        (-18, -16),
        (0, 17),
        (-18, 16),
        (0, -18),
        (18, 15),
    ]

    for row_index, row in model_table.iterrows():
        model_number = row_index + 1
        x_value = float(row["best_recall"])
        y_value = float(row["best_precision"])
        is_champion = bool(row["is_champion"])
        x_offset, y_offset = badge_offsets[
            row_index % len(badge_offsets)
        ]

        ax.annotate(
            str(model_number),
            xy=(x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=(
                COLORS["white"]
                if is_champion
                else COLORS["grey_900"]
            ),
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": (
                    COLORS["blue"]
                    if is_champion
                    else COLORS["white"]
                ),
                "edgecolor": (
                    COLORS["navy"]
                    if is_champion
                    else COLORS["grey_500"]
                ),
                "linewidth": 1.2,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["grey_500"],
                "linewidth": 0.8,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=7,
        )

    # Build a compact, readable key with actual precision and recall values.
    key_ax.text(
        0.00,
        0.98,
        "MODEL KEY",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
        transform=key_ax.transAxes,
    )
    key_ax.text(
        0.00,
        0.925,
        "Numbered markers match the plot.",
        ha="left",
        va="top",
        fontsize=8,
        color=COLORS["grey_500"],
        transform=key_ax.transAxes,
    )

    row_positions = np.linspace(
        0.84,
        0.08,
        len(model_table),
    )

    for row_index, (y_position, (_, row)) in enumerate(
        zip(row_positions, model_table.iterrows())
    ):
        model_number = row_index + 1
        is_champion = bool(row["is_champion"])
        marker_colour = (
            COLORS["blue"]
            if is_champion
            else COLORS["grey_500"]
        )

        # Number badge mirrors the numbered point in the scatter plot.
        key_ax.text(
            0.02,
            y_position,
            str(model_number),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS["white"],
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": marker_colour,
                "edgecolor": (
                    COLORS["navy"]
                    if is_champion
                    else COLORS["white"]
                ),
                "linewidth": 1.0,
            },
            transform=key_ax.transAxes,
        )

        # Wrap long names inside the fixed side-panel width.
        wrapped_name = "\n".join(
            textwrap.wrap(
                str(row["model_name"]),
                width=27,
                max_lines=2,
                placeholder="…",
            )
        )

        key_ax.text(
            0.10,
            y_position + 0.018,
            wrapped_name,
            ha="left",
            va="center",
            fontsize=7.8,
            fontweight="bold" if is_champion else "normal",
            color=(
                COLORS["navy"]
                if is_champion
                else COLORS["grey_700"]
            ),
            transform=key_ax.transAxes,
        )

        key_ax.text(
            0.10,
            y_position - 0.035,
            (
                f"P {float(row['best_precision']):.1%}  |  "
                f"R {float(row['best_recall']):.1%}"
            ),
            ha="left",
            va="center",
            fontsize=7.2,
            color=COLORS["grey_500"],
            transform=key_ax.transAxes,
        )

        if is_champion:
            key_ax.text(
                0.98,
                y_position,
                "CHAMPION",
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=COLORS["amber"],
                transform=key_ax.transAxes,
            )

    # Add sensible padding around the observed precision/recall range.
    recall_values = model_table["best_recall"].astype(float)
    precision_values = model_table["best_precision"].astype(float)

    ax.set_xlim(
        max(0.0, float(recall_values.min()) - 0.035),
        min(1.0, float(recall_values.max()) + 0.035),
    )
    ax.set_ylim(
        max(0.0, float(precision_values.min()) - 0.035),
        min(1.0, float(precision_values.max()) + 0.035),
    )

    ax.set_xlabel("Recall at selected threshold")
    ax.set_ylabel("Precision at selected threshold")
    ax.xaxis.set_major_formatter(percent_formatter(decimals=0))
    ax.yaxis.set_major_formatter(percent_formatter(decimals=0))
    style_axes(ax)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )

def create_multi_metric_ranking(
    model_table: pd.DataFrame,
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-A02 Multi-Metric Model Ranking.

    Input:
        Validation-level model metrics.

    Output:
        QA-validated executive heatmap with raw values and normalised colour.

    Used next:
        Viewers can compare model strength across several metrics at once.
    """

    title = "MLV-A02 | Multi-Metric Model Ranking"
    subtitle = (
        "Models are ordered by validation business score; cell colour "
        "shows relative strength and text shows the actual metric value."
    )

    normalised = normalise_ranking_metrics(model_table)

    # Align the normalised matrix with the sorted model table.
    model_names = model_table["model_name"].astype(str).tolist()
    matrix = normalised.loc[
        model_names,
        [column for column, _ in RANKING_METRICS],
    ].to_numpy()

    # Use a restrained project-specific colour ramp.
    heatmap_colour = LinearSegmentedColormap.from_list(
        "ml_ranking",
        [
            COLORS["grey_100"],
            "#D9E8FA",
            COLORS["blue"],
            COLORS["navy"],
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
            "Source: final_true_champion_comparison.csv | "
            "Ranking split: validation"
        ),
        interpretation_note=(
            "Colour is relative within this candidate set; text remains "
            "the actual metric."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.22,
        right=0.96,
        bottom=0.15,
        top=0.80,
    )

    # Draw the normalised matrix while preserving raw values as labels.
    ax.imshow(
        matrix,
        cmap=heatmap_colour,
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(RANKING_METRICS)),
        labels=[label for _, label in RANKING_METRICS],
    )
    ax.set_yticks(
        np.arange(len(model_names)),
        labels=model_names,
    )

    # Place metric labels above the matrix for faster scanning.
    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=10,
    )

    # Write actual values inside every cell.
    for row_index, model_row in model_table.iterrows():
        for column_index, (metric_column, _) in enumerate(
            RANKING_METRICS
        ):
            raw_value = float(model_row[metric_column])
            relative_score = float(
                matrix[row_index, column_index]
            )

            text_colour = (
                COLORS["white"]
                if relative_score >= 58
                else COLORS["grey_900"]
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
                    if bool(model_row["is_champion"])
                    else "normal"
                ),
                color=text_colour,
            )

    # Outline the champion row rather than changing the underlying values.
    champion_positions = np.flatnonzero(
        model_table["is_champion"].to_numpy()
    )

    for champion_position in champion_positions:
        ax.add_patch(
            Rectangle(
                (-0.5, champion_position - 0.5),
                width=len(RANKING_METRICS),
                height=1,
                fill=False,
                edgecolor=COLORS["amber"],
                linewidth=2.8,
                zorder=5,
            )
        )

        ax.text(
            len(RANKING_METRICS) - 0.55,
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

    # Use white separators for a clean matrix/table appearance.
    ax.set_xticks(
        np.arange(-0.5, len(RANKING_METRICS), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(model_names), 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=COLORS["white"],
        linewidth=2.0,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove all outside spines because the matrix has its own boundaries.
    for spine in ax.spines.values():
        spine.set_visible(False)

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


def create_generalisation_slopegraph(
    generalisation_table: pd.DataFrame,
    champion_name: str,
    champion_metrics: dict[str, float | int | str],
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-A03 Validation-to-Holdout Generalisation Slopegraph.

    Input:
        Paired PR-AUC values, champion identity, and holdout context.

    Output:
        QA-validated slopegraph plus an evidence summary panel.

    Used next:
        Viewers can judge whether selection performance held on untouched data.
    """

    title = (
        "MLV-A03 | Validation-to-Holdout "
        "Generalisation Slopegraph"
    )
    subtitle = (
        "Authoritative paired PR-AUC evidence from model selection to the "
        "untouched final holdout; holdout context is shown at right."
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
            "Sources: final_true_champion_comparison.csv and "
            "final_champion_metadata.json"
        ),
        interpretation_note=(
            "Current paired evidence is limited to models with verified "
            "validation and final-holdout metrics."
        ),
    )

    # Reserve the right side for evidence cards.
    # This avoids a sparse chart when only the champion has a verified pair.
    reserve_chart_space(
        fig,
        left=0.10,
        right=0.66,
        bottom=0.14,
        top=0.82,
    )
    evidence_ax = fig.add_axes([0.70, 0.15, 0.27, 0.66])
    evidence_ax.axis("off")

    # Plot every authoritative validation-to-holdout pair.
    for row_index, row in generalisation_table.iterrows():
        is_champion = bool(row["is_champion"])
        line_colour = (
            COLORS["blue"]
            if is_champion
            else MODEL_PALETTE[
                row_index % len(MODEL_PALETTE)
            ]
        )
        line_width = 3.6 if is_champion else 2.0
        line_alpha = 1.0 if is_champion else 0.64

        validation_value = float(row["validation_pr_auc"])
        holdout_value = float(row["holdout_pr_auc"])
        model_name = str(row["model_name"])

        ax.plot(
            [0, 1],
            [validation_value, holdout_value],
            marker="o",
            markersize=10 if is_champion else 8,
            linewidth=line_width,
            color=line_colour,
            alpha=line_alpha,
            zorder=4 if is_champion else 2,
        )

        # Left endpoint identifies the model and validation value.
        ax.text(
            -0.05,
            validation_value,
            f"{model_name}\n{validation_value:.3f}",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold" if is_champion else "normal",
            color=(
                COLORS["navy"]
                if is_champion
                else COLORS["grey_700"]
            ),
        )

        # Right endpoint shows the holdout value and actual change.
        delta = holdout_value - validation_value
        delta_sign = "+" if delta >= 0 else ""

        ax.text(
            1.05,
            holdout_value,
            f"{holdout_value:.3f}\n{delta_sign}{delta:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold" if is_champion else "normal",
            color=(
                COLORS["navy"]
                if is_champion
                else COLORS["grey_700"]
            ),
        )

    # Stage labels make the comparison explicit.
    ax.set_xticks(
        [0, 1],
        labels=[
            "Validation\n(model selection)",
            "Final holdout\n(untouched evaluation)",
        ],
    )
    ax.set_xlim(-0.20, 1.20)

    all_values = np.concatenate(
        [
            generalisation_table["validation_pr_auc"].to_numpy(
                dtype=float
            ),
            generalisation_table["holdout_pr_auc"].to_numpy(
                dtype=float
            ),
        ]
    )
    value_range = float(all_values.max() - all_values.min())
    padding = max(0.012, value_range * 0.20)

    ax.set_ylim(
        max(0.0, float(all_values.min()) - padding),
        min(1.0, float(all_values.max()) + padding),
    )
    ax.set_ylabel("PR-AUC")
    ax.yaxis.set_major_formatter(percent_formatter(decimals=0))
    style_axes(
        ax,
        show_x_grid=False,
        show_y_grid=True,
    )

    # Read the saved champion pair for the evidence panel.
    champion_row = generalisation_table.loc[
        generalisation_table["model_name"] == champion_name
    ]

    if champion_row.empty:
        raise ValueError(
            "Champion is missing from the generalisation table."
        )

    champion_row = champion_row.iloc[0]
    validation_value = float(
        champion_row["validation_pr_auc"]
    )
    holdout_value = float(
        champion_row["holdout_pr_auc"]
    )
    delta_value = float(champion_row["pr_auc_delta"])
    delta_sign = "+" if delta_value >= 0 else ""

    evidence_ax.text(
        0.00,
        0.98,
        "EVIDENCE SUMMARY",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
        transform=evidence_ax.transAxes,
    )
    evidence_ax.text(
        0.00,
        0.925,
        champion_name,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["blue"],
        transform=evidence_ax.transAxes,
    )

    summary_items = [
        (
            "Validation PR-AUC",
            f"{validation_value:.3f}",
            "Model-selection stage",
            COLORS["grey_500"],
        ),
        (
            "Final holdout PR-AUC",
            f"{holdout_value:.3f}",
            "Untouched evaluation",
            COLORS["blue"],
        ),
        (
            "Generalisation change",
            f"{delta_sign}{delta_value:.3f}",
            "Holdout minus validation",
            COLORS["teal"],
        ),
        (
            "Holdout positives",
            f"{int(champion_metrics['positive_rows']):,}",
            (
                f"of {int(champion_metrics['holdout_rows']):,} "
                "evaluated rows"
            ),
            COLORS["amber"],
        ),
    ]

    card_y_positions = [0.72, 0.53, 0.34, 0.15]

    for y_position, (label, value, note, accent) in zip(
        card_y_positions,
        summary_items,
    ):
        card = FancyBboxPatch(
            (0.00, y_position),
            0.98,
            0.145,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=evidence_ax.transAxes,
            facecolor=COLORS["white"],
            edgecolor=COLORS["grey_200"],
            linewidth=1.1,
        )
        evidence_ax.add_patch(card)

        evidence_ax.add_patch(
            Rectangle(
                (0.00, y_position),
                0.015,
                0.145,
                transform=evidence_ax.transAxes,
                facecolor=accent,
                edgecolor="none",
            )
        )

        evidence_ax.text(
            0.055,
            y_position + 0.112,
            label.upper(),
            ha="left",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            color=COLORS["grey_500"],
            transform=evidence_ax.transAxes,
        )
        evidence_ax.text(
            0.055,
            y_position + 0.065,
            value,
            ha="left",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=COLORS["navy"],
            transform=evidence_ax.transAxes,
        )
        evidence_ax.text(
            0.055,
            y_position + 0.018,
            note,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=COLORS["grey_700"],
            transform=evidence_ax.transAxes,
        )

    # State the evidence boundary honestly when only one pair is available.
    if len(generalisation_table) == 1:
        evidence_ax.text(
            0.00,
            0.03,
            (
                "Only the saved champion has an authoritative paired "
                "validation–holdout record in the current source."
            ),
            ha="left",
            va="bottom",
            fontsize=7.8,
            color=COLORS["grey_700"],
            wrap=True,
            transform=evidence_ax.transAxes,
        )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )

def _draw_metric_card(
    ax,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value: str,
    note: str,
    accent: str,
) -> None:
    """Draw one reusable executive metric card inside the scorecard."""

    # Rounded card boundary creates clear hierarchy without chart clutter.
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.2,
    )
    ax.add_patch(card)

    # Small accent strip identifies each KPI group.
    ax.add_patch(
        Rectangle(
            (x, y),
            0.008,
            height,
            transform=ax.transAxes,
            facecolor=accent,
            edgecolor="none",
        )
    )

    ax.text(
        x + 0.025,
        y + height - 0.035,
        label.upper(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    ax.text(
        x + 0.025,
        y + height * 0.52,
        value,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        x + 0.025,
        y + 0.028,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=COLORS["grey_700"],
    )


def create_champion_scorecard(
    champion_metrics: dict[str, float | int | str],
    lineage: dict[str, Any],
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-A04 Champion Scorecard / Identity Card.

    Input:
        Final holdout metrics and MLflow registered-model lineage.

    Output:
        QA-validated executive champion identity image.

    Used next:
        The artifact proves what was selected, how it performs, and where it
        is registered for controlled reuse.
    """

    title = "MLV-A04 | Champion Scorecard and Identity"
    subtitle = (
        "Saved production candidate, final holdout performance, operating "
        "threshold, audience volume, and MLflow registry lineage."
    )

    fig, ax = create_figure()
    ax.axis("off")

    add_title_block(
        fig,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        fig,
        source_note=(
            "Sources: final_champion_metadata.json and "
            "mlflow_champion_lineage.json"
        ),
        interpretation_note=(
            "Final holdout contains 318 positives; production outcome "
            "monitoring remains delayed-label dependent."
        ),
    )
    reserve_chart_space(
        fig,
        left=0.05,
        right=0.95,
        bottom=0.10,
        top=0.82,
    )

    # Identity header clearly names the selected model and family.
    ax.text(
        0.02,
        0.92,
        str(champion_metrics["model_name"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.02,
        0.855,
        (
            f"{champion_metrics['model_family']} model  •  "
            f"Registered alias: "
            f"{lineage.get('registered_model_alias', 'not recorded')}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=COLORS["grey_700"],
    )

    # First row: ranking quality and operating threshold.
    _draw_metric_card(
        ax,
        x=0.02,
        y=0.56,
        width=0.22,
        height=0.20,
        label="Final holdout PR-AUC",
        value=f"{float(champion_metrics['pr_auc']):.3f}",
        note="Primary rare-event ranking metric",
        accent=COLORS["blue"],
    )
    _draw_metric_card(
        ax,
        x=0.265,
        y=0.56,
        width=0.22,
        height=0.20,
        label="ROC-AUC",
        value=f"{float(champion_metrics['roc_auc']):.3f}",
        note="Overall ranking separation",
        accent=COLORS["teal"],
    )
    _draw_metric_card(
        ax,
        x=0.51,
        y=0.56,
        width=0.22,
        height=0.20,
        label="F1 at threshold",
        value=f"{float(champion_metrics['f1_score']):.3f}",
        note="Precision–recall balance",
        accent=COLORS["purple"],
    )
    _draw_metric_card(
        ax,
        x=0.755,
        y=0.56,
        width=0.22,
        height=0.20,
        label="Operating threshold",
        value=f"{float(champion_metrics['threshold']):.2f}",
        note="Saved decision boundary",
        accent=COLORS["amber"],
    )

    # Second row: action quality and evaluation volume.
    _draw_metric_card(
        ax,
        x=0.02,
        y=0.29,
        width=0.22,
        height=0.20,
        label="Precision",
        value=f"{float(champion_metrics['precision']):.1%}",
        note="Share of targeted visitors converting",
        accent=COLORS["green"],
    )
    _draw_metric_card(
        ax,
        x=0.265,
        y=0.29,
        width=0.22,
        height=0.20,
        label="Recall",
        value=f"{float(champion_metrics['recall']):.1%}",
        note="Share of converters captured",
        accent=COLORS["red"],
    )
    _draw_metric_card(
        ax,
        x=0.51,
        y=0.29,
        width=0.22,
        height=0.20,
        label="Holdout rows",
        value=f"{int(champion_metrics['holdout_rows']):,}",
        note=(
            f"{int(champion_metrics['positive_rows']):,} "
            "actual positives"
        ),
        accent=COLORS["blue"],
    )
    _draw_metric_card(
        ax,
        x=0.755,
        y=0.29,
        width=0.22,
        height=0.20,
        label="Predicted positives",
        value=(
            f"{int(champion_metrics['predicted_positive_rows']):,}"
        ),
        note="Visitors selected at threshold",
        accent=COLORS["teal"],
    )

    # Bottom lineage panel proves reproducibility and registry identity.
    lineage_panel = FancyBboxPatch(
        (0.02, 0.04),
        0.955,
        0.16,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=COLORS["grey_100"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.0,
    )
    ax.add_patch(lineage_panel)

    registered_name = str(
        lineage.get("registered_model_name", "not recorded")
    )
    version = str(
        lineage.get("registered_model_version", "not recorded")
    )
    run_id = str(lineage.get("run_id", "not recorded"))
    run_id_display = (
        f"{run_id[:12]}…{run_id[-6:]}"
        if len(run_id) > 21
        else run_id
    )

    ax.text(
        0.045,
        0.165,
        "MLFLOW LINEAGE",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    ax.text(
        0.045,
        0.105,
        (
            f"Registered model: {registered_name}   |   "
            f"Version: {version}   |   Run: {run_id_display}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.045,
        0.060,
        (
            "Business action: use the saved threshold for controlled "
            "high-intent targeting; continue monitoring drift and delayed "
            "outcomes before claiming live production performance."
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        color=COLORS["grey_700"],
    )

    return save_figure_with_qa(
        fig,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# Interpretation and package generation
# ---------------------------------------------------------------------------


def build_insight_markdown(
    model_table: pd.DataFrame,
    generalisation_table: pd.DataFrame,
    champion_metrics: dict[str, float | int | str],
    lineage: dict[str, Any],
) -> str:
    """Create actual-number conclusions for all four visuals.

    Input:
        Prepared ranking, generalisation, scorecard, and lineage data.

    Output:
        Markdown explaining what each visual proves and its limitation.

    Used next:
        The Markdown is logged beside PNG artifacts and reused in reports.
    """

    champion_name = str(champion_metrics["model_name"])

    champion_validation = model_table.loc[
        model_table["model_name"] == champion_name
    ]

    if champion_validation.empty:
        champion_rank_text = (
            "not present in the common validation comparison"
        )
        champion_business_score = float("nan")
    else:
        champion_rank = int(
            champion_validation["validation_rank"].iloc[0]
        )
        champion_business_score = float(
            champion_validation["business_score"].iloc[0]
        )
        champion_rank_text = (
            f"rank {champion_rank} of {len(model_table)}"
        )

    top_model = model_table.iloc[0]
    frontier = find_pareto_frontier(model_table)
    frontier_names = ", ".join(
        frontier["model_name"].astype(str).tolist()
    )

    champion_generalisation = generalisation_table.loc[
        generalisation_table["model_name"] == champion_name
    ]

    if champion_generalisation.empty:
        pr_delta_text = "not available"
    else:
        delta = float(
            champion_generalisation["pr_auc_delta"].iloc[0]
        )
        pr_delta_text = f"{delta:+.3f}"

    registered_model = str(
        lineage.get("registered_model_name", "not recorded")
    )
    registered_version = str(
        lineage.get("registered_model_version", "not recorded")
    )

    return f"""# Champion Selection Visual Intelligence — Actual-Number Findings

## MLV-A01 — Model Performance Frontier

**What it shows:** Validation precision and recall at each model's selected threshold. Marker size represents PR-AUC, while the connected frontier identifies models that are not dominated on both precision and recall.

**Actual finding:** The saved champion is **{champion_name}**. The validation Pareto frontier contains **{frontier_names}**.

**Business conclusion:** Use the frontier to understand the targeting trade-off instead of selecting a model from one metric alone.

**Recommended action:** Keep the saved champion threshold under controlled monitoring and compare any challenger against the same validation split and business objective.

**Limitation:** This view describes validation behaviour. It does not prove current live-production performance.

## MLV-A02 — Multi-Metric Model Ranking

**What it shows:** Candidate strength across business score, PR-AUC, F1, precision, recall, and ROC-AUC.

**Actual finding:** **{top_model['model_name']}** has the highest validation business score at **{float(top_model['business_score']):.3f}**. The saved champion is **{champion_rank_text}** with a validation business score of **{champion_business_score:.3f}**.

**Business conclusion:** The champion decision is supported by multiple metrics rather than accuracy alone.

**Recommended action:** Continue using business score as the primary selection signal, while retaining PR-AUC and threshold metrics as mandatory supporting evidence.

**Limitation:** Heatmap colour is relative to the candidates in this table and must not be interpreted as an absolute production grade.

## MLV-A03 — Validation-to-Holdout Generalisation

**What it shows:** PR-AUC movement from validation to the untouched final holdout.

**Actual finding:** The champion PR-AUC changed by **{pr_delta_text}**, from the validation stage to a final holdout PR-AUC of **{float(champion_metrics['pr_auc']):.3f}**.

**Business conclusion:** The final holdout provides a stronger and more honest proof point than validation alone.

**Recommended action:** Preserve the untouched-holdout result as the model-selection evidence and avoid repeatedly tuning against it.

**Limitation:** The final holdout contains only **{int(champion_metrics['positive_rows']):,} positive rows**, so uncertainty remains material.

## MLV-A04 — Champion Scorecard and Identity

**What it shows:** The selected model, operating threshold, final holdout metrics, evaluation volume, and MLflow registry identity.

**Actual finding:** **{champion_name}** uses threshold **{float(champion_metrics['threshold']):.2f}**. On **{int(champion_metrics['holdout_rows']):,} holdout rows**, precision is **{float(champion_metrics['precision']):.1%}**, recall is **{float(champion_metrics['recall']):.1%}**, F1 is **{float(champion_metrics['f1_score']):.3f}**, PR-AUC is **{float(champion_metrics['pr_auc']):.3f}**, and ROC-AUC is **{float(champion_metrics['roc_auc']):.3f}**.

**Business conclusion:** The operating point deliberately targets a very small high-intent audience rather than maximising broad prediction volume.

**Recommended action:** Use MLflow registered model **{registered_model}**, version **{registered_version}**, as the controlled champion reference.

**Limitation:** Real delayed production labels are not yet available beyond the source-data boundary, so live conversion performance must not be invented or claimed.
"""


def generate_champion_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate MLV-A01 to MLV-A04 and their supporting evidence.

    Input:
        Project root and optional repository-relative output directory.

    Output:
        Manifest dictionary containing source files, artifact paths, and QA.

    Used next:
        The runner prints a concise summary and MLflow can log the directory.
    """

    root = Path(project_root)

    # Use the standard package folder unless a test supplies a temporary path.
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

    # Load and prepare the existing verified project evidence.
    comparison, metadata, lineage = load_champion_sources(root)
    model_table = prepare_model_comparison(
        comparison,
        metadata,
    )
    generalisation_table = build_generalisation_table(
        comparison,
        metadata,
    )
    champion_metrics = build_champion_metrics(metadata)

    # Stable filenames keep MLflow, Streamlit, and reports aligned.
    visual_paths = {
        "MLV-A01": final_output_dir
        / f"{safe_filename('MLV-A01 Model Performance Frontier')}.png",
        "MLV-A02": final_output_dir
        / f"{safe_filename('MLV-A02 Multi-Metric Model Ranking')}.png",
        "MLV-A03": final_output_dir
        / (
            safe_filename(
                "MLV-A03 Validation to Holdout "
                "Generalisation Slopegraph"
            )
            + ".png"
        ),
        "MLV-A04": final_output_dir
        / f"{safe_filename('MLV-A04 Champion Scorecard Identity')}.png",
    }

    # Generate each visual through the shared QA-controlled exporter.
    qa_results = {
        "MLV-A01": create_model_performance_frontier(
            model_table,
            visual_paths["MLV-A01"],
        ),
        "MLV-A02": create_multi_metric_ranking(
            model_table,
            visual_paths["MLV-A02"],
        ),
        "MLV-A03": create_generalisation_slopegraph(
            generalisation_table,
            str(champion_metrics["model_name"]),
            champion_metrics,
            visual_paths["MLV-A03"],
        ),
        "MLV-A04": create_champion_scorecard(
            champion_metrics,
            lineage,
            visual_paths["MLV-A04"],
        ),
    }

    # Save the exact ranking values shown in A01 and A02.
    ranking_csv = (
        final_output_dir / "champion_model_ranking.csv"
    )
    model_table.to_csv(ranking_csv, index=False)

    # Save the validation-to-holdout values shown in A03.
    generalisation_csv = (
        final_output_dir / "champion_generalisation_metrics.csv"
    )
    generalisation_table.to_csv(
        generalisation_csv,
        index=False,
    )

    # Write actual-number conclusions beside the visual files.
    insight_markdown = (
        final_output_dir / "champion_selection_insights.md"
    )
    insight_markdown.write_text(
        build_insight_markdown(
            model_table,
            generalisation_table,
            champion_metrics,
            lineage,
        ),
        encoding="utf-8",
    )

    # Create a machine-readable artifact and source manifest.
    manifest_path = (
        final_output_dir / "champion_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Champion selection",
        "sources": [
            str(COMPARISON_PATH),
            str(CHAMPION_METADATA_PATH),
            str(MLFLOW_LINEAGE_PATH),
        ],
        "artifacts": {
            visual_id: str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path)
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "ranking_csv": str(ranking_csv.relative_to(root))
            if ranking_csv.is_relative_to(root)
            else str(ranking_csv),
            "generalisation_csv": (
                str(generalisation_csv.relative_to(root))
                if generalisation_csv.is_relative_to(root)
                else str(generalisation_csv)
            ),
            "insights": str(insight_markdown.relative_to(root))
            if insight_markdown.is_relative_to(root)
            else str(insight_markdown),
        },
        "champion": champion_metrics,
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
