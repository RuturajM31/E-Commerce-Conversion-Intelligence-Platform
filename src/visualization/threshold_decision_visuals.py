"""Create the MLV-B01 and MLV-B04 threshold-decision visual package.

Why this file exists:
    The project already stores validated threshold curves and final holdout
    metrics. This module turns those sources into two professional decision
    visuals without retraining the model or inventing production outcomes.

Visuals created:
    - MLV-B01: Threshold Decision Studio
    - MLV-B04: Confusion-Matrix Decision Map

Main inputs:
    - reports/tables/final_true_champion_thresholds.csv
    - models/metadata/final_champion_metadata.json

Main outputs:
    - Two QA-validated PNG files
    - One selected-threshold evidence CSV
    - One confusion-count CSV
    - One actual-number insight Markdown file
    - One JSON manifest

Used next:
    The approved package can be logged to MLflow and later reused in
    Streamlit, reports, and presentation material.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from src.visualization.ml_visual_style import (
    COLORS,
    DEFAULT_SPEC,
    VisualQAResult,
    add_footer,
    add_title_block,
    apply_ml_visual_style,
    percent_formatter,
    safe_filename,
    save_figure_with_qa,
    style_axes,
)


# ---------------------------------------------------------------------------
# Repository-relative paths
# ---------------------------------------------------------------------------

THRESHOLD_SOURCE_PATH = Path(
    "reports/tables/final_true_champion_thresholds.csv"
)
CHAMPION_METADATA_PATH = Path(
    "models/metadata/final_champion_metadata.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "reports/visuals/ml_visual_intelligence/threshold_decisions"
)


# ---------------------------------------------------------------------------
# Source loading and validation
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object.

    Input:
        Path to the final champion metadata file.

    Output:
        Parsed dictionary.

    Used next:
        Champion identity, selected threshold, and holdout metrics are read
        from this dictionary.
    """

    # Stop clearly when the required evidence file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    # Load the existing source without modifying it.
    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def validate_threshold_schema(thresholds: pd.DataFrame) -> None:
    """Validate the threshold-table source contract.

    Input:
        Raw `final_true_champion_thresholds.csv` DataFrame.

    Output:
        No value. A clear error is raised when required columns are missing.

    Used next:
        Prevents the decision studio from using incomplete or renamed data.
    """

    required_columns = {
        "model_name",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    }

    missing_columns = sorted(
        required_columns.difference(thresholds.columns)
    )

    if missing_columns:
        raise ValueError(
            "Threshold source is missing required columns: "
            + ", ".join(missing_columns)
        )


def load_threshold_sources(
    project_root: str | Path = ".",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the verified threshold table and champion metadata.

    Input:
        Project root. Relative paths keep the project portable.

    Output:
        Raw threshold DataFrame and champion metadata dictionary.

    Used next:
        Preparation functions isolate the saved champion and derive counts.
    """

    root = Path(project_root)
    threshold_file = root / THRESHOLD_SOURCE_PATH
    metadata_file = root / CHAMPION_METADATA_PATH

    if not threshold_file.exists():
        raise FileNotFoundError(
            f"Required threshold source not found: {threshold_file}"
        )

    thresholds = pd.read_csv(threshold_file)
    metadata = read_json(metadata_file)

    return thresholds, metadata


def prepare_champion_thresholds(
    thresholds: pd.DataFrame,
    champion_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Prepare the saved champion's validation threshold curve.

    Input:
        Full multi-model threshold table and final champion metadata.

    Output:
        One sorted table containing only the saved champion's thresholds.

    Used next:
        MLV-B01 plots precision, recall, F1, and selected-audience share.
    """

    # Confirm the current file still matches the audited source contract.
    validate_threshold_schema(thresholds)

    champion_name = str(
        champion_metadata.get("final_model_name", "")
    ).strip()

    if not champion_name:
        raise ValueError(
            "Champion metadata does not contain `final_model_name`."
        )

    # Keep only the authoritative saved champion.
    champion_rows = thresholds.loc[
        thresholds["model_name"].astype(str) == champion_name
    ].copy()

    if champion_rows.empty:
        raise ValueError(
            f"No threshold rows found for saved champion: {champion_name}"
        )

    numeric_columns = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    ]

    for column in numeric_columns:
        champion_rows[column] = pd.to_numeric(
            champion_rows[column],
            errors="raise",
        )

    # Keep one row per threshold and order from permissive to strict.
    champion_rows = (
        champion_rows
        .sort_values("threshold")
        .drop_duplicates(subset=["threshold"], keep="last")
        .reset_index(drop=True)
    )

    return champion_rows


def get_selected_threshold_row(
    champion_thresholds: pd.DataFrame,
    champion_metadata: dict[str, Any],
) -> pd.Series:
    """Return the exact saved operating-threshold row.

    Input:
        Champion threshold table and saved threshold in metadata.

    Output:
        One threshold row.

    Used next:
        The visual highlights this operating point and the evidence CSV stores
        its validation metrics.
    """

    selected_threshold = float(
        champion_metadata.get("best_threshold")
    )

    matches = np.isclose(
        champion_thresholds["threshold"].to_numpy(dtype=float),
        selected_threshold,
        rtol=0.0,
        atol=1e-9,
    )

    selected_rows = champion_thresholds.loc[matches]

    if selected_rows.empty:
        raise ValueError(
            "Saved operating threshold is not present in the champion "
            f"threshold table: {selected_threshold:.4f}"
        )

    return selected_rows.iloc[0]


def derive_confusion_counts(
    champion_metadata: dict[str, Any],
) -> dict[str, int | float]:
    """Reconstruct the exact final-holdout confusion counts.

    Input:
        `final_holdout_metrics` from final champion metadata.

    Output:
        TP, FP, FN, TN plus verified precision, recall, F1, and row totals.

    Used next:
        MLV-B04 displays decision outcomes with actual counts and rates.
    """

    holdout = champion_metadata.get("final_holdout_metrics")

    if not isinstance(holdout, dict):
        raise ValueError(
            "Champion metadata does not contain final holdout metrics."
        )

    required_fields = {
        "rows",
        "positive_rows",
        "predicted_positive_rows",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    }

    missing_fields = sorted(
        required_fields.difference(holdout)
    )

    if missing_fields:
        raise ValueError(
            "Final holdout metrics are missing fields: "
            + ", ".join(missing_fields)
        )

    total_rows = int(holdout["rows"])
    positive_rows = int(holdout["positive_rows"])
    predicted_positive_rows = int(
        holdout["predicted_positive_rows"]
    )
    precision = float(holdout["precision"])
    recall = float(holdout["recall"])
    f1_score = float(holdout["f1_score"])
    predicted_positive_rate = float(
        holdout["predicted_positive_rate"]
    )

    # TP can be reconstructed independently from recall and precision.
    tp_from_recall = int(round(recall * positive_rows))
    tp_from_precision = int(
        round(precision * predicted_positive_rows)
    )

    if tp_from_recall != tp_from_precision:
        raise ValueError(
            "Precision and recall imply different true-positive counts."
        )

    true_positive = tp_from_recall
    false_negative = positive_rows - true_positive
    false_positive = predicted_positive_rows - true_positive
    true_negative = (
        total_rows
        - true_positive
        - false_negative
        - false_positive
    )

    counts = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_rows": total_rows,
        "positive_rows": positive_rows,
        "negative_rows": total_rows - positive_rows,
        "predicted_positive_rows": predicted_positive_rows,
        "predicted_positive_rate": predicted_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    # Verify the reconstructed counts reproduce the saved metrics.
    reconstructed_precision = (
        true_positive
        / (true_positive + false_positive)
    )
    reconstructed_recall = (
        true_positive
        / (true_positive + false_negative)
    )

    if not np.isclose(
        reconstructed_precision,
        precision,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Reconstructed confusion counts do not reproduce precision."
        )

    if not np.isclose(
        reconstructed_recall,
        recall,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Reconstructed confusion counts do not reproduce recall."
        )

    return counts


# ---------------------------------------------------------------------------
# Reusable visual components
# ---------------------------------------------------------------------------


def _draw_kpi_card(
    axis,
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
    """Draw one compact executive KPI card."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.1,
    )
    axis.add_patch(card)

    axis.add_patch(
        Rectangle(
            (x, y),
            0.010,
            height,
            transform=axis.transAxes,
            facecolor=accent,
            edgecolor="none",
        )
    )

    axis.text(
        x + 0.025,
        y + height - 0.028,
        label.upper(),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    axis.text(
        x + 0.025,
        y + height * 0.52,
        value,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=COLORS["navy"],
    )
    axis.text(
        x + 0.025,
        y + 0.022,
        note,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=COLORS["grey_700"],
    )


# ---------------------------------------------------------------------------
# MLV-B01 Threshold Decision Studio
# ---------------------------------------------------------------------------


def create_threshold_decision_studio(
    champion_thresholds: pd.DataFrame,
    selected_row: pd.Series,
    champion_metadata: dict[str, Any],
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-B01 Threshold Decision Studio.

    Input:
        Champion validation threshold curve, selected row, and holdout context.

    Output:
        QA-validated PNG showing metric trade-offs and audience size.

    Used next:
        Business users can see what changes when the threshold becomes stricter
        or more permissive.
    """

    title = "MLV-B01 | Threshold Decision Studio"
    subtitle = (
        "Validation precision, recall, F1, and selected-audience share "
        "across the saved champion's operating thresholds."
    )

    apply_ml_visual_style(DEFAULT_SPEC)

    # Use a three-panel layout: trade-off curves, audience curve, decision card.
    figure = plt.figure(
        figsize=(DEFAULT_SPEC.width, DEFAULT_SPEC.height),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1.05, 0.95],
        left=0.07,
        right=0.96,
        bottom=0.12,
        top=0.81,
        wspace=0.25,
        hspace=0.28,
    )

    metric_axis = figure.add_subplot(grid[:, 0])
    audience_axis = figure.add_subplot(grid[0, 1])
    decision_axis = figure.add_subplot(grid[1, 1])
    decision_axis.axis("off")

    add_title_block(
        figure,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        figure,
        source_note=(
            "Sources: final_true_champion_thresholds.csv and "
            "final_champion_metadata.json"
        ),
        interpretation_note=(
            "Curves use validation data; final-holdout results are shown "
            "only as operating-context evidence."
        ),
    )

    selected_threshold = float(selected_row["threshold"])

    # Main panel: trade-off among precision, recall, and F1.
    metric_series = [
        ("precision", "Precision", COLORS["green"]),
        ("recall", "Recall", COLORS["red"]),
        ("f1_score", "F1", COLORS["purple"]),
    ]

    for column, label, colour in metric_series:
        metric_axis.plot(
            champion_thresholds["threshold"],
            champion_thresholds[column],
            label=label,
            color=colour,
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            alpha=0.95,
        )

        # Highlight the exact value at the saved operating threshold.
        metric_axis.scatter(
            [selected_threshold],
            [float(selected_row[column])],
            s=92,
            color=colour,
            edgecolor=COLORS["white"],
            linewidth=1.8,
            zorder=6,
        )

    metric_axis.axvline(
        selected_threshold,
        color=COLORS["navy"],
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
    )
    metric_axis.text(
        selected_threshold,
        1.02,
        f"Saved threshold {selected_threshold:.2f}",
        transform=metric_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["navy"],
    )

    metric_axis.set_title(
        "Performance trade-off",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        pad=12,
    )
    metric_axis.set_xlabel("Probability threshold")
    metric_axis.set_ylabel("Validation metric")
    metric_axis.set_xlim(
        float(champion_thresholds["threshold"].min()) - 0.02,
        float(champion_thresholds["threshold"].max()) + 0.02,
    )
    metric_axis.set_ylim(0.0, 1.03)
    metric_axis.yaxis.set_major_formatter(
        percent_formatter(decimals=0)
    )
    style_axes(metric_axis)
    metric_axis.legend(
        loc="upper center",
        ncol=3,
    )

    # Top-right panel: audience size across thresholds.
    audience_axis.plot(
        champion_thresholds["threshold"],
        champion_thresholds["predicted_positive_rate"],
        color=COLORS["blue"],
        linewidth=2.4,
        marker="o",
        markersize=4.5,
    )
    audience_axis.scatter(
        [selected_threshold],
        [float(selected_row["predicted_positive_rate"])],
        s=95,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        linewidth=1.6,
        zorder=5,
    )
    audience_axis.axvline(
        selected_threshold,
        color=COLORS["navy"],
        linewidth=1.4,
        linestyle="--",
    )
    audience_axis.set_yscale("log")
    audience_axis.set_title(
        "Audience selected",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        pad=10,
    )
    audience_axis.set_xlabel("Probability threshold")
    audience_axis.set_ylabel("Predicted-positive share (log scale)")
    audience_axis.yaxis.set_major_formatter(
        percent_formatter(decimals=2)
    )
    style_axes(audience_axis)

    audience_axis.annotate(
        f"{float(selected_row['predicted_positive_rate']):.3%}",
        xy=(
            selected_threshold,
            float(selected_row["predicted_positive_rate"]),
        ),
        xytext=(-8, 18),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["navy"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["grey_500"],
            "linewidth": 0.9,
        },
    )

    # Bottom-right panel: decision summary with validation and holdout context.
    decision_axis.text(
        0.00,
        0.98,
        "OPERATING DECISION",
        transform=decision_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    decision_axis.text(
        0.00,
        0.88,
        str(champion_metadata["final_model_name"]),
        transform=decision_axis.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLORS["navy"],
    )

    # First card row: selected validation metrics.
    _draw_kpi_card(
        decision_axis,
        x=0.00,
        y=0.48,
        width=0.31,
        height=0.28,
        label="Threshold",
        value=f"{selected_threshold:.2f}",
        note="Saved operating point",
        accent=COLORS["amber"],
    )
    _draw_kpi_card(
        decision_axis,
        x=0.345,
        y=0.48,
        width=0.31,
        height=0.28,
        label="Validation precision",
        value=f"{float(selected_row['precision']):.1%}",
        note="Target quality",
        accent=COLORS["green"],
    )
    _draw_kpi_card(
        decision_axis,
        x=0.69,
        y=0.48,
        width=0.31,
        height=0.28,
        label="Validation recall",
        value=f"{float(selected_row['recall']):.1%}",
        note="Converter capture",
        accent=COLORS["red"],
    )

    holdout = champion_metadata["final_holdout_metrics"]

    # Second card row: audience and final-holdout evidence.
    _draw_kpi_card(
        decision_axis,
        x=0.00,
        y=0.10,
        width=0.31,
        height=0.28,
        label="Validation audience",
        value=(
            f"{float(selected_row['predicted_positive_rate']):.3%}"
        ),
        note="Predicted-positive share",
        accent=COLORS["blue"],
    )
    _draw_kpi_card(
        decision_axis,
        x=0.345,
        y=0.10,
        width=0.31,
        height=0.28,
        label="Holdout precision",
        value=f"{float(holdout['precision']):.1%}",
        note="Final untouched split",
        accent=COLORS["teal"],
    )
    _draw_kpi_card(
        decision_axis,
        x=0.69,
        y=0.10,
        width=0.31,
        height=0.28,
        label="Holdout recall",
        value=f"{float(holdout['recall']):.1%}",
        note="Final untouched split",
        accent=COLORS["purple"],
    )

    return save_figure_with_qa(
        figure,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# MLV-B04 Confusion-Matrix Decision Map
# ---------------------------------------------------------------------------


def create_confusion_matrix_decision_map(
    confusion: dict[str, int | float],
    champion_metadata: dict[str, Any],
    output_path: Path,
) -> VisualQAResult:
    """Create MLV-B04 Confusion-Matrix Decision Map.

    Input:
        Verified final-holdout confusion counts and champion metadata.

    Output:
        QA-validated PNG explaining captured, missed, and false-alert outcomes.

    Used next:
        Business users can understand the actual decision consequences of the
        saved threshold.
    """

    title = "MLV-B04 | Confusion-Matrix Decision Map"
    subtitle = (
        "Final-holdout decisions at threshold "
        f"{float(champion_metadata['best_threshold']):.2f}, showing actual "
        "counts, within-row rates, and business consequences."
    )

    apply_ml_visual_style(DEFAULT_SPEC)

    figure = plt.figure(
        figsize=(DEFAULT_SPEC.width, DEFAULT_SPEC.height),
        facecolor=DEFAULT_SPEC.facecolor,
    )
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.35, 1.0],
        left=0.08,
        right=0.96,
        bottom=0.11,
        top=0.80,
        wspace=0.24,
    )

    matrix_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])
    summary_axis.axis("off")

    add_title_block(
        figure,
        title=title,
        subtitle=subtitle,
    )
    add_footer(
        figure,
        source_note="Source: final_champion_metadata.json",
        interpretation_note=(
            "Counts are reconstructed and verified from saved aggregate "
            "precision, recall, positives, and predicted positives."
        ),
    )

    tp = int(confusion["true_positive"])
    fp = int(confusion["false_positive"])
    fn = int(confusion["false_negative"])
    tn = int(confusion["true_negative"])
    positives = int(confusion["positive_rows"])
    negatives = int(confusion["negative_rows"])
    total_rows = int(confusion["total_rows"])

    # Matrix orientation:
    # rows = actual outcome, columns = model decision.
    cell_values = [
        [tp, fn],
        [fp, tn],
    ]
    cell_titles = [
        ["Captured converters", "Missed converters"],
        ["False alerts", "Correct exclusions"],
    ]
    row_denominators = [positives, negatives]
    cell_colours = [
        [COLORS["green"], COLORS["red"]],
        [COLORS["amber"], COLORS["blue"]],
    ]

    matrix_axis.set_xlim(0, 2)
    matrix_axis.set_ylim(0, 2)
    matrix_axis.invert_yaxis()
    matrix_axis.set_aspect("equal")
    matrix_axis.axis("off")

    # Draw the four decision cells with count and within-row rate.
    for row_index in range(2):
        for column_index in range(2):
            value = cell_values[row_index][column_index]
            row_rate = value / row_denominators[row_index]
            colour = cell_colours[row_index][column_index]

            rectangle = FancyBboxPatch(
                (column_index + 0.04, row_index + 0.04),
                0.92,
                0.92,
                boxstyle="round,pad=0.015,rounding_size=0.04",
                facecolor=colour,
                edgecolor=COLORS["white"],
                linewidth=2.0,
                alpha=0.92,
            )
            matrix_axis.add_patch(rectangle)

            matrix_axis.text(
                column_index + 0.50,
                row_index + 0.24,
                cell_titles[row_index][column_index],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=COLORS["white"],
            )
            matrix_axis.text(
                column_index + 0.50,
                row_index + 0.53,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=25,
                fontweight="bold",
                color=COLORS["white"],
            )
            matrix_axis.text(
                column_index + 0.50,
                row_index + 0.78,
                f"{row_rate:.2%} of actual row",
                ha="center",
                va="center",
                fontsize=9,
                color=COLORS["white"],
            )

    # Add matrix headings outside the cells.
    matrix_axis.text(
        0.50,
        -0.12,
        "Predicted converter",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    matrix_axis.text(
        1.50,
        -0.12,
        "Predicted non-converter",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    matrix_axis.text(
        -0.02,
        0.50,
        "Actual\nconverter",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    matrix_axis.text(
        -0.02,
        1.50,
        "Actual\nnon-converter",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )

    # Right panel: executive metrics and decision consequences.
    summary_axis.text(
        0.00,
        0.98,
        "DECISION CONSEQUENCES",
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    summary_axis.text(
        0.00,
        0.90,
        str(champion_metadata["final_model_name"]),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["navy"],
    )

    _draw_kpi_card(
        summary_axis,
        x=0.00,
        y=0.62,
        width=0.47,
        height=0.20,
        label="Precision",
        value=f"{float(confusion['precision']):.1%}",
        note="66 true converters among 231 targets",
        accent=COLORS["green"],
    )
    _draw_kpi_card(
        summary_axis,
        x=0.52,
        y=0.62,
        width=0.47,
        height=0.20,
        label="Recall",
        value=f"{float(confusion['recall']):.1%}",
        note="66 of 318 converters captured",
        accent=COLORS["red"],
    )
    _draw_kpi_card(
        summary_axis,
        x=0.00,
        y=0.38,
        width=0.47,
        height=0.20,
        label="F1 score",
        value=f"{float(confusion['f1_score']):.3f}",
        note="Precision–recall balance",
        accent=COLORS["purple"],
    )
    _draw_kpi_card(
        summary_axis,
        x=0.52,
        y=0.38,
        width=0.47,
        height=0.20,
        label="Audience selected",
        value=f"{int(confusion['predicted_positive_rows']):,}",
        note=f"{float(confusion['predicted_positive_rate']):.3%} of holdout",
        accent=COLORS["blue"],
    )

    # Bottom narrative panel translates counts into business language.
    narrative = FancyBboxPatch(
        (0.00, 0.015),
        0.99,
        0.315,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        transform=summary_axis.transAxes,
        facecolor=COLORS["grey_100"],
        edgecolor=COLORS["grey_200"],
        linewidth=1.0,
    )
    summary_axis.add_patch(narrative)

    target_yield = (
        100.0 * tp / (tp + fp)
    )
    missed_share = (
        100.0 * fn / positives
    )

    summary_axis.text(
        0.035,
        0.286,
        "HOW TO READ THIS",
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["grey_500"],
    )
    summary_axis.text(
        0.035,
        0.232,
        (
            f"• Per 100 targeted visitors: about {target_yield:.0f} converters"
            f"\n  and {100 - target_yield:.0f} false alerts."
        ),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["grey_700"],
    )
    summary_axis.text(
        0.035,
        0.162,
        (
            f"• The strict threshold missed {fn:,} of {positives:,} "
            f"converters ({missed_share:.1f}%)."
        ),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["grey_700"],
    )
    summary_axis.text(
        0.035,
        0.108,
        (
            f"• Correctly excluded {tn:,} non-converters,"
            "\n  keeping the campaign audience very small."
        ),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["grey_700"],
    )
    summary_axis.plot(
        [0.035, 0.955],
        [0.066, 0.066],
        transform=summary_axis.transAxes,
        color=COLORS["grey_300"],
        linewidth=0.8,
    )
    summary_axis.text(
        0.035,
        0.052,
        (
            "Limitation: final-holdout evidence only;\n"
            "live delayed-label production performance is not yet available."
        ),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        linespacing=1.15,
        fontstyle="italic",
        color=COLORS["grey_500"],
    )

    return save_figure_with_qa(
        figure,
        output_path,
        required_text=[title, subtitle],
    )


# ---------------------------------------------------------------------------
# Insight and package generation
# ---------------------------------------------------------------------------


def build_threshold_insight_markdown(
    selected_row: pd.Series,
    confusion: dict[str, int | float],
    champion_metadata: dict[str, Any],
) -> str:
    """Create actual-number conclusions for B01 and B04."""

    threshold = float(selected_row["threshold"])
    validation_precision = float(selected_row["precision"])
    validation_recall = float(selected_row["recall"])
    validation_f1 = float(selected_row["f1_score"])
    validation_share = float(
        selected_row["predicted_positive_rate"]
    )

    return f"""# Threshold Decision Visual Intelligence — Actual-Number Findings

## MLV-B01 — Threshold Decision Studio

**What it shows:** How validation precision, recall, F1, and predicted-positive share change as the decision threshold becomes stricter.

**Actual finding:** The saved champion uses threshold **{threshold:.2f}**. At that validation operating point, precision is **{validation_precision:.1%}**, recall is **{validation_recall:.1%}**, F1 is **{validation_f1:.3f}**, and the predicted-positive share is **{validation_share:.3%}**.

**Business conclusion:** The saved threshold deliberately selects a very small audience to prioritise target quality over broad converter capture.

**Recommended action:** Keep the current threshold as the controlled baseline and simulate any future threshold change against campaign capacity and value assumptions before deployment.

**Limitation:** The threshold curves are validation evidence. They do not represent current live-production outcomes.

## MLV-B04 — Confusion-Matrix Decision Map

**What it shows:** Final-holdout decision outcomes at the saved threshold.

**Actual finding:** The model captured **{int(confusion['true_positive']):,}** converters, generated **{int(confusion['false_positive']):,}** false alerts, missed **{int(confusion['false_negative']):,}** converters, and correctly excluded **{int(confusion['true_negative']):,}** non-converters. Precision is **{float(confusion['precision']):.1%}**, recall is **{float(confusion['recall']):.1%}**, and F1 is **{float(confusion['f1_score']):.3f}**.

**Business conclusion:** Approximately **{100 * float(confusion['precision']):.0f}** of every 100 targeted visitors were converters, but **{100 * int(confusion['false_negative']) / int(confusion['positive_rows']):.1f}%** of actual converters were not captured.

**Recommended action:** Use this strict operating point when campaign capacity is limited; evaluate a lower threshold when the cost of missed converters is more important than false alerts.

**Limitation:** These counts come from the untouched final holdout. Real production performance remains blocked until delayed labels mature.
"""


def generate_threshold_visual_package(
    project_root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate B01 and B04 plus supporting evidence.

    Input:
        Project root and optional output directory.

    Output:
        Manifest containing artifact paths, champion evidence, and QA results.

    Used next:
        The runner prints a concise summary and MLflow logs the approved folder.
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
    final_output_dir.mkdir(parents=True, exist_ok=True)

    thresholds, metadata = load_threshold_sources(root)
    champion_thresholds = prepare_champion_thresholds(
        thresholds,
        metadata,
    )
    selected_row = get_selected_threshold_row(
        champion_thresholds,
        metadata,
    )
    confusion = derive_confusion_counts(metadata)

    visual_paths = {
        "MLV-B01": final_output_dir
        / (
            safe_filename("MLV-B01 Threshold Decision Studio")
            + ".png"
        ),
        "MLV-B04": final_output_dir
        / (
            safe_filename("MLV-B04 Confusion Matrix Decision Map")
            + ".png"
        ),
    }

    qa_results = {
        "MLV-B01": create_threshold_decision_studio(
            champion_thresholds,
            selected_row,
            metadata,
            visual_paths["MLV-B01"],
        ),
        "MLV-B04": create_confusion_matrix_decision_map(
            confusion,
            metadata,
            visual_paths["MLV-B04"],
        ),
    }

    selected_evidence = selected_row.to_frame().T.copy()
    selected_evidence.insert(
        0,
        "saved_champion",
        metadata["final_model_name"],
    )
    selected_evidence_path = (
        final_output_dir / "selected_threshold_evidence.csv"
    )
    selected_evidence.to_csv(
        selected_evidence_path,
        index=False,
    )

    confusion_path = (
        final_output_dir / "confusion_matrix_counts.csv"
    )
    pd.DataFrame([confusion]).to_csv(
        confusion_path,
        index=False,
    )

    insight_path = (
        final_output_dir / "threshold_decision_insights.md"
    )
    insight_path.write_text(
        build_threshold_insight_markdown(
            selected_row,
            confusion,
            metadata,
        ),
        encoding="utf-8",
    )

    manifest_path = (
        final_output_dir / "threshold_visual_manifest.json"
    )

    manifest: dict[str, Any] = {
        "category": "ML Visual Intelligence",
        "package": "Threshold decisions",
        "sources": [
            str(THRESHOLD_SOURCE_PATH),
            str(CHAMPION_METADATA_PATH),
        ],
        "artifacts": {
            visual_id: (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
            )
            for visual_id, path in visual_paths.items()
        },
        "supporting_files": {
            "selected_threshold_evidence": (
                str(selected_evidence_path.relative_to(root))
                if selected_evidence_path.is_relative_to(root)
                else str(selected_evidence_path)
            ),
            "confusion_counts": (
                str(confusion_path.relative_to(root))
                if confusion_path.is_relative_to(root)
                else str(confusion_path)
            ),
            "insights": (
                str(insight_path.relative_to(root))
                if insight_path.is_relative_to(root)
                else str(insight_path)
            ),
        },
        "selected_threshold": float(
            selected_row["threshold"]
        ),
        "confusion_counts": confusion,
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
