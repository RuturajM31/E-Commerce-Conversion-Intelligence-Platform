"""Tests for the shared ML Visual Intelligence style and QA helpers.

Why this file exists:
    The visual foundation is reused by every future ML chart. These tests
    make sure filename creation, formatting, layout checks, and PNG export
    remain stable before individual charts depend on them.

Inputs:
    - Small synthetic Matplotlib figures
    - Temporary output directories supplied by pytest

Outputs:
    - Pass/fail evidence for visual foundation behaviour

Used next:
    CI and local validation run this file before ML visuals are committed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.visualization.ml_visual_style import (
    DEFAULT_SPEC,
    add_footer,
    add_title_block,
    create_figure,
    percent_formatter,
    reserve_chart_space,
    safe_filename,
    save_figure_with_qa,
    style_axes,
    validate_figure_layout,
)


def test_safe_filename_creates_stable_artifact_name() -> None:
    """A visual title should become a predictable artifact filename."""

    # Arrange: this title contains spaces and an em dash.
    visual_title = "MLV-A01 — Model Performance Frontier"

    # Act: convert it into a safe filename stem.
    result = safe_filename(visual_title)

    # Assert: the output is lowercase, stable, and filesystem-safe.
    assert result == "mlv-a01_model_performance_frontier"


def test_safe_filename_rejects_empty_name() -> None:
    """An empty title must not create an unsafe output path."""

    # Arrange: whitespace becomes empty after cleaning.
    empty_title = "   "

    # Act and assert: the helper must stop with a clear error.
    with pytest.raises(ValueError):
        safe_filename(empty_title)


def test_percent_formatter_uses_fraction_values() -> None:
    """Fraction values should display as readable percentages."""

    # Arrange: request one decimal place.
    formatter = percent_formatter(decimals=1)

    # Act: format a precision value stored as a fraction.
    formatted_value = formatter(0.285714, 0)

    # Assert: the visual displays the business-friendly percentage.
    assert formatted_value == "28.6%"


def test_layout_validation_accepts_clean_visual() -> None:
    """A correctly spaced chart should pass pre-export QA."""

    # Arrange: create the same standard figure future visuals will use.
    fig, ax = create_figure()

    # Add the required business title and reading context.
    add_title_block(
        fig,
        title="Model performance frontier",
        subtitle="Precision and recall trade-off across candidates.",
    )

    # Add a source note so evidence remains traceable.
    add_footer(
        fig,
        source_note="Source: validated model comparison table",
    )

    # Reserve space so titles, axes, and footer do not overlap.
    reserve_chart_space(fig)

    # Add a small synthetic line only to test the layout framework.
    ax.plot([0.1, 0.2], [0.2, 0.3])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    style_axes(ax)

    # Act: validate required text and visual boundaries.
    result = validate_figure_layout(
        fig,
        required_text=[
            "Model performance frontier",
            "Precision and recall trade-off across candidates.",
        ],
    )

    # Close the test figure so the test suite does not leak memory.
    plt.close(fig)

    # Assert: every mandatory visual-hygiene check passed.
    assert result.passed is True
    assert all(result.checks.values())


def test_layout_validation_detects_missing_required_text() -> None:
    """Missing explanatory text should fail visual QA."""

    # Arrange: create a figure without the required subtitle.
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])

    # Act: request a subtitle that is not present.
    result = validate_figure_layout(
        fig,
        required_text=["Required subtitle"],
    )

    # Close the figure after validation.
    plt.close(fig)

    # Assert: QA catches the missing explanation.
    assert result.passed is False
    assert result.checks["required_text_present"] is False


def test_save_figure_with_qa_creates_large_clean_export(
    tmp_path: Path,
) -> None:
    """The shared saver should create a readable PNG artifact."""

    # Arrange: create a realistic metric chart using project-style helpers.
    fig, ax = create_figure()

    add_title_block(
        fig,
        title="Champion identity",
        subtitle="Validated production model and operating threshold.",
    )

    add_footer(
        fig,
        source_note="Source: final champion metadata",
    )

    reserve_chart_space(fig)

    # Use simple known values so the test checks export logic, not modelling.
    metric_names = ["Precision", "Recall", "F1"]
    metric_values = [0.286, 0.208, 0.240]

    ax.bar(metric_names, metric_values)
    ax.yaxis.set_major_formatter(percent_formatter(decimals=0))
    ax.set_ylim(0, 0.35)
    style_axes(ax)

    # Temporary path keeps test artifacts outside the repository.
    output = tmp_path / "champion_identity.png"

    # Act: validate the layout, save the PNG, and validate its dimensions.
    result = save_figure_with_qa(
        fig,
        output,
        required_text=[
            "Champion identity",
            "Validated production model and operating threshold.",
        ],
    )

    # Assert: the file exists and meets the minimum quality requirements.
    assert result.passed is True
    assert output.exists()
    assert output.stat().st_size > 10_000
    assert result.width_px is not None
    assert result.height_px is not None
    assert result.width_px >= DEFAULT_SPEC.min_width_px
    assert result.height_px >= DEFAULT_SPEC.min_height_px
