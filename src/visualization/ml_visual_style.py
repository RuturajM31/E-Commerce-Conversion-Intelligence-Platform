"""Shared visual design and QA utilities for ML Visual Intelligence.

Why this file exists:
    Every ML chart must use the same professional design rules.
    This prevents inconsistent fonts, margins, export sizes, and annotations.

Main inputs:
    - Matplotlib figures and axes
    - A target output path
    - Optional title, subtitle, source note, and visual specification

Main outputs:
    - Clean PNG exports
    - A structured QA result describing whether the visual passed

Used next:
    Individual ML visual modules import these helpers before creating
    artifacts for MLflow, Streamlit, reports, or presentations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Shared design tokens
# ---------------------------------------------------------------------------

# Use a font bundled with Matplotlib so local, CI, and Docker rendering match.
FONT_FAMILY = "DejaVu Sans"

# Central colour dictionary.
# All future ML visuals should reuse these names instead of hard-coded colours.
COLORS = {
    "navy": "#0F2747",
    "blue": "#246BCE",
    "teal": "#0F8B8D",
    "green": "#2E8B57",
    "amber": "#D99000",
    "red": "#C23B3B",
    "purple": "#6F52A2",
    "grey_900": "#1F2937",
    "grey_700": "#4B5563",
    "grey_500": "#6B7280",
    "grey_300": "#D1D5DB",
    "grey_200": "#E5E7EB",
    "grey_100": "#F3F4F6",
    "white": "#FFFFFF",
}

# Reusable sequence for comparing several models.
# The order stays stable so the same model families look consistent.
MODEL_PALETTE = [
    COLORS["blue"],
    COLORS["teal"],
    COLORS["purple"],
    COLORS["amber"],
    COLORS["green"],
    COLORS["red"],
]


@dataclass(frozen=True)
class VisualSpec:
    """Configuration for one exported ML visual.

    Input:
        Values such as width, height, DPI, font sizes, and minimum resolution.

    Output:
        An immutable specification reused by figure creation and export QA.

    Used next:
        `create_figure()` and `save_figure_with_qa()` read these settings.
    """

    # Figure dimensions in inches.
    width: float = 13.5
    height: float = 7.6

    # Export quality.
    dpi: int = 180

    # Typography sizes.
    title_size: int = 18
    subtitle_size: int = 11
    label_size: int = 10
    tick_size: int = 9
    annotation_size: int = 9
    source_size: int = 8

    # Background colour used by both figure and export.
    facecolor: str = COLORS["white"]

    # Minimum rendered image size required by visual QA.
    min_width_px: int = 1_600
    min_height_px: int = 850


# Default project-wide visual specification.
DEFAULT_SPEC = VisualSpec()


@dataclass
class VisualQAResult:
    """Structured result returned by layout and export checks.

    Input:
        Boolean checks and any issues found during validation.

    Output:
        One object that can be tested, logged, or raised as an error.

    Used next:
        Tests and visual-generation scripts call `raise_for_failure()`.
    """

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    output_path: str | None = None
    width_px: int | None = None
    height_px: int | None = None

    def raise_for_failure(self) -> None:
        """Raise a clear error when any mandatory QA check failed."""

        # Stop immediately when every check passed.
        if self.passed:
            return

        # Combine all detected issues into one readable exception message.
        details = "; ".join(self.issues) or "Unknown visual QA failure."
        raise ValueError(f"Visual QA failed: {details}")


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------


def apply_ml_visual_style(spec: VisualSpec = DEFAULT_SPEC) -> None:
    """Apply the shared ML Visual Intelligence Matplotlib theme.

    Input:
        A `VisualSpec` containing font, size, and background settings.

    Output:
        Updated Matplotlib runtime settings.

    Used next:
        `create_figure()` calls this before creating a new figure.
    """

    # Update Matplotlib once with project-wide design rules.
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": spec.label_size,
            "axes.titlesize": spec.title_size,
            "axes.labelsize": spec.label_size,
            "axes.labelcolor": COLORS["grey_700"],
            "axes.edgecolor": COLORS["grey_300"],
            "axes.linewidth": 0.8,
            "axes.facecolor": spec.facecolor,
            "figure.facecolor": spec.facecolor,
            "xtick.labelsize": spec.tick_size,
            "ytick.labelsize": spec.tick_size,
            "xtick.color": COLORS["grey_700"],
            "ytick.color": COLORS["grey_700"],
            "grid.color": COLORS["grey_200"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": spec.tick_size,
            "savefig.facecolor": spec.facecolor,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def create_figure(
    spec: VisualSpec = DEFAULT_SPEC,
) -> tuple[Figure, Axes]:
    """Create a clean figure and one axes.

    Input:
        A `VisualSpec` describing size and background.

    Output:
        `(fig, ax)` used by the individual chart function.

    Used next:
        The calling visual adds data, titles, annotations, and footer text.
    """

    # Apply the shared theme before the figure is created.
    apply_ml_visual_style(spec)

    # Create one standard figure and one plotting area.
    fig, ax = plt.subplots(
        figsize=(spec.width, spec.height),
        facecolor=spec.facecolor,
    )

    return fig, ax


def style_axes(
    ax: Axes,
    *,
    show_x_grid: bool = False,
    show_y_grid: bool = True,
    remove_top_right: bool = True,
) -> None:
    """Apply consistent axis hygiene.

    Input:
        The axes created by the visual plus grid/spine preferences.

    Output:
        The same axes, styled in place.

    Used next:
        The chart is annotated and then passed to export QA.
    """

    # Start clean so old/default grid settings cannot leak into the chart.
    ax.grid(False)

    # Horizontal guides help compare values in most business charts.
    if show_y_grid:
        ax.yaxis.grid(True)

    # Vertical guides are optional because they can make charts feel crowded.
    if show_x_grid:
        ax.xaxis.grid(True)

    # Keep data marks above the grid lines.
    ax.set_axisbelow(True)

    # Remove decorative top and right borders unless a chart needs them.
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Keep the remaining borders subtle and consistent.
    ax.spines["left"].set_color(COLORS["grey_300"])
    ax.spines["bottom"].set_color(COLORS["grey_300"])


def add_title_block(
    fig: Figure,
    *,
    title: str,
    subtitle: str,
    spec: VisualSpec = DEFAULT_SPEC,
) -> None:
    """Add the title and subtitle above the plotting area.

    Input:
        Figure object plus chart title and business subtitle.

    Output:
        Two figure-level text artists with fixed spacing.

    Used next:
        `reserve_chart_space()` keeps the axes below this title block.
    """

    # Main title: concise message the viewer should understand first.
    fig.text(
        0.06,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=spec.title_size,
        fontweight="bold",
        color=COLORS["navy"],
    )

    # Subtitle: explains business context, comparison, or reading guidance.
    fig.text(
        0.06,
        0.915,
        subtitle,
        ha="left",
        va="top",
        fontsize=spec.subtitle_size,
        color=COLORS["grey_700"],
    )


def add_footer(
    fig: Figure,
    *,
    source_note: str,
    interpretation_note: str | None = None,
    spec: VisualSpec = DEFAULT_SPEC,
) -> None:
    """Add source and optional interpretation notes.

    Input:
        Source text and an optional short reading note.

    Output:
        Footer text placed below the plotting area.

    Used next:
        Export QA checks that the footer remains inside the image.
    """

    # Left footer records where the numbers came from.
    fig.text(
        0.06,
        0.025,
        source_note,
        ha="left",
        va="bottom",
        fontsize=spec.source_size,
        color=COLORS["grey_500"],
    )

    # Right footer can explain scale, scope, or an honest limitation.
    if interpretation_note:
        fig.text(
            0.94,
            0.025,
            interpretation_note,
            ha="right",
            va="bottom",
            fontsize=spec.source_size,
            color=COLORS["grey_500"],
        )


def reserve_chart_space(
    fig: Figure,
    *,
    left: float = 0.08,
    right: float = 0.96,
    bottom: float = 0.11,
    top: float = 0.84,
) -> None:
    """Reserve reliable space around the chart.

    Input:
        Figure plus normalised left/right/bottom/top margins.

    Output:
        Updated subplot boundaries.

    Used next:
        Titles, axis labels, legends, and footers render without overlap.
    """

    # Leave room above for the title block and below for the source footer.
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def percent_formatter(decimals: int = 0) -> FuncFormatter:
    """Create an axis formatter for fraction values.

    Input:
        Number of decimal places, for example `1` for `28.6%`.

    Output:
        A Matplotlib `FuncFormatter`.

    Used next:
        Axis ticks display `0.286` as `28.6%`.
    """

    # The source values are stored as fractions, so multiply by 100.
    return FuncFormatter(
        lambda value, _: f"{value * 100:.{decimals}f}%"
    )


def number_formatter(decimals: int = 0) -> FuncFormatter:
    """Create a compact number formatter.

    Input:
        Number of decimal places.

    Output:
        A formatter using K for thousands and M for millions.

    Used next:
        Audience sizes and row counts remain readable on chart axes.
    """

    def _format(value: float, _: int) -> str:
        """Convert one numeric tick into a compact label."""

        # Absolute value decides the display unit while preserving the sign.
        absolute = abs(value)

        if absolute >= 1_000_000:
            return f"{value / 1_000_000:.{decimals}f}M"

        if absolute >= 1_000:
            return f"{value / 1_000:.{decimals}f}K"

        return f"{value:.{decimals}f}"

    return FuncFormatter(_format)


def safe_filename(text: str) -> str:
    """Convert a visual name into a safe artifact filename.

    Input:
        A human-readable visual name.

    Output:
        A lowercase filesystem-safe filename stem.

    Used next:
        The stem becomes the PNG, CSV, and interpretation artifact name.
    """

    # Replace unsupported characters with underscores.
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())

    # Collapse repeated underscores and remove unsafe edge punctuation.
    clean = re.sub(r"_+", "_", clean).strip("._")

    # Refuse an empty filename because it could create unsafe paths.
    if not clean:
        raise ValueError("A visual filename cannot be empty.")

    return clean.lower()


# ---------------------------------------------------------------------------
# Visual QA
# ---------------------------------------------------------------------------


def _artist_is_inside_figure(
    artist,
    fig_bbox,
    renderer,
    tolerance_px: float = 4.0,
) -> bool:
    """Check whether one visible artist remains inside the canvas.

    Input:
        A text or legend artist, the figure boundary, and renderer.

    Output:
        `True` when the artist is inside the figure.

    Used next:
        `validate_figure_layout()` checks every visible text and legend.
    """

    # Invisible artists do not affect the exported visual.
    if not artist.get_visible():
        return True

    try:
        # Convert the artist into pixel coordinates for boundary comparison.
        bbox = artist.get_window_extent(renderer=renderer)
    except Exception:
        # Some custom artists cannot expose a bounding box.
        # They are not failed automatically because the caller may be valid.
        return True

    # Allow a very small tolerance for antialiasing and renderer rounding.
    return (
        bbox.x0 >= fig_bbox.x0 - tolerance_px
        and bbox.y0 >= fig_bbox.y0 - tolerance_px
        and bbox.x1 <= fig_bbox.x1 + tolerance_px
        and bbox.y1 <= fig_bbox.y1 + tolerance_px
    )


def validate_figure_layout(
    fig: Figure,
    *,
    required_text: Iterable[str] = (),
) -> VisualQAResult:
    """Validate visible chart layout before export.

    Input:
        A completed figure and text that must appear in it.

    Output:
        `VisualQAResult` with pass/fail checks and readable issues.

    Used next:
        `save_figure_with_qa()` blocks export when mandatory checks fail.
    """

    # Render once so Matplotlib calculates exact text and legend boundaries.
    fig.canvas.draw()

    # Renderer and figure boundary are required for pixel-level checks.
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    # Collect all checks and issues in a structured form.
    issues: list[str] = []
    checks: dict[str, bool] = {}

    # Confirm the in-memory canvas has usable dimensions.
    width_px, height_px = fig.canvas.get_width_height()
    checks["positive_canvas_size"] = width_px > 0 and height_px > 0

    if not checks["positive_canvas_size"]:
        issues.append("Figure canvas has invalid dimensions.")

    # Keep every visible, non-empty text artist for required-text checks.
    # This includes titles, labels, annotations, legends, and tick labels.
    visible_text = [
        text
        for text in fig.findobj(mpl.text.Text)
        if text.get_visible() and text.get_text().strip()
    ]

    # Convert visible text into a set for exact required-text checks.
    visible_strings = {
        text.get_text().strip()
        for text in visible_text
    }

    # Boundary checks should focus on business-critical text.
    # Normal tick labels are excluded here because `bbox_inches="tight"`
    # expands the exported image to preserve them. Treating every tick as
    # clipped creates false failures even when the final PNG is correct.
    boundary_text = list(fig.texts)

    for ax in fig.axes:
        # Axis title and labels must stay readable inside the export.
        boundary_text.extend(
            [
                ax.title,
                ax.xaxis.label,
                ax.yaxis.label,
            ]
        )

        # `ax.texts` contains annotations and callouts added by chart code.
        boundary_text.extend(ax.texts)

    boundary_text = [
        text
        for text in boundary_text
        if text.get_visible() and text.get_text().strip()
    ]

    # Identify any required title/subtitle that was not added.
    missing_required = [
        text
        for text in required_text
        if text.strip() and text.strip() not in visible_strings
    ]
    checks["required_text_present"] = not missing_required

    if missing_required:
        issues.append(
            "Missing required text: " + ", ".join(missing_required)
        )

    # Check only business-critical text against the canvas boundary.
    # Tick labels remain protected by the tight export bounding box and are
    # validated indirectly through the final rendered image dimensions.
    clipped_text = [
        text.get_text().strip()
        for text in boundary_text
        if not _artist_is_inside_figure(
            text,
            fig_bbox,
            renderer,
        )
    ]
    checks["text_inside_canvas"] = not clipped_text

    if clipped_text:
        issues.append(
            "Text extends outside canvas: "
            + ", ".join(clipped_text[:5])
        )

    # Check legends created at both figure level and axes level.
    clipped_legends: list[str] = []

    for legend in fig.legends:
        if not _artist_is_inside_figure(
            legend,
            fig_bbox,
            renderer,
        ):
            clipped_legends.append("figure legend")

    for ax in fig.axes:
        legend = ax.get_legend()

        if (
            legend is not None
            and not _artist_is_inside_figure(
                legend,
                fig_bbox,
                renderer,
            )
        ):
            clipped_legends.append("axes legend")

    checks["legends_inside_canvas"] = not clipped_legends

    if clipped_legends:
        issues.append("Legend extends outside the canvas.")

    # Confirm every plotting area stays inside normalised figure bounds.
    axes_inside = True

    for ax in fig.axes:
        bounds = ax.get_position()

        if (
            bounds.x0 < 0
            or bounds.y0 < 0
            or bounds.x1 > 1
            or bounds.y1 > 1
        ):
            axes_inside = False
            break

    checks["axes_inside_figure"] = axes_inside

    if not axes_inside:
        issues.append("At least one axes extends outside the figure.")

    # Return one result that later code can inspect or raise.
    return VisualQAResult(
        passed=all(checks.values()),
        checks=checks,
        issues=issues,
        width_px=width_px,
        height_px=height_px,
    )


def save_figure_with_qa(
    fig: Figure,
    output_path: str | Path,
    *,
    spec: VisualSpec = DEFAULT_SPEC,
    required_text: Iterable[str] = (),
    close: bool = True,
) -> VisualQAResult:
    """Validate, save, and re-check one visual artifact.

    Input:
        Completed figure, destination path, specification, and required text.

    Output:
        `VisualQAResult` containing final image dimensions and checks.

    Used next:
        The validated PNG can be logged to MLflow or reused elsewhere.
    """

    # Convert the supplied path and create its parent output directory.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Run layout checks before writing a broken image to disk.
    layout_result = validate_figure_layout(
        fig,
        required_text=required_text,
    )
    layout_result.raise_for_failure()

    # Save the image using the shared DPI and background.
    fig.savefig(
        output,
        dpi=spec.dpi,
        facecolor=spec.facecolor,
    )

    # Close the figure by default to prevent memory leaks in batch jobs.
    if close:
        plt.close(fig)

    # Start the final result with all pre-export checks.
    issues = list(layout_result.issues)
    checks = dict(layout_result.checks)

    # Confirm the file exists and is large enough to be a real image.
    checks["file_created"] = output.exists()
    checks["file_not_empty"] = (
        output.exists() and output.stat().st_size > 10_000
    )

    width_px: int | None = None
    height_px: int | None = None

    if output.exists():
        # Read the exported image so QA uses actual output dimensions.
        image = mpimg.imread(output)
        height_px, width_px = image.shape[:2]

        # Enforce minimum width and height for MLflow/report readability.
        checks["minimum_export_width"] = (
            width_px >= spec.min_width_px
        )
        checks["minimum_export_height"] = (
            height_px >= spec.min_height_px
        )
    else:
        # Missing output automatically fails both resolution checks.
        checks["minimum_export_width"] = False
        checks["minimum_export_height"] = False

    # Convert failed checks into clear user-facing issue messages.
    if not checks["file_created"]:
        issues.append("Export file was not created.")

    if not checks["file_not_empty"]:
        issues.append("Export file is unexpectedly small.")

    if not checks["minimum_export_width"]:
        issues.append(
            f"Export width is below {spec.min_width_px}px."
        )

    if not checks["minimum_export_height"]:
        issues.append(
            f"Export height is below {spec.min_height_px}px."
        )

    # Build one final result used by tests and chart-generation scripts.
    result = VisualQAResult(
        passed=all(checks.values()),
        checks=checks,
        issues=issues,
        output_path=str(output),
        width_px=width_px,
        height_px=height_px,
    )

    # Fail immediately so broken visuals cannot be logged to MLflow.
    result.raise_for_failure()

    return result
