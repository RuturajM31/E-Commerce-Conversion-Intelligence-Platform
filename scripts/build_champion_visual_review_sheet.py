"""Build one review sheet from the four champion-selection visuals.

Why this file exists:
    Automated checks can confirm file size and canvas boundaries, but they
    cannot fully judge label overlap, spacing, balance, or executive-level
    readability. This script combines the four real PNGs into one image for
    a fast visual-hygiene review.

Inputs:
    Four generated PNG files under:
    reports/visuals/ml_visual_intelligence/champion_selection/

Output:
    reports/qa/champion_visual_review_sheet.png

Used next:
    Review the single sheet before committing or logging artifacts to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Repository-relative folder containing the real generated visuals.
VISUAL_DIR = Path(
    "reports/visuals/ml_visual_intelligence/champion_selection"
)

# Output stays under reports/qa because it is review evidence.
OUTPUT_PATH = Path(
    "reports/qa/champion_visual_review_sheet.png"
)

# Fixed visual order keeps the review aligned with the coverage matrix.
VISUALS = [
    (
        "MLV-A01",
        VISUAL_DIR / "mlv-a01_model_performance_frontier.png",
    ),
    (
        "MLV-A02",
        VISUAL_DIR / "mlv-a02_multi-metric_model_ranking.png",
    ),
    (
        "MLV-A03",
        VISUAL_DIR
        / "mlv-a03_validation_to_holdout_generalisation_slopegraph.png",
    ),
    (
        "MLV-A04",
        VISUAL_DIR / "mlv-a04_champion_scorecard_identity.png",
    ),
]


def main() -> None:
    """Create a two-by-two review sheet from the four PNG artifacts."""

    # Stop early when any required visual is missing.
    missing_files = [
        str(path)
        for _, path in VISUALS
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing generated visual files: "
            + ", ".join(missing_files)
        )

    # Create one large canvas so each visual remains readable.
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(22, 14),
        facecolor="white",
    )

    # Flatten the axes so the loop maps directly to the visual order.
    flat_axes = axes.ravel()

    for axis, (visual_id, image_path) in zip(
        flat_axes,
        VISUALS,
    ):
        # Load the actual exported PNG rather than re-rendering source data.
        image = mpimg.imread(image_path)

        # Place the complete image without stretching its aspect ratio.
        axis.imshow(image)
        axis.set_title(
            visual_id,
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        axis.axis("off")

    # Keep consistent spacing between all four panels.
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.95,
        wspace=0.04,
        hspace=0.10,
    )

    # Add one clear review heading above the four visuals.
    figure.suptitle(
        "Champion Selection Visual Hygiene Review",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )

    # Create the QA folder and save a high-resolution combined image.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    print(f"Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
