"""Build one visual review sheet for MLV-B01 and MLV-B04.

Why this file exists:
    Automated QA checks cannot fully judge spacing, overlap, balance, or
    readability. This script combines both real PNGs into one review image.

Inputs:
    Two generated PNG files under the threshold-decisions output folder.

Output:
    reports/qa/threshold_visual_review_sheet.png

Used next:
    The combined sheet is manually inspected before commit or MLflow logging.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


VISUAL_DIR = Path(
    "reports/visuals/ml_visual_intelligence/threshold_decisions"
)
OUTPUT_PATH = Path(
    "reports/qa/threshold_visual_review_sheet.png"
)

VISUALS = [
    (
        "MLV-B01",
        VISUAL_DIR / "mlv-b01_threshold_decision_studio.png",
    ),
    (
        "MLV-B04",
        VISUAL_DIR
        / "mlv-b04_confusion_matrix_decision_map.png",
    ),
]


def main() -> None:
    """Create a large side-by-side review sheet."""

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

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(24, 9),
        facecolor="white",
    )

    for axis, (visual_id, image_path) in zip(
        axes,
        VISUALS,
    ):
        image = mpimg.imread(image_path)
        axis.imshow(image)
        axis.set_title(
            visual_id,
            fontsize=15,
            fontweight="bold",
            pad=10,
        )
        axis.axis("off")

    figure.suptitle(
        "Threshold Decision Visual Hygiene Review",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.03,
        top=0.93,
        wspace=0.035,
    )

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
