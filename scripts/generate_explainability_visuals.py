"""Generate MLV-E01 to MLV-E06 explainability artifacts.

Why this script exists:
    The reusable calculation and rendering logic lives under `src/`.
    This runner provides one clear project command.

Output:
    Six PNGs and supporting evidence under:
    reports/visuals/ml_visual_intelligence/explainability/
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.explainability_visuals import (
    generate_explainability_visual_package,
)


def main() -> None:
    """Generate the package and print concise automated-QA results."""

    manifest = generate_explainability_visual_package()

    print("Explainability visual package created.")

    for visual_id, qa_result in manifest["qa"].items():
        print(
            f"{visual_id}: "
            f"passed={qa_result['passed']} | "
            f"{qa_result['width_px']}x"
            f"{qa_result['height_px']} px"
        )

    print(
        "Insights: "
        f"{manifest['supporting_files']['insights']}"
    )
    print(
        "Manifest: "
        f"{manifest['supporting_files']['manifest']}"
    )


if __name__ == "__main__":
    main()
