"""Generate the MLV-B01 and MLV-B04 threshold-decision package.

Why this script exists:
    The reusable visual logic lives under `src/visualization/`. This runner
    gives one direct command for generating all threshold-decision artifacts.

Input:
    Existing threshold table and final champion metadata.

Output:
    Two PNG files and supporting evidence under:
    reports/visuals/ml_visual_intelligence/threshold_decisions/
"""

from __future__ import annotations

from pathlib import Path
import sys


# Add the project root so direct script execution can import `src/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.threshold_decision_visuals import (
    generate_threshold_visual_package,
)


def main() -> None:
    """Generate the package and print a concise QA summary."""

    manifest = generate_threshold_visual_package()

    print("Threshold decision visual package created.")

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
