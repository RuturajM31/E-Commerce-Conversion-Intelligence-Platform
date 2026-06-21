"""Generate the MLV-A01 to MLV-A04 champion-selection package.

Why this script exists:
    The visual functions live in a reusable source module. This small runner
    gives the user one clear command to generate all champion artifacts.

Input:
    Existing project comparison tables and champion metadata.

Output:
    Four PNG files, two CSV evidence files, one insight Markdown file, and
    one JSON manifest under reports/visuals/ml_visual_intelligence/.

Used next:
    Review the rendered images, then log the approved folder to MLflow.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add the project root to Python's import path.
# This lets the script import `src/...` even when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.champion_selection_visuals import (
    generate_champion_visual_package,
)


def main() -> None:
    """Generate the package and print only a concise QA summary."""

    # Generate all artifacts from the current project root.
    manifest = generate_champion_visual_package()

    print("Champion visual package created.")

    # Print one compact line per visual instead of a large terminal dump.
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
