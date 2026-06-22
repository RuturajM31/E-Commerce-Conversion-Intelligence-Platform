"""Generate MLV-H01, MLV-H02, and MLV-H04 feature-health artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.feature_health_visuals import (
    generate_feature_health_visual_package,
)


def main() -> None:
    """Generate the package and print concise automated-QA results."""

    manifest = generate_feature_health_visual_package()

    print("Feature-health visual package created.")

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
