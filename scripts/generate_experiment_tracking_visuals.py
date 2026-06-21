"""Generate MLflow experiment-tracking visual intelligence artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.experiment_tracking_visuals import (
    generate_experiment_tracking_visual_package,
)


def main() -> None:
    """Generate I01, I02, I04, and I05 and print concise QA output."""

    manifest = generate_experiment_tracking_visual_package()

    print("Experiment-tracking visual package created.")

    for visual_id, qa_result in manifest["qa"].items():
        print(
            f"{visual_id}: "
            f"passed={qa_result['passed']} | "
            f"{qa_result['width_px']}x"
            f"{qa_result['height_px']} px"
        )

    print(
        "MLV-I03: "
        + (
            "ready"
            if manifest["i03_ready"]
            else "conditional"
        )
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
