"""Generate MLV-J04 and MLV-J08 production-monitoring artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.monitoring_visuals import (
    generate_monitoring_visual_package,
)


def main() -> None:
    """Generate supported monitoring visuals and concise QA output."""

    manifest = generate_monitoring_visual_package()

    print("Monitoring visual package created.")

    for visual_id, qa_result in manifest["qa"].items():
        print(
            f"{visual_id}: "
            f"passed={qa_result['passed']} | "
            f"{qa_result['width_px']}x"
            f"{qa_result['height_px']} px"
        )

    print("Conditional: MLV-J01, MLV-J02, MLV-J05")
    print("Blocked: MLV-J03, MLV-J06, MLV-J07")
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
