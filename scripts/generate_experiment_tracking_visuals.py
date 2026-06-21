"""Generate the consolidated Experiment Tracking Readiness page."""

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
    """Generate one readiness page and print concise QA output."""

    manifest = generate_experiment_tracking_visual_package()
    qa = manifest["qa"]["experiment_tracking_readiness"]

    print("Experiment Tracking Readiness package created.")
    print(
        "Readiness page: "
        f"passed={qa['passed']} | "
        f"{qa['width_px']}x{qa['height_px']} px"
    )
    print("Supported experiment visuals: 0")
    print(
        "Conditional: MLV-I01, MLV-I02, MLV-I03, MLV-I04, MLV-I05"
    )
    print(
        f"Readiness: {manifest['readiness_artifact']}"
    )
    print(
        f"Manifest: {manifest['supporting_files']['manifest']}"
    )


if __name__ == "__main__":
    main()
