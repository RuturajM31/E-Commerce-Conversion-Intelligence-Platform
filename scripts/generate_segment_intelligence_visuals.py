"""Generate MLV-G02 segment intelligence artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.segment_intelligence_visuals import (
    generate_segment_intelligence_visual_package,
)


def main() -> None:
    """Generate G02 and print concise QA output."""

    manifest = (
        generate_segment_intelligence_visual_package()
    )

    print(
        "Segment intelligence visual package created."
    )

    qa_result = manifest["qa"]["MLV-G02"]
    print(
        "MLV-G02: "
        f"passed={qa_result['passed']} | "
        f"{qa_result['width_px']}x"
        f"{qa_result['height_px']} px"
    )
    print(
        "Conditional: MLV-G01, MLV-G03, MLV-G04"
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
