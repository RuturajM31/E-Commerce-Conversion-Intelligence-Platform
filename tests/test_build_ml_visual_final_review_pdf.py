"""Tests for the final review with one readiness page."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from scripts.build_ml_visual_final_review_pdf import (
    MANIFEST_NAME,
    READINESS_RELATIVE_PATH,
    SUPPORTED_VISUALS,
    build_review_pdf,
    validate_visual_coverage,
)


def create_dummy(
    path: Path,
    label: str,
) -> None:
    """Create a deterministic high-resolution review image."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image = Image.new(
        "RGB",
        (1800, 1000),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (40, 40, 1760, 960),
        outline="navy",
        width=8,
    )
    draw.text(
        (100, 100),
        label,
        fill="navy",
    )
    image.save(path, "PNG")


def populate(root: Path) -> None:
    """Create 23 supported visuals, readiness, and experiment manifest."""

    visual_root = (
        root
        / "reports"
        / "visuals"
        / "ml_visual_intelligence"
        / "synthetic"
    )

    for visual_id, _, _ in SUPPORTED_VISUALS:
        create_dummy(
            visual_root
            / f"mlv-{visual_id.lower()}_dummy.png",
            f"MLV-{visual_id}",
        )

    create_dummy(
        root
        / "reports"
        / "visuals"
        / "ml_visual_intelligence"
        / READINESS_RELATIVE_PATH,
        "Experiment Tracking Readiness",
    )

    manifest_dir = (
        root
        / "reports"
        / "visuals"
        / "ml_visual_intelligence"
        / "experiment_tracking"
    )
    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        manifest_dir
        / "experiment_tracking_visual_manifest.json"
    ).write_text(
        json.dumps(
            {
                "supported_visuals": [],
                "conditional_visuals": {
                    "MLV-I01": "conditional",
                    "MLV-I02": "conditional",
                    "MLV-I03": "conditional",
                    "MLV-I04": "conditional",
                    "MLV-I05": "conditional",
                },
            }
        ),
        encoding="utf-8",
    )


def test_final_pdf_has_23_supported_and_one_readiness_page(
    tmp_path: Path,
) -> None:
    """The portfolio PDF should have 24 review pages and 26 total pages."""

    populate(tmp_path)
    output = tmp_path / "review"

    result = build_review_pdf(
        project_root=tmp_path,
        output_dir=output,
        quality=72,
    )
    manifest = json.loads(
        (
            output / MANIFEST_NAME
        ).read_text(encoding="utf-8")
    )

    assert result["supported_visual_count"] == 23
    assert result["review_visual_count"] == 24
    assert result["page_count"] == 26
    assert result["conditional_count"] == 24
    assert result["blocked_count"] == 3

    assert manifest["readiness_page_count"] == 1
    assert sum(
        item.get("page_type") == "readiness"
        for item in manifest["visuals"]
    ) == 1
    assert len(manifest["visuals"]) == 24


def test_missing_readiness_stops_build(
    tmp_path: Path,
) -> None:
    """A missing readiness page must stop the final PDF build."""

    populate(tmp_path)
    (
        tmp_path
        / "reports"
        / "visuals"
        / "ml_visual_intelligence"
        / READINESS_RELATIVE_PATH
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="readiness",
    ):
        build_review_pdf(
            project_root=tmp_path,
            output_dir=tmp_path / "review",
            quality=72,
        )


def test_experiment_placeholder_images_are_not_required(
    tmp_path: Path,
) -> None:
    """No I01/I02/I04/I05 chart file should be required."""

    populate(tmp_path)
    records = validate_visual_coverage(
        tmp_path
        / "reports"
        / "visuals"
        / "ml_visual_intelligence"
    )

    assert len(records) == 23
    assert not any(
        record.visual_id.startswith("I")
        for record in records
    )
