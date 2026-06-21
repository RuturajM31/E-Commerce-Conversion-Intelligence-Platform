"""Build one final combined review PDF for all supported ML visuals.

Why this file exists:
    The project review workflow requires one upload containing every supported
    ML Visual Intelligence chart at readable size. This script discovers the
    approved chart files, validates exact coverage, preserves their order, and
    builds one multi-page PDF plus a source manifest.

Main input:
    reports/visuals/ml_visual_intelligence/**/mlv-*.png

Main outputs:
    - reports/visuals/ml_visual_intelligence/final_review/
      ML_VISUAL_INTELLIGENCE_FINAL_REVIEW.pdf
    - reports/visuals/ml_visual_intelligence/final_review/
      ML_VISUAL_INTELLIGENCE_FINAL_REVIEW_MANIFEST.json

Review design:
    - Cover page
    - One full-size page per supported visual
    - Final scope/status page

Important rule:
    Missing or duplicate visual IDs stop the build. The script never silently
    omits a required chart or chooses between conflicting files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


VISUAL_ROOT = Path(
    "reports/visuals/ml_visual_intelligence"
)
OUTPUT_DIR = VISUAL_ROOT / "final_review"
PDF_NAME = "ML_VISUAL_INTELLIGENCE_FINAL_REVIEW.pdf"
MANIFEST_NAME = "ML_VISUAL_INTELLIGENCE_FINAL_REVIEW_MANIFEST.json"

# The order follows the controlling ML Visual Intelligence coverage matrix.
SUPPORTED_VISUALS = [
    ("A01", "Champion selection", "Model Performance Frontier"),
    ("A02", "Champion selection", "Multi-Metric Model Ranking"),
    ("A03", "Champion selection", "Validation-to-Holdout Generalisation"),
    ("A04", "Champion selection", "Champion Scorecard"),
    ("B01", "Threshold intelligence", "Threshold Decision Studio"),
    ("B04", "Threshold intelligence", "Confusion-Matrix Decision Map"),
    ("E01", "Explainability", "Global SHAP Beeswarm"),
    ("E02", "Explainability", "Global Feature-Impact Ranking"),
    ("E03", "Explainability", "SHAP Dependence Plot"),
    ("E04", "Explainability", "Visitor-Level Waterfall"),
    ("E05", "Explainability", "Representative Visitor Decision Paths"),
    ("E06", "Explainability", "Feature Interaction Heatmap"),
    ("F01", "Robustness and stability", "Model Robustness Heatmap"),
    ("F02", "Robustness and stability", "Metric Variability Distribution"),
    ("F03", "Robustness and stability", "Champion Stability Profile"),
    ("F04", "Robustness and stability", "Sensitivity Tornado"),
    ("F05", "Robustness and stability", "Performance Degradation Waterfall"),
    ("G02", "Segment intelligence", "Score Distribution by Segment"),
    ("H01", "Data and feature health", "Feature Distribution Profile"),
    ("H02", "Data and feature health", "Feature Correlation Cluster Map"),
    ("H04", "Data and feature health", "Missingness and Validity Map"),
    ("J04", "Production monitoring", "Delayed-Label Maturity Funnel"),
    ("J08", "Production monitoring", "Monitoring Freshness and Data-Coverage Card"),
]

READINESS_RELATIVE_PATH = Path(
    "experiment_tracking/experiment_tracking_readiness.png"
)

CONDITIONAL_VISUALS = [
    "B02", "B03",
    "C01", "C02", "C03", "C04", "C05", "C06",
    "D01", "D02", "D03",
    "G01", "G03", "G04",
    "H03", "H05",
    "I03",
    "J01", "J02", "J05",
]

BLOCKED_VISUALS = ["J03", "J06", "J07"]

PAGE_WIDTH = 2400
PAGE_HEIGHT = 1600
PAGE_BACKGROUND = (248, 250, 252)
WHITE = (255, 255, 255)
NAVY = (22, 49, 82)
BLUE = (46, 111, 176)
TEAL = (42, 157, 143)
AMBER = (225, 164, 56)
RED = (199, 78, 78)
GREY_900 = (36, 44, 52)
GREY_700 = (79, 91, 105)
GREY_500 = (126, 137, 148)
GREY_300 = (205, 213, 221)
GREY_200 = (226, 231, 236)

VISUAL_ID_PATTERN = re.compile(
    r"^mlv-([a-j]\d{2})(?:[_-]|$)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class VisualRecord:
    """One validated source visual and its review metadata."""

    visual_id: str
    category: str
    title: str
    path: Path
    width_px: int
    height_px: int
    sha256: str


def _font_candidates(bold: bool) -> list[Path]:
    """Return common cross-platform font candidates."""

    if bold:
        names = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/local/share/fontss/DejaVuSans-Bold.ttf",
        ]
    else:
        names = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/local/share/fontss/DejaVuSans.ttf",
        ]

    return [Path(name) for name in names]


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable font and fall back safely if none is available."""

    for candidate in _font_candidates(bold):
        if candidate.exists():
            return ImageFont.truetype(
                str(candidate),
                size=size,
            )

    return ImageFont.load_default()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one source image."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def discover_visual_files(visual_root: Path) -> dict[str, list[Path]]:
    """Discover PNG files grouped by parsed MLV visual ID."""

    discovered: dict[str, list[Path]] = {}

    for path in sorted(visual_root.rglob("*.png")):
        # Ignore final-review products if the script is rerun.
        if OUTPUT_DIR.name in path.parts:
            continue

        match = VISUAL_ID_PATTERN.match(path.stem)

        if not match:
            continue

        visual_id = match.group(1).upper()
        discovered.setdefault(visual_id, []).append(path)

    return discovered


def validate_visual_coverage(
    visual_root: Path,
) -> list[VisualRecord]:
    """Validate that every supported visual exists exactly once."""

    if not visual_root.exists():
        raise FileNotFoundError(
            f"ML Visual Intelligence root not found: {visual_root}"
        )

    discovered = discover_visual_files(visual_root)
    expected_ids = [item[0] for item in SUPPORTED_VISUALS]

    missing = [
        visual_id
        for visual_id in expected_ids
        if visual_id not in discovered
    ]
    duplicates = {
        visual_id: paths
        for visual_id, paths in discovered.items()
        if visual_id in expected_ids and len(paths) > 1
    }

    if missing:
        raise ValueError(
            "Missing supported visual IDs: " + ", ".join(missing)
        )

    if duplicates:
        details = "; ".join(
            f"{visual_id}: "
            + ", ".join(str(path) for path in paths)
            for visual_id, paths in sorted(duplicates.items())
        )
        raise ValueError(
            "Duplicate supported visual IDs detected: " + details
        )

    records: list[VisualRecord] = []

    for visual_id, category, title in SUPPORTED_VISUALS:
        path = discovered[visual_id][0]

        with Image.open(path) as image:
            width_px, height_px = image.size

        if width_px < 1600 or height_px < 850:
            raise ValueError(
                f"Visual {visual_id} is below the review resolution floor: "
                f"{width_px}x{height_px}px"
            )

        records.append(
            VisualRecord(
                visual_id=visual_id,
                category=category,
                title=title,
                path=path,
                width_px=width_px,
                height_px=height_px,
                sha256=sha256_file(path),
            )
        )

    return records


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 12,
) -> int:
    """Draw wrapped text and return the final y position."""

    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    x, y = xy

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing

    return y



def resolve_review_status(
    project_root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve supported/conditional/blocked IDs from current manifests."""

    expected_ids = [item[0] for item in SUPPORTED_VISUALS]
    conditional = set(CONDITIONAL_VISUALS)
    blocked = set(BLOCKED_VISUALS)
    experiment_manifest = (
        project_root
        / VISUAL_ROOT
        / "experiment_tracking"
        / "experiment_tracking_visual_manifest.json"
    )

    if experiment_manifest.exists():
        try:
            content = json.loads(
                experiment_manifest.read_text(encoding="utf-8")
            )
            dynamic = content.get("conditional_visuals", {})
            if isinstance(dynamic, dict):
                conditional.update(
                    visual_id.removeprefix("MLV-")
                    for visual_id in dynamic
                )
            elif isinstance(dynamic, list):
                conditional.update(
                    str(visual_id).removeprefix("MLV-")
                    for visual_id in dynamic
                )
        except Exception:
            pass

    supported = [
        visual_id for visual_id in expected_ids
        if visual_id not in conditional and visual_id not in blocked
    ]

    return supported, sorted(conditional), sorted(blocked)


def draw_cover_page(
    records: list[VisualRecord],
    *,
    generated_at: str,
    supported_ids: list[str],
    conditional_ids: list[str],
    blocked_ids: list[str],
) -> Image.Image:
    """Create the final-review cover page."""

    page = Image.new(
        "RGB",
        (PAGE_WIDTH, PAGE_HEIGHT),
        PAGE_BACKGROUND,
    )
    draw = ImageDraw.Draw(page)

    title_font = load_font(78, bold=True)
    subtitle_font = load_font(32)
    section_font = load_font(28, bold=True)
    body_font = load_font(25)
    small_font = load_font(20)

    draw.rounded_rectangle(
        (80, 80, PAGE_WIDTH - 80, PAGE_HEIGHT - 80),
        radius=36,
        fill=WHITE,
        outline=GREY_200,
        width=3,
    )
    draw.rectangle(
        (80, 80, 105, PAGE_HEIGHT - 80),
        fill=NAVY,
    )

    draw.text(
        (150, 145),
        "ML VISUAL INTELLIGENCE",
        font=small_font,
        fill=BLUE,
    )
    draw.text(
        (150, 205),
        "Final Combined Review",
        font=title_font,
        fill=NAVY,
    )
    draw.text(
        (150, 315),
        "E-Commerce Conversion Intelligence Platform",
        font=subtitle_font,
        fill=GREY_700,
    )

    draw.rounded_rectangle(
        (150, 425, 760, 610),
        radius=24,
        fill=(238, 245, 252),
        outline=(198, 218, 238),
        width=2,
    )
    draw.text(
        (190, 460),
        str(len(records)),
        font=title_font,
        fill=BLUE,
    )
    draw.text(
        (380, 490),
        "supported visuals",
        font=section_font,
        fill=NAVY,
    )
    draw.text(
        (380, 540),
        "plus one readiness evidence page",
        font=body_font,
        fill=GREY_700,
    )

    draw.rounded_rectangle(
        (815, 425, 1425, 610),
        radius=24,
        fill=(236, 248, 246),
        outline=(190, 224, 218),
        width=2,
    )
    draw.text(
        (855, 460),
        str(len(conditional_ids)),
        font=title_font,
        fill=TEAL,
    )
    draw.text(
        (1045, 490),
        "conditional visuals",
        font=section_font,
        fill=NAVY,
    )
    draw.text(
        (1045, 540),
        "tracked with explicit blockers",
        font=body_font,
        fill=GREY_700,
    )

    draw.rounded_rectangle(
        (1480, 425, 2090, 610),
        radius=24,
        fill=(252, 242, 240),
        outline=(234, 202, 196),
        width=2,
    )
    draw.text(
        (1520, 460),
        str(len(blocked_ids)),
        font=title_font,
        fill=RED,
    )
    draw.text(
        (1710, 490),
        "blocked visuals",
        font=section_font,
        fill=NAVY,
    )
    draw.text(
        (1710, 540),
        "future-label data boundary",
        font=body_font,
        fill=GREY_700,
    )

    draw.text(
        (150, 715),
        "Review scope",
        font=section_font,
        fill=NAVY,
    )

    group_lines: list[str] = []
    current_group = None
    current_ids: list[str] = []

    for visual_id, category, _ in SUPPORTED_VISUALS:
        if category != current_group:
            if current_group is not None:
                group_lines.append(
                    f"{current_group}: {', '.join(current_ids)}"
                )
            current_group = category
            current_ids = [visual_id]
        else:
            current_ids.append(visual_id)

    if current_group is not None:
        group_lines.append(
            f"{current_group}: {', '.join(current_ids)}"
        )

    group_lines.insert(
        -1,
        "Experiment tracking: one readiness page; I01, I02, I03, I04, I05 conditional",
    )

    y = 770

    for line in group_lines:
        y = draw_wrapped_text(
            draw,
            line,
            xy=(150, y),
            font=body_font,
            fill=GREY_700,
            max_width=1940,
            line_spacing=8,
        ) + 10

    draw.text(
        (150, 1365),
        "Manual status: pending one consolidated visual review",
        font=section_font,
        fill=AMBER,
    )
    draw.text(
        (150, 1425),
        f"Generated: {generated_at}",
        font=small_font,
        fill=GREY_500,
    )

    return page


def fit_image(
    source: Image.Image,
    *,
    max_width: int,
    max_height: int,
) -> Image.Image:
    """Resize an image proportionally without stretching or cropping."""

    image = source.convert("RGB")
    scale = min(
        max_width / image.width,
        max_height / image.height,
        1.0,
    )

    if scale >= 1.0:
        return image.copy()

    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    return image.resize(size, resampling)


def draw_visual_page(
    record: VisualRecord,
    *,
    page_number: int,
    total_pages: int,
    project_root: Path,
) -> Image.Image:
    """Create one full-size review page for a chart."""

    page = Image.new(
        "RGB",
        (PAGE_WIDTH, PAGE_HEIGHT),
        PAGE_BACKGROUND,
    )
    draw = ImageDraw.Draw(page)
    header_font = load_font(24, bold=True)
    small_font = load_font(18)

    draw.rounded_rectangle(
        (30, 30, PAGE_WIDTH - 30, PAGE_HEIGHT - 30),
        radius=24,
        fill=WHITE,
        outline=GREY_200,
        width=2,
    )

    draw.text(
        (70, 55),
        f"MLV-{record.visual_id} | {record.category}",
        font=header_font,
        fill=NAVY,
    )
    draw.text(
        (PAGE_WIDTH - 70, 58),
        f"Page {page_number} of {total_pages}",
        font=small_font,
        fill=GREY_500,
        anchor="ra",
    )

    image_left = 70
    image_top = 105
    image_right = PAGE_WIDTH - 70
    image_bottom = PAGE_HEIGHT - 95
    max_width = image_right - image_left
    max_height = image_bottom - image_top

    with Image.open(record.path) as source:
        fitted = fit_image(
            source,
            max_width=max_width,
            max_height=max_height,
        )

    x = image_left + (max_width - fitted.width) // 2
    y = image_top + (max_height - fitted.height) // 2
    page.paste(fitted, (x, y))

    draw.rectangle(
        (x - 1, y - 1, x + fitted.width, y + fitted.height),
        outline=GREY_300,
        width=1,
    )

    try:
        relative_path = record.path.relative_to(project_root)
    except ValueError:
        relative_path = record.path

    footer_text = (
        f"{record.title} | Source: {relative_path} | "
        f"{record.width_px}x{record.height_px}px"
    )
    draw.text(
        (70, PAGE_HEIGHT - 66),
        footer_text,
        font=small_font,
        fill=GREY_500,
    )

    return page



def draw_status_page(
    records: list[VisualRecord],
    *,
    total_pages: int,
    supported_ids: list[str],
    conditional_ids: list[str],
    blocked_ids: list[str],
) -> Image.Image:
    """Create the final dynamic scope and honesty-status page."""

    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), PAGE_BACKGROUND)
    draw = ImageDraw.Draw(page)
    title_font = load_font(58, bold=True)
    section_font = load_font(30, bold=True)
    body_font = load_font(23)
    small_font = load_font(18)
    draw.rounded_rectangle(
        (70, 70, PAGE_WIDTH - 70, PAGE_HEIGHT - 70),
        radius=30, fill=WHITE, outline=GREY_200, width=2,
    )
    draw.text((135, 125), "Coverage and Review Status", font=title_font, fill=NAVY)
    draw.text(
        (PAGE_WIDTH - 135, 145),
        f"Page {total_pages} of {total_pages}",
        font=small_font, fill=GREY_500, anchor="ra",
    )
    sections = [
        ("SUPPORTED AND INCLUDED", ", ".join(supported_ids), BLUE),
        (
            "CONDITIONAL - SOURCE EVIDENCE NOT YET SUFFICIENT",
            ", ".join(conditional_ids), AMBER,
        ),
        (
            "BLOCKED - FUTURE LABEL / PRODUCTION OUTCOME BOUNDARY",
            ", ".join(blocked_ids), RED,
        ),
    ]
    y = 260
    for heading, values, colour in sections:
        draw.rounded_rectangle(
            (135, y, PAGE_WIDTH - 135, y + 230),
            radius=22, fill=PAGE_BACKGROUND, outline=GREY_200, width=2,
        )
        draw.rectangle((135, y, 150, y + 230), fill=colour)
        draw.text((185, y + 34), heading, font=section_font, fill=colour)
        draw_wrapped_text(
            draw, values, xy=(185, y + 96), font=body_font,
            fill=GREY_700, max_width=2020, line_spacing=10,
        )
        y += 270
    draw.text((135, 1115), "Completion rule", font=section_font, fill=NAVY)
    draw_wrapped_text(
        draw,
        (
            "A visual page may be supported or an explicit conditional status card. "
            "Final sign-off requires readable layout, truthful evidence scope, actual-number "
            "interpretation where available, and one consolidated manual review."
        ),
        xy=(135, 1175), font=body_font, fill=GREY_700,
        max_width=2050, line_spacing=10,
    )
    draw.text(
        (135, 1410),
        "No unsupported result, demo metric, alert, drift event, or production outcome is fabricated.",
        font=section_font, fill=TEAL,
    )
    return page



def validate_readiness_artifact(
    visual_root: Path,
) -> VisualRecord:
    """Validate the single experiment readiness image."""

    path = visual_root / READINESS_RELATIVE_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment readiness artifact not found: {path}"
        )

    with Image.open(path) as image:
        width_px, height_px = image.size

    if width_px < 1600 or height_px < 850:
        raise ValueError(
            "Experiment readiness artifact is below the review resolution floor: "
            f"{width_px}x{height_px}px"
        )

    return VisualRecord(
        visual_id="READINESS",
        category="Experiment tracking",
        title="Experiment Tracking Readiness & Evidence Gate",
        path=path,
        width_px=width_px,
        height_px=height_px,
        sha256=sha256_file(path),
    )


def draw_readiness_page(
    record: VisualRecord,
    *,
    page_number: int,
    total_pages: int,
    project_root: Path,
) -> Image.Image:
    """Create the single experiment readiness review page."""

    page = draw_visual_page(
        record,
        page_number=page_number,
        total_pages=total_pages,
        project_root=project_root,
    )
    draw = ImageDraw.Draw(page)
    header_font = load_font(24, bold=True)
    draw.rectangle((55, 42, 1200, 88), fill=WHITE)
    draw.text(
        (70, 55),
        "Experiment tracking | Readiness & Evidence Gate",
        font=header_font,
        fill=NAVY,
    )
    return page

def save_pages_as_pdf(
    pages: list[Image.Image],
    output_path: Path,
    *,
    quality: int = 94,
) -> None:
    """Save RGB page images as one multi-page PDF."""

    if not pages:
        raise ValueError("At least one PDF page is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *remaining = [page.convert("RGB") for page in pages]
    first.save(
        output_path,
        "PDF",
        save_all=True,
        append_images=remaining,
        resolution=150.0,
        quality=quality,
        subsampling=0,
        title="ML Visual Intelligence Final Combined Review",
        author="E-Commerce Conversion Intelligence Platform",
        subject="Consolidated visual QA review",
    )


def build_review_pdf(
    project_root: str | Path = ".",
    *,
    output_dir: str | Path | None = None,
    quality: int = 94,
) -> dict[str, Any]:
    """Validate sources and build the final combined review PDF."""

    root = Path(project_root).resolve()
    visual_root = root / VISUAL_ROOT
    final_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / OUTPUT_DIR
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)

    records = validate_visual_coverage(visual_root)
    readiness_record = validate_readiness_artifact(visual_root)
    supported_ids, conditional_ids, blocked_ids = resolve_review_status(root)
    generated_at = datetime.now(timezone.utc).isoformat()
    total_pages = len(records) + 3

    pages: list[Image.Image] = [
        draw_cover_page(
            records,
            generated_at=generated_at,
            supported_ids=supported_ids,
            conditional_ids=conditional_ids,
            blocked_ids=blocked_ids,
        )
    ]

    page_number = 2
    readiness_page_number = None

    for record in records:
        if record.visual_id == "J04" and readiness_page_number is None:
            readiness_page_number = page_number
            pages.append(
                draw_readiness_page(
                    readiness_record,
                    page_number=page_number,
                    total_pages=total_pages,
                    project_root=root,
                )
            )
            page_number += 1

        pages.append(
            draw_visual_page(
                record,
                page_number=page_number,
                total_pages=total_pages,
                project_root=root,
            )
        )
        page_number += 1

    if readiness_page_number is None:
        readiness_page_number = page_number
        pages.append(
            draw_readiness_page(
                readiness_record,
                page_number=page_number,
                total_pages=total_pages,
                project_root=root,
            )
        )
        page_number += 1

    pages.append(
        draw_status_page(
            records,
            total_pages=total_pages,
            supported_ids=supported_ids,
            conditional_ids=conditional_ids,
            blocked_ids=blocked_ids,
        )
    )

    pdf_path = final_output_dir / PDF_NAME
    manifest_path = final_output_dir / MANIFEST_NAME
    save_pages_as_pdf(
        pages,
        pdf_path,
        quality=quality,
    )

    visual_manifest = []
    current_page = 2
    readiness_inserted = False

    for record in records:
        if record.visual_id == "J04" and not readiness_inserted:
            visual_manifest.append(
                {
                    "page": current_page,
                    "page_type": "readiness",
                    "visual_id": None,
                    "category": readiness_record.category,
                    "title": readiness_record.title,
                    "source_path": str(readiness_record.path.relative_to(root)),
                    "width_px": readiness_record.width_px,
                    "height_px": readiness_record.height_px,
                    "sha256": readiness_record.sha256,
                }
            )
            current_page += 1
            readiness_inserted = True

        visual_manifest.append(
            {
                "page": current_page,
                "page_type": "supported_visual",
                "visual_id": record.visual_id,
                "category": record.category,
                "title": record.title,
                "source_path": str(record.path.relative_to(root)),
                "width_px": record.width_px,
                "height_px": record.height_px,
                "sha256": record.sha256,
            }
        )
        current_page += 1

    if not readiness_inserted:
        visual_manifest.append(
            {
                "page": readiness_page_number,
                "page_type": "readiness",
                "visual_id": None,
                "category": readiness_record.category,
                "title": readiness_record.title,
                "source_path": str(readiness_record.path.relative_to(root)),
                "width_px": readiness_record.width_px,
                "height_px": readiness_record.height_px,
                "sha256": readiness_record.sha256,
            }
        )
        visual_manifest.sort(key=lambda item: item["page"])

    manifest: dict[str, Any] = {
        "document": "ML Visual Intelligence Final Combined Review",
        "generated_at_utc": generated_at,
        "pdf_path": str(pdf_path.relative_to(root)),
        "page_count": total_pages,
        "review_visual_count": len(records) + 1,
        "readiness_page_count": 1,
        "supported_visual_count": len(supported_ids),
        "supported_visuals": supported_ids,
        "conditional_visuals": conditional_ids,
        "blocked_visuals": blocked_ids,
        "manual_review_status": "pending_consolidated_review",
        "visuals": visual_manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "pdf_path": str(pdf_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "page_count": total_pages,
        "review_visual_count": len(records) + 1,
        "readiness_page_count": 1,
        "supported_visual_count": len(supported_ids),
        "conditional_count": len(conditional_ids),
        "blocked_count": len(blocked_ids),
    }


def main() -> None:
    """Build the final PDF and print concise output paths."""

    result = build_review_pdf()

    print("Final ML Visual Intelligence review PDF created.")
    print(
        f"Review pages: {result['review_visual_count']} | Supported: {result['supported_visual_count']}"
    )
    print(f"Pages: {result['page_count']}")
    print(f"PDF: {result['pdf_path']}")
    print(f"Manifest: {result['manifest_path']}")
    print(
        "Next action: upload the PDF once for consolidated visual review."
    )


if __name__ == "__main__":
    main()
