"""Truthfulness checks for the Streamlit enhancement matrix checkpoints."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "streamlit"
MASTER_CSV = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.csv"
MASTER_MD = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.md"
MANIFEST = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_MATRIX_MANIFEST.json"
PACKAGE_01 = DOCS / "PACKAGE_01_MATRIX_EVIDENCE.csv"
PACKAGE_02 = DOCS / "PACKAGE_02_MATRIX_EVIDENCE.csv"
PACKAGE_02_REPORT = DOCS / "PACKAGE_02_FINAL_RECONCILIATION.md"

ALLOWED_STATUSES = {
    "PLANNED",
    "IN PROGRESS",
    "IMPLEMENTED",
    "VERIFIED",
    "CONDITIONAL",
    "BLOCKED",
    "EXCLUDED",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Return every CSV row so tests can reconcile IDs and statuses."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_master_matrix_has_204_unique_rows_and_allowed_statuses() -> None:
    """The controlling CSV must remain complete and structurally valid."""

    rows = read_csv(MASTER_CSV)
    ids = [row["ID"] for row in rows]

    assert len(rows) == 204
    assert len(ids) == len(set(ids))
    assert {row["Status"] for row in rows}.issubset(ALLOWED_STATUSES)


def test_markdown_statuses_match_the_controlling_csv() -> None:
    """The human-readable matrix must not drift from the CSV statuses."""

    csv_statuses = {row["ID"]: row["Status"] for row in read_csv(MASTER_CSV)}
    markdown = MASTER_MD.read_text(encoding="utf-8")
    markdown_statuses: dict[str, str] = {}

    for line in markdown.splitlines():
        match = re.match(r"\| (SVE-[A-Z]+-\d{2}) \|.*\| ([A-Z ]+) \|$", line)
        if match:
            markdown_statuses[match.group(1)] = match.group(2)

    assert markdown_statuses == csv_statuses


def test_manifest_counts_match_the_master_matrix() -> None:
    """Manifest status totals must be calculated from the controlling rows."""

    rows = read_csv(MASTER_CSV)
    expected = dict(sorted(Counter(row["Status"] for row in rows).items()))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["matrix_version"] == "1.1"
    assert manifest["total_rows"] == 204
    assert manifest["statuses"] == expected
    assert manifest["package_checkpoints"]["package_01"]["commit"] == "82c3f66"
    assert manifest["package_checkpoints"]["package_02"]["verified_rows"] == 10
    assert manifest["package_checkpoints"]["package_02"]["excluded_rows"] == 2


def test_package_01_evidence_references_the_real_commit() -> None:
    """Committed Package 1 rows must no longer say QA and commit are pending."""

    rows = read_csv(PACKAGE_01)

    assert len(rows) == 42
    for row in rows:
        assert "82c3f66" in row["verification_evidence"]
        assert "pending final user-environment QA and commit" not in row["closure_state"]


def test_package_02_has_ten_verified_and_two_explicit_exclusions() -> None:
    """Accepted omissions must never be represented as completed controls."""

    rows = read_csv(PACKAGE_02)
    statuses = Counter(row["Status"] for row in rows)
    by_id = {row["ID"]: row for row in rows}

    assert len(rows) == 12
    assert statuses == {"VERIFIED": 10, "EXCLUDED": 2}
    assert by_id["SVE-EXEC-11"]["Status"] == "EXCLUDED"
    assert by_id["SVE-EXEC-12"]["Status"] == "EXCLUDED"
    assert "User-accepted omission" in by_id["SVE-EXEC-11"]["Evidence"]
    assert "User-accepted omission" in by_id["SVE-EXEC-12"]["Evidence"]


def test_package_02_report_records_the_actual_qa_result() -> None:
    """The closure report must retain the user-environment evidence."""

    report = PACKAGE_02_REPORT.read_text(encoding="utf-8")

    assert "31 passed in 7.00 seconds" in report
    assert "accepted by the user on 30 June 2026" in report
    assert "Ruturaj Mokashi — Data Analyst" in report
    normalized = " ".join(report.split())
    assert "does not stage, commit, push, merge" in normalized
