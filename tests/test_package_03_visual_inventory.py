"""Final Package 3 inventory and documented-exception controls."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


DOCS = Path("docs/streamlit")
INVENTORY = DOCS / "PACKAGE_03_CURRENT_VISUAL_INVENTORY.csv"
QUARANTINE = DOCS / "PACKAGE_03_QUARANTINE_DECISION_REGISTER.csv"
AUDIT_EVIDENCE = DOCS / "PACKAGE_03_AUDIT_EVIDENCE.csv"
CAPTURE_EXCEPTION = DOCS / "PACKAGE_03_CAPTURE_EXCEPTION.md"
FINAL_RESULT = DOCS / "PACKAGE_03_FINAL_RECONCILIATION.md"
MASTER = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.csv"
MANIFEST = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_MATRIX_MANIFEST.json"

AUDIT_IDS = {
    "SVE-AUD-01", "SVE-AUD-02", "SVE-AUD-03", "SVE-AUD-04",
    "SVE-AUD-05", "SVE-AUD-06", "SVE-AUD-07", "SVE-AUD-08",
    "SVE-AUD-11", "SVE-AUD-12", "SVE-AUD-13", "SVE-AUD-14",
    "SVE-AUD-16",
}

EXPECTED_PAGES = {
    "Executive Overview",
    "Visitor Intent Predictor",
    "Batch Scoring",
    "Model Benchmark Selection",
    "Business KPI Forecasting",
    "Anomaly and Outlier",
    "Monitoring, Drift and Health",
    "MLOps Architecture",
    "ML Validation & Evidence",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_inventory_remains_complete_and_source_grounded() -> None:
    rows = read_csv(INVENTORY)
    identifiers = [row["inventory_id"] for row in rows]

    assert len(rows) == 175
    assert len(identifiers) == len(set(identifiers))
    assert EXPECTED_PAGES == {row["page"] for row in rows}

    for row in rows:
        source = Path(row["source_file"])
        assert source.is_file(), row
        assert row["source_anchor"] in source.read_text(encoding="utf-8"), row


def test_all_thirteen_audit_rows_are_verified_truthfully() -> None:
    matrix = {row["ID"]: row for row in read_csv(MASTER)}
    evidence = {row["ID"]: row for row in read_csv(AUDIT_EVIDENCE)}

    assert set(evidence) == AUDIT_IDS

    for audit_id in AUDIT_IDS:
        assert matrix[audit_id]["Status"] == "VERIFIED"
        assert "Automated screenshots were not produced" in (
            matrix[audit_id]["Acceptance_Evidence"]
        )
        assert evidence[audit_id]["Package_03_State"] == "VERIFIED"
        assert "screenshot gate waived" in evidence[audit_id]["Remaining_Gate"]


def test_final_matrix_counts_reconcile() -> None:
    rows = read_csv(MASTER)
    counts = Counter(row["Status"] for row in rows)

    assert len(rows) == 204
    # Package 3 remains closed while the master closure moves every remaining
    # row to a terminal status.
    assert counts == {
        "VERIFIED": 177,
        "CONDITIONAL": 25,
        "EXCLUDED": 2,
    }


def test_manifest_records_the_exact_qa_exception() -> None:
    rows = read_csv(MASTER)
    expected = dict(sorted(Counter(row["Status"] for row in rows).items()))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["statuses"] == expected
    package_3 = manifest["package_checkpoints"]["package_03"]
    assert package_3["state"] == "verified_with_documented_qa_exception"
    assert package_3["verified_audit_rows"] == 13
    assert package_3["browser_screenshots_produced"] == 0
    assert package_3["browser_capture_status"] == (
        "waived_after_environment_incompatibility"
    )


def test_quarantine_remains_classified_without_restore() -> None:
    rows = read_csv(QUARANTINE)

    assert len(rows) == 33
    assert len({row["path"] for row in rows}) == 33
    assert all(row["decision"] != "RESTORE" for row in rows)
    assert all(
        row["decision"] == "DELETE_METADATA"
        for row in rows
        if row["source_type"] == "MACOS_METADATA"
    )


def test_capture_exception_records_all_failed_methods() -> None:
    text = CAPTURE_EXCEPTION.read_text(encoding="utf-8")

    assert "Playwright driver incompatibility" in text
    assert "Chrome DOM-export mode timeout" in text
    assert "Chrome screenshot mode timeout" in text
    assert "Browser screenshots produced: **0**" in text
    assert "Broken browser-capture scripts committed: **0**" in text


def test_failed_capture_scripts_are_not_committed() -> None:
    assert not Path("scripts/run_package_03_visual_capture.py").exists()
    assert not Path("scripts/run_package_03_visual_capture.sh").exists()


def test_final_result_does_not_invent_screenshot_evidence() -> None:
    text = FINAL_RESULT.read_text(encoding="utf-8")

    assert "VERIFIED WITH A DOCUMENTED QA EXCEPTION" in text
    assert "Automated screenshots were **not produced**" in text
    assert "Streamlit application files changed: **0**" in text
    assert "Package 4 — Batch Scoring and Campaign Intelligence" in text
