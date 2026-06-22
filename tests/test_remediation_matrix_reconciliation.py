"""Guard the first-pass remediation-matrix reconciliation."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "docs/remediation/REMEDIATION_COVERAGE_MATRIX.md"
ZERO_PATH = PROJECT_ROOT / "docs/remediation/ZERO_COST_MLOPS_REMAINING_COVERAGE_MATRIX.md"
REPORT_PATH = PROJECT_ROOT / "docs/remediation/MATRIX_RECONCILIATION_REPORT.md"

MASTER_PATTERN = re.compile(r"^- \[([ xX])\] `([^`]+)` ")
ZERO_PATTERN = re.compile(r"^\| `([^`]+)` \| .* \| `([^`]+)` \|$")


def master_states() -> dict[str, bool]:
    """Return master checkbox states by ID."""

    states: dict[str, bool] = {}
    for line in MASTER_PATH.read_text(encoding="utf-8").splitlines():
        match = MASTER_PATTERN.match(line)
        if match:
            states[match.group(2)] = match.group(1).lower() == "x"
    return states


def zero_statuses() -> dict[str, str]:
    """Return zero-cost statuses by ID."""

    statuses: dict[str, str] = {}
    for line in ZERO_PATH.read_text(encoding="utf-8").splitlines():
        match = ZERO_PATTERN.match(line)
        if match and "-" in match.group(1):
            statuses[match.group(1)] = match.group(2)
    return statuses


def test_core_remediation_items_are_resolved() -> None:
    """Core implemented areas should be checked in the master matrix."""

    states = master_states()
    required = {
        "GIT-06",
        "STR-01",
        "COM-01",
        "DATA-03",
        "FEAT-06",
        "LEAK-10",
        "MOD-04",
        "SCORE-01",
        "PROD-08",
        "DOWN-03",
        "MON-16",
        "APP-02",
        "DEP-07",
        "TEST-10",
        "CI-06",
        "DOCK-10",
        "K8S-12",
        "OUT-05",
        "DOC-19",
    }

    missing = sorted(item_id for item_id in required if not states.get(item_id))
    assert not missing, f"Expected resolved master items are still open: {missing}"


def test_final_closure_has_no_open_master_rows() -> None:
    """All master rows should be closed after approval and merge."""

    states = master_states()

    final_gates = {
        "GIT-09",
        "QA-13",
        "QA-14",
    }

    incorrectly_open_final_gates = sorted(
        item_id for item_id in final_gates if not states.get(item_id)
    )

    assert not incorrectly_open_final_gates, (
        "Final approval or merge rows remain open: " f"{incorrectly_open_final_gates}"
    )

    all_open_rows = sorted(item_id for item_id, is_closed in states.items() if not is_closed)

    assert not all_open_rows, "Master remediation rows remain open: " f"{all_open_rows}"


def test_evidently_and_delayed_label_rows_are_current() -> None:
    """Previously stale Evidently and delayed-label rows should be completed."""

    statuses = zero_statuses()

    for number in range(1, 19):
        item_id = f"EVD-{number:02d}"
        expected = "DEFERRED" if item_id == "EVD-13" else "COMPLETED"
        assert statuses[item_id] == expected

    for number in range(1, 16):
        assert statuses[f"LBL-{number:02d}"] == "COMPLETED"


def test_later_cloud_work_is_deferred() -> None:
    """Cloud and hosted Streamlit work belongs to the later phase."""

    statuses = zero_statuses()

    for number in range(1, 15):
        assert statuses[f"STDEP-{number:02d}"] == "DEFERRED"

    for item_id in [
        "GRA-03",
        "GRA-04",
        "GRA-05",
        "GRA-06",
        "GRA-07",
        "GRA-08",
        "GRA-10",
        "GRA-11",
        "GRA-12",
    ]:
        assert statuses[item_id] == "DEFERRED"


def test_new_ci_evidence_is_recorded() -> None:
    """All implemented CI controls should be marked complete."""

    statuses = zero_statuses()

    completed_items = {
        "CI-03",
        "CI-04",
        "CI-05",
        "CI-06",
        "CI-07",
        "CI-08",
        "CI-09",
        "CI-10",
        "CI-11",
        "CI-12",
        "CI-13",
        "CI-14",
        "CI-15",
    }

    for item_id in completed_items:
        assert statuses[item_id] == "COMPLETED"


def test_reconciliation_report_exists() -> None:
    """Closure decisions should have one readable evidence report."""

    text = REPORT_PATH.read_text(encoding="utf-8")

    assert "# Remediation Matrix Reconciliation Report" in text
    assert "Final QA" in text
    assert "df0bc2b" in text
    assert "Items deliberately left open" in text
