"""Reconcile the two remediation matrices with verified project work.

This script performs the first closure pass only.

It marks work that is supported by committed code, tests, reports, or the
verified project audit. It does not close final QA, user approval, merge,
cloud deployment, or items that still need a separate proof step.

Inputs:
    docs/remediation/REMEDIATION_COVERAGE_MATRIX.md
    docs/remediation/ZERO_COST_MLOPS_REMAINING_COVERAGE_MATRIX.md

Outputs:
    updated matrix files
    docs/remediation/MATRIX_RECONCILIATION_REPORT.md

Used next:
    the remaining open items become the final remediation QA checklist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "docs/remediation/REMEDIATION_COVERAGE_MATRIX.md"
ZERO_PATH = PROJECT_ROOT / "docs/remediation/ZERO_COST_MLOPS_REMAINING_COVERAGE_MATRIX.md"
REPORT_PATH = PROJECT_ROOT / "docs/remediation/MATRIX_RECONCILIATION_REPORT.md"


# These items are directly supported by remediation commits and tests.
MASTER_FIXED = {
    "GIT-06", "GIT-07", "GIT-08",
    "DATA-03", "DATA-04", "DATA-05", "DATA-06", "DATA-07",
    "DATA-08", "DATA-09", "DATA-10", "DATA-11", "DATA-13", "DATA-14",
    *(f"FEAT-{number:02d}" for number in range(1, 19)),
    *(f"LEAK-{number:02d}" for number in range(1, 14)),
    *(f"MOD-{number:02d}" for number in range(1, 16) if number != 9),
    *(f"SCORE-{number:02d}" for number in range(1, 13)),
    *(f"PROD-{number:02d}" for number in range(1, 12)),
    "DOWN-01", "DOWN-02", "DOWN-03", "DOWN-04", "DOWN-05",
    "DOWN-07", "DOWN-09",
    *(f"MON-{number:02d}" for number in range(1, 17)),
    *(f"APP-{number:02d}" for number in range(1, 14)),
    *(f"DEP-{number:02d}" for number in range(1, 10)),
    *(f"TEST-{number:02d}" for number in range(2, 16)),
    "TEST-18", "TEST-19",
    *(f"CI-{number:02d}" for number in range(1, 10)),
    "CI-12", "CI-14",
    *(f"DOCK-{number:02d}" for number in range(1, 13)),
    *(f"K8S-{number:02d}" for number in range(1, 17)),
    "OUT-01", "OUT-02", "OUT-03", "OUT-04", "OUT-05",
    "OUT-10", "OUT-12", "OUT-13",
    *(f"DOC-{number:02d}" for number in range(2, 14)),
    "DOC-15", "DOC-16", "DOC-17", "DOC-18", "DOC-19",
}


# These are quality rules verified by the structure audit and comment inventory.
MASTER_VERIFIED = {
    *(f"STR-{number:02d}" for number in range(1, 11)),
    *(f"COM-{number:02d}" for number in range(1, 13)),
}


# These two requirements do not apply to the approved seven-feature numeric schema.
MASTER_EXCLUDED = {
    "FEAT-11": (
        "Recency, frequency, diversity, and activity-duration fields are not "
        "members of the approved seven-feature production schema."
    ),
    "FEAT-18": (
        "The approved production feature schema is numeric and contains no "
        "boolean or categorical model inputs requiring a separate encoding path."
    ),
}


# Final QA or genuinely unfinished work remains open in this first pass.
MASTER_EXPECTED_OPEN = {
    "GIT-09",
    "DATA-12",
    "MOD-09",
    "DOWN-06", "DOWN-08",
    "DEP-10",
    "SEC-07", "SEC-08", "SEC-09", "SEC-10",
    "TEST-16", "TEST-17",
    "CI-10", "CI-11", "CI-13",
    "OUT-06", "OUT-07", "OUT-08", "OUT-09", "OUT-11",
    "DOC-01", "DOC-14", "DOC-20",
    *(f"QA-{number:02d}" for number in range(1, 15)),
}


ZERO_COMPLETE = {
    "GOV-05", "GOV-06", "GOV-07", "GOV-08",
    "MLF-13", "MLF-20",
    *(f"EVD-{number:02d}" for number in range(1, 19) if number != 13),
    *(f"LBL-{number:02d}" for number in range(1, 16)),
    "CI-03", "CI-05", "CI-06", "CI-07", "CI-09", "CI-10",
    "CI-11", "CI-12", "CI-13", "CI-14", "CI-15",
    "K8S-07", "K8S-08",
    "SEC-02", "SEC-08",
    "DOC-03", "DOC-04", "DOC-05", "DOC-08", "DOC-09", "DOC-13",
    "QA-02", "QA-03", "QA-08",
}


ZERO_DEFERRED = {
    "MLF-17",
    "EVD-13",
    *(f"STDEP-{number:02d}" for number in range(1, 15)),
    "GRA-03", "GRA-04", "GRA-05", "GRA-06", "GRA-07", "GRA-08",
    "GRA-10", "GRA-11", "GRA-12",
    "SEC-05",
    "DOC-06",
}


ZERO_OPTIONAL = {"K8S-06"}


ZERO_REMAIN_OPEN = {
    "CI-04", "CI-08",
    "SEC-06", "SEC-07", "SEC-09", "SEC-10",
    "DOC-01", "DOC-02", "DOC-07", "DOC-10", "DOC-11", "DOC-12", "DOC-14",
    "QA-04", "QA-06", "QA-10", "QA-11", "QA-12", "QA-13", "QA-14",
    "QA-15", "QA-16", "QA-17", "QA-18", "QA-19", "QA-20",
}


COMMIT_EVIDENCE = [
    ("37b50ea", "Add shared feature engineering and validation tests"),
    ("2be7862", "Unify training and Streamlit feature engineering"),
    ("58f2ff8", "Use canonical feature schema in model training"),
    ("fc1eea5", "Add shared production model resolver"),
    ("8f54d79", "Use shared production model resolver in Streamlit"),
    ("0f575f4", "Use production model resolver in anomaly scoring"),
    ("154547e", "Use production model resolver in forecasting"),
    ("44f46ca", "Use production model resolver in segmentation"),
    ("00cc48f", "Add traceable final champion score export"),
    ("6501365", "Validate cached scores against production manifest"),
    ("c5a1b71", "Build leakage-safe rolling visitor snapshots"),
    ("3165892", "Separate training snapshots from production features"),
    ("8b222b5", "Use validation and untouched final holdout"),
    ("b3e890c", "Use chronological splits across model workflows"),
    ("2042def", "Finalize champion with untouched holdout"),
    ("4e89db5", "Refresh champion and monitoring metadata"),
    ("8aa32d2", "Add secure local alert delivery"),
    ("74380ac", "Secure Kubernetes and Helm alerting"),
    ("dbb9253", "Make production outcome reporting truthful"),
    ("267338e", "Pin runtime and record model provenance"),
    ("30a5458", "Make Kubernetes demo startup rollout-safe"),
    ("2f2466b", "Strengthen CI and neutralize model winner tests"),
    ("13ab284", "Add isolated MLflow champion tracking"),
    ("f25e482", "Add isolated Evidently drift monitoring"),
    ("26eefd8", "Align monitoring dashboard and documentation with outcome truthfulness"),
    ("94875cc", "Add production prediction ledger foundation"),
    ("f66a6bf", "Add safe prediction ledger writing"),
    ("f6afa76", "Add delayed label input contract"),
    ("44f5b54", "Add delayed label maturity validation"),
    ("5cd5c1e", "Add delayed label production performance reporting"),
    ("d5217e8", "Add end-to-end delayed label evaluation runner"),
    ("bdc2f0d", "Add delayed label monitoring runbook"),
    ("aa2263d", "Add delayed label monitoring tests to CI"),
    ("169f3ee", "Add shared ML visual style and QA"),
    ("c238a7a", "Add MLflow logging for champion visual artifacts"),
    ("8952737", "Add MLflow logging for threshold visual artifacts"),
    ("c38ec02", "Finalize ML visual intelligence review package"),
    ("df0bc2b", "Add lightweight MLflow and Evidently CI checks"),
]


MASTER_PATTERN = re.compile(r"^- \[([ xX])\] `([^`]+)` (.*)$")
ZERO_PATTERN = re.compile(
    r"^\| `([^`]+)` \| (.*?) \| (.*?) \| `([^`]+)` \|$"
)


def read_required(path: Path) -> str:
    """Read a required project file with a clear error."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return path.read_text(encoding="utf-8")


def master_ids(text: str) -> set[str]:
    """Return every checklist ID from the master matrix."""

    return {
        match.group(2)
        for line in text.splitlines()
        if (match := MASTER_PATTERN.match(line))
    }


def zero_ids(text: str) -> set[str]:
    """Return every requirement ID from the zero-cost matrix."""

    return {
        match.group(1)
        for line in text.splitlines()
        if (match := ZERO_PATTERN.match(line))
        and "-" in match.group(1)
    }


def validate_target_ids(master_text: str, zero_text: str) -> None:
    """Stop before writing when a target ID is absent."""

    available_master = master_ids(master_text)
    available_zero = zero_ids(zero_text)

    master_targets = (
        MASTER_FIXED
        | MASTER_VERIFIED
        | set(MASTER_EXCLUDED)
        | MASTER_EXPECTED_OPEN
    )
    zero_targets = (
        ZERO_COMPLETE
        | ZERO_DEFERRED
        | ZERO_OPTIONAL
        | ZERO_REMAIN_OPEN
    )

    missing_master = sorted(master_targets - available_master)
    missing_zero = sorted(zero_targets - available_zero)

    if missing_master or missing_zero:
        raise RuntimeError(
            "Matrix IDs did not match the expected structure. "
            f"Missing master IDs: {missing_master}; "
            f"missing zero-cost IDs: {missing_zero}."
        )


def update_master(text: str) -> tuple[str, dict[str, str]]:
    """Check only master items that have verified resolution evidence."""

    resolutions: dict[str, str] = {}
    output: list[str] = []

    for line in text.splitlines():
        match = MASTER_PATTERN.match(line)

        if not match:
            output.append(line)
            continue

        item_id = match.group(2)
        title = match.group(3)

        if item_id in MASTER_FIXED:
            output.append(f"- [x] `{item_id}` {title}")
            resolutions[item_id] = "Fixed and Tested"
        elif item_id in MASTER_VERIFIED:
            output.append(f"- [x] `{item_id}` {title}")
            resolutions[item_id] = "Verified Already Correct"
        elif item_id in MASTER_EXCLUDED:
            output.append(f"- [x] `{item_id}` {title}")
            resolutions[item_id] = "Intentionally Excluded"
        else:
            output.append(line)

    return "\n".join(output).rstrip() + "\n", resolutions


def zero_evidence(item_id: str, current: str) -> str:
    """Return concise acceptance evidence for updated zero-cost rows."""

    evidence = {
        "GOV-05": "Commit, test, report, and generated-artifact evidence is recorded before closure",
        "GOV-06": "Runbooks and monitoring outputs distinguish local validation from managed production",
        "GOV-07": "Required remediation tools use local open-source software or GitHub free automation",
        "GOV-08": "Future labels, large data, free-tier limits, and later cloud work are documented",
        "MLF-13": "Champion and threshold visual artifacts were logged and verified in MLflow",
        "MLF-17": "Optional Compose service is deferred because local isolated MLflow already satisfies this phase",
        "MLF-20": "Local MLflow startup, validation, registry, storage, and limitations are documented",
        "EVD-13": "Streamlit presentation remains scheduled for the dedicated visual-enhancement branch",
        "CI-03": "GitHub Actions now includes workflow_dispatch",
        "CI-05": "The CI guide records why full-data GitHub retraining is not practical",
        "CI-06": "Normal CI uses repository tests and does not upload the large local dataset",
        "CI-07": "GitHub Actions now includes a weekly compact validation schedule",
        "CI-09": "Manual and scheduled runs execute the isolated MLflow bridge test",
        "CI-10": "Manual and scheduled runs execute Evidently bridge and drift-summary tests",
        "CI-11": "Delayed-label tests are included in the normal CI workflow",
        "CI-13": "Failure-only evidence upload is configured",
        "CI-14": "Failure evidence is retained for seven days",
        "CI-15": "CI_WORKFLOW_GUIDE.md explains the full-retraining cost and runtime trade-off",
        "K8S-06": "Optional free-hosting experiments are not required for local Kubernetes closure",
        "K8S-07": "Documentation and runbooks avoid claims of managed cloud production",
        "K8S-08": "CI continues to validate Helm and Kubernetes after MLflow and Evidently work",
        "SEC-02": "Generated Evidently runtime outputs are controlled and repository hygiene is documented",
        "SEC-05": "Cloud-token handling is deferred to the later zero-cost deployment phase",
        "SEC-08": "MLflow and Evidently dependency files are pinned",
        "DOC-03": "MLflow startup, stop, inspect, validation, and local-storage steps are documented",
        "DOC-04": "Evidently generation, outputs, interpretation, and limitations are documented",
        "DOC-05": "The delayed-label monitoring runbook documents the full operational process",
        "DOC-08": "Limitations are separated from historical validation and current unlabeled scoring",
        "DOC-09": "Comment-preservation inventory and summary exist",
        "DOC-13": "Architecture and business explanations exist in project guides and visual evidence",
        "QA-02": "Focused Evidently bridge, schema, cohort, and drift tests passed during remediation",
        "QA-03": "Delayed-label contract, maturity, evaluation, and CI tests passed",
        "QA-08": "Evidently reports were generated from canonical project artifacts",
    }

    if item_id.startswith("EVD-") and item_id != "EVD-13":
        return "Implemented and tested in commit f25e482, with later CI validation"

    if item_id.startswith("LBL-"):
        return "Implemented, tested, documented, and connected through the delayed-label runner"

    if item_id.startswith("STDEP-"):
        return "Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase"

    if item_id.startswith("GRA-") and item_id in ZERO_DEFERRED:
        return "Public Grafana publication is deferred to the later zero-cost cloud phase"

    return evidence.get(item_id, current)


def update_zero(text: str) -> tuple[str, dict[str, str]]:
    """Update zero-cost rows without changing their requirement wording."""

    resolutions: dict[str, str] = {}
    output: list[str] = []

    for line in text.splitlines():
        match = ZERO_PATTERN.match(line)

        if not match:
            output.append(line)
            continue

        item_id, requirement, current_evidence, current_status = match.groups()

        if item_id in ZERO_COMPLETE:
            status = "COMPLETED"
        elif item_id in ZERO_DEFERRED:
            status = "DEFERRED"
        elif item_id in ZERO_OPTIONAL:
            status = "OPTIONAL"
        else:
            output.append(line)
            continue

        evidence = zero_evidence(item_id, current_evidence)
        output.append(
            f"| `{item_id}` | {requirement} | {evidence} | `{status}` |"
        )
        resolutions[item_id] = status

    return "\n".join(output).rstrip() + "\n", resolutions


def parse_master_counts(text: str) -> Counter[str]:
    """Count checked and unchecked master items."""

    counts: Counter[str] = Counter()

    for line in text.splitlines():
        match = MASTER_PATTERN.match(line)
        if not match:
            continue
        counts["checked" if match.group(1).lower() == "x" else "unchecked"] += 1

    return counts


def parse_zero_counts(text: str) -> Counter[str]:
    """Count zero-cost statuses."""

    counts: Counter[str] = Counter()

    for line in text.splitlines():
        match = ZERO_PATTERN.match(line)
        if match and "-" in match.group(1):
            counts[match.group(4)] += 1

    return counts


def ids_by_master_state(text: str, checked: bool) -> list[str]:
    """Return master IDs in one checkbox state."""

    result: list[str] = []

    for line in text.splitlines():
        match = MASTER_PATTERN.match(line)
        if not match:
            continue
        is_checked = match.group(1).lower() == "x"
        if is_checked == checked:
            result.append(match.group(2))

    return result


def ids_by_zero_status(text: str, statuses: set[str]) -> list[str]:
    """Return zero-cost IDs with selected statuses."""

    result: list[str] = []

    for line in text.splitlines():
        match = ZERO_PATTERN.match(line)
        if match and match.group(4) in statuses and "-" in match.group(1):
            result.append(match.group(1))

    return result


def zero_status_map(text: str) -> dict[str, str]:
    """Return every zero-cost matrix status by requirement ID."""

    result: dict[str, str] = {}

    for line in text.splitlines():
        match = ZERO_PATTERN.match(line)
        if match and "-" in match.group(1):
            result[match.group(1)] = match.group(4)

    return result


def grouped_master_resolutions(resolutions: dict[str, str]) -> dict[str, list[str]]:
    """Group master IDs by resolution label for the report."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id, status in resolutions.items():
        grouped[status].append(item_id)
    return grouped


def make_report(
    master_text: str,
    zero_text: str,
    master_resolutions: dict[str, str],
    zero_resolutions: dict[str, str],
) -> str:
    """Build the first-pass reconciliation report."""

    master_counts = parse_master_counts(master_text)
    zero_counts = parse_zero_counts(zero_text)
    master_open = ids_by_master_state(master_text, checked=False)
    zero_open = ids_by_zero_status(zero_text, {"NOT STARTED", "IN PROGRESS"})
    grouped = grouped_master_resolutions(master_resolutions)

    lines = [
        "# Remediation Matrix Reconciliation Report",
        "",
        "## Purpose",
        "",
        "This report records the first evidence-based closure pass across both remediation matrices.",
        "Only work supported by committed code, tests, generated evidence, or the verified project audit was closed.",
        "Final QA, user approval, merge, and later cloud deployment remain separate controlled steps.",
        "",
        "## Current totals after reconciliation",
        "",
        f"- Master matrix: {master_counts['checked']} resolved, {master_counts['unchecked']} still open",
    ]

    zero_summary = ", ".join(
        f"{status}: {count}"
        for status, count in sorted(zero_counts.items())
    )
    lines.append(f"- Zero-cost matrix: {zero_summary}")

    lines.extend([
        "",
        "## Master-matrix resolutions in this pass",
        "",
    ])

    for status in ["Fixed and Tested", "Verified Already Correct", "Intentionally Excluded"]:
        ids = sorted(grouped.get(status, []))
        lines.append(f"### {status}")
        lines.append("")
        lines.append(", ".join(f"`{item_id}`" for item_id in ids) or "None")
        lines.append("")

    lines.extend([
        "## Intentional exclusions",
        "",
    ])

    for item_id, reason in sorted(MASTER_EXCLUDED.items()):
        lines.append(f"- `{item_id}`: {reason}")

    lines.extend([
        "",
        "## Zero-cost status changes in this pass",
        "",
    ])

    zero_grouped: dict[str, list[str]] = defaultdict(list)
    for item_id, status in zero_resolutions.items():
        zero_grouped[status].append(item_id)

    for status in ["COMPLETED", "DEFERRED", "OPTIONAL"]:
        ids = sorted(zero_grouped.get(status, []))
        lines.append(f"### {status}")
        lines.append("")
        lines.append(", ".join(f"`{item_id}`" for item_id in ids) or "None")
        lines.append("")

    lines.extend([
        "## Items deliberately left open",
        "",
        "These items require final QA, a separate proof step, user approval, merge, or later cloud work.",
        "They were not closed from commit names alone.",
        "",
        "### Master matrix",
        "",
        ", ".join(f"`{item_id}`" for item_id in master_open),
        "",
        "### Zero-cost matrix",
        "",
        ", ".join(f"`{item_id}`" for item_id in zero_open),
        "",
        "## Main commit evidence",
        "",
    ])

    for commit, message in COMMIT_EVIDENCE:
        lines.append(f"- `{commit}` — {message}")

    lines.extend([
        "",
        "## Next action",
        "",
        "1. Review the remaining open IDs.",
        "2. Run the final project QA package.",
        "3. Close only the items proven by that QA run.",
        "4. Push the documentation update.",
        "5. Obtain user approval before merge.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    """Run the first evidence-based matrix reconciliation pass."""

    master_text = read_required(MASTER_PATH)
    zero_text = read_required(ZERO_PATH)

    validate_target_ids(master_text, zero_text)

    updated_master, master_resolutions = update_master(master_text)
    updated_zero, zero_resolutions = update_zero(zero_text)

    # Verify that items outside this reconciliation pass kept their
    # original state. Some local matrices may already contain valid completed
    # evidence from earlier QA runs, so we preserve that evidence instead of
    # forcing those rows back to an open status.
    original_master_states = {
        item_id: item_id not in set(
            ids_by_master_state(master_text, checked=False)
        )
        for item_id in MASTER_EXPECTED_OPEN
    }
    updated_master_states = {
        item_id: item_id not in set(
            ids_by_master_state(updated_master, checked=False)
        )
        for item_id in MASTER_EXPECTED_OPEN
    }

    changed_master = sorted(
        item_id
        for item_id in MASTER_EXPECTED_OPEN
        if original_master_states[item_id] != updated_master_states[item_id]
    )
    if changed_master:
        raise RuntimeError(
            "Items outside this pass changed state unexpectedly: "
            f"{changed_master}"
        )

    original_zero_statuses = zero_status_map(zero_text)
    updated_zero_statuses = zero_status_map(updated_zero)

    changed_zero = sorted(
        item_id
        for item_id in ZERO_REMAIN_OPEN
        if original_zero_statuses.get(item_id)
        != updated_zero_statuses.get(item_id)
    )
    if changed_zero:
        raise RuntimeError(
            "Items outside this pass changed status unexpectedly: "
            f"{changed_zero}"
        )

    report = make_report(
        updated_master,
        updated_zero,
        master_resolutions,
        zero_resolutions,
    )

    MASTER_PATH.write_text(updated_master, encoding="utf-8")
    ZERO_PATH.write_text(updated_zero, encoding="utf-8")
    REPORT_PATH.write_text(report.rstrip() + "\n", encoding="utf-8")

    master_counts = parse_master_counts(updated_master)
    zero_counts = parse_zero_counts(updated_zero)

    print("Remediation matrices reconciled.")
    print(
        "Master matrix: "
        f"{master_counts['checked']} resolved | "
        f"{master_counts['unchecked']} still open"
    )
    print("Zero-cost matrix statuses:")
    for status, count in sorted(zero_counts.items()):
        print(f"  {status}: {count}")
    print("Report:", REPORT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
