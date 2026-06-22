"""Log the approved threshold-decision visual package to MLflow.

Why this file exists:
    MLV-B01 and MLV-B04 have passed automated tests and manual visual review.
    This script logs the approved package to the existing champion MLflow run.

Inputs:
    - reports/visuals/ml_visual_intelligence/threshold_decisions/
    - reports/qa/threshold_visual_review_sheet.png
    - models/metadata/mlflow_champion_lineage.json
    - Current Git commit and branch

Outputs:
    - Threshold visual artifacts stored under the existing MLflow champion run
    - Traceability tags for coverage, QA method, Git commit, and branch
    - A concise terminal verification summary

Used next:
    MLflow becomes the controlled evidence location for MLV-B01 and MLV-B04.
    After verification, explainability visuals E01 to E06 can begin.

Safety:
    - No model training occurs.
    - No existing metrics or parameters are changed.
    - Missing or failed visual evidence blocks the upload.
    - Re-running is safe because the same artifact paths are reused.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# Repository-relative package configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_DIR = Path(
    "reports/visuals/ml_visual_intelligence/threshold_decisions"
)
MANIFEST_PATH = PACKAGE_DIR / "threshold_visual_manifest.json"
REVIEW_SHEET_PATH = Path(
    "reports/qa/threshold_visual_review_sheet.png"
)
LINEAGE_PATH = Path(
    "models/metadata/mlflow_champion_lineage.json"
)

EXPECTED_VISUAL_IDS = {
    "MLV-B01",
    "MLV-B04",
}

EXPECTED_PACKAGE_FILES = {
    "mlv-b01_threshold_decision_studio.png",
    "mlv-b04_confusion_matrix_decision_map.png",
    "selected_threshold_evidence.csv",
    "confusion_matrix_counts.csv",
    "threshold_decision_insights.md",
    "threshold_visual_manifest.json",
}

MLFLOW_ARTIFACT_ROOT = "visual_intelligence/threshold_decisions"


# ---------------------------------------------------------------------------
# File and Git helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object.

    Input:
        Absolute path to a JSON file.

    Output:
        Parsed dictionary.

    Used next:
        Manifest validation and MLflow run identification.
    """

    # Stop clearly when required evidence is missing.
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    # Load the existing JSON without modifying it.
    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    # Named-field access requires a dictionary.
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def run_git_command(*arguments: str) -> str:
    """Run one read-only Git command from the project root.

    Input:
        Git arguments such as `rev-parse HEAD`.

    Output:
        Clean command output.

    Used next:
        MLflow tags record the exact source commit and branch.
    """

    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def get_git_context() -> dict[str, str]:
    """Return the current Git commit and branch."""

    return {
        "commit": run_git_command("rev-parse", "HEAD"),
        "branch": run_git_command(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
    }


# ---------------------------------------------------------------------------
# Package validation
# ---------------------------------------------------------------------------


def validate_approved_package(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate all approved threshold visual evidence.

    Input:
        Project root containing the package and QA review sheet.

    Output:
        Parsed manifest when every mandatory check passes.

    Used next:
        MLflow logging starts only after this validation succeeds.
    """

    package_dir = project_root / PACKAGE_DIR
    manifest_file = project_root / MANIFEST_PATH
    review_sheet = project_root / REVIEW_SHEET_PATH

    # Confirm the generated package directory exists.
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Threshold visual package directory not found: {package_dir}"
        )

    manifest = read_json(manifest_file)

    # Confirm the manifest belongs to the expected controlled package.
    if manifest.get("category") != "ML Visual Intelligence":
        raise ValueError(
            "Manifest category is not `ML Visual Intelligence`."
        )

    if manifest.get("package") != "Threshold decisions":
        raise ValueError(
            "Manifest package is not `Threshold decisions`."
        )

    qa = manifest.get("qa")

    if not isinstance(qa, dict):
        raise ValueError("Manifest does not contain a valid QA section.")

    actual_visual_ids = set(qa.keys())

    if actual_visual_ids != EXPECTED_VISUAL_IDS:
        raise ValueError(
            "Manifest visual coverage mismatch. "
            f"Expected {sorted(EXPECTED_VISUAL_IDS)}, "
            f"found {sorted(actual_visual_ids)}."
        )

    # Every controlled visual must have passed automated QA.
    failed_visuals = sorted(
        visual_id
        for visual_id, result in qa.items()
        if not isinstance(result, dict)
        or result.get("passed") is not True
    )

    if failed_visuals:
        raise ValueError(
            "Visual QA is not approved for: "
            + ", ".join(failed_visuals)
        )

    # Confirm every expected package file exists.
    existing_files = {
        path.name
        for path in package_dir.iterdir()
        if path.is_file()
    }
    missing_files = sorted(
        EXPECTED_PACKAGE_FILES.difference(existing_files)
    )

    if missing_files:
        raise FileNotFoundError(
            "Approved package is missing files: "
            + ", ".join(missing_files)
        )

    # The manually approved review sheet is mandatory evidence.
    if not review_sheet.exists():
        raise FileNotFoundError(
            f"Threshold review sheet not found: {review_sheet}"
        )

    # Refuse empty output files.
    empty_files = sorted(
        path.name
        for path in package_dir.iterdir()
        if path.is_file() and path.stat().st_size == 0
    )

    if empty_files:
        raise ValueError(
            "Package contains empty files: "
            + ", ".join(empty_files)
        )

    if review_sheet.stat().st_size < 10_000:
        raise ValueError(
            "Threshold review sheet is unexpectedly small."
        )

    return manifest


def expected_mlflow_artifact_paths() -> set[str]:
    """Build exact MLflow-relative artifact paths expected after upload."""

    package_paths = {
        f"{MLFLOW_ARTIFACT_ROOT}/{filename}"
        for filename in EXPECTED_PACKAGE_FILES
    }

    review_path = (
        f"{MLFLOW_ARTIFACT_ROOT}/qa/"
        f"{REVIEW_SHEET_PATH.name}"
    )

    return package_paths | {review_path}


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def list_artifacts_recursively(
    client,
    run_id: str,
    artifact_path: str = "",
) -> set[str]:
    """List every file artifact under one MLflow location."""

    discovered: set[str] = set()

    for item in client.list_artifacts(
        run_id,
        artifact_path or None,
    ):
        if item.is_dir:
            discovered.update(
                list_artifacts_recursively(
                    client,
                    run_id,
                    item.path,
                )
            )
        else:
            discovered.add(item.path)

    return discovered


def log_package_to_mlflow(
    *,
    tracking_uri: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and log the threshold package to the champion run.

    Input:
        MLflow tracking URI and optional dry-run flag.

    Output:
        Verification summary containing run ID, coverage, Git context, and
        uploaded artifact count.
    """

    # Validate local evidence before contacting MLflow.
    manifest = validate_approved_package(PROJECT_ROOT)
    lineage = read_json(PROJECT_ROOT / LINEAGE_PATH)
    git_context = get_git_context()

    run_id = str(lineage.get("run_id", "")).strip()

    if not run_id:
        raise ValueError(
            "MLflow champion lineage does not contain a run ID."
        )

    expected_paths = expected_mlflow_artifact_paths()

    # Dry-run confirms package integrity without writing anything.
    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "tracking_uri": tracking_uri,
            "expected_artifact_count": len(expected_paths),
            "visual_ids": sorted(manifest["qa"].keys()),
            "git_commit": git_context["commit"],
            "git_branch": git_context["branch"],
        }

    # Import MLflow only for a real upload.
    from mlflow import set_tracking_uri
    from mlflow.tracking import MlflowClient

    set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Confirm the target champion run still exists and is active.
    run = client.get_run(run_id)

    if run.info.lifecycle_stage != "active":
        raise ValueError(
            f"Target MLflow run is not active: {run_id}"
        )

    package_dir = PROJECT_ROOT / PACKAGE_DIR
    review_sheet = PROJECT_ROOT / REVIEW_SHEET_PATH

    # Upload the complete package under one stable artifact root.
    client.log_artifacts(
        run_id=run_id,
        local_dir=str(package_dir),
        artifact_path=MLFLOW_ARTIFACT_ROOT,
    )

    # Upload the manual review sheet as explicit QA evidence.
    client.log_artifact(
        run_id=run_id,
        local_path=str(review_sheet),
        artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/qa",
    )

    # Add traceability tags without changing model metrics or parameters.
    tags = {
        "visual_intelligence.threshold_decisions.status": "approved",
        "visual_intelligence.threshold_decisions.coverage": (
            "MLV-B01,MLV-B04"
        ),
        "visual_intelligence.threshold_decisions.qa": (
            "automated_and_manual"
        ),
        "visual_intelligence.threshold_decisions.git_commit": (
            git_context["commit"]
        ),
        "visual_intelligence.threshold_decisions.git_branch": (
            git_context["branch"]
        ),
        "visual_intelligence.threshold_decisions.artifact_root": (
            MLFLOW_ARTIFACT_ROOT
        ),
    }

    for key, value in tags.items():
        client.set_tag(
            run_id=run_id,
            key=key,
            value=value,
        )

    # Verify every expected artifact exists after upload.
    uploaded_paths = list_artifacts_recursively(
        client,
        run_id,
        MLFLOW_ARTIFACT_ROOT,
    )
    missing_uploaded = sorted(
        expected_paths.difference(uploaded_paths)
    )

    if missing_uploaded:
        raise RuntimeError(
            "MLflow upload verification failed. Missing: "
            + ", ".join(missing_uploaded)
        )

    return {
        "dry_run": False,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "uploaded_artifact_count": len(expected_paths),
        "visual_ids": sorted(manifest["qa"].keys()),
        "git_commit": git_context["commit"],
        "git_branch": git_context["branch"],
        "verification_passed": True,
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Log the approved threshold-decision visual package "
            "to the existing MLflow champion run."
        )
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        ),
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate package evidence without uploading artifacts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run validation, optional upload, and concise verification output."""

    args = parse_args()

    summary = log_package_to_mlflow(
        tracking_uri=args.tracking_uri,
        dry_run=args.dry_run,
    )

    if summary["dry_run"]:
        print("GOOD: Threshold visual MLflow dry-run passed")
        print(f"Run ID: {summary['run_id']}")
        print(
            "Expected artifacts: "
            f"{summary['expected_artifact_count']}"
        )
        print(
            "Coverage: "
            + ", ".join(summary["visual_ids"])
        )
        print(f"Commit: {summary['git_commit']}")
        print(f"Branch: {summary['git_branch']}")
        return

    print("GOOD: Threshold visual artifacts logged to MLflow")
    print(f"Run ID: {summary['run_id']}")
    print(
        "Uploaded artifacts verified: "
        f"{summary['uploaded_artifact_count']}"
    )
    print(
        "Coverage: "
        + ", ".join(summary["visual_ids"])
    )
    print(f"Commit: {summary['git_commit']}")
    print(f"Branch: {summary['git_branch']}")


if __name__ == "__main__":
    main()
