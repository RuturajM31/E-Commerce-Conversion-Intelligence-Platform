"""Log the approved champion-selection visual package to MLflow.

Why this file exists:
    The champion-selection visuals have passed automated tests and manual
    visual review. This script logs the approved artifacts to the existing
    MLflow champion run without retraining or changing model metrics.

Inputs:
    - Approved visual package directory
    - Visual review sheet
    - Champion visual manifest
    - MLflow champion lineage metadata
    - Current Git commit and branch

Outputs:
    - Visual artifacts stored under the existing champion MLflow run
    - Traceability tags describing approval status, coverage, commit, branch,
      and QA method
    - A concise verification summary printed to the terminal

Used next:
    MLflow becomes the controlled evidence location for MLV-A01 to MLV-A04.
    The same approved artifacts can later be reused in Streamlit and reports.

Safety:
    - No model training occurs.
    - No model metrics or parameters are overwritten.
    - The script refuses to log when visual QA is incomplete.
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
# Repository-relative paths and expected visual coverage
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_DIR = Path(
    "reports/visuals/ml_visual_intelligence/champion_selection"
)
MANIFEST_PATH = PACKAGE_DIR / "champion_visual_manifest.json"
REVIEW_SHEET_PATH = Path(
    "reports/qa/champion_visual_review_sheet.png"
)
LINEAGE_PATH = Path(
    "models/metadata/mlflow_champion_lineage.json"
)

# The four approved visuals controlled by the current coverage matrix.
EXPECTED_VISUAL_IDS = {
    "MLV-A01",
    "MLV-A02",
    "MLV-A03",
    "MLV-A04",
}

# Files that must exist before MLflow logging can begin.
EXPECTED_PACKAGE_FILES = {
    "mlv-a01_model_performance_frontier.png",
    "mlv-a02_multi-metric_model_ranking.png",
    "mlv-a03_validation_to_holdout_generalisation_slopegraph.png",
    "mlv-a04_champion_scorecard_identity.png",
    "champion_model_ranking.csv",
    "champion_generalisation_metrics.csv",
    "champion_selection_insights.md",
    "champion_visual_manifest.json",
}

# Stable artifact root used inside the existing MLflow champion run.
MLFLOW_ARTIFACT_ROOT = "visual_intelligence/champion_selection"


# ---------------------------------------------------------------------------
# General file and Git helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object.

    Input:
        Absolute path to a JSON file.

    Output:
        Parsed dictionary.

    Used next:
        Manifest validation and champion run identification.
    """

    # Stop with a clear message when required evidence is missing.
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    # Load the existing JSON without changing it.
    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    # The script relies on named fields, so a dictionary is mandatory.
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def run_git_command(*arguments: str) -> str:
    """Run one read-only Git command from the project root.

    Input:
        Git command arguments, for example `rev-parse HEAD`.

    Output:
        Clean single-line command output.

    Used next:
        MLflow tags record the exact commit and branch of the visual package.
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
    """Return the current Git commit and branch.

    Input:
        Current repository state.

    Output:
        Dictionary with `commit` and `branch`.

    Used next:
        These values are written as MLflow traceability tags.
    """

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
    """Validate all approved champion visual evidence.

    Input:
        Project root containing the generated package and QA evidence.

    Output:
        Parsed manifest when every mandatory check passes.

    Used next:
        MLflow logging starts only after this function succeeds.
    """

    package_dir = project_root / PACKAGE_DIR
    manifest_file = project_root / MANIFEST_PATH
    review_sheet = project_root / REVIEW_SHEET_PATH

    # Confirm the approved artifact directory exists.
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Champion visual package directory not found: {package_dir}"
        )

    # Load the machine-readable QA and artifact manifest.
    manifest = read_json(manifest_file)

    # Confirm the package identity before trusting its contents.
    if manifest.get("category") != "ML Visual Intelligence":
        raise ValueError(
            "Manifest category is not `ML Visual Intelligence`."
        )

    if manifest.get("package") != "Champion selection":
        raise ValueError(
            "Manifest package is not `Champion selection`."
        )

    qa = manifest.get("qa")

    if not isinstance(qa, dict):
        raise ValueError("Manifest does not contain a valid QA section.")

    # All four controlled visuals must be present in the QA record.
    actual_visual_ids = set(qa.keys())

    if actual_visual_ids != EXPECTED_VISUAL_IDS:
        raise ValueError(
            "Manifest visual coverage mismatch. "
            f"Expected {sorted(EXPECTED_VISUAL_IDS)}, "
            f"found {sorted(actual_visual_ids)}."
        )

    # Refuse to log any visual that failed automated QA.
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

    # Check every expected package file.
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

    # The manually inspected review sheet is mandatory evidence.
    if not review_sheet.exists():
        raise FileNotFoundError(
            f"Visual review sheet not found: {review_sheet}"
        )

    # Reject empty or suspiciously tiny evidence files.
    small_files = sorted(
        path.name
        for path in package_dir.iterdir()
        if path.is_file() and path.stat().st_size == 0
    )

    if small_files:
        raise ValueError(
            "Package contains empty files: "
            + ", ".join(small_files)
        )

    if review_sheet.stat().st_size < 10_000:
        raise ValueError(
            "Visual review sheet is unexpectedly small."
        )

    return manifest


def expected_mlflow_artifact_paths() -> set[str]:
    """Build the exact MLflow artifact paths expected after logging.

    Input:
        Stable package filenames and artifact root constants.

    Output:
        Set of MLflow-relative artifact paths.

    Used next:
        Post-upload verification checks every expected artifact.
    """

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
# MLflow interaction
# ---------------------------------------------------------------------------


def list_artifacts_recursively(
    client,
    run_id: str,
    artifact_path: str = "",
) -> set[str]:
    """List every artifact path below one MLflow location.

    Input:
        MLflow client, run ID, and optional starting path.

    Output:
        Set of file artifact paths.

    Used next:
        The logger verifies that every approved file reached MLflow.
    """

    discovered: set[str] = set()

    # MLflow returns files and directories for one level at a time.
    for item in client.list_artifacts(
        run_id,
        artifact_path or None,
    ):
        if item.is_dir:
            # Recurse into directories to collect their file paths.
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
    """Validate and log the approved package to the champion run.

    Input:
        MLflow tracking URI and optional dry-run flag.

    Output:
        Summary containing run ID, artifact count, and verification result.

    Used next:
        Terminal output confirms that MLflow contains the approved evidence.
    """

    # Validate local evidence before importing or contacting MLflow.
    manifest = validate_approved_package(PROJECT_ROOT)
    lineage = read_json(PROJECT_ROOT / LINEAGE_PATH)
    git_context = get_git_context()

    run_id = str(lineage.get("run_id", "")).strip()

    if not run_id:
        raise ValueError(
            "MLflow champion lineage does not contain a run ID."
        )

    expected_paths = expected_mlflow_artifact_paths()

    # Dry-run is useful for local validation and unit tests.
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

    # Import MLflow only when a real upload is requested.
    # This keeps pure package-validation tests independent of MLflow.
    from mlflow import set_tracking_uri
    from mlflow.tracking import MlflowClient

    # Use the already-running local MLflow server.
    set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Confirm the target run exists before writing artifacts or tags.
    run = client.get_run(run_id)

    if run.info.lifecycle_stage != "active":
        raise ValueError(
            f"Target MLflow run is not active: {run_id}"
        )

    package_dir = PROJECT_ROOT / PACKAGE_DIR
    review_sheet = PROJECT_ROOT / REVIEW_SHEET_PATH

    # Log the approved package contents under one stable artifact folder.
    client.log_artifacts(
        run_id=run_id,
        local_dir=str(package_dir),
        artifact_path=MLFLOW_ARTIFACT_ROOT,
    )

    # Log the manual review sheet separately as explicit QA evidence.
    client.log_artifact(
        run_id=run_id,
        local_path=str(review_sheet),
        artifact_path=f"{MLFLOW_ARTIFACT_ROOT}/qa",
    )

    # Add traceability tags without changing model metrics or parameters.
    tags = {
        "visual_intelligence.champion_selection.status": "approved",
        "visual_intelligence.champion_selection.coverage": (
            "MLV-A01,MLV-A02,MLV-A03,MLV-A04"
        ),
        "visual_intelligence.champion_selection.qa": (
            "automated_and_manual"
        ),
        "visual_intelligence.champion_selection.git_commit": (
            git_context["commit"]
        ),
        "visual_intelligence.champion_selection.git_branch": (
            git_context["branch"]
        ),
        "visual_intelligence.champion_selection.artifact_root": (
            MLFLOW_ARTIFACT_ROOT
        ),
    }

    for key, value in tags.items():
        client.set_tag(
            run_id=run_id,
            key=key,
            value=value,
        )

    # Verify every expected file exists inside the MLflow artifact tree.
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
    """Read command-line options for local execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Log the approved champion-selection visual package "
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
            "Validate files and show the target run without "
            "uploading artifacts."
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
        print("GOOD: Champion visual MLflow dry-run passed")
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

    print("GOOD: Champion visual artifacts logged to MLflow")
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
