'''Check that the GitHub workflow contains the required safety features.'''

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def read_workflow() -> str:
    '''Read the workflow once for the checks below.'''

    assert WORKFLOW_PATH.exists(), "CI workflow is missing."
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_manual_and_scheduled_runs_exist() -> None:
    '''GitHub should support manual and weekly light checks.'''

    text = read_workflow()

    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert 'cron: "17 5 * * 1"' in text


def test_mlflow_and_evidently_checks_exist() -> None:
    '''The two MLOps tools should have focused CI checks.'''

    text = read_workflow()

    assert "name: Validate MLflow integration" in text
    assert "requirements-mlflow.txt" in text
    assert "tests/test_mlflow_bridge.py" in text

    assert "name: Validate Evidently integration" in text
    assert "requirements-evidently.txt" in text
    assert "tests/test_evidently_bridge.py" in text
    assert "tests/test_drift_summary.py" in text


def test_existing_safety_checks_remain() -> None:
    '''The update must not remove the existing important checks.'''

    text = read_workflow()

    required_text = [
        "name: Run automated tests",
        "name: Validate Docker build",
        "name: Validate Docker Compose configuration",
        "name: Validate Helm chart",
        "name: Validate Kubernetes YAML",
        "name: Validate delayed-label monitoring",
    ]

    for item in required_text:
        assert item in text, f"Missing existing CI check: {item}"


def test_failure_evidence_has_short_retention() -> None:
    '''Failure evidence should be kept for only seven days.'''

    text = read_workflow()

    assert "name: Upload compact failure evidence" in text
    assert "actions/upload-artifact@v4" in text
    assert "retention-days: 7" in text
