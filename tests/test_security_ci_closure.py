"""Check the final security and CI closure state."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_retired_demo_password_is_removed() -> None:
    """Current documentation must not expose the old demo value."""

    paths = [
        PROJECT_ROOT / "docs" / "GRAFANA_MISSION_CONTROL_QA_QC_STATUS.md",
        PROJECT_ROOT / "docs" / "GRAFANA_MISSION_CONTROL_README.md",
    ]
    for path in paths:
        assert "Techno#123" not in path.read_text(encoding="utf-8")


def test_ci_contains_required_quality_checks() -> None:
    """CI must run lint, vulnerability, secret, and smoke checks."""

    text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    required = [
        "requirements-ci.txt",
        "ruff check",
        "ruff format --check",
        "pip_audit",
        "audit_security_repro.py --ci",
        "run_smoke_training.py",
    ]
    for token in required:
        assert token in text


def test_representative_sample_is_present() -> None:
    """The repository must include a compact CI dataset."""

    path = PROJECT_ROOT / "data" / "sample" / "visitor_features_sample.csv"
    assert path.exists()
    assert path.stat().st_size < 100_000


def test_ci_dependencies_are_pinned() -> None:
    """CI-only tools must have reproducible versions."""

    text = (PROJECT_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    assert "ruff==" in text
    assert "pip-audit==" in text
