"""Tests for reproducible dependencies and model provenance."""

from pathlib import Path

from src.models.environment_provenance import (
    get_environment_provenance,
)


def load_requirement_pins(path: str) -> dict:
    """Read exact package pins from one requirements file."""

    pins = {}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        assert "==" in line, f"Unpinned dependency in {path}: {line}"

        package_name, package_version = line.split("==", 1)
        pins[package_name] = package_version

    return pins


def test_core_requirements_are_exactly_pinned():
    """Production installations must not silently upgrade core packages."""

    pins = load_requirement_pins("requirements.txt")

    assert pins["numpy"] == "1.26.4"
    assert pins["pandas"] == "2.1.4"
    assert pins["scikit-learn"] == "1.4.2"
    assert pins["joblib"] == "1.3.2"
    assert pins["xgboost"] == "3.2.0"
    assert pins["imbalanced-learn"] == "0.14.2"


def test_app_runtime_can_load_xgboost_champion():
    """The lightweight application environment must load the saved model."""

    pins = load_requirement_pins("requirements-app.txt")

    assert pins["scikit-learn"] == "1.4.2"
    assert pins["joblib"] == "1.3.2"
    assert pins["xgboost"] == "3.2.0"


def test_environment_provenance_has_required_versions():
    """Saved model metadata must explain its serialization environment."""

    provenance = get_environment_provenance("test_capture")

    assert provenance["capture_context"] == "test_capture"
    assert provenance["python_version"]
    assert provenance["python_implementation"]

    versions = provenance["package_versions"]

    for required_key in [
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "joblib",
        "xgboost",
    ]:
        assert required_key in versions
        assert versions[required_key] != "not_installed"


def test_champion_workflows_save_environment_provenance():
    """Both champion metadata workflows must call the shared helper."""

    paths = [
        Path("src/models/model_selection.py"),
        Path("src/models/finalize_true_champion.py"),
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")

        assert "get_environment_provenance" in text
        assert '"environment": get_environment_provenance()' in text
