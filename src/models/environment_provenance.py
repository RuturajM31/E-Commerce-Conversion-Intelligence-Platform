"""Capture the runtime used to serialize a trained model artifact."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform


MODEL_PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
    "xgboost": "xgboost",
    "imbalanced_learn": "imbalanced-learn",
}


def get_installed_version(package_name: str) -> str:
    """Return an installed package version or an explicit unavailable value."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not_installed"


def get_environment_provenance(
    capture_context: str = "captured_during_model_save",
) -> dict:
    """Return reproducibility details for a saved model artifact."""

    package_versions = {
        metadata_key: get_installed_version(package_name)
        for metadata_key, package_name in MODEL_PACKAGE_NAMES.items()
    }

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_context": capture_context,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "machine_architecture": platform.machine(),
        "package_versions": package_versions,
    }
