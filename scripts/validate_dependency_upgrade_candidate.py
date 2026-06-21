"""Validate secure dependency candidates in an isolated environment."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "qa"
    / "dependency_preflight"
    / "dependency_candidate_validation.json"
)

EXPECTED_VERSIONS = {
    "pyarrow": "23.0.1",
    "scikit-learn": "1.5.0",
    "streamlit": "1.54.0",
    "pillow": "12.2.0",
}

MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "final_champion_model.joblib"
METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "metadata"
    / "final_champion_metadata.json"
)
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "visitor_features_sample.csv"
)


def installed_versions() -> dict[str, str]:
    """Return exact versions from the candidate environment."""

    return {
        package: version(package)
        for package in EXPECTED_VERSIONS
    }


def validate_model_load() -> dict:
    """Load the current model and detect version-persistence risk."""

    result = {
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "metadata_path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "loaded": False,
        "predicted": False,
        "retrain_required": False,
        "detail": "",
    }

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        result["retrain_required"] = True
        result["detail"] = "Current champion model or metadata is missing."
        return result

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", [])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)

        result["loaded"] = True

        if SAMPLE_PATH.exists() and feature_columns:
            sample = pd.read_csv(SAMPLE_PATH)

            missing = [
                column
                for column in feature_columns
                if column not in sample.columns
            ]
            if missing:
                result["detail"] = (
                    "Model loaded, but sample data is missing: "
                    + ", ".join(missing)
                )
                return result

            probabilities = model.predict_proba(
                sample[feature_columns].head(5)
            )[:, 1]

            if not (
                (probabilities >= 0).all()
                and (probabilities <= 1).all()
            ):
                raise ValueError(
                    "Candidate probabilities are outside 0 to 1."
                )

            result["predicted"] = True
            result["probability_min"] = float(probabilities.min())
            result["probability_max"] = float(probabilities.max())

        result["detail"] = (
            "Current champion loads and predicts in the candidate environment."
        )

    except InconsistentVersionWarning as warning:
        result["retrain_required"] = True
        result["detail"] = (
            "scikit-learn version mismatch detected: "
            f"{warning}. Retrain before changing live pins."
        )

    except Exception as error:
        result["retrain_required"] = True
        result["detail"] = (
            "Current champion is not safely compatible: "
            f"{type(error).__name__}: {error}"
        )

    return result


def main() -> None:
    """Run candidate version and model validation."""

    versions = installed_versions()

    mismatches = {
        package: {
            "expected": expected,
            "installed": versions.get(package),
        }
        for package, expected in EXPECTED_VERSIONS.items()
        if versions.get(package) != expected
    }

    model_result = validate_model_load()

    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "purpose": (
            "Dependency preflight only; live requirements are unchanged."
        ),
        "versions": versions,
        "version_mismatches": mismatches,
        "model_compatibility": model_result,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    if mismatches:
        raise RuntimeError(
            "Candidate environment versions do not match secure pins."
        )

    print("GOOD: secure dependency candidate versions installed")
    for package, installed in versions.items():
        print(f"{package}: {installed}")

    if model_result["retrain_required"]:
        print("MODEL STATUS: RETRAIN REQUIRED")
    elif model_result["predicted"]:
        print("MODEL STATUS: LOAD AND PREDICTION PASSED")
    else:
        print("MODEL STATUS: LOAD PASSED")

    print(model_result["detail"])
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
