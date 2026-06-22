"""Build reproducible provenance records for monitoring artifacts.

This module uses only the Python standard library.

Inputs:
    Paths to the model, metadata, reference data, current data, and scores.

Outputs:
    SHA-256 hashes, file sizes, project-relative paths, feature-schema hash,
    and optional MLflow champion lineage.

Used next:
    ``evidently_drift.py`` stores this provenance inside the compact
    monitoring summary so every drift result can be traced back to the
    exact files and model version that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# Find the repository root from src/monitoring/monitoring_provenance.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLflow lineage is optional because monitoring must still work
# when the local MLflow server or registry is unavailable.
MLFLOW_LINEAGE_PATH = (
    PROJECT_ROOT
    / "models/metadata/mlflow_champion_lineage.json"
)


def relative_project_path(path: Path) -> str:
    """Return a project-relative path when possible.

    Args:
        path:
            File path that belongs to this project.

    Returns:
        A readable relative path for JSON reports. An absolute path is
        returned only when the file sits outside the project root.
    """

    resolved_path = path.resolve()

    try:
        return str(
            resolved_path.relative_to(
                PROJECT_ROOT.resolve()
            )
        )
    except ValueError:
        return str(resolved_path)


def file_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the full file into memory.

    Args:
        path:
            Existing file that should be fingerprinted.

        chunk_size:
            Number of bytes read per iteration. One-megabyte chunks keep
            memory use small when hashing the large training CSV.

    Returns:
        A lowercase SHA-256 hexadecimal digest.

    Raises:
        FileNotFoundError:
            Raised when the required provenance file does not exist.

        ValueError:
            Raised when the supplied path is not a normal file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Provenance file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Provenance path is not a file: {path}"
        )

    # SHA-256 creates a stable fingerprint for the exact file contents.
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            # Read one small block so large datasets never fill memory.
            chunk = file_handle.read(chunk_size)

            # An empty block means the end of the file was reached.
            if not chunk:
                break

            # Add this block to the running file fingerprint.
            digest.update(chunk)

    return digest.hexdigest()


def file_provenance(
    path: Path,
) -> dict[str, Any]:
    """Create the provenance record for one input file.

    Input:
        One required project file.

    Output:
        Project-relative path, byte size, and SHA-256 hash.

    Used next:
        The result is stored under ``provenance.files`` in the compact
        Evidently monitoring summary.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Provenance file was not found: {path}"
        )

    return {
        "path": relative_project_path(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def feature_schema_sha256(
    feature_columns: list[str],
) -> str:
    """Create a deterministic hash for the ordered feature schema.

    Feature order matters because the champion model expects the same
    seven columns in the same order during production scoring.
    """

    # Compact JSON provides one consistent representation of the schema.
    canonical_schema = json.dumps(
        feature_columns,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_schema.encode("utf-8")
    ).hexdigest()


def load_optional_mlflow_lineage() -> dict[str, Any] | None:
    """Load the current champion registry lineage when available.

    Returns:
        Selected model-registry fields when the lineage file exists.

        ``None`` when MLflow lineage is unavailable. Monitoring remains
        valid because MLflow is an optional local integration.
    """

    if not MLFLOW_LINEAGE_PATH.exists():
        return None

    lineage = json.loads(
        MLFLOW_LINEAGE_PATH.read_text(
            encoding="utf-8"
        )
    )

    # Keep only fields useful for tracing this monitoring report.
    return {
        "run_id": lineage.get("run_id"),
        "registered_model_name": lineage.get(
            "registered_model_name"
        ),
        "registered_model_version": lineage.get(
            "registered_model_version"
        ),
        "registered_model_alias": lineage.get(
            "registered_model_alias"
        ),
        "model_uri": lineage.get("model_uri"),
    }


def build_monitoring_provenance(
    metadata_path: Path,
    model_path: Path,
    training_path: Path,
    scoring_path: Path,
    scores_path: Path,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Build complete file, schema, and model-registry provenance.

    Inputs:
        The exact artifacts used to create the monitoring report.

    Output:
        A compact dictionary proving which model, feature schema,
        reference source, current source, and score file created the
        drift results.
    """

    # Hash every important input so later runs can detect any change.
    files = {
        "champion_metadata": file_provenance(
            metadata_path
        ),
        "champion_model": file_provenance(
            model_path
        ),
        "training_reference_source": file_provenance(
            training_path
        ),
        "current_scoring_source": file_provenance(
            scoring_path
        ),
        "production_scores": file_provenance(
            scores_path
        ),
    }

    return {
        "feature_schema": {
            "columns": feature_columns,
            "column_count": len(feature_columns),
            "ordered_schema_sha256": (
                feature_schema_sha256(
                    feature_columns
                )
            ),
        },
        "files": files,
        "mlflow_champion": (
            load_optional_mlflow_lineage()
        ),
    }
