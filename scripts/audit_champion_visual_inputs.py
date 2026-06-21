"""Audit source data and repository structure for MLV-A01 to MLV-A04.

Why this file exists:
    The first ML Visual Intelligence package covers champion selection and
    model comparison. Before chart code is written, we need to confirm the
    exact source files, columns, data types, row counts, and existing visual
    helpers in the current repository.

What this script reads:
    - Validated model-comparison CSV files under reports/tables/
    - Champion metadata JSON files under models/metadata/
    - Existing Python files under src/visualization/

What this script writes:
    - reports/qa/champion_visual_input_audit.json
    - reports/qa/champion_visual_input_audit.txt

How the outputs are used next:
    The audit controls the implementation of MLV-A01 to MLV-A04. It prevents
    guessed column names, duplicated helpers, and charts built from the wrong
    model-comparison source.

Safety:
    This is a read-only audit. It does not modify models, source tables,
    MLflow data, or existing visual files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# The script should be run from the project root.
# Using relative paths keeps the project portable across machines.
PROJECT_ROOT = Path.cwd()

# These are the validated sources identified by the earlier coverage audit.
TABLE_SOURCES = [
    Path("reports/tables/manual_model_comparison.csv"),
    Path("reports/tables/automl_benchmark_results.csv"),
    Path("reports/tables/final_model_selection.csv"),
    Path("reports/tables/final_true_champion_comparison.csv"),
    Path("reports/tables/final_true_champion_summary.csv"),
]

# Champion identity and MLflow lineage provide the evidence for MLV-A04.
METADATA_SOURCES = [
    Path("models/metadata/final_champion_metadata.json"),
    Path("models/metadata/mlflow_champion_lineage.json"),
]

# Existing visual modules are inspected so new code extends the current
# architecture instead of creating duplicate or conflicting helpers.
VISUALIZATION_DIR = Path("src/visualization")

# Audit outputs stay under reports/qa because they are generated evidence,
# not source code.
OUTPUT_DIR = Path("reports/qa")
JSON_OUTPUT = OUTPUT_DIR / "champion_visual_input_audit.json"
TEXT_OUTPUT = OUTPUT_DIR / "champion_visual_input_audit.txt"


def make_json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-safe Python values.

    Input:
        A value read from a DataFrame, Series, or JSON document.

    Output:
        A standard Python value accepted by `json.dump()`.

    Used next:
        Sample rows and metadata previews are stored in the audit JSON.
    """

    # Keep normal primitive values unchanged.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Convert pandas missing values into JSON null.
    if pd.isna(value):
        return None

    # Convert NumPy scalar objects into their native Python value.
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    # Fall back to a readable string for uncommon objects.
    return str(value)


def audit_csv(relative_path: Path) -> dict[str, Any]:
    """Inspect one CSV source without changing it.

    Input:
        Repository-relative path to a model-comparison CSV file.

    Output:
        Existence, shape, columns, dtypes, null counts, and sample rows.

    Used next:
        Chart functions use the confirmed schema instead of guessed columns.
    """

    # Resolve the file against the current project root.
    full_path = PROJECT_ROOT / relative_path

    # Return an explicit missing-file record instead of crashing immediately.
    if not full_path.exists():
        return {
            "path": str(relative_path),
            "exists": False,
            "issue": "Source file does not exist.",
        }

    # Read the source table exactly as stored in the repository.
    dataframe = pd.read_csv(full_path)

    # Keep only a small preview so the audit remains concise and reviewable.
    preview = dataframe.head(3).copy()

    # Convert sample values into JSON-safe native Python values.
    sample_rows = [
        {
            str(column): make_json_safe(value)
            for column, value in row.items()
        }
        for row in preview.to_dict(orient="records")
    ]

    # Return the complete schema evidence needed for implementation.
    return {
        "path": str(relative_path),
        "exists": True,
        "rows": int(dataframe.shape[0]),
        "columns_count": int(dataframe.shape[1]),
        "columns": [str(column) for column in dataframe.columns],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "null_counts": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().items()
            if int(count) > 0
        },
        "sample_rows": sample_rows,
    }


def audit_json(relative_path: Path) -> dict[str, Any]:
    """Inspect one champion metadata JSON file.

    Input:
        Repository-relative path to a metadata JSON file.

    Output:
        Existence, top-level keys, and the parsed metadata content.

    Used next:
        MLV-A04 uses the verified champion identity and lineage fields.
    """

    # Resolve the metadata path from the project root.
    full_path = PROJECT_ROOT / relative_path

    # Record missing metadata explicitly so it can be fixed before charting.
    if not full_path.exists():
        return {
            "path": str(relative_path),
            "exists": False,
            "issue": "Metadata file does not exist.",
        }

    # Load the existing JSON without changing it.
    with full_path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    # A dictionary provides useful key-level evidence.
    if isinstance(content, dict):
        top_level_keys = sorted(str(key) for key in content.keys())
    else:
        top_level_keys = []

    return {
        "path": str(relative_path),
        "exists": True,
        "content_type": type(content).__name__,
        "top_level_keys": top_level_keys,
        "content": content,
    }


def audit_visualization_package() -> dict[str, Any]:
    """Inspect existing visual source files.

    Input:
        The repository's `src/visualization/` directory.

    Output:
        File list, line counts, and whether the shared style file exists.

    Used next:
        New champion visual modules are placed consistently in this package.
    """

    # Return a clear record if the package directory is missing.
    if not (PROJECT_ROOT / VISUALIZATION_DIR).exists():
        return {
            "path": str(VISUALIZATION_DIR),
            "exists": False,
            "issue": "Visualization directory does not exist.",
        }

    file_records: list[dict[str, Any]] = []

    # Inspect Python modules only; generated images do not belong here.
    for file_path in sorted(
        (PROJECT_ROOT / VISUALIZATION_DIR).glob("*.py")
    ):
        # Count source lines to understand the current package size.
        line_count = len(
            file_path.read_text(encoding="utf-8").splitlines()
        )

        file_records.append(
            {
                "path": str(file_path.relative_to(PROJECT_ROOT)),
                "lines": int(line_count),
            }
        )

    return {
        "path": str(VISUALIZATION_DIR),
        "exists": True,
        "python_files": file_records,
        "shared_style_present": (
            PROJECT_ROOT
            / VISUALIZATION_DIR
            / "ml_visual_style.py"
        ).exists(),
    }


def build_text_report(audit: dict[str, Any]) -> str:
    """Create a concise human-readable summary.

    Input:
        Full structured audit dictionary.

    Output:
        Plain-text report containing only implementation-critical details.

    Used next:
        The user can upload this smaller file instead of pasting terminal logs.
    """

    lines: list[str] = []

    lines.append("CHAMPION VISUAL INPUT AUDIT")
    lines.append("=" * 80)
    lines.append(f"Project root: {audit['project_root']}")
    lines.append("")

    lines.append("TABLE SOURCES")
    lines.append("-" * 80)

    for table in audit["tables"]:
        lines.append(f"Path: {table['path']}")
        lines.append(f"Exists: {table['exists']}")

        if table["exists"]:
            lines.append(
                f"Shape: {table['rows']} rows x "
                f"{table['columns_count']} columns"
            )
            lines.append(
                "Columns: " + ", ".join(table["columns"])
            )
            lines.append(
                "Columns with nulls: "
                + (
                    ", ".join(table["null_counts"].keys())
                    if table["null_counts"]
                    else "None"
                )
            )
        else:
            lines.append(f"Issue: {table['issue']}")

        lines.append("")

    lines.append("METADATA SOURCES")
    lines.append("-" * 80)

    for metadata in audit["metadata"]:
        lines.append(f"Path: {metadata['path']}")
        lines.append(f"Exists: {metadata['exists']}")

        if metadata["exists"]:
            lines.append(
                "Top-level keys: "
                + ", ".join(metadata["top_level_keys"])
            )
        else:
            lines.append(f"Issue: {metadata['issue']}")

        lines.append("")

    lines.append("VISUALIZATION PACKAGE")
    lines.append("-" * 80)

    visual_package = audit["visualization_package"]
    lines.append(f"Path: {visual_package['path']}")
    lines.append(f"Exists: {visual_package['exists']}")

    if visual_package["exists"]:
        lines.append(
            "Shared style present: "
            f"{visual_package['shared_style_present']}"
        )

        for file_record in visual_package["python_files"]:
            lines.append(
                f"- {file_record['path']} "
                f"({file_record['lines']} lines)"
            )
    else:
        lines.append(f"Issue: {visual_package['issue']}")

    lines.append("")
    lines.append("NEXT USE")
    lines.append("-" * 80)
    lines.append(
        "Use this audit to map exact columns and metadata fields "
        "into MLV-A01, MLV-A02, MLV-A03, and MLV-A04."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the complete read-only champion visual input audit."""

    # Create the output folder only; source data remains untouched.
    (PROJECT_ROOT / OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # Audit every validated comparison table.
    table_audits = [
        audit_csv(relative_path)
        for relative_path in TABLE_SOURCES
    ]

    # Audit champion identity and MLflow lineage metadata.
    metadata_audits = [
        audit_json(relative_path)
        for relative_path in METADATA_SOURCES
    ]

    # Inspect the current visual source package and shared style file.
    visualization_audit = audit_visualization_package()

    # Combine all findings into one structured result.
    audit = {
        "project_root": str(PROJECT_ROOT),
        "tables": table_audits,
        "metadata": metadata_audits,
        "visualization_package": visualization_audit,
    }

    # Save machine-readable evidence for precise implementation mapping.
    with (PROJECT_ROOT / JSON_OUTPUT).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Save a concise human-readable summary for quick review.
    text_report = build_text_report(audit)

    (PROJECT_ROOT / TEXT_OUTPUT).write_text(
        text_report,
        encoding="utf-8",
    )

    # Print only the two generated paths and a compact completion summary.
    print(f"Created: {JSON_OUTPUT}")
    print(f"Created: {TEXT_OUTPUT}")
    print(
        "Tables found: "
        f"{sum(item['exists'] for item in table_audits)}"
        f"/{len(table_audits)}"
    )
    print(
        "Metadata files found: "
        f"{sum(item['exists'] for item in metadata_audits)}"
        f"/{len(metadata_audits)}"
    )
    print(
        "Shared visual style found: "
        f"{visualization_audit.get('shared_style_present', False)}"
    )


if __name__ == "__main__":
    main()
