"""Build secure MLflow and Evidently requirement candidates.

The live requirement files are not modified.
"""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "qa" / "dependency_preflight"

SERVICE_PINS = {
    "mlflow": {
        "mlflow": "3.14.0",
        "pyarrow": "23.0.1",
        "scikit-learn": "1.5.0",
        "cryptography": "48.0.1",
    },
    "evidently": {
        "pyarrow": "23.0.1",
        "scikit-learn": "1.5.0",
    },
}


def package_name(line: str) -> str | None:
    """Extract a normalized package name from one requirement line."""

    clean = line.strip()

    if not clean or clean.startswith("#") or clean.startswith("-"):
        return None

    match = re.match(r"([A-Za-z0-9_.-]+)", clean)
    if not match:
        return None

    return match.group(1).lower().replace("_", "-")


def build_candidate(
    source: Path,
    target: Path,
    pins: dict[str, str],
) -> None:
    """Replace selected package pins while preserving the rest."""

    if not source.exists():
        raise FileNotFoundError(f"Missing requirements file: {source}")

    source_lines = source.read_text(encoding="utf-8").splitlines()
    output_lines: list[str] = []
    seen: set[str] = set()

    for line in source_lines:
        name = package_name(line)

        if name in pins:
            if name not in seen:
                output_lines.append(f"{name}=={pins[name]}")
                seen.add(name)
            continue

        output_lines.append(line)

    missing = [name for name in pins if name not in seen]

    if missing:
        output_lines.extend(
            [
                "",
                "# Secure dependency candidate pins.",
                "# Live requirements remain unchanged during preflight.",
            ]
        )
        output_lines.extend(
            f"{name}=={pins[name]}"
            for name in missing
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(output_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Create isolated candidates for both service environments."""

    candidates = {
        "mlflow": (
            PROJECT_ROOT / "requirements-mlflow.txt",
            OUTPUT_DIR / "requirements-mlflow-secure-candidate.txt",
        ),
        "evidently": (
            PROJECT_ROOT / "requirements-evidently.txt",
            OUTPUT_DIR / "requirements-evidently-secure-candidate.txt",
        ),
    }

    for service_name, (source, target) in candidates.items():
        build_candidate(
            source,
            target,
            SERVICE_PINS[service_name],
        )
        print(
            f"{service_name}: "
            f"{target.relative_to(PROJECT_ROOT)}"
        )

    print("Live requirement files were not changed.")


if __name__ == "__main__":
    main()
