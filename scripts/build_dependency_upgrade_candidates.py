"""Build secure dependency candidate files without changing live pins."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "qa" / "dependency_preflight"

# Minimum secure versions supported by the current audit evidence.
SECURE_PINS = {
    "pyarrow": "23.0.1",
    "scikit-learn": "1.5.0",
    "streamlit": "1.54.0",
    "pillow": "12.2.0",
}


def normalise_name(line: str) -> str | None:
    """Return a normalised package name for one requirement line."""

    clean = line.strip()

    if not clean or clean.startswith("#") or clean.startswith("-"):
        return None

    match = re.match(r"([A-Za-z0-9_.-]+)", clean)
    if not match:
        return None

    return match.group(1).lower().replace("_", "-")


def build_candidate(source_path: Path, output_path: Path) -> dict[str, str]:
    """Replace vulnerable pins and add any missing secure pins."""

    if not source_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {source_path}")

    source_lines = source_path.read_text(encoding="utf-8").splitlines()

    updated_lines: list[str] = []
    replaced: dict[str, str] = {}
    seen: set[str] = set()

    for line in source_lines:
        name = normalise_name(line)

        if name in SECURE_PINS:
            secure_line = f"{name}=={SECURE_PINS[name]}"
            updated_lines.append(secure_line)
            replaced[name] = secure_line
            seen.add(name)
        else:
            updated_lines.append(line)

    missing = [name for name in SECURE_PINS if name not in seen]

    if missing:
        updated_lines.extend(
            [
                "",
                "# Security validation overrides.",
                "# Candidate only; live requirements remain unchanged.",
            ]
        )
        for name in missing:
            secure_line = f"{name}=={SECURE_PINS[name]}"
            updated_lines.append(secure_line)
            replaced[name] = secure_line

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(updated_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    return replaced


def main() -> None:
    """Create secure candidate files for main and app environments."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_updates = build_candidate(
        PROJECT_ROOT / "requirements.txt",
        OUTPUT_DIR / "requirements-main-secure-candidate.txt",
    )
    app_updates = build_candidate(
        PROJECT_ROOT / "requirements-app.txt",
        OUTPUT_DIR / "requirements-app-secure-candidate.txt",
    )

    print("Secure dependency candidate files created.")
    print("Live requirements files were not changed.")

    print("\nMain candidate:")
    for name, value in sorted(main_updates.items()):
        print(f"- {name}: {value}")

    print("\nApp candidate:")
    for name, value in sorted(app_updates.items()):
        print(f"- {name}: {value}")


if __name__ == "__main__":
    main()
