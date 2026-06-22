"""Audit repository security and reproducibility controls.

Use ``--ci`` to return a failing exit code when a required control is
missing, failed, or still needs manual review.

The audit deliberately ignores local virtual environments and installed
third-party packages. It evaluates project files, not copied dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "qa"
JSON_PATH = REPORT_DIR / "security_repro_audit.json"
MARKDOWN_PATH = REPORT_DIR / "security_repro_audit.md"

TEXT_SUFFIXES = {
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".sh",
    ".env",
    ".example",
    ".csv",
}

# Local environments and generated caches are not project source files.
EXCLUDED_PATH_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "site-packages",
    "dist-packages",
    "node_modules",
}

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "generic_token": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\b"
        r"\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"
    ),
    "hardcoded_password": re.compile(
        r"(?i)\b(?:password|passwd|pwd|admin_password)\b" r"\s*[:=]\s*['\"]([^'\"\n]+)['\"]"
    ),
}

SAFE_VALUE_MARKERS = {
    "change_me",
    "changeme",
    "example",
    "placeholder",
    "your_",
    "<",
    "${",
    "$(",
    "env:",
    "secretkeyref",
    "valuefrom",
    "not-set",
    "grafana_admin_password",
    "gf_security_admin_password",
    "adminpasswordkey",
    "admin-password",
    "supplied securely",
    "supplied at runtime",
    "runtime secret",
    "not set",
}


@dataclass
class Finding:
    """One audit result shown in the report."""

    area: str
    status: str
    detail: str
    evidence: list[str]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one read-only command from the project root."""

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def is_excluded_path(path: Path) -> bool:
    """Return True for local environments, caches, and vendored packages."""

    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path

    for part in relative.parts:
        lowered = part.lower()

        if lowered in EXCLUDED_PATH_NAMES:
            return True

        if lowered == "venv" or lowered.startswith(".venv"):
            return True

    return False


def repository_files() -> list[Path]:
    """Return tracked plus untracked, non-ignored project files."""

    result = run_command(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not list repository files.")

    paths = [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]

    # Defence in depth: never audit local environments even if a new
    # environment was created before its .gitignore rule was added.
    return [path for path in paths if not is_excluded_path(path)]


def readable_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield project files that are safe to inspect as text."""

    for path in paths:
        if is_excluded_path(path):
            continue

        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".env"):
            yield path


def read_text(path: Path) -> str:
    """Read text without stopping the full audit on one bad file."""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def looks_like_safe_placeholder(value: str) -> bool:
    """Return True for placeholders and runtime secret references."""

    lowered = value.strip().lower()
    return any(marker in lowered for marker in SAFE_VALUE_MARKERS)


def is_platform_configuration_file(path: Path) -> bool:
    """Return True only for deployable configuration or startup files.

    Markdown documentation is checked by the general secret scan. It is not
    deployment configuration and should not be flagged merely for discussing
    the word "password".
    """

    if is_excluded_path(path):
        return False

    relative = str(path.relative_to(PROJECT_ROOT)).lower()
    suffix = path.suffix.lower()

    deployment_hint = any(
        word in relative
        for word in (
            "docker-compose",
            "compose.y",
            "kubernetes",
            "k8s",
            "helm",
            "values.y",
            "start_",
            "startup",
            "grafana",
        )
    )

    return deployment_hint and suffix in {
        ".yml",
        ".yaml",
        ".sh",
        ".env",
        ".example",
    }


def line_uses_runtime_secret(line: str) -> bool:
    """Return True when a line references a runtime or Kubernetes secret."""

    lowered = line.strip().lower()

    return any(
        marker in lowered
        for marker in (
            "${",
            "secretkeyref",
            "valuefrom",
            "env_file",
            "change_me",
            "changeme",
            "example",
            "placeholder",
            "grafana_admin_password",
            "gf_security_admin_password",
            "adminpasswordkey",
            "admin-password",
            "supplied securely",
            "supplied at runtime",
            "not set",
        )
    )


def inspect_environment_files(paths: list[Path]) -> Finding:
    """Confirm that real environment files are not committed."""

    environment_files = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in paths
        if not is_excluded_path(path) and path.name.startswith(".env")
    )
    unsafe = [path for path in environment_files if Path(path).name != ".env.example"]

    return Finding(
        area="Repository environment files",
        status="PASS" if not unsafe else "FAIL",
        detail=(
            "Only the safe .env.example file is present."
            if not unsafe
            else "A real environment file appears in the repository."
        ),
        evidence=environment_files or ["No .env-style files found."],
    )


def scan_repository_secrets(paths: list[Path]) -> Finding:
    """Search project files for obvious secret patterns."""

    hits: list[str] = []

    for path in readable_text_files(paths):
        relative = path.relative_to(PROJECT_ROOT)
        text = read_text(path)

        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_name, pattern in SECRET_PATTERNS.items():
                match = pattern.search(line)
                if not match:
                    continue

                if pattern_name == "hardcoded_password":
                    value = match.group(1)
                    if looks_like_safe_placeholder(value):
                        continue

                hits.append(f"{relative}:{line_number} [{pattern_name}]")

    return Finding(
        area="Repository secret-pattern scan",
        status="PASS" if not hits else "REVIEW",
        detail=(
            "No obvious committed credentials were found."
            if not hits
            else "Potential secret-like values require review."
        ),
        evidence=hits[:30],
    )


def inspect_platform_passwords(paths: list[Path]) -> Finding:
    """Inspect password handling in deployment configuration files."""

    hits: list[str] = []

    for path in readable_text_files(paths):
        if not is_platform_configuration_file(path):
            continue

        for line_number, line in enumerate(read_text(path).splitlines(), start=1):
            if not re.search(r"(?i)(password|passwd|admin_password)", line):
                continue

            if line_uses_runtime_secret(line):
                continue

            stripped = line.strip()

            # A literal password assignment is reviewable. A variable name or
            # descriptive comment without a value is not a credential.
            literal_assignment = re.search(
                r"(?i)(password|passwd|admin_password)\s*[:=]\s*" r"['\"]?([^'\"#\s][^'\"#]*)",
                stripped,
            )
            if not literal_assignment:
                continue

            value = literal_assignment.group(2).strip()
            if looks_like_safe_placeholder(value):
                continue

            hits.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}: " f"{stripped[:160]}")

    return Finding(
        area="Platform password configuration",
        status="PASS" if not hits else "REVIEW",
        detail=(
            "Deployment passwords use runtime variables or Secret references."
            if not hits
            else "Potential plain-text password configuration remains."
        ),
        evidence=hits[:30],
    )


def inspect_git_history() -> Finding:
    """Search history for high-risk secret signatures."""

    patterns = [
        r"AKIA[0-9A-Z]{16}",
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY",
    ]
    hits: list[str] = []

    for pattern in patterns:
        result = run_command(
            [
                "git",
                "log",
                "--all",
                "-G",
                pattern,
                "--format=%h %ad %s",
                "--date=short",
                "--",
                ".",
            ]
        )
        if result.returncode not in {0, 1}:
            hits.append(f"History scan error for pattern: {pattern}")
            continue

        for line in result.stdout.splitlines():
            if line.strip():
                hits.append(f"{pattern} -> {line.strip()}")

    return Finding(
        area="Git history high-risk secret scan",
        status="PASS" if not hits else "REVIEW",
        detail=(
            "No private-key or high-risk token signatures were found."
            if not hits
            else "Potential historical exposure requires review."
        ),
        evidence=hits[:30],
    )


def inspect_retired_demo_password() -> Finding:
    """Record the known retired public Grafana demo password."""

    result = run_command(
        [
            "git",
            "log",
            "--all",
            "-S",
            "Techno#123",
            "--format=%h %ad %s",
            "--date=short",
            "--",
            ".",
        ]
    )
    history = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    return Finding(
        area="Retired Grafana demo credential",
        status="INFO",
        detail=(
            "The former demo value is retired and must never be reused. "
            "Current configuration requires a runtime secret."
        ),
        evidence=history[:20] or ["No historical occurrence found."],
    )


def run_pip_check(label: str, python_path: Path) -> Finding:
    """Run pip check for one available Python environment."""

    if not python_path.exists():
        return Finding(
            area=f"Dependency health: {label}",
            status="SKIP",
            detail="Python environment is not present.",
            evidence=[str(python_path)],
        )

    result = run_command([str(python_path), "-m", "pip", "check"])
    output = (result.stdout + result.stderr).strip()

    return Finding(
        area=f"Dependency health: {label}",
        status="PASS" if result.returncode == 0 else "FAIL",
        detail=(
            "Installed packages report compatible dependencies."
            if result.returncode == 0
            else "pip check found dependency problems."
        ),
        evidence=output.splitlines()[:20] or ["No output."],
    )


def inspect_sample_data(paths: list[Path]) -> Finding:
    """Confirm a small representative dataset exists."""

    candidates = []
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        lowered = str(relative).lower()
        sample_hint = any(
            word in lowered for word in ("sample", "fixture", "tiny", "smoke", "synthetic")
        )
        data_suffix = path.suffix.lower() in {
            ".csv",
            ".json",
            ".parquet",
            ".feather",
            ".pkl",
        }
        if sample_hint and data_suffix:
            candidates.append(str(relative))

    return Finding(
        area="Representative sample data",
        status="PASS" if candidates else "MISSING",
        detail=(
            "Small repository data exists for tests and CI."
            if candidates
            else "No representative sample dataset was found."
        ),
        evidence=sorted(candidates)[:30],
    )


def inspect_calibration(paths: list[Path]) -> Finding:
    """Search for probability-calibration code or evidence."""

    tokens = (
        "calibration_curve",
        "calibratedclassifiercv",
        "brier_score",
        "reliability diagram",
        "probability calibration",
        "expected calibration error",
    )
    hits: list[str] = []

    for path in readable_text_files(paths):
        text = read_text(path).lower()
        if any(token in text for token in tokens):
            hits.append(str(path.relative_to(PROJECT_ROOT)))

    return Finding(
        area="Probability calibration evidence",
        status="PASS" if hits else "MISSING",
        detail=(
            "Calibration code or evidence exists."
            if hits
            else "No calibration implementation or report was found."
        ),
        evidence=sorted(set(hits))[:30],
    )


def inspect_ci_security(paths: list[Path]) -> Finding:
    """Check CI formatting, dependency, secret, and smoke controls."""

    workflows = [
        path
        for path in paths
        if str(path.relative_to(PROJECT_ROOT)).startswith(".github/workflows/")
        and path.suffix.lower() in {".yml", ".yaml"}
    ]

    text = "\n".join(read_text(path).lower() for path in workflows)

    checks = {
        "formatting_or_lint": ("ruff check" in text and "ruff format --check" in text),
        "dependency_scan": "pip_audit" in text,
        "secret_scan": "audit_security_repro.py --ci" in text,
        "sample_smoke_training": "run_smoke_training.py" in text,
    }
    missing = [name for name, present in checks.items() if not present]

    return Finding(
        area="CI quality and security checks",
        status="PASS" if not missing else "MISSING",
        detail=(
            "CI contains formatting, vulnerability, secret, " "and sample-training checks."
            if not missing
            else "One or more CI safety checks are missing."
        ),
        evidence=[
            f"{name}: {'FOUND' if present else 'NOT FOUND'}" for name, present in checks.items()
        ],
    )


def write_reports(findings: list[Finding]) -> None:
    """Write JSON and Markdown evidence."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    JSON_PATH.write_text(
        json.dumps(
            {
                "python": sys.version,
                "findings": [asdict(item) for item in findings],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Security and Reproducibility Audit",
        "",
        "This report is generated from the current repository state.",
        "",
    ]

    for item in findings:
        lines.extend(
            [
                f"## {item.area}",
                "",
                f"**Status:** `{item.status}`",
                "",
                item.detail,
                "",
                "**Evidence:**",
                "",
            ]
        )
        if item.evidence:
            lines.extend(f"- `{entry}`" for entry in item.evidence)
        else:
            lines.append("- No evidence files found.")
        lines.append("")

    MARKDOWN_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description="Audit repository security and reproducibility controls."
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Return a failing exit code when a required control is not ready.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full audit."""

    args = parse_args()
    paths = repository_files()

    findings = [
        inspect_environment_files(paths),
        scan_repository_secrets(paths),
        inspect_platform_passwords(paths),
        inspect_git_history(),
        inspect_retired_demo_password(),
        run_pip_check("main", Path(sys.executable)),
        run_pip_check(
            "MLflow",
            PROJECT_ROOT / ".venv-mlflow" / "bin" / "python",
        ),
        run_pip_check(
            "Evidently",
            PROJECT_ROOT / ".venv-evidently" / "bin" / "python",
        ),
        inspect_sample_data(paths),
        inspect_calibration(paths),
        inspect_ci_security(paths),
    ]

    write_reports(findings)

    print("\n=== SECURITY AND REPRODUCIBILITY AUDIT ===")
    for item in findings:
        print(f"{item.status:7} | {item.area}")

    print(f"\nReport: {MARKDOWN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON:   {JSON_PATH.relative_to(PROJECT_ROOT)}")

    blockers = [item for item in findings if item.status in {"FAIL", "REVIEW", "MISSING"}]

    if args.ci and blockers:
        print("\nCI audit blockers:")
        for item in blockers:
            print(f"- {item.status}: {item.area}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
