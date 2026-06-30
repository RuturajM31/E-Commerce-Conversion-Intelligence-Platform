"""Functional tests for the secure project wrapper."""
from __future__ import annotations

from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "E-Commerce-Conversion-Intelligence-Platform"


def fake_project(tmp_path: Path) -> Path:
    """Create a disposable project with fake Kubernetes scripts."""

    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)

    wrapper = project / SOURCE.name
    shutil.copy2(SOURCE, wrapper)
    wrapper.chmod(0o755)

    start = scripts / "k8s_demo_start.sh"
    start.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
test -n "${GRAFANA_ADMIN_USER:-}"
test -n "${GRAFANA_ADMIN_PASSWORD:-}"
printf '%s' "$GRAFANA_ADMIN_PASSWORD" > .received_password
echo "Kubernetes demo is ready:"
echo "Grafana: http://localhost:3000"
""",
        encoding="utf-8",
    )
    start.chmod(0o755)

    stop = scripts / "k8s_demo_stop.sh"
    stop.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
touch .stop_called
echo "Kubernetes port-forwards stopped: 7"
""",
        encoding="utf-8",
    )
    stop.chmod(0o755)

    return project


def run_wrapper(
    project: Path,
    command: str,
) -> subprocess.CompletedProcess[str]:
    """Run one wrapper command in the disposable project."""

    return subprocess.run(
        [str(project / SOURCE.name), command],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )


def test_start_creates_secure_password(tmp_path: Path) -> None:
    """Start creates a strong password and passes it to the child."""

    project = fake_project(tmp_path)
    result = run_wrapper(project, "start")

    assert result.returncode == 0, result.stderr

    runtime = project / ".runtime"
    password_file = runtime / "grafana_admin_password"
    password = password_file.read_text(encoding="utf-8").strip()
    received = (project / ".received_password").read_text(encoding="utf-8")

    assert len(password) == 48
    assert all(character in "0123456789abcdef" for character in password)
    assert received == password
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert stat.S_IMODE(password_file.stat().st_mode) == 0o600


def test_stop_removes_password(tmp_path: Path) -> None:
    """Stop removes the secret and calls the existing stop script."""

    project = fake_project(tmp_path)
    assert run_wrapper(project, "start").returncode == 0

    password_file = project / ".runtime" / "grafana_admin_password"
    assert password_file.is_file()

    result = run_wrapper(project, "stop")

    assert result.returncode == 0, result.stderr
    assert not password_file.exists()
    assert (project / ".stop_called").is_file()


def test_password_command_requires_active_session(tmp_path: Path) -> None:
    """Password is shown only while the local session is active."""

    project = fake_project(tmp_path)
    assert run_wrapper(project, "password").returncode == 1

    run_wrapper(project, "start")
    shown = run_wrapper(project, "password")

    assert shown.returncode == 0
    assert len(shown.stdout.strip()) == 48

    run_wrapper(project, "stop")
    assert run_wrapper(project, "password").returncode == 1


def test_unknown_command_returns_error(tmp_path: Path) -> None:
    """Unknown commands fail without running child scripts."""

    project = fake_project(tmp_path)
    result = run_wrapper(project, "launch")

    assert result.returncode == 2
    assert "ERROR: Unknown command: launch" in result.stdout
