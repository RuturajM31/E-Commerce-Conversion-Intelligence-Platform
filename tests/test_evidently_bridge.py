"""Tests for the optional isolated Evidently bridge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.monitoring import evidently_bridge


def test_monitoring_is_skipped_by_default(
    monkeypatch,
) -> None:
    """Normal execution must not require Evidently."""

    # Remove the opt-in flag to reproduce the default workflow.
    monkeypatch.delenv(
        "RUN_EVIDENTLY_MONITORING",
        raising=False,
    )

    result = (
        evidently_bridge
        .run_optional_evidently_monitoring()
    )

    assert result is False


def test_monitoring_uses_isolated_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Enabled monitoring must use .venv-evidently."""

    # Create a harmless fake Python path for the subprocess check.
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "RUN_EVIDENTLY_MONITORING",
        "1",
    )

    monkeypatch.setattr(
        evidently_bridge,
        "EVIDENTLY_PYTHON",
        fake_python,
    )

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        """Capture the subprocess inputs without running Evidently."""

        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]

        return SimpleNamespace(
            returncode=0,
            stdout="Monitoring completed",
            stderr="",
        )

    monkeypatch.setattr(
        evidently_bridge.subprocess,
        "run",
        fake_run,
    )

    result = (
        evidently_bridge
        .run_optional_evidently_monitoring()
    )

    assert result is True

    # The bridge must call the real monitoring module.
    assert captured["command"] == [
        str(fake_python),
        "-m",
        "src.monitoring.evidently_drift",
    ]

    # The subprocess must run from the repository root so imports work.
    assert (
        captured["cwd"]
        == evidently_bridge.PROJECT_ROOT
    )


def test_monitoring_failure_is_non_fatal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An Evidently failure must return False instead of crashing."""

    fake_python = tmp_path / "python"
    fake_python.write_text(
        "",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "RUN_EVIDENTLY_MONITORING",
        "1",
    )

    monkeypatch.setattr(
        evidently_bridge,
        "EVIDENTLY_PYTHON",
        fake_python,
    )

    # Simulate a monitoring subprocess failure.
    monkeypatch.setattr(
        evidently_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Evidently report generation failed",
        ),
    )

    result = (
        evidently_bridge
        .run_optional_evidently_monitoring()
    )

    assert result is False
