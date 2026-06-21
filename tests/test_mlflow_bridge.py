"""Tests for the optional MLflow subprocess bridge."""

from pathlib import Path
from types import SimpleNamespace

from src.models import mlflow_bridge


def test_tracking_is_skipped_by_default(
    monkeypatch,
) -> None:
    """Normal project execution must not require MLflow."""

    monkeypatch.delenv(
        "RUN_MLFLOW_TRACKING",
        raising=False,
    )

    result = mlflow_bridge.run_optional_mlflow_tracking()

    assert result is False


def test_tracking_uses_isolated_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Enabled tracking must call the isolated Python environment."""

    fake_python = tmp_path / "python"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("RUN_MLFLOW_TRACKING", "1")
    monkeypatch.setattr(
        mlflow_bridge,
        "MLFLOW_PYTHON",
        fake_python,
    )

    captured_command = {}

    def fake_run(command, **kwargs):
        captured_command["command"] = command
        captured_command["cwd"] = kwargs["cwd"]

        return SimpleNamespace(
            returncode=0,
            stdout="Champion registered",
            stderr="",
        )

    monkeypatch.setattr(
        mlflow_bridge.subprocess,
        "run",
        fake_run,
    )

    result = mlflow_bridge.run_optional_mlflow_tracking()

    assert result is True
    assert captured_command["command"] == [
        str(fake_python),
        "-m",
        "src.models.mlflow_tracking",
    ]
    assert captured_command["cwd"] == mlflow_bridge.PROJECT_ROOT


def test_tracking_failure_is_non_fatal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An MLflow service failure must return False, not crash training."""

    fake_python = tmp_path / "python"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("RUN_MLFLOW_TRACKING", "1")
    monkeypatch.setattr(
        mlflow_bridge,
        "MLFLOW_PYTHON",
        fake_python,
    )

    monkeypatch.setattr(
        mlflow_bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Tracking server unavailable",
        ),
    )

    result = mlflow_bridge.run_optional_mlflow_tracking()

    assert result is False
