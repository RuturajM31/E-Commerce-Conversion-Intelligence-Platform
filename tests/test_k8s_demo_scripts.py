"""Regression tests for the local Kubernetes demo scripts."""

from pathlib import Path
import subprocess


START_SCRIPT = Path("scripts/k8s_demo_start.sh")
STOP_SCRIPT = Path("scripts/k8s_demo_stop.sh")


def test_start_script_waits_for_stable_deployments():
    """A rolling update must not fail because an old pod disappears."""

    text = START_SCRIPT.read_text(encoding="utf-8")

    assert "kubectl rollout status" in text
    assert "--for=condition=Ready" not in text

    for deployment_name in [
        "alert-webhook",
        "alertmanager",
        "blackbox-exporter",
        "grafana",
        "metrics-exporter",
        "prometheus",
        "streamlit-app",
    ]:
        assert f'"{deployment_name}"' in text


def test_start_script_configures_seven_port_forwards():
    """Every dashboard and monitoring endpoint needs a local forward."""

    text = START_SCRIPT.read_text(encoding="utf-8")

    for service_name in [
        "streamlit-service",
        "grafana-service",
        "prometheus-service",
        "metrics-exporter-service",
        "blackbox-exporter-service",
        "alertmanager-service",
        "alert-webhook-service",
    ]:
        assert f'"{service_name}"' in text

    assert "Verified ${#FORWARD_NAMES[@]} port-forward processes." in text


def test_stop_script_handles_pid_files_safely():
    """The stop script must not kill unrelated processes through stale PIDs."""

    text = STOP_SCRIPT.read_text(encoding="utf-8")

    assert "shopt -s nullglob" in text
    assert '"$LOG_DIR"/*.pid' in text
    assert "kubectl port-forward" in text
    assert "kill $(cat" not in text


def test_demo_scripts_have_valid_shell_syntax():
    """Both shell scripts must pass Bash syntax validation."""

    for script in [START_SCRIPT, STOP_SCRIPT]:
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
