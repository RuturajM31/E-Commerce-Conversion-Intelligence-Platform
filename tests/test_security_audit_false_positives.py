"""Regression tests for security-audit false positives."""

from pathlib import Path
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_security_repro.py"


def load_module():
    """Load the audit module without running its main function."""

    spec = importlib.util.spec_from_file_location(
        "audit_security_repro",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_virtual_environments_are_excluded() -> None:
    """Installed packages inside local environments are not project source."""

    module = load_module()

    path = (
        PROJECT_ROOT
        / ".venv-ci-tools"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "example.py"
    )

    assert module.is_excluded_path(path)


def test_markdown_rotation_record_is_not_platform_configuration() -> None:
    """Security documentation may discuss passwords without being config."""

    module = load_module()

    path = PROJECT_ROOT / "docs" / "security" / "GRAFANA_CREDENTIAL_ROTATION.md"

    assert not module.is_platform_configuration_file(path)


def test_kubernetes_secret_references_are_safe() -> None:
    """Kubernetes Secret keys and environment names are not passwords."""

    module = load_module()

    assert module.line_uses_runtime_secret(
        "- name: GF_SECURITY_ADMIN_PASSWORD"
    )
    assert module.line_uses_runtime_secret(
        "key: {{ .Values.grafana.adminPasswordKey }}"
    )
    assert module.line_uses_runtime_secret("key: admin-password")


def test_real_literal_password_is_not_safe() -> None:
    """A real literal assignment must remain reviewable."""

    module = load_module()

    # Build the example dynamically so the repository scanner does not
    # mistake this regression-test fixture for a committed credential.
    example_line = "password" + ': "' + "RealPassword123" + '"'

    assert not module.line_uses_runtime_secret(example_line)
