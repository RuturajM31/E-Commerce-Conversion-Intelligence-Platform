"""Tests for the repository security audit."""

from pathlib import Path
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_security_repro.py"


def load_module():
    """Load the audit module safely."""

    spec = importlib.util.spec_from_file_location("audit_security_repro", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_secret_references_are_recognised() -> None:
    """Runtime environment references are not hardcoded secrets."""

    module = load_module()
    assert module.looks_like_safe_placeholder("${GRAFANA_ADMIN_PASSWORD}")
    assert module.looks_like_safe_placeholder("supplied at runtime")


def test_real_password_is_not_a_placeholder() -> None:
    """A normal password-like value still requires review."""

    module = load_module()
    assert not module.looks_like_safe_placeholder("RealPassword123")


def test_repository_file_listing_includes_untracked_files() -> None:
    """The audit should see package files before they are staged."""

    module = load_module()
    paths = module.repository_files()
    relative = {str(path.relative_to(PROJECT_ROOT)) for path in paths}
    assert "requirements-ci.txt" in relative
