"""Tests for secure dependency candidate generation."""

from pathlib import Path
import importlib.util
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "build_dependency_upgrade_candidates.py"
)


def load_module():
    """Load the candidate builder."""

    spec = importlib.util.spec_from_file_location(
        "build_dependency_upgrade_candidates",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_candidate_uses_pyarrow_23_security_fix(
    tmp_path: Path,
) -> None:
    """The candidate must replace vulnerable PyArrow versions."""

    module = load_module()

    source = tmp_path / "requirements.txt"
    output = tmp_path / "candidate.txt"

    source.write_text(
        "pyarrow==14.0.2\n"
        "scikit-learn==1.4.2\n"
        "streamlit==1.37.1\n"
        "pillow==10.4.0\n",
        encoding="utf-8",
    )

    module.build_candidate(source, output)

    text = output.read_text(encoding="utf-8")

    assert "pyarrow==23.0.1" in text
    assert "scikit-learn==1.5.0" in text
    assert "streamlit==1.54.0" in text
    assert "pillow==12.2.0" in text

    assert "pyarrow==14.0.2" not in text
