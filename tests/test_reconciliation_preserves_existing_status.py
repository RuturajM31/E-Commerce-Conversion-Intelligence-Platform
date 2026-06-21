"""Check that completed rows outside the update stay completed."""

from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "reconcile_remediation_matrices.py"


def load_module():
    """Load the reconciliation script without running its main function."""

    spec = importlib.util.spec_from_file_location(
        "reconcile_remediation_matrices",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_status_map_reads_completed_qa_row() -> None:
    """A completed QA row remains valid existing evidence."""

    module = load_module()

    text = (
        "| `QA-04` | Full normal pytest suite passes | "
        "Approved skips only | `COMPLETED` |\n"
    )

    assert module.zero_status_map(text)["QA-04"] == "COMPLETED"
