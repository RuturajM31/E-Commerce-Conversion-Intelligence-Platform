"""Tests for representative sample-data smoke training."""

from pathlib import Path
import importlib.util
import json
import sys

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_smoke_training.py"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "visitor_features_sample.csv"


def load_module():
    """Load the smoke-training module."""

    spec = importlib.util.spec_from_file_location("run_smoke_training", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_data_has_both_target_classes() -> None:
    """The sample must support a real classification fit."""

    module = load_module()
    data = module.load_sample_data(SAMPLE_PATH)
    assert set(data["converted"].unique()) == {0, 1}
    assert data["visitorid"].is_unique
    assert len(data) == 120


def test_smoke_training_writes_valid_evidence(tmp_path: Path) -> None:
    """The small training path must fit and save metrics."""

    module = load_module()
    output_path = tmp_path / "smoke_result.json"
    result = module.run_smoke_training(
        data_path=SAMPLE_PATH,
        output_path=output_path,
    )

    assert result["status"] == "PASS"
    assert result["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["pr_auc"] <= 1
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["rows"] == 120


def test_script_bootstraps_project_root_before_src_import() -> None:
    """Direct execution must be able to import the project package."""

    text = SCRIPT_PATH.read_text(encoding="utf-8")

    bootstrap_position = text.index("sys.path.insert")
    src_import_position = text.index("from src.data.feature_engineering")

    assert bootstrap_position < src_import_position
