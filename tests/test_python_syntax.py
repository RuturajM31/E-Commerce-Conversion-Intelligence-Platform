from pathlib import Path
import py_compile


def test_key_python_files_compile():
    """Key project Python files should compile without syntax errors."""

    files = [
        "app/Executive_Overview.py",
        "app/app_utils.py",
        "src/models/model_config.py",
        "src/models/model_evaluation.py",
        "src/models/model_selection.py",
        "src/models/run_model_selection.py",
        "src/models/finalize_true_champion.py",
        "src/models/build_mvd_method_coverage.py",
    ]

    for file in files:
        path = Path(file)
        assert path.exists(), f"Missing Python file: {file}"
        py_compile.compile(str(path), doraise=True)
