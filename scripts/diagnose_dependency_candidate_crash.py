"""Diagnose native crashes in the secure dependency candidate environment.

The parent process runs every risky step in a separate child process. This
prevents one segmentation fault from hiding which import, model-load, or
prediction stage failed.

The script does not modify live dependency pins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "qa" / "dependency_preflight"
JSON_PATH = REPORT_DIR / "dependency_candidate_crash_diagnosis.json"
MARKDOWN_PATH = REPORT_DIR / "dependency_candidate_crash_diagnosis.md"

CANDIDATE_PYTHON = (
    PROJECT_ROOT
    / ".venv-security-upgrade"
    / "bin"
    / "python"
)
CI_TOOLS_PYTHON = (
    PROJECT_ROOT
    / ".venv-ci-tools"
    / "bin"
    / "python"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "final_champion_model.joblib"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "metadata"
    / "final_champion_metadata.json"
)
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "visitor_features_sample.csv"
)


@dataclass
class StageResult:
    """One isolated diagnostic stage."""

    name: str
    status: str
    command: list[str]
    return_code: int | None
    signal_name: str | None
    stdout: str
    stderr: str
    note: str


def signal_name_from_return_code(return_code: int | None) -> str | None:
    """Convert a negative subprocess return code into a signal name."""

    if return_code is None or return_code >= 0:
        return None

    signal_number = -return_code

    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"SIGNAL_{signal_number}"


def run_stage(
    name: str,
    command: list[str],
    *,
    timeout_seconds: int = 120,
    note: str = "",
) -> StageResult:
    """Run one child process and capture native crashes safely."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["MPLBACKEND"] = "Agg"

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )

        return_code = completed.returncode
        signal_name = signal_name_from_return_code(return_code)

        if return_code == 0:
            status = "PASS"
        elif signal_name:
            status = "CRASH"
        else:
            status = "FAIL"

        return StageResult(
            name=name,
            status=status,
            command=command,
            return_code=return_code,
            signal_name=signal_name,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            note=note,
        )

    except subprocess.TimeoutExpired as error:
        return StageResult(
            name=name,
            status="TIMEOUT",
            command=command,
            return_code=None,
            signal_name=None,
            stdout=(error.stdout or "").strip(),
            stderr=(error.stderr or "").strip(),
            note=(
                f"{note} Timed out after {timeout_seconds} seconds."
            ).strip(),
        )


def python_code_stage(
    name: str,
    python_path: Path,
    code: str,
    *,
    note: str = "",
) -> StageResult:
    """Run one Python snippet with faulthandler enabled."""

    return run_stage(
        name,
        [
            str(python_path),
            "-X",
            "faulthandler",
            "-c",
            code,
        ],
        note=note,
    )


def build_import_stages() -> list[StageResult]:
    """Test candidate imports in small groups."""

    stages: list[StageResult] = []

    groups = {
        "candidate_import_numeric_stack": """
import numpy
import pandas
import scipy
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
""",
        "candidate_import_sklearn_joblib": """
import sklearn
import joblib
print("scikit-learn", sklearn.__version__)
print("joblib", joblib.__version__)
""",
        "candidate_import_xgboost": """
import xgboost
print("xgboost", xgboost.__version__)
""",
        "candidate_import_pyarrow": """
import pyarrow
print("pyarrow", pyarrow.__version__)
""",
        "candidate_import_streamlit_pillow": """
import streamlit
from PIL import Image
import PIL
print("streamlit", streamlit.__version__)
print("pillow", PIL.__version__)
print("image-class", Image.Image.__name__)
""",
    }

    for name, code in groups.items():
        stages.append(
            python_code_stage(
                name,
                CANDIDATE_PYTHON,
                code,
                note="Isolates binary and import compatibility.",
            )
        )

    return stages


def build_model_stages() -> list[StageResult]:
    """Test metadata, current model, candidate load, and prediction."""

    metadata_code = f"""
import json
from pathlib import Path
path = Path({str(METADATA_PATH)!r})
data = json.loads(path.read_text(encoding="utf-8"))
print("metadata-loaded", path)
print("feature-count", len(data.get("feature_columns", [])))
print("model-name", data.get("model_name"))
"""

    current_model_code = f"""
import joblib
from pathlib import Path
path = Path({str(MODEL_PATH)!r})
print("loading-current-model", path, flush=True)
model = joblib.load(path)
print("current-model-type", type(model).__module__, type(model).__name__)
"""

    candidate_model_code = f"""
import faulthandler
faulthandler.enable(all_threads=True)
import joblib
from pathlib import Path
path = Path({str(MODEL_PATH)!r})
print("loading-candidate-model", path, flush=True)
model = joblib.load(path)
print("candidate-model-type", type(model).__module__, type(model).__name__)
"""

    candidate_prediction_code = f"""
import faulthandler
faulthandler.enable(all_threads=True)
import json
import joblib
import pandas as pd
from pathlib import Path

model_path = Path({str(MODEL_PATH)!r})
metadata_path = Path({str(METADATA_PATH)!r})
sample_path = Path({str(SAMPLE_PATH)!r})

print("prediction-stage-start", flush=True)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
features = metadata.get("feature_columns", [])
print("features", features, flush=True)

model = joblib.load(model_path)
print("model-loaded", type(model).__name__, flush=True)

sample = pd.read_csv(sample_path)
missing = [name for name in features if name not in sample.columns]
if missing:
    raise RuntimeError("Missing sample columns: " + ", ".join(missing))

probabilities = model.predict_proba(sample[features].head(5))[:, 1]
print("probabilities", probabilities.tolist(), flush=True)
"""

    return [
        python_code_stage(
            "metadata_load",
            Path(sys.executable),
            metadata_code,
            note="Reads JSON only; no model deserialisation.",
        ),
        python_code_stage(
            "current_environment_model_load",
            Path(sys.executable),
            current_model_code,
            note=(
                "Confirms whether the existing model artifact is healthy "
                "in the current environment."
            ),
        ),
        python_code_stage(
            "candidate_environment_model_load",
            CANDIDATE_PYTHON,
            candidate_model_code,
            note=(
                "Isolates candidate-environment model deserialisation."
            ),
        ),
        python_code_stage(
            "candidate_environment_prediction",
            CANDIDATE_PYTHON,
            candidate_prediction_code,
            note=(
                "Runs only if the child process can load the model."
            ),
        ),
    ]


def build_test_stage() -> StageResult:
    """Run focused tests that do not require the production model."""

    candidate_tests = [
        "tests/test_app_utils.py",
        "tests/test_business_rules.py",
        "tests/test_smoke_training.py",
        "tests/test_streamlit_chart_coverage.py",
    ]

    existing = [
        test_path
        for test_path in candidate_tests
        if (PROJECT_ROOT / test_path).exists()
    ]

    if not existing:
        return StageResult(
            name="candidate_focused_tests",
            status="SKIP",
            command=[],
            return_code=None,
            signal_name=None,
            stdout="",
            stderr="",
            note="No focused non-model test files were found.",
        )

    return run_stage(
        "candidate_focused_tests",
        [
            str(CANDIDATE_PYTHON),
            "-m",
            "pytest",
            *existing,
            "-q",
            "-ra",
            "-p",
            "no:cacheprovider",
        ],
        timeout_seconds=300,
        note="Excludes production-model loading until crash source is known.",
    )


def build_pip_audit_stages() -> list[StageResult]:
    """Run both candidate vulnerability scans."""

    if not CI_TOOLS_PYTHON.exists():
        return [
            StageResult(
                name="candidate_main_pip_audit",
                status="SKIP",
                command=[],
                return_code=None,
                signal_name=None,
                stdout="",
                stderr="",
                note=".venv-ci-tools is not available.",
            )
        ]

    candidate_files = {
        "candidate_main_pip_audit": (
            REPORT_DIR / "requirements-main-secure-candidate.txt"
        ),
        "candidate_app_pip_audit": (
            REPORT_DIR / "requirements-app-secure-candidate.txt"
        ),
    }

    stages: list[StageResult] = []

    for name, requirement_path in candidate_files.items():
        stages.append(
            run_stage(
                name,
                [
                    str(CI_TOOLS_PYTHON),
                    "-m",
                    "pip_audit",
                    "-r",
                    str(requirement_path),
                    "--progress-spinner",
                    "off",
                ],
                timeout_seconds=600,
                note="Checks the secure candidate requirement set.",
            )
        )

    return stages


def run_streamlit_health_check() -> StageResult:
    """Start Streamlit and verify the local health endpoint."""

    streamlit_path = (
        PROJECT_ROOT
        / ".venv-security-upgrade"
        / "bin"
        / "streamlit"
    )
    app_path = PROJECT_ROOT / "app" / "Executive_Overview.py"
    log_path = REPORT_DIR / "streamlit_candidate_startup.log"

    if not streamlit_path.exists():
        return StageResult(
            name="candidate_streamlit_startup",
            status="SKIP",
            command=[],
            return_code=None,
            signal_name=None,
            stdout="",
            stderr="",
            note="Candidate Streamlit executable is unavailable.",
        )

    if not app_path.exists():
        return StageResult(
            name="candidate_streamlit_startup",
            status="SKIP",
            command=[],
            return_code=None,
            signal_name=None,
            stdout="",
            stderr="",
            note="Executive Overview app entry point is unavailable.",
        )

    command = [
        str(streamlit_path),
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        "8765",
    ]

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    healthy = False
    return_code: int | None = None

    try:
        for _ in range(30):
            return_code = process.poll()

            if return_code is not None:
                break

            try:
                with urlopen(
                    "http://127.0.0.1:8765/_stcore/health",
                    timeout=2,
                ) as response:
                    if response.status == 200:
                        healthy = True
                        break
            except (URLError, TimeoutError):
                pass

            time.sleep(1)

    finally:
        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    return_code = process.returncode
    signal_name = signal_name_from_return_code(return_code)
    log_text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if healthy:
        status = "PASS"
        note = "Streamlit health endpoint returned HTTP 200."
    elif signal_name:
        status = "CRASH"
        note = "Streamlit process exited because of a native signal."
    else:
        status = "FAIL"
        note = "Streamlit did not become healthy within 30 seconds."

    return StageResult(
        name="candidate_streamlit_startup",
        status=status,
        command=command,
        return_code=return_code,
        signal_name=signal_name,
        stdout="",
        stderr="\n".join(log_text.splitlines()[-60:]),
        note=note,
    )


def derive_conclusion(stages: list[StageResult]) -> str:
    """Summarise the most likely failure boundary without guessing."""

    by_name = {stage.name: stage for stage in stages}

    import_failures = [
        stage
        for stage in stages
        if stage.name.startswith("candidate_import_")
        and stage.status != "PASS"
    ]

    if import_failures:
        names = ", ".join(stage.name for stage in import_failures)
        return (
            "The candidate environment has an import or binary compatibility "
            f"failure before model loading: {names}."
        )

    current_load = by_name.get("current_environment_model_load")
    candidate_load = by_name.get("candidate_environment_model_load")
    candidate_prediction = by_name.get("candidate_environment_prediction")

    if (
        current_load
        and current_load.status == "PASS"
        and candidate_load
        and candidate_load.status == "CRASH"
    ):
        return (
            "The existing model artifact is healthy in the current "
            "environment but crashes during candidate-environment "
            "deserialisation. Do not adopt the candidate scikit-learn pin "
            "without retraining or recreating the persisted model."
        )

    if (
        candidate_load
        and candidate_load.status == "PASS"
        and candidate_prediction
        and candidate_prediction.status != "PASS"
    ):
        return (
            "The candidate environment can deserialize the model, but "
            "prediction is not yet compatible. Live pins must remain "
            "unchanged until that prediction failure is fixed."
        )

    if candidate_prediction and candidate_prediction.status == "PASS":
        return (
            "Candidate model loading and prediction passed. Dependency "
            "adoption can proceed only after the remaining tests, audits, "
            "and Streamlit startup also pass."
        )

    return (
        "The diagnostic did not establish a safe upgrade path. Review the "
        "individual stage results before changing live dependency pins."
    )


def write_reports(stages: list[StageResult], conclusion: str) -> None:
    """Write machine-readable and human-readable evidence."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "current_python": sys.executable,
        "candidate_python": str(CANDIDATE_PYTHON),
        "conclusion": conclusion,
        "stages": [asdict(stage) for stage in stages],
    }

    JSON_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Dependency Candidate Crash Diagnosis",
        "",
        conclusion,
        "",
    ]

    for stage in stages:
        lines.extend(
            [
                f"## {stage.name}",
                "",
                f"- Status: `{stage.status}`",
                f"- Return code: `{stage.return_code}`",
                f"- Signal: `{stage.signal_name}`",
                f"- Note: {stage.note}",
                "",
                "### Standard output",
                "",
                "```text",
                stage.stdout or "(none)",
                "```",
                "",
                "### Standard error",
                "",
                "```text",
                stage.stderr or "(none)",
                "```",
                "",
            ]
        )

    MARKDOWN_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run every diagnostic stage."""

    if not CANDIDATE_PYTHON.exists():
        raise FileNotFoundError(
            "Candidate environment not found: "
            f"{CANDIDATE_PYTHON}"
        )

    stages: list[StageResult] = []

    stages.extend(build_import_stages())
    stages.extend(build_model_stages())
    stages.append(build_test_stage())
    stages.extend(build_pip_audit_stages())
    stages.append(run_streamlit_health_check())

    conclusion = derive_conclusion(stages)
    write_reports(stages, conclusion)

    print("\n=== DEPENDENCY CANDIDATE CRASH DIAGNOSIS ===")
    for stage in stages:
        signal_text = (
            f" | signal={stage.signal_name}"
            if stage.signal_name
            else ""
        )
        print(
            f"{stage.status:7} | {stage.name}"
            f" | return_code={stage.return_code}"
            f"{signal_text}"
        )

    print("\nCONCLUSION")
    print(conclusion)
    print(f"\nReport: {MARKDOWN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON:   {JSON_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
