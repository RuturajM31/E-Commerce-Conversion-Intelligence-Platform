"""Master closure controls for Packages 4-9."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "streamlit"
MATRIX = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.csv"
IMPLEMENTATION_MAP = DOCS / "CLOSURE_SPRINT_IMPLEMENTATION_MAP.csv"
MANIFEST = DOCS / "STREAMLIT_VISUAL_ENHANCEMENT_MATRIX_MANIFEST.json"


EXPECTED_FINAL_COUNTS = {
    "VERIFIED": 177,
    "CONDITIONAL": 25,
    "EXCLUDED": 2,
}


EXPECTED_EXTENSIONS = {
    "app/pages/1_Visitor_Intent_Predictor.py": (
        "render_visitor_similarity_explainability",
    ),
    "app/pages/2_Batch_Scoring.py": (
        "render_batch_campaign_intelligence",
    ),
    "app/pages/4_Business_KPI_Forecasting.py": (
        "render_forecast_decision_intelligence",
    ),
    "app/pages/5_Anomaly_Outlier.py": (
        "render_anomaly_investigation_intelligence",
    ),
    "app/pages/7_MLOps_Architecture.py": (
        "render_architecture_governance_intelligence",
    ),
}


NEW_INTELLIGENCE_FILES = (
    "app/pages/9_Customer_Segmentation_Journey.py",
    "app/ui/architecture_intelligence.py",
    "app/ui/campaign_intelligence.py",
    "app/ui/closure_core.py",
    "app/ui/customer_segmentation_journey.py",
    "app/ui/model_decision_intelligence.py",
    "app/ui/operations_intelligence.py",
    "app/ui/visitor_similarity_explainability.py",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV without changing its row grain."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_matrix_rows_have_terminal_statuses() -> None:
    """Every row must end VERIFIED, CONDITIONAL, or EXCLUDED."""

    rows = read_csv(MATRIX)
    counts = Counter(row["Status"] for row in rows)

    assert len(rows) == 204
    assert counts == EXPECTED_FINAL_COUNTS
    assert not ({"PLANNED", "IN PROGRESS", "BLOCKED"} & set(counts))


def test_conditional_rows_match_the_implementation_map() -> None:
    """Missing evidence must remain conditional rather than invented."""

    rows = {row["ID"]: row for row in read_csv(MATRIX)}
    implementation = read_csv(IMPLEMENTATION_MAP)

    conditional_ids = {
        row["ID"]
        for row in implementation
        if row["Closure_State_Before_Final_QA"] == "CONDITIONAL_PENDING_QA"
    }
    excluded_ids = {
        row["ID"]
        for row in implementation
        if row["Closure_State_Before_Final_QA"] == "EXCLUDED_ACCEPTED"
    }

    assert len(conditional_ids) == 25
    assert len(excluded_ids) == 2
    assert {identifier for identifier, row in rows.items() if row["Status"] == "CONDITIONAL"} == conditional_ids
    assert {identifier for identifier, row in rows.items() if row["Status"] == "EXCLUDED"} == excluded_ids


def test_every_existing_page_has_its_isolated_extension() -> None:
    """Current pages remain intact and call one isolated closure renderer."""

    for relative, required_tokens in EXPECTED_EXTENSIONS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in text, relative


def test_new_intelligence_modules_and_page_exist() -> None:
    """All coordinated closure modules must be present."""

    missing = [relative for relative in NEW_INTELLIGENCE_FILES if not (ROOT / relative).is_file()]
    assert not missing


def test_manifest_records_all_six_remaining_packages() -> None:
    """Packages 4-9 must be represented in the final manifest."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["statuses"] == {
        "CONDITIONAL": 25,
        "EXCLUDED": 2,
        "VERIFIED": 177,
    }
    for package_number in range(4, 9):
        checkpoint = manifest["package_checkpoints"][f"package_{package_number:02d}"]
        assert checkpoint["state"] == "verified_by_master_closure"

    package_09 = manifest["package_checkpoints"]["package_09"]
    assert package_09["state"] == "repository_ready_for_streamlit_community_cloud"
    assert package_09["deployment_branch"] == "main"
    assert package_09["entrypoint"] == "app/Executive_Overview.py"
    assert package_09["python_version"] == "3.10"
    assert package_09["secrets_required"] is False


def test_streamlit_community_cloud_files_are_ready_and_truthful() -> None:
    """Persistent deployment readiness must be explicit and account-bound."""

    requirements = (ROOT / "app/requirements.txt").read_text(encoding="utf-8")
    documentation = (DOCS / "STREAMLIT_COMMUNITY_CLOUD_DEPLOYMENT.md").read_text(encoding="utf-8")

    for dependency in (
        "streamlit==1.54.0",
        "pandas==2.1.4",
        "scikit-learn==1.5.0",
        "xgboost==3.2.0",
        "plotly==5.24.1",
    ):
        assert dependency in requirements

    assert "branch `main`" in documentation
    assert "app/Executive_Overview.py" in documentation
    assert "Python `3.10`" in documentation
    assert "Secrets empty" in documentation
    assert "project owner" in documentation
    assert not (ROOT / "scripts/start_zero_cost_public_demo.sh").exists()
    assert not (ROOT / "scripts/stop_zero_cost_public_demo.sh").exists()


def test_final_review_pack_covers_all_ten_pages() -> None:
    """The readable review pack must cover every final application page."""

    text = (DOCS / "CLOSURE_SPRINT_REVIEW_PACK.md").read_text(encoding="utf-8")
    page_names = (
        "Executive Overview",
        "Visitor Intent Predictor",
        "Batch Scoring",
        "Model Benchmark Selection",
        "Business KPI Forecasting",
        "Anomaly Outlier",
        "Monitoring Drift Health",
        "MLOps Architecture",
        "ML Validation Evidence",
        "Customer Segmentation Journey",
    )

    for page_name in page_names:
        assert page_name in text

def test_standalone_pages_keep_governed_closure_without_duplicate_renderers() -> None:
    """Model and monitoring pages now own their governed implementations."""

    checks = {
        "app/pages/3_Model_Benchmark_Selection.py": (
            (
                "build_model_comparison_chart",
                "build_precision_recall_chart",
                "build_threshold_chart",
                "build_stability_chart",
                "holdout_lift",
            ),
            "render_model_decision_intelligence",
        ),
        "app/pages/6_Monitoring_Drift_Health.py": (
            (
                "MINIMUM_LIVE_SAMPLE = 30",
                "def calculate_psi(",
                "Insufficient live data",
                "One prediction cannot establish production drift.",
                "def parse_drift_report(",
            ),
            "render_monitoring_health_intelligence",
        ),
    }

    for relative, (required_tokens, forbidden_token) in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")

        for token in required_tokens:
            assert token in text, relative

        assert forbidden_token not in text, relative
