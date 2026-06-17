from pathlib import Path


def test_core_project_folders_exist():
    """The project should keep code, app, reports, models, and data separated."""

    required_folders = [
        "app",
        "app/pages",
        "src",
        "src/models",
        "src/data",
        "reports",
        "models",
        "data",
    ]

    missing = [folder for folder in required_folders if not Path(folder).exists()]

    assert not missing, f"Missing required project folders: {missing}"


def test_main_streamlit_entrypoint_exists():
    """The Streamlit app should have one clear entrypoint."""

    assert Path("app/Executive_Overview.py").exists()


def test_key_streamlit_pages_exist():
    """The main dashboard pages should be present."""

    expected_pages = [
        "1_Visitor_Intent_Predictor.py",
        "2_Batch_Scoring.py",
        "3_Model_Benchmark_Selection.py",
        "4_Business_KPI_Forecasting.py",
        "5_Anomaly_Outlier.py",
        "6_Monitoring_Drift_Health.py",
        "7_MLOps_Architecture.py",
        "8_MVD_Coverage_Proof.py",
    ]

    existing_pages = {path.name for path in Path("app/pages").glob("*.py")}
    missing = [page for page in expected_pages if page not in existing_pages]

    assert not missing, f"Missing Streamlit pages: {missing}"
