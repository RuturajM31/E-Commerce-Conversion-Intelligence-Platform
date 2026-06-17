from pathlib import Path

import pytest


PAGES_WITH_CHARTS = [
    Path("app/pages/3_Model_Benchmark_Selection.py"),
    Path("app/pages/4_Business_KPI_Forecasting.py"),
    Path("app/pages/5_Anomaly_Outlier.py"),
    Path("app/pages/6_Monitoring_Drift_Health.py"),
    Path("app/pages/7_MLOps_Architecture.py"),
    Path("app/pages/8_MVD_Coverage_Proof.py"),
]


def test_streamlit_charts_have_explanations():
    """Every Streamlit chart page should include chart explanations.

    This is a static chart-coverage test.
    True visual rendering tests require a browser tool such as Playwright and are heavier.
    """

    missing_pages = [path for path in PAGES_WITH_CHARTS if not path.exists()]

    if missing_pages:
        pytest.skip(f"Some chart pages are not available yet: {missing_pages}")

    for path in PAGES_WITH_CHARTS:
        text = path.read_text(encoding="utf-8")

        chart_count = text.count("st.plotly_chart")
        explanation_count = text.count("render_chart_explanation(") - 1

        assert chart_count > 0, f"No Plotly charts found in {path}"
        assert explanation_count >= chart_count, (
            f"{path} has {chart_count} charts but only {explanation_count} explanations."
        )


def test_streamlit_pages_compile():
    """Streamlit pages should compile before manual visual QA."""

    import py_compile

    for path in PAGES_WITH_CHARTS:
        if not path.exists():
            pytest.skip(f"Missing page: {path}")

        py_compile.compile(str(path), doraise=True)
