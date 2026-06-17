# 7_MLOps_Architecture.py
# Streamlit page for the MLOps architecture story.
#
# Purpose:
#   Explain how this ML + BI project can run like a production system.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Project files and target MLOps files
#   4. Helper functions
#   5. Charts
#   6. Load champion context
#   7. Sidebar and header
#   8. Architecture flow
#   9. Current implementation status
#   10. Monitoring and deployment path
#   11. Interview story and evidence tables

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from app.app_utils import (
    escape_text,
    format_percent,
    get_best_threshold,
    get_champion_metrics,
    get_champion_model_name,
    inject_global_css,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="MLOps Architecture",
    page_icon="☸️",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .architecture-hero {
            padding: 36px 40px;
            border-radius: 34px;
            background:
                radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.28), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(34, 197, 94, 0.20), transparent 34%),
                radial-gradient(circle at 58% 16%, rgba(168, 85, 247, 0.18), transparent 28%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.50);
            margin-bottom: 22px;
        }

        .architecture-title {
            color: #F8FAFC;
            font-size: 2.40rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .architecture-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1080px;
        }

        .system-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 182px;
            margin-bottom: 18px;
        }

        .system-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .system-value {
            color: #F8FAFC;
            font-size: 1.42rem;
            line-height: 1.15;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .system-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.50;
        }

        .section-kicker {
            color: #7DD3FC;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-top: 16px;
            margin-bottom: 4px;
        }

        .section-title {
            color: #F8FAFC;
            font-size: 1.45rem;
            font-weight: 950;
            margin-bottom: 12px;
        }

        .story-card {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(2, 6, 23, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.34);
            margin-bottom: 18px;
        }

        .story-title {
            color: #F8FAFC;
            font-size: 1.24rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .story-text {
            color: #CBD5E1;
            font-size: 0.94rem;
            line-height: 1.58;
        }

        .flow-wrapper {
            display: grid;
            grid-template-columns: repeat(5, minmax(140px, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }

        .flow-box {
            padding: 20px 18px;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
            min-height: 160px;
        }

        .flow-icon {
            font-size: 1.55rem;
            margin-bottom: 8px;
        }

        .flow-title {
            color: #F8FAFC;
            font-size: 0.98rem;
            font-weight: 900;
            margin-bottom: 7px;
        }

        .flow-text {
            color: #CBD5E1;
            font-size: 0.82rem;
            line-height: 1.46;
        }

        .status-ok {
            color: #86EFAC;
            font-weight: 900;
        }

        .status-planned {
            color: #FDE68A;
            font-weight: 900;
        }

        @media (max-width: 1100px) {
            .flow-wrapper {
                grid-template-columns: repeat(2, minmax(140px, 1fr));
            }
        }
    </style>
    """
)


# --------------------------------------------------
# 3. PROJECT FILES AND TARGET MLOPS FILES
# --------------------------------------------------

PROJECT_FILES: Dict[str, Path] = {
    "Feature table": Path("data/processed/visitor_features.csv"),
    "Final champion model": Path("models/trained/final_champion_model.joblib"),
    "Final champion metadata": Path("models/metadata/final_champion_metadata.json"),
    "Champion visitor scores": Path("data/processed/champion_visitor_scores.csv"),
    "Forecast outputs": Path("reports/tables/business_forecast_future.csv"),
    "Anomaly outputs": Path("reports/tables/anomaly_summary.csv"),
    "MVD coverage matrix": Path("reports/tables/mvd_method_coverage_matrix.csv"),
    "Single prediction log": Path("monitoring/prediction_logs/prediction_log.csv"),
    "Batch scoring log": Path("monitoring/prediction_logs/batch_scoring_log.csv"),
}

MLOPS_TARGET_FILES: Dict[str, Path] = {
    "requirements.txt": Path("requirements.txt"),
    "README.md": Path("README.md"),
    ".env.example": Path(".env.example"),
    "Makefile": Path("Makefile"),
    "Dockerfile": Path("Dockerfile"),
    "docker-compose.yml": Path("docker-compose.yml"),
    "GitHub Actions CI": Path(".github/workflows/ci.yml"),
    "Prometheus config": Path("monitoring/prometheus/prometheus.yml"),
    "Alert rules": Path("monitoring/prometheus/alert_rules.yml"),
    "Alertmanager config": Path("monitoring/alertmanager/alertmanager.yml"),
    "Grafana dashboards": Path("monitoring/grafana/dashboards"),
    "Kubernetes manifests": Path("k8s"),
    "Helm chart": Path("helm/ecommerce-conversion-platform"),
}


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def file_status(path: Path) -> str:
    """Return OK if a file/folder exists, otherwise Planned."""

    return "OK" if path.exists() else "Planned"


def count_existing(items: Dict[str, Path]) -> int:
    """Count how many files/folders exist."""

    return sum(1 for path in items.values() if path.exists())


def build_status_table(items: Dict[str, Path]) -> pd.DataFrame:
    """Build a readable file-status table."""

    rows = []

    for component, path in items.items():
        rows.append(
            {
                "Component": component,
                "Path": str(path),
                "Status": file_status(path),
            }
        )

    return pd.DataFrame(rows)


def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load optional CSV if it exists."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def status_html(status: str) -> str:
    """Create colored HTML status text."""

    class_name = "status-ok" if status == "OK" else "status-planned"

    return f'<span class="{class_name}">{escape_text(status)}</span>'


def format_count(value: float) -> str:
    """Format counts for dashboard text."""

    if pd.isna(value):
        return "NA"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"




def render_chart_explanation(title: str, shows: str, conclusion: str) -> None:
    """Add a compact explanation and conclusion under a chart."""

    render_html(
        (
            f'<div class="story-card">'
            f'<div class="story-title">{escape_text(title)}</div>'
            f'<div class="story-text">'
            f'<b>What this chart shows:</b> {escape_text(shows)}<br><br>'
            f'<b>Conclusion:</b> {escape_text(conclusion)}'
            f'</div></div>'
        )
    )

# --------------------------------------------------
# 5. CHART FUNCTIONS
# --------------------------------------------------

def build_architecture_progress_chart(
    project_done: int,
    project_total: int,
    mlops_done: int,
    mlops_total: int,
):
    """Build implemented vs planned architecture chart."""

    if not PLOTLY_AVAILABLE:
        return None

    chart_data = pd.DataFrame(
        [
            {
                "Area": "ML product core",
                "Implemented": project_done,
                "Planned / missing": project_total - project_done,
            },
            {
                "Area": "MLOps package",
                "Implemented": mlops_done,
                "Planned / missing": mlops_total - mlops_done,
            },
        ]
    )

    long_data = chart_data.melt(
        id_vars="Area",
        var_name="Status",
        value_name="Components",
    )

    fig = px.bar(
        long_data,
        x="Area",
        y="Components",
        color="Status",
        text="Components",
        title="Project readiness: implemented vs planned components",
        barmode="stack",
    )

    fig.update_traces(
        textposition="inside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Architecture area",
        yaxis_title="Component count",
        legend_title="Status",
        title_font=dict(size=20),
    )

    return fig


# --------------------------------------------------
# 6. LOAD CHAMPION CONTEXT
# --------------------------------------------------

champion_model_name = get_champion_model_name()
best_threshold = get_best_threshold()
champion_metrics = get_champion_metrics()

precision = float(champion_metrics.get("best_precision", 0))
recall = float(champion_metrics.get("best_recall", 0))
pr_auc = float(champion_metrics.get("pr_auc", 0))

anomaly_summary = load_optional_csv(Path("reports/tables/anomaly_summary.csv"))
forecast_future = load_optional_csv(Path("reports/tables/business_forecast_future.csv"))

project_status = build_status_table(PROJECT_FILES)
mlops_status = build_status_table(MLOPS_TARGET_FILES)

project_done = count_existing(PROJECT_FILES)
project_total = len(PROJECT_FILES)
mlops_done = count_existing(MLOPS_TARGET_FILES)
mlops_total = len(MLOPS_TARGET_FILES)


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## ☸️ MLOps Architecture")
st.sidebar.caption("Production story")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Core files:** {project_done}/{project_total}")
st.sidebar.markdown(f"**MLOps files:** {mlops_done}/{mlops_total}")
st.sidebar.markdown("---")
st.sidebar.caption("Shows the path from local ML app to production-style monitoring and deployment.")


# --------------------------------------------------
# 8. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="architecture-hero">'
        f'<div class="eyebrow">MLOps and production architecture</div>'
        f'<div class="architecture-title">From ML project to production-style intelligence platform</div>'
        f'<div class="architecture-subtitle">'
        f'This page explains how the project is structured as a production-style system: data pipeline, '
        f'final champion model, Streamlit app, prediction logs, monitoring, alerts, Docker path, and Kubernetes/Helm deployment plan.'
        f'</div><br>'
        f'<span class="success-pill">Final champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Monitoring + alerting path included</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 9. SYSTEM KPI CARDS
# --------------------------------------------------

card_1, card_2, card_3, card_4 = st.columns(4)

with card_1:
    render_html(
        (
            f'<div class="system-card">'
            f'<div class="system-label">ML product core</div>'
            f'<div class="system-value">{project_done}/{project_total} ready</div>'
            f'<div class="system-help">Data, model, app outputs, monitoring logs, and proof files.</div>'
            f'</div>'
        )
    )

with card_2:
    render_html(
        (
            f'<div class="system-card">'
            f'<div class="system-label">Final champion</div>'
            f'<div class="system-value">{escape_text(champion_model_name)}</div>'
            f'<div class="system-help">PR-AUC {pr_auc:.3f}, precision {format_percent(precision)}, recall {format_percent(recall)}.</div>'
            f'</div>'
        )
    )

with card_3:
    render_html(
        (
            f'<div class="system-card">'
            f'<div class="system-label">MLOps package</div>'
            f'<div class="system-value">{mlops_done}/{mlops_total} ready</div>'
            f'<div class="system-help">README, Docker, CI/CD, monitoring configs, Kubernetes, and Helm path.</div>'
            f'</div>'
        )
    )

with card_4:
    render_html(
        (
            f'<div class="system-card">'
            f'<div class="system-label">Deployment target</div>'
            f'<div class="system-value">Local → Docker → K8s</div>'
            f'<div class="system-help">Start simple locally, then package and deploy with production patterns.</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 10. ARCHITECTURE FLOW
# --------------------------------------------------

render_html('<div class="section-kicker">System flow</div><div class="section-title">How data becomes a business decision</div>')

render_html(
    """
    <div class="flow-wrapper">
        <div class="flow-box">
            <div class="flow-icon">📥</div>
            <div class="flow-title">Raw events</div>
            <div class="flow-text">RetailRocket behaviour events enter the data layer.</div>
        </div>
        <div class="flow-box">
            <div class="flow-icon">🧱</div>
            <div class="flow-title">Feature pipeline</div>
            <div class="flow-text">Visitor-level features are built with safe grain and no target leakage.</div>
        </div>
        <div class="flow-box">
            <div class="flow-icon">🏆</div>
            <div class="flow-title">Champion model</div>
            <div class="flow-text">Final tuned model scores visitor purchase intent.</div>
        </div>
        <div class="flow-box">
            <div class="flow-icon">📊</div>
            <div class="flow-title">BI app</div>
            <div class="flow-text">Streamlit shows scoring, benchmarks, forecasts, anomaly review, and monitoring.</div>
        </div>
        <div class="flow-box">
            <div class="flow-icon">🚨</div>
            <div class="flow-title">Monitoring</div>
            <div class="flow-text">Prediction logs, drift checks, Prometheus/Grafana path, and alerts complete the loop.</div>
        </div>
    </div>
    """
)


# --------------------------------------------------
# 11. IMPLEMENTATION STATUS
# --------------------------------------------------

render_html('<div class="section-kicker">Implementation status</div><div class="section-title">What is implemented and what is planned</div>')

chart = build_architecture_progress_chart(
    project_done=project_done,
    project_total=project_total,
    mlops_done=mlops_done,
    mlops_total=mlops_total,
)

if chart is not None:
    st.plotly_chart(chart, use_container_width=True)
    render_chart_explanation(
        "Chart explanation: architecture readiness",
        "It compares implemented ML product components with planned MLOps package components.",
        "The ML product core is already built around data, model, app, logs, monitoring, and proof outputs. Remaining MLOps files complete the production packaging story.",
    )
else:
    st.info("Plotly is not installed. Status tables are shown below.")


# --------------------------------------------------
# 12. MONITORING AND DEPLOYMENT STORY
# --------------------------------------------------

story_1, story_2 = st.columns(2)

with story_1:
    render_html(
        '<div class="story-card">'
        '<div class="story-title">Monitoring stack story</div>'
        '<div class="story-text">'
        'The app writes prediction logs. The monitoring page reads logs and compares live scores with baseline scores. '
        'Prometheus can scrape app/system metrics, Grafana can show dashboards, and Alertmanager can notify when drift or error rates become risky.'
        '</div>'
        '</div>'
    )

with story_2:
    render_html(
        '<div class="story-card">'
        '<div class="story-title">Deployment path</div>'
        '<div class="story-text">'
        'The practical path is: run locally first, package with Docker, orchestrate with docker-compose, then document Kubernetes manifests and a Helm chart for production-style deployment.'
        '</div>'
        '</div>'
    )


# --------------------------------------------------
# 13. INTERVIEW STORY
# --------------------------------------------------

render_html(
    '<div class="story-card">'
    '<div class="story-title">Interview-ready summary</div>'
    '<div class="story-text">'
    'I built this as a conversion-intelligence platform, not only a notebook. '
    'It covers data engineering, model selection, final champion hardening, BI dashboards, forecasting, anomaly review, monitoring, and a clear production deployment path.'
    '</div>'
    '</div>'
)


# --------------------------------------------------
# 14. EVIDENCE TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Evidence tables</div><div class="section-title">Architecture readiness checklist</div>')

tab_1, tab_2, tab_3 = st.tabs(
    [
        "Implemented product files",
        "MLOps target files",
        "Supporting outputs",
    ]
)

with tab_1:
    st.dataframe(
        project_status,
        use_container_width=True,
        hide_index=True,
    )

with tab_2:
    st.dataframe(
        mlops_status,
        use_container_width=True,
        hide_index=True,
    )

with tab_3:
    output_rows = [
        {
            "Output": "Anomaly summary",
            "Rows": len(anomaly_summary),
            "Status": file_status(Path("reports/tables/anomaly_summary.csv")),
        },
        {
            "Output": "Forecast future",
            "Rows": len(forecast_future),
            "Status": file_status(Path("reports/tables/business_forecast_future.csv")),
        },
        {
            "Output": "Prediction log",
            "Rows": len(load_optional_csv(Path("monitoring/prediction_logs/prediction_log.csv"))),
            "Status": file_status(Path("monitoring/prediction_logs/prediction_log.csv")),
        },
        {
            "Output": "Batch scoring log",
            "Rows": len(load_optional_csv(Path("monitoring/prediction_logs/batch_scoring_log.csv"))),
            "Status": file_status(Path("monitoring/prediction_logs/batch_scoring_log.csv")),
        },
    ]

    st.dataframe(
        pd.DataFrame(output_rows),
        use_container_width=True,
        hide_index=True,
    )
