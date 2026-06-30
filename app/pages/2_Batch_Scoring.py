# 2_Batch_Scoring.py
# Streamlit page for batch campaign scoring.
#
# Purpose:
#   Score many visitors from a CSV file and create a campaign-ready audience list.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Load final champion context
#   4. Helper functions
#   5. Sidebar and header
#   6. Upload or use sample data
#   7. Validate and score batch
#   8. Show KPIs, charts, table, and download
#   9. Log batch summary for monitoring

from __future__ import annotations

# STREAMLIT CLOUD PATH BOOTSTRAP
# Streamlit executes page files from inside the app folder.
# Add the repository root so imports such as `app.*` and `src.*`
# work consistently locally and on Streamlit Community Cloud.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

while (
    _PROJECT_ROOT != _PROJECT_ROOT.parent
    and not (_PROJECT_ROOT / "src").is_dir()
):
    _PROJECT_ROOT = _PROJECT_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from app.app_utils import (
    assign_intent_segment,
    assign_recommended_action,
    escape_text,
    format_lift,
    format_percent,
    get_best_threshold,
    get_champion_metrics,
    get_champion_model_name,
    get_champion_track,
    inject_global_css,
    prepare_batch_model_input,
    predict_batch_purchase_intent,
    render_html,
)

# The project logger may exist.
# A fallback logger is included so this page does not crash if the logger changes.
try:
    from src.monitoring.prediction_logger import log_batch_summary
except Exception:
    log_batch_summary = None


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Batch Scoring",
    page_icon="📦",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .batch-hero {
            padding: 32px 36px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.26), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(34, 197, 94, 0.20), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .batch-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .batch-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1050px;
        }

        .workflow-card {
            padding: 20px 22px;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
            min-height: 140px;
            margin-bottom: 16px;
        }

        .workflow-icon {
            font-size: 1.55rem;
            margin-bottom: 8px;
        }

        .workflow-title {
            color: #F8FAFC;
            font-size: 1.00rem;
            font-weight: 900;
            margin-bottom: 7px;
        }

        .workflow-text {
            color: #CBD5E1;
            font-size: 0.86rem;
            line-height: 1.48;
        }

        .upload-panel {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), transparent 28%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            margin-bottom: 18px;
        }

        .section-kicker {
            color: #7DD3FC;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-top: 8px;
            margin-bottom: 4px;
        }

        .section-title {
            color: #F8FAFC;
            font-size: 1.45rem;
            font-weight: 950;
            margin-bottom: 12px;
        }

        .compact-note {
            color: #94A3B8;
            font-size: 0.84rem;
            line-height: 1.48;
            margin-bottom: 10px;
        }

        .result-banner {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(2, 6, 23, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.34);
            margin-top: 18px;
            margin-bottom: 20px;
        }

        .result-title {
            color: #F8FAFC;
            font-size: 1.35rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .result-text {
            color: #CBD5E1;
            font-size: 0.95rem;
            line-height: 1.58;
        }

        .stButton > button {
            border-radius: 16px !important;
            font-weight: 900 !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(125, 211, 252, 0.38) !important;
            background: linear-gradient(135deg, #0369A1, #059669) !important;
            box-shadow: 0 14px 34px rgba(14, 165, 233, 0.22) !important;
            min-height: 48px !important;
        }

        .stButton > button:hover {
            border: 1px solid rgba(186, 230, 253, 0.70) !important;
            background: linear-gradient(135deg, #0284C7, #16A34A) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        .stDownloadButton > button {
            border-radius: 16px !important;
            font-weight: 900 !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(125, 211, 252, 0.38) !important;
            background: linear-gradient(135deg, #334155, #0F766E) !important;
            min-height: 46px !important;
        }
    </style>
    """
)


# --------------------------------------------------
# 3. LOAD FINAL CHAMPION CONTEXT
# --------------------------------------------------

champion_metrics = get_champion_metrics()

champion_model_name = get_champion_model_name()
champion_track = get_champion_track()
best_threshold = get_best_threshold()

best_precision = float(champion_metrics.get("best_precision", 0))
best_recall = float(champion_metrics.get("best_recall", 0))
best_f1 = float(champion_metrics.get("best_f1_score", 0))
pr_auc = float(champion_metrics.get("pr_auc", 0))


# --------------------------------------------------
# 4. CONSTANTS
# --------------------------------------------------

REQUIRED_RAW_COLUMNS = [
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
]

DISPLAY_COLUMNS = [
    "campaign_rank",
    "visitorid",
    "purchase_intent_score_pct",
    "intent_segment",
    "recommended_action",
    "target_now",
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
]

SEGMENT_ORDER = [
    "High Intent",
    "Strong Intent",
    "Warm Intent",
    "Low Intent",
    "Cold Visitor",
]


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

def load_optional_csv(path: str) -> pd.DataFrame:
    """Load a CSV if it exists."""

    csv_path = Path(path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def get_natural_conversion_rate() -> float:
    """Return the labeled final-holdout buyer rate used for lift.

    The production visitor table intentionally has no outcome label.
    Therefore, lift must use the untouched labeled evaluation holdout,
    not the current production scoring population.
    """

    holdout = load_optional_csv(
        "reports/tables/final_true_champion_holdout.csv"
    )

    if holdout.empty:
        raise RuntimeError(
            "Final holdout results are unavailable. "
            "Run: python3 -m src.models.finalize_true_champion"
        )

    if "positive_rate" in holdout.columns:
        rates = pd.to_numeric(
            holdout["positive_rate"],
            errors="coerce",
        ).dropna()

        if not rates.empty and 0 < float(rates.iloc[0]) <= 1:
            return float(rates.iloc[0])

    required = {"rows", "positive_rows"}

    if required.issubset(holdout.columns):
        rows = float(holdout.iloc[0]["rows"])
        positives = float(holdout.iloc[0]["positive_rows"])

        if rows > 0 and 0 <= positives <= rows:
            return positives / rows

    raise RuntimeError(
        "Final holdout results do not contain a valid labeled buyer rate."
    )

def calculate_lift() -> float:
    """Calculate lift using final precision and natural conversion rate."""

    natural_conversion_rate = get_natural_conversion_rate()

    if natural_conversion_rate <= 0:
        return 0.0

    return best_precision / natural_conversion_rate


def readable_track_name(track_name: str) -> str:
    """Convert internal track names into readable dashboard labels."""

    track_lower = str(track_name).lower()

    if "final" in track_lower:
        return "Final True Champion"

    if "manual" in track_lower:
        return "Manual Champion/Challenger"

    if "automl" in track_lower:
        return "AutoML-style Benchmark"

    return str(track_name)


def create_sample_batch_template() -> pd.DataFrame:
    """Create a realistic sample file for batch scoring."""

    return pd.DataFrame(
        [
            {
                "visitorid": "sample_001",
                "total_events": 4,
                "view_count": 4,
                "addtocart_count": 0,
                "unique_items": 4,
                "activity_span_ms": 2 * 60 * 1000,
            },
            {
                "visitorid": "sample_002",
                "total_events": 18,
                "view_count": 18,
                "addtocart_count": 0,
                "unique_items": 12,
                "activity_span_ms": 22 * 60 * 1000,
            },
            {
                "visitorid": "sample_003",
                "total_events": 34,
                "view_count": 26,
                "addtocart_count": 5,
                "unique_items": 9,
                "activity_span_ms": 42 * 60 * 1000,
            },
            {
                "visitorid": "sample_004",
                "total_events": 58,
                "view_count": 42,
                "addtocart_count": 10,
                "unique_items": 8,
                "activity_span_ms": 76 * 60 * 1000,
            },
            {
                "visitorid": "sample_005",
                "total_events": 92,
                "view_count": 65,
                "addtocart_count": 16,
                "unique_items": 11,
                "activity_span_ms": 110 * 60 * 1000,
            },
        ]
    )


def validate_uploaded_batch(data: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """Validate uploaded batch data and return cleaned data."""

    errors: List[str] = []

    if data.empty:
        return False, ["Uploaded CSV is empty."], data

    cleaned_data = data.copy()

    # Visitor ID is useful but not mandatory.
    if "visitorid" not in cleaned_data.columns:
        cleaned_data["visitorid"] = [
            f"uploaded_{row_number + 1:05d}"
            for row_number in range(len(cleaned_data))
        ]

    missing_columns = [
        column for column in REQUIRED_RAW_COLUMNS
        if column not in cleaned_data.columns
    ]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors, cleaned_data

    for column in REQUIRED_RAW_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

        if cleaned_data[column].isna().any():
            errors.append(f"Column '{column}' contains missing or non-numeric values.")

        if (cleaned_data[column] < 0).any():
            errors.append(f"Column '{column}' contains negative values.")

    if (cleaned_data["total_events"] <= 0).any():
        errors.append("total_events must be greater than 0.")

    if (cleaned_data["unique_items"] <= 0).any():
        errors.append("unique_items must be greater than 0.")

    if (cleaned_data["view_count"] > cleaned_data["total_events"]).any():
        errors.append("view_count cannot be greater than total_events.")

    if (cleaned_data["addtocart_count"] > cleaned_data["total_events"]).any():
        errors.append("addtocart_count cannot be greater than total_events.")

    cleaned_data["visitorid"] = cleaned_data["visitorid"].astype(str)

    return len(errors) == 0, errors, cleaned_data


def score_batch_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Score every visitor using the final champion model."""

    model_input = prepare_batch_model_input(cleaned_data)
    scores = predict_batch_purchase_intent(model_input)

    scored_data = cleaned_data.copy()

    scored_data["purchase_intent_score"] = scores.values
    scored_data["purchase_intent_score_pct"] = scored_data["purchase_intent_score"] * 100
    scored_data["intent_segment"] = scored_data["purchase_intent_score"].apply(assign_intent_segment)
    scored_data["recommended_action"] = scored_data["purchase_intent_score"].apply(assign_recommended_action)
    scored_data["target_now"] = scored_data["purchase_intent_score"] >= best_threshold

    scored_data = scored_data.sort_values(
        "purchase_intent_score",
        ascending=False,
    ).reset_index(drop=True)

    scored_data["campaign_rank"] = scored_data.index + 1

    return scored_data


def summarize_scored_batch(scored_data: pd.DataFrame) -> Dict[str, float]:
    """Create business summary numbers for the scored batch."""

    total_visitors = len(scored_data)
    target_now_count = int(scored_data["target_now"].sum())
    high_intent_count = int((scored_data["intent_segment"] == "High Intent").sum())
    average_score = float(scored_data["purchase_intent_score"].mean())

    expected_buyers_in_target_group = float(target_now_count * best_precision)

    target_share = target_now_count / total_visitors if total_visitors > 0 else 0

    return {
        "total_visitors": total_visitors,
        "target_now_count": target_now_count,
        "target_share": target_share,
        "high_intent_count": high_intent_count,
        "average_score": average_score,
        "expected_buyers_in_target_group": expected_buyers_in_target_group,
    }


def fallback_log_batch_summary(summary: Dict[str, float]) -> str:
    """Write batch scoring summary to CSV if project logger fails."""

    log_path = Path("monitoring/prediction_logs/batch_scoring_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "page_name": "Batch Scoring",
        "champion_model": champion_model_name,
        "threshold": best_threshold,
        **summary,
    }

    pd.DataFrame([log_row]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    return "fallback_csv_logger"


def safe_log_batch_summary(summary: Dict[str, float]) -> str:
    """Log batch summary without crashing the page."""

    if log_batch_summary is not None:
        try:
            log_batch_summary(
                page_name="Batch Scoring",
                champion_model=champion_model_name,
                threshold=best_threshold,
                total_visitors=summary["total_visitors"],
                target_now_count=summary["target_now_count"],
                high_intent_count=summary["high_intent_count"],
                average_score=summary["average_score"],
            )
            return "project_logger"
        except TypeError:
            try:
                log_batch_summary(summary)
                return "project_logger_reduced_signature"
            except Exception:
                pass
        except Exception:
            pass

    return fallback_log_batch_summary(summary)


# --------------------------------------------------
# 6. CHART FUNCTIONS
# --------------------------------------------------

def build_segment_chart(scored_data: pd.DataFrame):
    """Create visitor segment distribution chart."""

    if not PLOTLY_AVAILABLE:
        return None

    segment_counts = (
        scored_data["intent_segment"]
        .value_counts()
        .reindex(SEGMENT_ORDER)
        .dropna()
        .reset_index()
    )

    segment_counts.columns = ["Segment", "Visitors"]

    fig = px.bar(
        segment_counts,
        x="Segment",
        y="Visitors",
        text="Visitors",
        title="Visitor audience by purchase-intent segment",
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=20),
        xaxis_title="",
        yaxis_title="Visitor count",
        showlegend=False,
    )

    return fig


def build_score_distribution_chart(scored_data: pd.DataFrame):
    """Create purchase-intent score distribution chart."""

    if not PLOTLY_AVAILABLE:
        return None

    fig = px.histogram(
        scored_data,
        x="purchase_intent_score_pct",
        nbins=30,
        title="Purchase-intent score distribution",
    )

    fig.add_vline(
        x=best_threshold * 100,
        line_dash="dash",
        annotation_text=f"Threshold {best_threshold:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=20),
        xaxis_title="Purchase intent score (%)",
        yaxis_title="Visitor count",
        showlegend=False,
    )

    return fig


def build_priority_chart(scored_data: pd.DataFrame, top_n: int = 20):
    """Create top campaign-priority chart."""

    if not PLOTLY_AVAILABLE:
        return None

    top_visitors = scored_data.head(top_n).copy()
    top_visitors["visitorid"] = top_visitors["visitorid"].astype(str)

    fig = px.bar(
        top_visitors.sort_values("purchase_intent_score", ascending=True),
        x="purchase_intent_score_pct",
        y="visitorid",
        orientation="h",
        color="intent_segment",
        hover_data={
            "recommended_action": True,
            "campaign_rank": True,
            "purchase_intent_score_pct": ":.1f",
        },
        title=f"Top {min(top_n, len(top_visitors))} campaign priorities",
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=20),
        xaxis_title="Purchase intent score (%)",
        yaxis_title="Visitor ID",
        legend_title="Segment",
    )

    return fig


# --------------------------------------------------
# 7. DERIVED METRICS
# --------------------------------------------------

best_lift = calculate_lift()


# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 📦 Batch Scoring")
st.sidebar.caption("Campaign audience scoring")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Source track:** {readable_track_name(champion_track)}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Precision:** {format_percent(best_precision)}")
st.sidebar.markdown(f"**Lift:** {format_lift(best_lift)}")
st.sidebar.markdown("---")
st.sidebar.caption("Upload many visitors, score them, and download a campaign-ready list.")


# --------------------------------------------------
# 9. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="batch-hero">'
        f'<div class="eyebrow">Campaign scoring engine</div>'
        f'<div class="batch-title">Batch Scoring for Marketing Audiences</div>'
        f'<div class="batch-subtitle">'
        f'Upload a visitor CSV, score every visitor with the final champion model, '
        f'rank campaign priorities, and download a scored audience file for activation.'
        f'</div><br>'
        f'<span class="success-pill">Champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Lift: {format_lift(best_lift)} vs random</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 10. WORKFLOW CARDS
# --------------------------------------------------

workflow_1, workflow_2, workflow_3, workflow_4 = st.columns(4)

with workflow_1:
    render_html(
        '<div class="workflow-card">'
        '<div class="workflow-icon">📄</div>'
        '<div class="workflow-title">1. Upload CSV</div>'
        '<div class="workflow-text">Upload visitor behaviour features or use the sample template.</div>'
        '</div>'
    )

with workflow_2:
    render_html(
        '<div class="workflow-card">'
        '<div class="workflow-icon">🧠</div>'
        '<div class="workflow-title">2. Score visitors</div>'
        '<div class="workflow-text">Final champion predicts purchase intent for every visitor.</div>'
        '</div>'
    )

with workflow_3:
    render_html(
        '<div class="workflow-card">'
        '<div class="workflow-icon">🎯</div>'
        '<div class="workflow-title">3. Rank campaign</div>'
        '<div class="workflow-text">High-intent visitors are ranked by score and action priority.</div>'
        '</div>'
    )

with workflow_4:
    render_html(
        '<div class="workflow-card">'
        '<div class="workflow-icon">⬇️</div>'
        '<div class="workflow-title">4. Download output</div>'
        '<div class="workflow-text">Export scored visitors for campaign activation or BI reporting.</div>'
        '</div>'
    )


# --------------------------------------------------
# 11. UPLOAD PANEL
# --------------------------------------------------

render_html(
    '<div class="upload-panel">'
    '<div class="section-kicker">Input data</div>'
    '<div class="section-title">Upload campaign audience file</div>'
    '<div class="compact-note">'
    'Required columns: visitorid, total_events, view_count, addtocart_count, unique_items, activity_span_ms. '
    'If visitorid is missing, temporary IDs are created automatically.'
    '</div>'
    '</div>'
)

template_data = create_sample_batch_template()

upload_col_1, upload_col_2 = st.columns([0.52, 0.48])

with upload_col_1:
    uploaded_file = st.file_uploader(
        "Upload visitor CSV",
        type=["csv"],
        help="Upload a CSV with visitor behaviour features.",
    )

    st.download_button(
        label="Download sample batch CSV template",
        data=template_data.to_csv(index=False),
        file_name="sample_batch_scoring_visitors.csv",
        mime="text/csv",
        use_container_width=True,
    )

with upload_col_2:
    st.markdown("#### Sample input preview")
    st.dataframe(
        template_data,
        use_container_width=True,
        hide_index=True,
    )

use_sample_data = st.toggle(
    "Use sample data instead of uploading a file",
    value=uploaded_file is None,
    help="Useful for demoing the app without uploading a file.",
)


# --------------------------------------------------
# 12. READ SELECTED INPUT DATA
# --------------------------------------------------

raw_input_data = None

if use_sample_data:
    raw_input_data = template_data.copy()

elif uploaded_file is not None:
    try:
        raw_input_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read uploaded CSV: {error}")

else:
    st.info("Upload a CSV or enable sample data to score visitors.")


# --------------------------------------------------
# 13. VALIDATE AND SCORE
# --------------------------------------------------

if raw_input_data is not None:
    is_valid, validation_errors, cleaned_data = validate_uploaded_batch(raw_input_data)

    if not is_valid:
        st.error("CSV validation failed.")
        for error in validation_errors:
            st.warning(error)

    else:
        st.success(f"CSV validation passed. Ready to score {len(cleaned_data):,} visitors.")

        score_button_clicked = st.button(
            "🚀 Score Batch Audience",
            type="primary",
            use_container_width=True,
        )

        if score_button_clicked:
            scored_data = score_batch_data(cleaned_data)
            summary = summarize_scored_batch(scored_data)
            log_method = safe_log_batch_summary(summary)

            st.session_state.latest_batch_scoring = {
                "scored_data": scored_data,
                "summary": summary,
                "log_method": log_method,
                "scored_at_utc": datetime.now(timezone.utc).isoformat(),
            }


# --------------------------------------------------
# 14. DISPLAY LATEST BATCH RESULT
# --------------------------------------------------

if "latest_batch_scoring" in st.session_state:
    result = st.session_state.latest_batch_scoring

    scored_data = result["scored_data"]
    summary = result["summary"]
    log_method = result["log_method"]

    render_html(
        (
            f'<div class="result-banner">'
            f'<div class="result-title">Batch scoring complete</div>'
            f'<div class="result-text">'
            f'Scored <b>{summary["total_visitors"]:,}</b> visitors. '
            f'<b>{summary["target_now_count"]:,}</b> visitors passed the final production threshold. '
            f'Expected buyers in the priority group: <b>{summary["expected_buyers_in_target_group"]:.1f}</b>.'
            f'</div>'
            f'</div>'
        )
    )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

    with kpi_1:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Visitors scored</div>'
                f'<div class="metric-value">{summary["total_visitors"]:,}</div>'
                f'<div class="metric-help">Rows processed in this batch.</div>'
                f'</div>'
            )
        )

    with kpi_2:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Target now</div>'
                f'<div class="metric-value">{summary["target_now_count"]:,}</div>'
                f'<div class="metric-help">Visitors above final threshold.</div>'
                f'</div>'
            )
        )

    with kpi_3:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Target share</div>'
                f'<div class="metric-value">{format_percent(summary["target_share"])}</div>'
                f'<div class="metric-help">Share of audience selected.</div>'
                f'</div>'
            )
        )

    with kpi_4:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Expected buyers</div>'
                f'<div class="metric-value">{summary["expected_buyers_in_target_group"]:.1f}</div>'
                f'<div class="metric-help">Estimated buyers inside target-now group.</div>'
                f'</div>'
            )
        )

    with kpi_5:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Average score</div>'
                f'<div class="metric-value">{format_percent(summary["average_score"])}</div>'
                f'<div class="metric-help">Log method: {escape_text(log_method)}</div>'
                f'</div>'
            )
        )

    st.markdown("## Campaign audience visuals")

    if PLOTLY_AVAILABLE:
        visual_col_1, visual_col_2 = st.columns(2)

        with visual_col_1:
            st.plotly_chart(
                build_segment_chart(scored_data),
                use_container_width=True,
            )

        with visual_col_2:
            st.plotly_chart(
                build_score_distribution_chart(scored_data),
                use_container_width=True,
            )

        st.plotly_chart(
            build_priority_chart(scored_data, top_n=20),
            use_container_width=True,
        )

    else:
        st.info("Plotly is not installed. Charts will appear after installing plotly.")

    st.markdown("## Campaign priority table")

    available_display_columns = [
        column for column in DISPLAY_COLUMNS
        if column in scored_data.columns
    ]

    st.dataframe(
        scored_data[available_display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download scored campaign audience CSV",
        data=scored_data.to_csv(index=False),
        file_name="scored_campaign_audience.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Show model and batch metadata"):
        st.json(
            {
                "champion_model": champion_model_name,
                "champion_track": readable_track_name(champion_track),
                "threshold": best_threshold,
                "precision_at_threshold": best_precision,
                "recall_at_threshold": best_recall,
                "f1_at_threshold": best_f1,
                "pr_auc": pr_auc,
                "business_lift": best_lift,
                "scored_at_utc": result["scored_at_utc"],
                "log_method": log_method,
            }
        )

else:
    render_html(
        '<div class="result-banner">'
        '<div class="result-title">No batch scored yet</div>'
        '<div class="result-text">'
        'Upload a visitor CSV or use the sample template, then click '
        '<b>Score Batch Audience</b>. The page will create a campaign-ready scored audience.'
        '</div>'
        '</div>'
    )

# --------------------------------------------------
# FINAL CLOSURE EXTENSION: CAMPAIGN INTELLIGENCE
# --------------------------------------------------

from app.ui.campaign_intelligence import (
    render_batch_campaign_intelligence,
)

render_batch_campaign_intelligence(
    production_threshold=best_threshold,
    validated_precision=best_precision,
)
