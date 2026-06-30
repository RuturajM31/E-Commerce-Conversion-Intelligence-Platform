# 1_Visitor_Intent_Predictor.py
# Streamlit page for scoring one visitor with the final champion model.
#
# Purpose:
#   Score one visitor, explain the prediction, recommend an action, and log the result.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Load final champion context
#   4. Helper functions
#   5. Sidebar and header
#   6. Scenario presets
#   7. Manual visitor input
#   8. Score visitor
#   9. Show result and log proof

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from app.app_utils import (
    assign_intent_segment,
    assign_recommended_action,
    build_model_input,
    escape_text,
    explain_prediction,
    format_lift,
    format_percent,
    get_best_threshold,
    get_champion_metrics,
    get_champion_model_name,
    get_champion_track,
    inject_global_css,
    load_champion_metadata,
    predict_purchase_intent,
    render_html,
)

# The project logger may exist.
# We keep a fallback logger so the app never fails because of logging.
try:
    from src.monitoring.prediction_logger import log_single_prediction
except Exception:
    log_single_prediction = None


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Visitor Intent Predictor",
    page_icon="🎯",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .visitor-hero {
            padding: 34px 36px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.28), transparent 32%),
                radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.22), transparent 32%),
                linear-gradient(135deg, #0F172A 0%, #111827 50%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 24px 75px rgba(0, 0, 0, 0.45);
            margin-bottom: 24px;
        }

        .scenario-card {
            padding: 22px 22px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.25);
            min-height: 158px;
            margin-bottom: 10px;
        }

        .scenario-icon {
            font-size: 1.55rem;
            margin-bottom: 8px;
        }

        .scenario-title {
            color: #F8FAFC;
            font-size: 1.08rem;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .scenario-text {
            color: #CBD5E1;
            font-size: 0.90rem;
            line-height: 1.5;
            margin-bottom: 10px;
        }

        .scenario-stats {
            color: #7DD3FC;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .input-card {
            padding: 24px 26px;
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
            margin-top: 18px;
            margin-bottom: 18px;
        }

        .prediction-panel {
            padding: 30px 32px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.20), transparent 30%),
                radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.15), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(2, 6, 23, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 24px 75px rgba(0, 0, 0, 0.42);
            margin-top: 18px;
            margin-bottom: 22px;
        }

        .score-label {
            color: #94A3B8;
            font-size: 0.80rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .score-number {
            font-size: 4.4rem;
            font-weight: 950;
            line-height: 1;
            color: #F8FAFC;
            margin-top: 8px;
            margin-bottom: 10px;
        }

        .gauge-shell {
            position: relative;
            height: 24px;
            border-radius: 999px;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.20);
            overflow: hidden;
            margin-top: 18px;
            margin-bottom: 10px;
        }

        .gauge-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0EA5E9, #22C55E);
            box-shadow: 0 0 30px rgba(34, 197, 94, 0.42);
        }

        .threshold-marker {
            position: absolute;
            top: -7px;
            width: 3px;
            height: 38px;
            background: #FACC15;
            box-shadow: 0 0 20px rgba(250, 204, 21, 0.75);
        }

        .decision-card {
            padding: 24px 26px;
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
            min-height: 188px;
        }

        .decision-title {
            color: #F8FAFC;
            font-size: 1.20rem;
            font-weight: 900;
            margin-bottom: 9px;
        }

        .decision-text {
            color: #CBD5E1;
            font-size: 0.95rem;
            line-height: 1.65;
        }

        .mini-note {
            color: #94A3B8;
            font-size: 0.82rem;
            line-height: 1.45;
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
    </style>
    """
)


# --------------------------------------------------
# 3. LOAD FINAL CHAMPION CONTEXT
# --------------------------------------------------

metadata = load_champion_metadata()
champion_metrics = get_champion_metrics()

champion_model_name = get_champion_model_name()
champion_track = get_champion_track()
best_threshold = get_best_threshold()

best_precision = float(champion_metrics.get("best_precision", 0))
best_recall = float(champion_metrics.get("best_recall", 0))
best_f1 = float(champion_metrics.get("best_f1_score", 0))
pr_auc = float(champion_metrics.get("pr_auc", 0))
roc_auc = float(champion_metrics.get("roc_auc", 0))


# --------------------------------------------------
# 4. HELPER FUNCTIONS
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
    """Calculate lift from final precision and natural conversion rate."""

    natural_conversion_rate = get_natural_conversion_rate()

    if natural_conversion_rate <= 0:
        return 0.0

    return best_precision / natural_conversion_rate


def readable_track_name(track_name: str) -> str:
    """Convert internal track name into a readable label."""

    track_lower = str(track_name).lower()

    if "final" in track_lower:
        return "Final True Champion"

    if "manual" in track_lower:
        return "Manual Champion/Challenger"

    if "automl" in track_lower:
        return "AutoML-style Benchmark"

    return str(track_name)


def initialize_input_state() -> None:
    """Create default visitor input values."""

    defaults = {
        "total_events": 12,
        "view_count": 10,
        "addtocart_count": 1,
        "unique_items": 6,
        "activity_minutes": 18,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_preset(
    total_events: int,
    view_count: int,
    addtocart_count: int,
    unique_items: int,
    activity_minutes: int,
) -> None:
    """Apply one preset scenario to the input widgets."""

    st.session_state.total_events = total_events
    st.session_state.view_count = view_count
    st.session_state.addtocart_count = addtocart_count
    st.session_state.unique_items = unique_items
    st.session_state.activity_minutes = activity_minutes


def get_current_input_values() -> Dict[str, float]:
    """Read current visitor values from session state."""

    return {
        "total_events": float(st.session_state.total_events),
        "view_count": float(st.session_state.view_count),
        "addtocart_count": float(st.session_state.addtocart_count),
        "unique_items": float(st.session_state.unique_items),
        "activity_minutes": float(st.session_state.activity_minutes),
    }


def validate_current_inputs(values: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Validate manual visitor input values."""

    errors: List[str] = []

    if values["total_events"] <= 0:
        errors.append("Total events must be greater than 0.")

    if values["view_count"] > values["total_events"]:
        errors.append("View count cannot be greater than total events.")

    if values["addtocart_count"] > values["total_events"]:
        errors.append("Add-to-cart count cannot be greater than total events.")

    if values["unique_items"] > values["total_events"]:
        errors.append("Unique items should not be greater than total events for this demo.")

    return len(errors) == 0, errors


def create_model_input_from_values(values: Dict[str, float]) -> pd.DataFrame:
    """Convert user input into the model feature format."""

    activity_span_ms = values["activity_minutes"] * 60 * 1000

    return build_model_input(
        total_events=values["total_events"],
        view_count=values["view_count"],
        addtocart_count=values["addtocart_count"],
        unique_items=values["unique_items"],
        activity_span_ms=activity_span_ms,
    )


def fallback_log_prediction(
    visitor_id: str,
    purchase_intent_score: float,
    intent_segment: str,
    recommended_action: str,
    input_features: Dict,
) -> str:
    """Write a prediction log directly if the project logger fails."""

    log_path = Path("monitoring/prediction_logs/prediction_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "page_name": "Visitor Intent Predictor",
        "prediction_type": "single_visitor",
        "visitor_id": visitor_id,
        "purchase_intent_score": purchase_intent_score,
        "intent_segment": intent_segment,
        "recommended_action": recommended_action,
    }

    for feature_name, feature_value in input_features.items():
        log_row[f"feature_{feature_name}"] = feature_value

    pd.DataFrame([log_row]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )

    return "fallback_csv_logger"


def safe_log_prediction(
    visitor_id: str,
    purchase_intent_score: float,
    intent_segment: str,
    recommended_action: str,
    input_features: Dict,
) -> str:
    """Log a prediction without crashing the app."""

    if log_single_prediction is not None:
        try:
            log_single_prediction(
                page_name="Visitor Intent Predictor",
                visitor_id=visitor_id,
                purchase_intent_score=purchase_intent_score,
                intent_segment=intent_segment,
                recommended_action=recommended_action,
                input_features=input_features,
            )
            return "project_logger"
        except TypeError:
            try:
                log_single_prediction(
                    visitor_id=visitor_id,
                    purchase_intent_score=purchase_intent_score,
                    intent_segment=intent_segment,
                    recommended_action=recommended_action,
                    input_features=input_features,
                )
                return "project_logger_reduced_signature"
            except Exception:
                pass
        except Exception:
            pass

    return fallback_log_prediction(
        visitor_id=visitor_id,
        purchase_intent_score=purchase_intent_score,
        intent_segment=intent_segment,
        recommended_action=recommended_action,
        input_features=input_features,
    )


def score_current_visitor(source_label: str) -> None:
    """Score current visitor input and store result in session state."""

    values = get_current_input_values()

    is_valid, errors = validate_current_inputs(values)

    if not is_valid:
        for error in errors:
            st.error(error)
        return

    model_input = create_model_input_from_values(values)
    purchase_intent_score = predict_purchase_intent(model_input)

    intent_segment = assign_intent_segment(purchase_intent_score)
    recommended_action = assign_recommended_action(purchase_intent_score)
    prediction_explanation = explain_prediction(purchase_intent_score)

    visitor_id = f"demo_{uuid.uuid4().hex[:8]}"

    log_method = safe_log_prediction(
        visitor_id=visitor_id,
        purchase_intent_score=purchase_intent_score,
        intent_segment=intent_segment,
        recommended_action=recommended_action,
        input_features=model_input.iloc[0].to_dict(),
    )

    st.session_state.latest_prediction = {
        "visitor_id": visitor_id,
        "purchase_intent_score": purchase_intent_score,
        "intent_segment": intent_segment,
        "recommended_action": recommended_action,
        "prediction_explanation": prediction_explanation,
        "model_input": model_input.copy(),
        "source_label": source_label,
        "log_method": log_method,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------
# 5. INITIALIZE STATE AND DERIVED METRICS
# --------------------------------------------------

initialize_input_state()

best_lift = calculate_lift()


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🎯 Visitor Intent Predictor")
st.sidebar.caption("Final champion scoring page")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Source track:** {readable_track_name(champion_track)}")
st.sidebar.markdown(f"**Production threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Lift vs random:** {format_lift(best_lift)}")
st.sidebar.markdown("---")
st.sidebar.caption("Score one visitor manually or through a preset. Batch CSV scoring is on the Batch Scoring page.")


# --------------------------------------------------
# 7. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="visitor-hero">'
        f'<div class="eyebrow">Live scoring engine</div>'
        f'<div class="hero-title">Visitor Intent Predictor</div>'
        f'<div class="hero-subtitle">'
        f'Score one ecommerce visitor with the final production champion model. '
        f'The app converts behaviour signals into a purchase intent score, segment, and recommended action.'
        f'</div><br>'
        f'<span class="success-pill">Champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Lift: {format_lift(best_lift)} vs random</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 8. CHAMPION KPI CARDS
# --------------------------------------------------

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    render_html(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">Final champion</div>'
            f'<div class="metric-value">{escape_text(champion_model_name)}</div>'
            f'<div class="metric-help">Selected after tuning, boosting comparison, ensemble check, and stability testing.</div>'
            f'</div>'
        )
    )

with kpi_2:
    render_html(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">Production threshold</div>'
            f'<div class="metric-value">{best_threshold:.2f}</div>'
            f'<div class="metric-help">Only very high-scoring visitors become priority targets.</div>'
            f'</div>'
        )
    )

with kpi_3:
    render_html(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">Precision at threshold</div>'
            f'<div class="metric-value">{format_percent(best_precision)}</div>'
            f'<div class="metric-help">Out of targeted visitors, this share actually converted.</div>'
            f'</div>'
        )
    )

with kpi_4:
    render_html(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">Business lift</div>'
            f'<div class="metric-value">{format_lift(best_lift)}</div>'
            f'<div class="metric-help">Target quality compared with random marketing.</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 9. SCENARIO PRESETS
# --------------------------------------------------

render_html(
    """
    <div class="insight-card">
        <div class="insight-title">Choose a visitor scenario</div>
        <div class="insight-text">
            Use a preset for a fast demo, or adjust the input values manually.
            These scenarios make the model result easier to explain.
        </div>
    </div>
    """
)

scenario_1, scenario_2, scenario_3, scenario_4 = st.columns(4)

with scenario_1:
    render_html(
        """
        <div class="scenario-card">
            <div class="scenario-icon">❄️</div>
            <div class="scenario-title">Cold Visitor</div>
            <div class="scenario-text">Few views, no cart activity, weak buying signal.</div>
            <div class="scenario-stats">3 events · 0 carts · 2 minutes</div>
        </div>
        """
    )
    st.button(
        "Use Cold Visitor",
        use_container_width=True,
        on_click=apply_preset,
        args=(3, 3, 0, 3, 2),
    )

with scenario_2:
    render_html(
        """
        <div class="scenario-card">
            <div class="scenario-icon">👀</div>
            <div class="scenario-title">Browsing Only</div>
            <div class="scenario-text">Many product views, but no add-to-cart signal yet.</div>
            <div class="scenario-stats">18 events · 0 carts · 22 minutes</div>
        </div>
        """
    )
    st.button(
        "Use Browsing Only",
        use_container_width=True,
        on_click=apply_preset,
        args=(18, 18, 0, 12, 22),
    )

with scenario_3:
    render_html(
        """
        <div class="scenario-card">
            <div class="scenario-icon">🛒</div>
            <div class="scenario-title">Cart Abandoner</div>
            <div class="scenario-text">Strong cart behaviour, useful for retargeting.</div>
            <div class="scenario-stats">34 events · 5 carts · 42 minutes</div>
        </div>
        """
    )
    st.button(
        "Use Cart Abandoner",
        use_container_width=True,
        on_click=apply_preset,
        args=(34, 26, 5, 9, 42),
    )

with scenario_4:
    render_html(
        """
        <div class="scenario-card">
            <div class="scenario-icon">🔥</div>
            <div class="scenario-title">High Intent Buyer</div>
            <div class="scenario-text">Repeated actions and cart activity, likely to convert.</div>
            <div class="scenario-stats">58 events · 10 carts · 76 minutes</div>
        </div>
        """
    )
    st.button(
        "Use High Intent Buyer",
        use_container_width=True,
        on_click=apply_preset,
        args=(58, 42, 10, 8, 76),
    )


# --------------------------------------------------
# 10. MANUAL VISITOR INPUT
# --------------------------------------------------

render_html(
    """
    <div class="input-card">
        <div class="insight-title">Visitor behaviour input</div>
        <div class="insight-text">
            These fields describe what the visitor did before purchase.
            The model uses these signals to estimate purchase intent.
        </div>
    </div>
    """
)

input_col_1, input_col_2, input_col_3, input_col_4, input_col_5 = st.columns(5)

with input_col_1:
    st.number_input(
        "Total events",
        min_value=1,
        max_value=500,
        step=1,
        key="total_events",
        help="Total number of recorded actions for this visitor.",
    )

with input_col_2:
    st.number_input(
        "View count",
        min_value=0,
        max_value=500,
        step=1,
        key="view_count",
        help="Number of product views.",
    )

with input_col_3:
    st.number_input(
        "Add-to-cart count",
        min_value=0,
        max_value=500,
        step=1,
        key="addtocart_count",
        help="Number of add-to-cart actions.",
    )

with input_col_4:
    st.number_input(
        "Unique items",
        min_value=1,
        max_value=500,
        step=1,
        key="unique_items",
        help="Number of different products the visitor interacted with.",
    )

with input_col_5:
    st.number_input(
        "Activity minutes",
        min_value=0,
        max_value=1440,
        step=1,
        key="activity_minutes",
        help="Approximate visitor activity span in minutes.",
    )


# --------------------------------------------------
# 11. VALIDATE AND SCORE
# --------------------------------------------------

current_values = get_current_input_values()
is_valid, input_errors = validate_current_inputs(current_values)

if not is_valid:
    for error in input_errors:
        st.error(error)

score_clicked = st.button(
    "🚀 Score Visitor & Log Prediction",
    type="primary",
    use_container_width=True,
    disabled=not is_valid,
)

if score_clicked:
    score_current_visitor(source_label="manual_or_preset_input")
    st.success("Prediction completed. Result shown below.")


# --------------------------------------------------
# 12. DISPLAY RESULT
# --------------------------------------------------

if "latest_prediction" in st.session_state:
    prediction = st.session_state.latest_prediction

    score = float(prediction["purchase_intent_score"])
    score_percent = score * 100
    gauge_width = max(0, min(score_percent, 100))
    threshold_percent = best_threshold * 100

    is_above_threshold = score >= best_threshold

    decision_status = (
        "Above production threshold"
        if is_above_threshold
        else "Below production threshold"
    )

    decision_pill_class = (
        "success-pill"
        if is_above_threshold
        else "warning-pill"
    )

    status_pills_html = (
        f'<span class="{decision_pill_class}">{escape_text(decision_status)}</span>'
        f'&nbsp;<span class="info-pill">{escape_text(prediction["intent_segment"])}</span>'
    )

    prediction_panel_html = (
        f'<div class="prediction-panel">'
        f'<div class="score-label">Purchase intent score</div>'
        f'<div class="score-number">{score_percent:.1f}%</div>'
        f'{status_pills_html}'
        f'<div class="gauge-shell">'
        f'<div class="gauge-fill" style="width: {gauge_width:.1f}%;"></div>'
        f'<div class="threshold-marker" style="left: {threshold_percent:.1f}%;"></div>'
        f'</div>'
        f'<div class="mini-note">'
        f'Yellow marker = final production threshold at {best_threshold:.2f}. '
        f'Visitors above this marker are high-priority marketing targets.'
        f'</div>'
        f'</div>'
    )

    render_html(prediction_panel_html)

    result_col_1, result_col_2 = st.columns([1.08, 0.92])

    with result_col_1:
        render_html(
            (
                f'<div class="decision-card">'
                f'<div class="decision-title">Recommended business action</div>'
                f'<div class="decision-text">'
                f'<b>{escape_text(prediction["recommended_action"])}</b>'
                f'<br><br>'
                f'{escape_text(prediction["prediction_explanation"])}'
                f'</div>'
                f'</div>'
            )
        )

    with result_col_2:
        render_html(
            (
                f'<div class="decision-card">'
                f'<div class="decision-title">Why this model decision matters</div>'
                f'<div class="decision-text">'
                f'Random marketing targets 100 visitors and gets less than 1 buyer. '
                f'The final champion targets high-intent visitors and achieves '
                f'<b>{format_lift(best_lift)}</b> lift versus random targeting.'
                f'<br><br>'
                f'This means marketing budget can focus on visitors who are much more likely to buy.'
                f'</div>'
                f'</div>'
            )
        )

    st.markdown("### Final champion context")

    context_col_1, context_col_2, context_col_3, context_col_4, context_col_5 = st.columns(5)

    with context_col_1:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">PR-AUC</div>'
                f'<div class="metric-value">{pr_auc:.3f}</div>'
                f'<div class="metric-help">Rare-buyer ranking quality.</div>'
                f'</div>'
            )
        )

    with context_col_2:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Best F1</div>'
                f'<div class="metric-value">{best_f1:.3f}</div>'
                f'<div class="metric-help">Balance between precision and recall.</div>'
                f'</div>'
            )
        )

    with context_col_3:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Recall</div>'
                f'<div class="metric-value">{format_percent(best_recall)}</div>'
                f'<div class="metric-help">Real buyers captured at threshold.</div>'
                f'</div>'
            )
        )

    with context_col_4:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">ROC-AUC</div>'
                f'<div class="metric-value">{roc_auc:.3f}</div>'
                f'<div class="metric-help">General ranking strength.</div>'
                f'</div>'
            )
        )

    with context_col_5:
        render_html(
            (
                f'<div class="metric-card">'
                f'<div class="metric-label">Log method</div>'
                f'<div class="metric-value">Saved</div>'
                f'<div class="metric-help">{escape_text(prediction["log_method"])}</div>'
                f'</div>'
            )
        )

    with st.expander("Show exact model input row"):
        st.dataframe(
            prediction["model_input"],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Show prediction metadata"):
        st.json(
            {
                "visitor_id": prediction["visitor_id"],
                "predicted_at_utc": prediction["predicted_at_utc"],
                "champion_model": champion_model_name,
                "champion_track": readable_track_name(champion_track),
                "best_threshold": best_threshold,
                "score": score,
                "segment": prediction["intent_segment"],
                "recommended_action": prediction["recommended_action"],
                "log_method": prediction["log_method"],
            }
        )

else:
    render_html(
        """
        <div class="insight-card">
            <div class="insight-title">No prediction yet</div>
            <div class="insight-text">
                Choose a preset or enter visitor behaviour manually.
                Then click <b>Score Visitor & Log Prediction</b>.
            </div>
        </div>
        """
    )

# --------------------------------------------------
# FINAL CLOSURE EXTENSION: KNN AND EXPLAINABILITY
# --------------------------------------------------

from app.ui.visitor_similarity_explainability import (
    render_visitor_similarity_explainability,
)

render_visitor_similarity_explainability()
