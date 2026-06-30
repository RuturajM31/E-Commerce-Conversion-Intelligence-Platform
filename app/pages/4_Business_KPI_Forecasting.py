# 4_Business_KPI_Forecasting.py
# Streamlit page for business KPI forecasting.
#
# Purpose:
#   Show future operational demand after the conversion model has scored visitor intent.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Paths and labels
#   4. Helper functions
#   5. Chart functions
#   6. Load forecast outputs
#   7. Sidebar and header
#   8. Forecast KPI cards
#   9. Forecast charts
#   10. Model comparison and raw tables

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


from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from app.app_utils import (
    escape_text,
    get_best_threshold,
    get_champion_model_name,
    inject_global_css,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Business KPI Forecasting",
    page_icon="📈",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .forecast-hero {
            padding: 34px 38px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.28), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(34, 197, 94, 0.20), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .forecast-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .forecast-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1050px;
        }

        .forecast-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 184px;
            margin-bottom: 18px;
        }

        .forecast-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .forecast-value {
            color: #F8FAFC;
            font-size: 2.20rem;
            line-height: 1;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .forecast-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.48;
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
            font-size: 1.25rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .story-text {
            color: #CBD5E1;
            font-size: 0.94rem;
            line-height: 1.58;
        }

        .model-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #E2E8F0;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .small-note {
            color: #94A3B8;
            font-size: 0.84rem;
            line-height: 1.48;
            margin-bottom: 12px;
        }
    </style>
    """
)


# --------------------------------------------------
# 3. PATHS AND LABELS
# --------------------------------------------------

DAILY_KPI_PATH = Path("reports/tables/daily_business_kpis.csv")
FORECAST_COMPARISON_PATH = Path("reports/tables/business_forecast_comparison.csv")
FORECAST_FUTURE_PATH = Path("reports/tables/business_forecast_future.csv")
FORECAST_HISTORY_PATH = Path("reports/tables/business_forecast_history_with_predictions.csv")

TARGET_LABELS: Dict[str, str] = {
    "unique_visitors": "Unique Visitors",
    "event_volume": "Event Volume",
    "converted_visitor_count": "Converted Visitors",
    "high_intent_visitor_count": "High-Intent Visitors",
}

TARGET_EXPLANATIONS: Dict[str, str] = {
    "unique_visitors": "Expected traffic demand.",
    "event_volume": "Expected ecommerce activity and system load.",
    "converted_visitor_count": "Expected buyer demand.",
    "high_intent_visitor_count": "Expected marketing-ready audience.",
}

DEFAULT_TARGET_ORDER = [
    "unique_visitors",
    "event_volume",
    "converted_visitor_count",
    "high_intent_visitor_count",
]


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV safely and parse date column when available."""

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    return data


def format_count(value: float) -> str:
    """Format counts for dashboard cards."""

    if pd.isna(value):
        return "NA"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_error(value: float) -> str:
    """Format forecast error values."""

    if pd.isna(value):
        return "NA"

    return f"{float(value):,.1f}"


def normalize_bool_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert mixed true/false values into booleans."""

    if data.empty or column not in data.columns:
        return pd.Series(False, index=data.index)

    return data[column].astype(str).str.lower().isin(["true", "1", "yes"])


def get_best_model_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return one best forecast model per target."""

    if comparison.empty:
        return pd.DataFrame()

    data = comparison.copy()

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower() == "success"].copy()

    if data.empty:
        return pd.DataFrame()

    if "is_best_model" in data.columns:
        best_rows = data[normalize_bool_column(data, "is_best_model")].copy()

        if not best_rows.empty:
            return best_rows

    if "mae" in data.columns:
        return (
            data
            .sort_values(["target_name", "mae"], ascending=[True, True])
            .groupby("target_name", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

    return data.drop_duplicates("target_name").copy()


def get_best_future_rows(future: pd.DataFrame) -> pd.DataFrame:
    """Return future forecast rows for the best model only."""

    if future.empty:
        return pd.DataFrame()

    data = future.copy()

    if "is_best_model" in data.columns:
        best = data[normalize_bool_column(data, "is_best_model")].copy()

        if not best.empty:
            return best

    return data.copy()


def build_forecast_summary(best_future: pd.DataFrame) -> pd.DataFrame:
    """Summarise future forecasts by target."""

    if best_future.empty:
        return pd.DataFrame()

    required = {"target_name", "date", "predicted_value"}

    if not required.issubset(best_future.columns):
        return pd.DataFrame()

    summary = (
        best_future
        .groupby("target_name")
        .agg(
            forecast_days=("date", "nunique"),
            total_forecast=("predicted_value", "sum"),
            average_daily_forecast=("predicted_value", "mean"),
            min_forecast=("predicted_value", "min"),
            max_forecast=("predicted_value", "max"),
            model_name=("model_name", "first") if "model_name" in best_future.columns else ("target_name", "first"),
        )
        .reset_index()
    )

    return summary


def get_summary_value(summary: pd.DataFrame, target_name: str, column: str) -> float:
    """Read one value from forecast summary."""

    if summary.empty:
        return np.nan

    row = summary[summary["target_name"] == target_name]

    if row.empty or column not in row.columns:
        return np.nan

    return float(row.iloc[0][column])


def get_summary_model(summary: pd.DataFrame, target_name: str) -> str:
    """Read best model name for one target."""

    if summary.empty:
        return "NA"

    row = summary[summary["target_name"] == target_name]

    if row.empty or "model_name" not in row.columns:
        return "NA"

    return str(row.iloc[0]["model_name"])


def get_available_targets(summary: pd.DataFrame) -> List[str]:
    """Return targets in business-friendly order."""

    if summary.empty or "target_name" not in summary.columns:
        return DEFAULT_TARGET_ORDER

    targets = summary["target_name"].dropna().astype(str).unique().tolist()

    ordered = [target for target in DEFAULT_TARGET_ORDER if target in targets]
    extra = [target for target in targets if target not in ordered]

    return ordered + extra




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

def build_historical_forecast_chart(
    daily_kpis: pd.DataFrame,
    best_future: pd.DataFrame,
    target_name: str,
):
    """Build actual historical trend plus future forecast line chart."""

    if not PLOTLY_AVAILABLE:
        return None

    if daily_kpis.empty or "date" not in daily_kpis.columns or target_name not in daily_kpis.columns:
        return None

    historical = daily_kpis[["date", target_name]].copy()
    historical = historical.rename(columns={target_name: "value"})
    historical["series"] = "Historical actual"

    future_rows = best_future[best_future["target_name"] == target_name].copy()

    if future_rows.empty or "predicted_value" not in future_rows.columns:
        chart_data = historical
    else:
        future_rows = future_rows[["date", "predicted_value"]].rename(
            columns={"predicted_value": "value"}
        )
        future_rows["series"] = "Future forecast"
        chart_data = pd.concat([historical, future_rows], ignore_index=True)

    fig = px.line(
        chart_data,
        x="date",
        y="value",
        color="series",
        markers=True,
        title=f"{TARGET_LABELS.get(target_name, target_name)}: actual history and future forecast",
    )

    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title=TARGET_LABELS.get(target_name, target_name),
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


def build_model_error_chart(comparison: pd.DataFrame, target_name: str):
    """Build model error comparison chart for one target."""

    if not PLOTLY_AVAILABLE or comparison.empty:
        return None

    if "target_name" not in comparison.columns or "mae" not in comparison.columns:
        return None

    data = comparison[comparison["target_name"] == target_name].copy()

    if data.empty:
        return None

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower() == "success"].copy()

    if data.empty:
        return None

    data = data.sort_values("mae", ascending=True)

    fig = px.bar(
        data,
        x="model_name",
        y="mae",
        text="mae",
        title=f"Forecast model comparison: {TARGET_LABELS.get(target_name, target_name)}",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=410,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Forecast model",
        yaxis_title="MAE",
        title_font=dict(size=20),
        showlegend=False,
    )

    return fig


def build_future_multi_target_chart(best_future: pd.DataFrame):
    """Build one combined chart for all best-model future forecasts."""

    if not PLOTLY_AVAILABLE or best_future.empty:
        return None

    required = {"date", "target_name", "predicted_value"}

    if not required.issubset(best_future.columns):
        return None

    chart_data = best_future.copy()
    chart_data["target_label"] = chart_data["target_name"].map(TARGET_LABELS).fillna(chart_data["target_name"])

    fig = px.line(
        chart_data,
        x="date",
        y="predicted_value",
        color="target_label",
        markers=True,
        title="Future forecast across operational KPIs",
    )

    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Forecast value",
        legend_title="KPI",
        title_font=dict(size=20),
    )

    return fig


def build_actual_vs_predicted_chart(history: pd.DataFrame, target_name: str):
    """Build actual vs predicted test-period chart when history file has predictions."""

    if not PLOTLY_AVAILABLE or history.empty:
        return None

    required = {"date", "target_name", "actual_value", "predicted_value"}

    if not required.issubset(history.columns):
        return None

    data = history[history["target_name"] == target_name].copy()

    if data.empty:
        return None

    if "is_best_model" in data.columns:
        best_rows = data[normalize_bool_column(data, "is_best_model")].copy()
        if not best_rows.empty:
            data = best_rows

    chart_data = data[["date", "actual_value", "predicted_value"]].copy()
    chart_data = chart_data.melt(
        id_vars="date",
        var_name="series",
        value_name="value",
    )

    chart_data["series"] = chart_data["series"].replace(
        {
            "actual_value": "Actual",
            "predicted_value": "Predicted",
        }
    )

    fig = px.line(
        chart_data,
        x="date",
        y="value",
        color="series",
        markers=True,
        title=f"Backtest check: actual vs predicted for {TARGET_LABELS.get(target_name, target_name)}",
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


# --------------------------------------------------
# 6. LOAD FORECAST OUTPUTS
# --------------------------------------------------

daily_kpis = load_csv(DAILY_KPI_PATH)
comparison = load_csv(FORECAST_COMPARISON_PATH)
future = load_csv(FORECAST_FUTURE_PATH)
history = load_csv(FORECAST_HISTORY_PATH)

best_models = get_best_model_rows(comparison)
best_future = get_best_future_rows(future)
forecast_summary = build_forecast_summary(best_future)
available_targets = get_available_targets(forecast_summary)

champion_model_name = get_champion_model_name()
best_threshold = get_best_threshold()


# --------------------------------------------------
# 7. SIDEBAR AND HEADER
# --------------------------------------------------

st.sidebar.markdown("## 📈 Business KPI Forecasting")
st.sidebar.caption("Operational demand forecast")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Intent threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Forecast file:** {'OK' if not future.empty else 'Missing'}")
st.sidebar.markdown("---")
st.sidebar.caption("Forecasts traffic, events, conversions, and high-intent audience demand.")

render_html(
    (
        f'<div class="forecast-hero">'
        f'<div class="eyebrow">Business demand forecast</div>'
        f'<div class="forecast-title">Business KPI Forecasting</div>'
        f'<div class="forecast-subtitle">'
        f'The conversion model tells us who may buy. This page estimates how much traffic, activity, '
        f'conversion demand, and high-intent audience volume the business should expect next.'
        f'</div><br>'
        f'<span class="success-pill">Best models selected per KPI</span>'
        f'&nbsp;<span class="info-pill">Final champion threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Revenue not forecasted: no transaction value in dataset</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 8. MISSING FILE CHECK
# --------------------------------------------------

missing_files = []

for file_path in [
    DAILY_KPI_PATH,
    FORECAST_COMPARISON_PATH,
    FORECAST_FUTURE_PATH,
    FORECAST_HISTORY_PATH,
]:
    if not file_path.exists():
        missing_files.append(str(file_path))

if missing_files:
    st.warning("Some forecasting output files are missing.")
    st.code("\n".join(missing_files))
    st.info("Run: python3 -m src.forecasting.build_business_forecasts")


# --------------------------------------------------
# 9. FORECAST KPI CARDS
# --------------------------------------------------

render_html('<div class="section-kicker">Forecast summary</div><div class="section-title">Best-model future demand forecast</div>')

card_columns = st.columns(4)

for index, target_name in enumerate(DEFAULT_TARGET_ORDER):
    with card_columns[index]:
        total_forecast = get_summary_value(forecast_summary, target_name, "total_forecast")
        average_forecast = get_summary_value(forecast_summary, target_name, "average_daily_forecast")
        model_name = get_summary_model(forecast_summary, target_name)

        render_html(
            (
                f'<div class="forecast-card">'
                f'<div class="forecast-label">{escape_text(TARGET_LABELS.get(target_name, target_name))}</div>'
                f'<div class="forecast-value">{format_count(total_forecast)}</div>'
                f'<div class="forecast-help">'
                f'Total future forecast. Daily average: <b>{format_count(average_forecast)}</b>. '
                f'Best model: <b>{escape_text(model_name)}</b>.'
                f'</div>'
                f'</div>'
            )
        )


# --------------------------------------------------
# 10. FORECAST STORY
# --------------------------------------------------

if not best_models.empty:
    model_list_html = ""

    for _, row in best_models.iterrows():
        target_name = str(row.get("target_name", "Unknown"))
        model_name = str(row.get("model_name", "Unknown"))
        mae = row.get("mae", np.nan)

        model_list_html += (
            f'<span class="model-pill">'
            f'{escape_text(TARGET_LABELS.get(target_name, target_name))}: '
            f'{escape_text(model_name)} · MAE {format_error(mae)}'
            f'</span>'
        )

    render_html(
        (
            f'<div class="story-card">'
            f'<div class="story-title">Forecast model decision</div>'
            f'<div class="story-text">'
            f'The pipeline compares multiple forecast approaches and selects the lowest-error model for each KPI. '
            f'This keeps the forecast practical: each business metric can use the model that performs best for that pattern.'
            f'<br><br>{model_list_html}'
            f'</div>'
            f'</div>'
        )
    )
else:
    render_html(
        '<div class="story-card">'
        '<div class="story-title">Forecast model decision</div>'
        '<div class="story-text">Forecast comparison data is not available yet. Run the forecasting pipeline to generate it.</div>'
        '</div>'
    )


# --------------------------------------------------
# 11. MAIN FORECAST CHARTS
# --------------------------------------------------

render_html('<div class="section-kicker">Forecast visuals</div><div class="section-title">Operational demand trends</div>')

if PLOTLY_AVAILABLE:
    combined_chart = build_future_multi_target_chart(best_future)

    if combined_chart is not None:
        st.plotly_chart(
            combined_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: future KPI forecast",
            "It shows the future forecast for operational KPIs such as traffic, events, conversions, and high-intent visitors.",
            "This helps plan capacity, campaign timing, and expected demand before the business period happens.",
        )

    selected_target = st.selectbox(
        "Select KPI for detailed forecast view",
        options=available_targets,
        format_func=lambda target: TARGET_LABELS.get(target, target),
    )

    detail_col_1, detail_col_2 = st.columns(2)

    with detail_col_1:
        history_chart = build_historical_forecast_chart(
            daily_kpis=daily_kpis,
            best_future=best_future,
            target_name=selected_target,
        )

        if history_chart is not None:
            st.plotly_chart(history_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: historical and future trend",
                "It combines past actual KPI values with the selected forecast for the chosen KPI.",
                "If the forecast follows the historical pattern, it is easier to trust for planning. Sudden changes should be reviewed before decisions are made.",
            )
        else:
            st.info("Historical forecast chart is not available for this KPI.")

    with detail_col_2:
        model_error_chart = build_model_error_chart(
            comparison=comparison,
            target_name=selected_target,
        )

        if model_error_chart is not None:
            st.plotly_chart(model_error_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: forecast model error",
                "It compares forecasting models by error. Lower error means the model predicted historical validation data more accurately.",
                "The best forecasting model should have the lowest error for the selected KPI, because that model is more reliable for future planning.",
            )
        else:
            st.info("Model error comparison is not available for this KPI.")

    backtest_chart = build_actual_vs_predicted_chart(
        history=history,
        target_name=selected_target,
    )

    if backtest_chart is not None:
        st.plotly_chart(backtest_chart, use_container_width=True)
        render_chart_explanation(
            "Chart explanation: actual versus predicted backtest",
            "It compares historical actual values with predicted values for the selected KPI.",
            "When predicted and actual lines stay close, the model is learning the demand pattern well. Large gaps show where forecast risk is higher.",
        )

else:
    st.info("Plotly is not installed. Forecast charts will appear after installing plotly.")


# --------------------------------------------------
# 12. RAW FORECAST TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Raw evidence</div><div class="section-title">Forecast tables</div>')

with st.expander("Future forecast table", expanded=True):
    if not best_future.empty:
        st.dataframe(
            best_future,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download best-model future forecast CSV",
            data=best_future.to_csv(index=False),
            file_name="business_forecast_future_best_models.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Future forecast table not available.")

with st.expander("Forecast model comparison table"):
    if not comparison.empty:
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Forecast comparison table not available.")

with st.expander("Historical KPI table"):
    if not daily_kpis.empty:
        st.dataframe(
            daily_kpis,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Historical KPI table not available.")

with st.expander("History with predictions table"):
    if not history.empty:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("History with predictions table not available.")


# --------------------------------------------------
# 13. HONEST SCOPE NOTE
# --------------------------------------------------

render_html(
    '<div class="story-card">'
    '<div class="story-title">Honest forecasting scope</div>'
    '<div class="story-text">'
    'This page forecasts operational demand: visitors, events, conversions, and high-intent audience volume. '
    'It does not forecast revenue because the source dataset contains behaviour events but no reliable order value.'
    '</div>'
    '</div>'
)

# --------------------------------------------------
# FINAL CLOSURE EXTENSION: FORECAST DECISION INTELLIGENCE
# --------------------------------------------------

from app.ui.operations_intelligence import (
    render_forecast_decision_intelligence,
)

render_forecast_decision_intelligence()
