"""Executive Overview for the E-Commerce Conversion Intelligence Platform.

Package 2 turns the landing page into a decision workspace. Every metric comes
from approved project evidence or from clearly labelled scenario assumptions.
"""

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


import pandas as pd
import streamlit as st

from app.app_utils import escape_text, inject_global_css, render_sidebar_html
from app.ui.components import (
    inject_product_css,
    render_detail_cards,
    render_empty_state,
    render_interpretation,
    render_kpi_grid,
    render_page_header,
    render_section_header,
    render_source_note,
    show_table_with_download,
)
from app.ui.executive_intelligence import (
    anomaly_summary,
    available_composition_views,
    build_anomaly_figure,
    build_composition_figure,
    build_efficiency_figure,
    build_forecast_figure,
    build_funnel_figure,
    build_funnel_table,
    build_readiness_figure,
    build_selected_campaign_audience,
    calculate_scenario,
    controlling_holdout_row,
    decision_summary,
    executive_brief_table,
    load_executive_evidence,
    readiness_table,
    source_visitor_count,
)


PACKAGE_02_EXEC_REQUIREMENTS = (
    "SVE-EXEC-01",
    "SVE-EXEC-02",
    "SVE-EXEC-03",
    "SVE-EXEC-04",
    "SVE-EXEC-05",
    "SVE-EXEC-06",
    "SVE-EXEC-07",
    "SVE-EXEC-08",
    "SVE-EXEC-09",
    "SVE-EXEC-10",
    "SVE-EXEC-11",
    "SVE-EXEC-12",
)


def render_executive_sidebar(
    *,
    model_name: str,
    threshold: float,
    champion_track: str,
    evidence_timestamp: str,
) -> None:
    """Render the polished Executive Overview sidebar.

    Inputs come from the same approved holdout and evidence objects used by the
    main page. Escaping the dynamic text keeps model metadata safe when it is
    inserted into the custom HTML. The sidebar is presentation-only: it does
    not change campaign calculations, model selection, or evidence loading.
    """

    safe_model_name = escape_text(model_name)
    safe_track = escape_text(champion_track)
    safe_timestamp = escape_text(evidence_timestamp)

    sidebar_html = f"""
        <style>
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
                padding-top: 0.85rem;
            }}

            .eci-sidebar-stack {{
                display: flex;
                flex-direction: column;
                gap: 0.82rem;
                padding: 0 0.34rem 1.35rem;
            }}

            .eci-sidebar-brand-card,
            .eci-sidebar-owner-card,
            .eci-sidebar-snapshot-card,
            .eci-sidebar-evidence-card {{
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 18px;
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.19);
            }}

            .eci-sidebar-brand-card {{
                position: relative;
                overflow: hidden;
                padding: 1.05rem;
                background:
                    radial-gradient(circle at 92% 6%, rgba(56, 189, 248, 0.24), transparent 39%),
                    radial-gradient(circle at 10% 98%, rgba(52, 211, 153, 0.12), transparent 42%),
                    linear-gradient(145deg, rgba(20, 37, 62, 0.98), rgba(11, 20, 35, 0.99));
                border-color: rgba(125, 211, 252, 0.24);
            }}

            .eci-sidebar-brand-card::after {{
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                background: linear-gradient(118deg, rgba(255, 255, 255, 0.035), transparent 43%);
            }}

            .eci-sidebar-brand-row {{
                position: relative;
                z-index: 1;
                display: flex;
                align-items: center;
                gap: 0.82rem;
            }}

            .eci-sidebar-mark {{
                display: grid;
                place-items: center;
                flex: 0 0 48px;
                width: 48px;
                height: 48px;
                border: 1px solid rgba(125, 211, 252, 0.38);
                border-radius: 15px;
                background: linear-gradient(145deg, rgba(56, 189, 248, 0.24), rgba(129, 140, 248, 0.14));
                color: #F8FAFC;
                font-size: 0.90rem;
                font-weight: 900;
                letter-spacing: 0.08em;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
            }}

            .eci-sidebar-kicker,
            .eci-sidebar-section-label {{
                color: #7DD3FC;
                font-size: 0.63rem;
                font-weight: 850;
                letter-spacing: 0.15em;
                text-transform: uppercase;
            }}

            .eci-sidebar-product-name {{
                margin-top: 0.20rem;
                color: #F8FAFC;
                font-size: 1.05rem;
                font-weight: 850;
                line-height: 1.08;
                letter-spacing: -0.025em;
            }}

            .eci-sidebar-product-copy {{
                position: relative;
                z-index: 1;
                margin: 0.86rem 0 0;
                color: #B6C3D5;
                font-size: 0.78rem;
                line-height: 1.50;
            }}

            .eci-sidebar-chip-row {{
                position: relative;
                z-index: 1;
                display: flex;
                flex-wrap: wrap;
                gap: 0.38rem;
                margin-top: 0.80rem;
            }}

            .eci-sidebar-chip {{
                padding: 0.28rem 0.48rem;
                border: 1px solid rgba(125, 211, 252, 0.18);
                border-radius: 999px;
                background: rgba(7, 13, 24, 0.45);
                color: #D8E7F5;
                font-size: 0.64rem;
                font-weight: 700;
            }}

            .eci-sidebar-owner-card,
            .eci-sidebar-snapshot-card {{
                padding: 0.92rem;
                background: linear-gradient(145deg, rgba(17, 28, 47, 0.94), rgba(13, 22, 38, 0.96));
            }}

            .eci-sidebar-owner-row {{
                display: flex;
                align-items: center;
                gap: 0.72rem;
                margin-top: 0.70rem;
            }}

            .eci-sidebar-avatar {{
                display: grid;
                place-items: center;
                flex: 0 0 42px;
                width: 42px;
                height: 42px;
                border-radius: 50%;
                background: linear-gradient(145deg, #38BDF8, #818CF8);
                color: #07111F;
                font-size: 0.76rem;
                font-weight: 950;
                letter-spacing: 0.04em;
                box-shadow: 0 8px 22px rgba(56, 189, 248, 0.20);
            }}

            .eci-sidebar-owner-name {{
                color: #F8FAFC;
                font-size: 0.89rem;
                font-weight: 820;
                line-height: 1.25;
            }}

            .eci-sidebar-owner-role {{
                margin-top: 0.14rem;
                color: #94A3B8;
                font-size: 0.72rem;
                font-weight: 650;
            }}

            .eci-sidebar-owner-note {{
                margin-top: 0.72rem;
                padding-top: 0.68rem;
                border-top: 1px solid rgba(148, 163, 184, 0.12);
                color: #8FA0B7;
                font-size: 0.68rem;
                line-height: 1.48;
            }}

            .eci-sidebar-snapshot-heading {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.54rem;
            }}

            .eci-sidebar-live-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.32rem;
                padding: 0.23rem 0.42rem;
                border: 1px solid rgba(52, 211, 153, 0.19);
                border-radius: 999px;
                background: rgba(52, 211, 153, 0.08);
                color: #8AF0C5;
                font-size: 0.60rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .eci-sidebar-live-dot {{
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #34D399;
                box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.11);
            }}

            .eci-sidebar-metric-row {{
                display: grid;
                grid-template-columns: minmax(0, 0.82fr) minmax(0, 1.18fr);
                gap: 0.48rem;
                align-items: start;
                padding: 0.57rem 0;
                border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            }}

            .eci-sidebar-metric-row:last-child {{
                padding-bottom: 0;
                border-bottom: 0;
            }}

            .eci-sidebar-metric-label {{
                color: #8494AA;
                font-size: 0.67rem;
                font-weight: 650;
            }}

            .eci-sidebar-metric-value {{
                color: #E8F2FA;
                font-size: 0.68rem;
                font-weight: 760;
                line-height: 1.36;
                text-align: right;
                overflow-wrap: anywhere;
            }}

            .eci-sidebar-evidence-card {{
                display: flex;
                align-items: flex-start;
                gap: 0.64rem;
                padding: 0.82rem 0.88rem;
                background: linear-gradient(145deg, rgba(14, 45, 42, 0.72), rgba(12, 31, 37, 0.86));
                border-color: rgba(52, 211, 153, 0.20);
            }}

            .eci-sidebar-evidence-icon {{
                display: grid;
                place-items: center;
                flex: 0 0 26px;
                width: 26px;
                height: 26px;
                border-radius: 9px;
                background: rgba(52, 211, 153, 0.13);
                color: #6EE7B7;
                font-size: 0.75rem;
                font-weight: 900;
            }}

            .eci-sidebar-evidence-title {{
                color: #DDFBF0;
                font-size: 0.72rem;
                font-weight: 800;
            }}

            .eci-sidebar-evidence-time {{
                margin-top: 0.18rem;
                color: #8DB6AA;
                font-size: 0.63rem;
                line-height: 1.40;
                overflow-wrap: anywhere;
            }}

            @media (max-width: 768px) {{
                .eci-sidebar-stack {{
                    padding-right: 0.18rem;
                    padding-left: 0.18rem;
                }}
            }}
        </style>

        <div class="eci-sidebar-stack">
            <section class="eci-sidebar-brand-card" aria-label="Product identity">
                <div class="eci-sidebar-brand-row">
                    <div class="eci-sidebar-mark" aria-hidden="true">CI</div>
                    <div>
                        <div class="eci-sidebar-kicker">E-commerce AI</div>
                        <div class="eci-sidebar-product-name">Conversion Intelligence</div>
                    </div>
                </div>
                <p class="eci-sidebar-product-copy">
                    Executive decision workspace for targeting, economics,
                    forecast evidence, and production readiness.
                </p>
                <div class="eci-sidebar-chip-row" aria-label="Workspace capabilities">
                    <span class="eci-sidebar-chip">Audience</span>
                    <span class="eci-sidebar-chip">Forecast</span>
                    <span class="eci-sidebar-chip">Monitoring</span>
                </div>
            </section>

            <section class="eci-sidebar-owner-card" aria-label="Portfolio owner">
                <div class="eci-sidebar-section-label">Designed &amp; built by</div>
                <div class="eci-sidebar-owner-row">
                    <div class="eci-sidebar-avatar" aria-hidden="true">RM</div>
                    <div>
                        <div class="eci-sidebar-owner-name">Ruturaj Mokashi</div>
                        <div class="eci-sidebar-owner-role">Data Analyst</div>
                    </div>
                </div>
                <div class="eci-sidebar-owner-note">
                    End-to-end analytics and machine-learning portfolio project,
                    built around governed evidence and business decisions.
                </div>
            </section>

            <section class="eci-sidebar-snapshot-card" aria-label="Model snapshot">
                <div class="eci-sidebar-snapshot-heading">
                    <div class="eci-sidebar-section-label">Model snapshot</div>
                    <span class="eci-sidebar-live-pill">
                        <span class="eci-sidebar-live-dot" aria-hidden="true"></span>
                        Validated
                    </span>
                </div>
                <div class="eci-sidebar-metric-row">
                    <span class="eci-sidebar-metric-label">Champion</span>
                    <span class="eci-sidebar-metric-value">{safe_model_name}</span>
                </div>
                <div class="eci-sidebar-metric-row">
                    <span class="eci-sidebar-metric-label">Threshold</span>
                    <span class="eci-sidebar-metric-value">{threshold:.2f}</span>
                </div>
                <div class="eci-sidebar-metric-row">
                    <span class="eci-sidebar-metric-label">Track</span>
                    <span class="eci-sidebar-metric-value">{safe_track}</span>
                </div>
            </section>

            <section class="eci-sidebar-evidence-card" aria-label="Evidence status">
                <div class="eci-sidebar-evidence-icon" aria-hidden="true">✓</div>
                <div>
                    <div class="eci-sidebar-evidence-title">Approved evidence loaded</div>
                    <div class="eci-sidebar-evidence-time">Refreshed {safe_timestamp}</div>
                </div>
            </section>
        </div>
        """

    # Streamlit renders custom HTML through a Markdown parser. Blank lines
    # between indented sibling <section> elements can end the raw HTML block
    # and make the remaining markup appear as a visible code block. Remove
    # only those separator lines so the complete sidebar stays one HTML block.
    sidebar_html = sidebar_html.replace(
        "\n\n            <section",
        "\n            <section",
    )

    render_sidebar_html(sidebar_html)



def operational_score_summary(
    scores: pd.DataFrame,
    production_threshold: float,
) -> dict[str, object]:
    """Summarise the current scored audience without mixing it with holdout rows.

    The holdout controls model-quality metrics. This helper controls only the
    current operational audience count and threshold decision.
    """

    result: dict[str, object] = {
        "available": False,
        "rows": 0,
        "eligible_rows": 0,
        "message": "Approved cached visitor-score evidence is unavailable.",
    }

    if scores.empty:
        return result

    visitor_column = next(
        (
            column
            for column in ("visitorid", "visitor_id", "visitorId")
            if column in scores.columns
        ),
        None,
    )
    score_column = next(
        (
            column
            for column in (
                "purchase_intent_score",
                "conversion_probability",
                "score",
                "probability",
            )
            if column in scores.columns
        ),
        None,
    )

    if visitor_column is None or score_column is None:
        result["message"] = (
            "Visitor-score evidence exists but required ID or score columns "
            "are missing."
        )
        return result

    current = scores[[visitor_column, score_column]].copy()
    current[visitor_column] = (
        current[visitor_column].astype("string").str.strip()
    )
    current[score_column] = pd.to_numeric(
        current[score_column],
        errors="coerce",
    )
    current = (
        current.dropna(subset=[visitor_column, score_column])
        .drop_duplicates(subset=[visitor_column], keep="first")
        .copy()
    )

    if current.empty:
        result["message"] = "Visitor-score evidence contains no usable rows."
        return result

    if "predicted_conversion" in scores.columns:
        decision_source = scores[
            [visitor_column, "predicted_conversion"]
        ].copy()
        decision_source[visitor_column] = (
            decision_source[visitor_column]
            .astype("string")
            .str.strip()
        )
        decision_source["predicted_conversion"] = pd.to_numeric(
            decision_source["predicted_conversion"],
            errors="coerce",
        )
        decision_source = decision_source.drop_duplicates(
            subset=[visitor_column],
            keep="first",
        )
        current = current.merge(
            decision_source,
            on=visitor_column,
            how="left",
            validate="one_to_one",
        )
        eligible = (
            current["predicted_conversion"]
            .fillna(0)
            .astype(int)
            .eq(1)
        )
    elif "production_threshold" in scores.columns:
        threshold_source = scores[
            [visitor_column, "production_threshold"]
        ].copy()
        threshold_source[visitor_column] = (
            threshold_source[visitor_column]
            .astype("string")
            .str.strip()
        )
        threshold_source["production_threshold"] = pd.to_numeric(
            threshold_source["production_threshold"],
            errors="coerce",
        )
        threshold_source = threshold_source.drop_duplicates(
            subset=[visitor_column],
            keep="first",
        )
        current = current.merge(
            threshold_source,
            on=visitor_column,
            how="left",
            validate="one_to_one",
        )
        threshold_values = current["production_threshold"].fillna(
            float(production_threshold)
        )
        eligible = current[score_column].ge(threshold_values)
    else:
        eligible = current[score_column].ge(float(production_threshold))

    result.update(
        {
            "available": True,
            "rows": int(len(current)),
            "eligible_rows": int(eligible.sum()),
            "message": (
                f"{int(eligible.sum()):,} of {len(current):,} currently scored "
                "visitors pass the production threshold."
            ),
        }
    )
    return result


st.set_page_config(
    page_title="Executive Overview | Conversion Intelligence",
    page_icon="🧠",
    layout="wide",
)

inject_global_css()
inject_product_css()

evidence = load_executive_evidence()

# The Executive Overview reads validated metrics directly from the final
# holdout table. It must not load the full production model bundle merely to
# display model metadata.
fallback = {
    "model_name": "Unavailable",
    "threshold": 0.0,
    "rows": 0,
    "positive_rows": 0,
    "positive_rate": 0.0,
    "predicted_positive_rows": 0,
    "predicted_positive_rate": 0.0,
    "pr_auc": 0.0,
    "roc_auc": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
}
holdout = controlling_holdout_row(evidence.holdout, fallback)

champion_track = (
    "Final true champion"
    if not evidence.holdout.empty
    else "Evidence unavailable"
)

holdout_rows = int(holdout["rows"])
holdout_eligible_visitors = int(holdout["predicted_positive_rows"])
natural_rate = float(holdout["positive_rate"])
precision = float(holdout["precision"])
lift = precision / natural_rate if natural_rate > 0 else 0.0

operational_scores = operational_score_summary(
    evidence.scores,
    float(holdout["threshold"]),
)
source_visitors = int(operational_scores["rows"])
eligible_visitors = int(operational_scores["eligible_rows"])

render_executive_sidebar(
    model_name=str(holdout["model_name"]),
    threshold=float(holdout["threshold"]),
    champion_track=champion_track,
    evidence_timestamp=evidence.source_timestamp,
)

render_page_header(
    eyebrow="E-Commerce Conversion Intelligence Platform",
    title="Turn visitor intent into a measurable campaign decision.",
    subtitle=(
        "A single executive workspace for audience selection, campaign "
        "economics, model evidence, forecasts, anomaly exposure, and "
        "production readiness."
    ),
    badges=[
        (f"Champion: {holdout['model_name']}", "positive"),
        (f"Threshold: {float(holdout['threshold']):.2f}", "info"),
        (f"Track: {champion_track}", "neutral"),
        ("Evidence: validated holdout", "positive"),
    ],
)

render_kpi_grid(
    [
        (
            "Validated target quality",
            f"{precision:.1%}",
            "Expected buyers among threshold-selected visitors",
        ),
        (
            "Natural buyer rate",
            f"{natural_rate:.2%}",
            "Observed buyer rate in the final holdout",
        ),
        (
            "Targeting lift",
            f"{lift:.1f}×",
            "Selected-audience quality versus random targeting",
        ),
        (
            "Current threshold-eligible audience",
            (
                f"{eligible_visitors:,}"
                if operational_scores["available"]
                else "Unavailable"
            ),
            (
                f"From {source_visitors:,} currently scored visitors"
                if operational_scores["available"]
                else str(operational_scores["message"])
            ),
        ),
    ]
)

executive_sources = [
    "reports/tables/final_true_champion_holdout.csv",
    "reports/tables/ml_validation_projection_sample.csv",
    "reports/metadata/ml_validation_manifest.json",
]
if evidence.score_source:
    executive_sources.append(evidence.score_source)

render_source_note(
    source="; ".join(executive_sources),
    evidence_type=(
        "Untouched-holdout model evidence and current cached audience scores"
    ),
    refreshed_at=evidence.source_timestamp,
    extra=(
        "Final model metrics come only from the untouched holdout. Current "
        "audience counts and composition come only from cached visitor scores. "
        "Campaign value applies holdout precision as an explicit planning "
        "assumption because the source data has no reliable transaction revenue."
    ),
)

render_section_header(
    "Campaign scenario",
    "Choose the audience and test the commercial assumptions",
    (
        "The current score file supplies the operational audience. Validated "
        "holdout precision supplies the expected buyer-rate assumption. Contact "
        "cost and buyer value remain editable scenario assumptions."
    ),
)

if not operational_scores["available"]:
    render_empty_state(
        title="Current campaign audience is unavailable",
        message=str(operational_scores["message"]),
        next_action=(
            "Publish or regenerate the approved current visitor-score file. "
            "The page will not reuse holdout-positive rows as a live audience."
        ),
    )

# STREAMLIT CLOUD: safely handle zero or one current eligible visitor.
maximum_target = max(eligible_visitors, 0)
default_target = (
    min(maximum_target, max(1, round(maximum_target * 0.5)))
    if maximum_target > 0
    else 0
)

control_1, control_2, control_3 = st.columns(3)

with control_1:
    if maximum_target <= 1:
        target_size = maximum_target
        st.metric("Campaign target size", f"{target_size:,}")
        st.caption(
            "The available evidence currently has no adjustable target range."
        )
    else:
        target_size = st.slider(
            "Campaign target size",
            min_value=1,
            max_value=maximum_target,
            value=default_target,
            help="Cannot exceed the current threshold-eligible scored audience.",
        )

with control_2:
    contact_cost = st.number_input(
        "Contact cost per visitor",
        min_value=0.0,
        value=1.00,
        step=0.10,
        format="%.2f",
    )

with control_3:
    buyer_value = st.number_input(
        "Assumed value per buyer",
        min_value=0.0,
        value=50.00,
        step=5.00,
        format="%.2f",
    )

scenario = calculate_scenario(
    available_targets=eligible_visitors,
    target_size=target_size,
    precision=precision,
    baseline_rate=natural_rate,
    contact_cost=contact_cost,
    buyer_value=buyer_value,
)

render_kpi_grid(
    [
        (
            "Expected buyers",
            f"{scenario['expected_buyers']:.1f}",
            f"Holdout precision applied to {target_size:,} current selected visitors",
        ),
        (
            "Campaign cost",
            f"{scenario['campaign_cost']:,.2f}",
            "Target size × contact cost",
        ),
        (
            "Net expected value",
            f"{scenario['net_value']:,.2f}",
            "Expected buyer value minus campaign cost",
        ),
        (
            "Scenario ROI",
            f"{scenario['roi']:.1%}",
            "Assumption-based return before operational overhead",
        ),
    ]
)

funnel = pd.DataFrame(
    [
        ("Current scored visitors", max(source_visitors, 0)),
        ("Current threshold eligible", max(eligible_visitors, 0)),
        ("Campaign target", max(target_size, 0)),
        ("Expected buyers", max(float(scenario["expected_buyers"]), 0.0)),
    ],
    columns=["Stage", "Visitors"],
)

render_section_header(
    "Audience decision",
    "From broad traffic to a measurable buyer opportunity",
)

st.plotly_chart(
    build_funnel_figure(funnel),
    width="stretch",
    key="exec_targeting_funnel",
)
render_interpretation(
    what_it_shows=(
        "The operational decision path from currently scored visitors to "
        "threshold eligibility, campaign selection, and expected buyers."
    ),
    how_to_read=(
        "Every population stage uses the current scored audience. The final "
        "stage is an expectation calculated with untouched-holdout precision, "
        "not an observed campaign result."
    ),
    actual_finding=(
        f"{target_size:,} selected visitors are expected to contain "
        f"{scenario['expected_buyers']:.1f} buyers."
    ),
    conclusion=(
        "The model supports a deliberately narrow audience with much higher "
        "buyer concentration than random outreach."
    ),
    recommended_action=(
        "Use the campaign target as the operational shortlist and retain a "
        "separate control group for real campaign measurement."
    ),
    limitation=(
        "Audience counts come from current cached scores, while expected "
        "buyer yield comes from holdout precision and remains an assumption "
        "until campaign outcomes arrive."
    ),
)

st.plotly_chart(
    build_efficiency_figure(
        source_visitors=source_visitors,
        target_size=target_size,
        expected_buyers=scenario["expected_buyers"],
        baseline_rate=natural_rate,
    ),
    width="stretch",
    key="exec_efficiency_comparison",
)
render_interpretation(
    what_it_shows=(
        "Selected-audience share, expected buyer yield, and natural random "
        "buyer yield in one comparable view."
    ),
    how_to_read=(
        "A small audience share with a much larger buyer yield represents "
        "efficient prioritisation."
    ),
    actual_finding=(
        f"The scenario targets {(target_size / source_visitors if source_visitors else 0.0):.2%} of currently scored "
        f"visitors with an expected {precision:.1%} buyer yield based on the holdout."
    ),
    conclusion=(
        "Campaign efficiency comes from concentrating spend on the strongest "
        "validated intent signals."
    ),
    recommended_action=(
        "Start with the highest-ranked eligible visitors and increase target "
        "size only when incremental economics remain positive."
    ),
    limitation=(
        "Precision may change when audience size, channel, season, or live "
        "visitor behaviour differs from the holdout."
    ),
)

render_section_header(
    "Audience composition",
    "Understand who is inside the selected campaign audience",
)

selected_audience, composition_message = build_selected_campaign_audience(
    evidence.projection,
    evidence.scores,
    target_size,
)
composition_views = available_composition_views(selected_audience)

if not composition_views:
    render_empty_state(
        title="Selected-audience composition is unavailable",
        message=composition_message,
        next_action=(
            "Regenerate the approved final visitor-score file before using "
            "composition evidence for campaign activation."
        ),
    )
else:
    composition_view = st.radio(
        "Composition view",
        composition_views,
        horizontal=True,
    )

    composition_figure, composition_table = build_composition_figure(
        selected_audience,
        composition_view,
    )

    st.plotly_chart(
        composition_figure,
        width="stretch",
        key=f"exec_composition_{composition_view}",
    )
    render_interpretation(
        what_it_shows=(
            f"Distribution of the current threshold-selected campaign "
            f"audience by {composition_view.lower()}."
        ),
        how_to_read=(
            "Longer bars represent more selected visitors in that category. "
            "The category totals reconcile with the selected score rows, not "
            "with the complete validation projection."
        ),
        actual_finding=(
            f"{int(composition_table['Visitors'].sum()):,} selected visitors "
            f"are represented across {len(composition_table):,} categories. "
            f"{composition_message}"
        ),
        conclusion=(
            "Audience composition should influence campaign treatment rather "
            "than using one message for every selected visitor."
        ),
        recommended_action=(
            "Investigate the largest and highest-value groups on the "
            "segmentation and batch-scoring pages before activation."
        ),
        limitation=(
            "Intent tier comes from approved purchase-intent scores. Segment "
            "and anomaly attributes are available only where selected visitor "
            "IDs also appear in the governed projection sample; uncovered "
            "visitors remain visibly labelled rather than being replaced by "
            "the full sample."
        ),
    )

render_section_header(
    "Champion evidence",
    "Model quality, threshold behaviour, and reliability",
)

render_kpi_grid(
    [
        ("PR-AUC", f"{float(holdout['pr_auc']):.3f}", "Rare-buyer ranking"),
        ("ROC-AUC", f"{float(holdout['roc_auc']):.3f}", "Overall discrimination"),
        ("Precision", f"{precision:.1%}", "Target quality"),
        ("Recall", f"{float(holdout['recall']):.1%}", "Buyer capture"),
        ("F1 score", f"{float(holdout['f1_score']):.3f}", "Precision-recall balance"),
        (
            "Predicted-positive rate",
            f"{float(holdout['predicted_positive_rate']):.3%}",
            "Share passing the threshold",
        ),
    ]
)

render_detail_cards(
    [
        (
            "Evidence type",
            "Final untouched holdout",
            (
                f"{int(holdout['rows']):,} rows, "
                f"{int(holdout['positive_rows']):,} observed buyers"
            ),
        ),
        (
            "Decision rule",
            f"Score ≥ {float(holdout['threshold']):.2f}",
            (
                f"{holdout_eligible_visitors:,} holdout visitors pass the production threshold"
            ),
        ),
        (
            "Reliability",
            "Probability caution",
            (
                "Ranking quality is validated; business use should focus on "
                "threshold decisions rather than treating scores as guaranteed "
                "purchase probabilities."
            ),
        ),
    ]
)

render_section_header(
    "Forward outlook",
    "Compact KPI forecast with uncertainty when available",
)

forecast_figure, forecast_message = build_forecast_figure(evidence.forecast)

if forecast_figure is None:
    render_empty_state(
        title="Forecast evidence is not available",
        message=forecast_message,
        next_action=(
            "Generate the approved KPI forecast summary before treating "
            "forward demand as production evidence."
        ),
    )
else:
    st.plotly_chart(
        forecast_figure,
        width="stretch",
        key="exec_forward_outlook",
    )
    render_interpretation(
        what_it_shows="The approved forward KPI forecast and its uncertainty.",
        how_to_read=(
            "The central line is the expected path. The shaded interval is the "
            "range of plausible outcomes when interval data is available."
        ),
        actual_finding=forecast_message,
        conclusion=(
            "Forward planning should use the central estimate together with "
            "uncertainty, not the point forecast alone."
        ),
        recommended_action=(
            "Use the forecast for capacity and campaign timing, then review "
            "actual-versus-forecast error after each period."
        ),
        limitation=(
            "Forecast quality depends on the historical horizon, event "
            "coverage, and stability of future visitor behaviour."
        ),
    )

render_source_note(
    source=evidence.forecast_source or "No approved forecast source",
    evidence_type=(
        "Approved forecast evidence"
        if forecast_figure is not None
        else "Forecast availability check"
    ),
    refreshed_at=evidence.source_timestamp,
    extra=forecast_message,
)

render_section_header(
    "Risk and operations",
    "Anomaly exposure and production-readiness evidence",
)

anomaly = anomaly_summary(evidence.projection)
anomaly_figure = build_anomaly_figure(evidence.projection)

if anomaly_figure is None:
    render_empty_state(
        title="Anomaly evidence is unavailable",
        message="No LOF score evidence was found in the validation projection.",
        next_action="Regenerate Package 1 ML validation evidence.",
    )
else:
    st.plotly_chart(
        anomaly_figure,
        width="stretch",
        key="exec_anomaly_distribution",
    )
    render_interpretation(
        what_it_shows=(
            "Current LOF anomaly-score distribution for the governed "
            "validation sample."
        ),
        how_to_read=(
            "Higher scores indicate stronger local deviation. Flagged visitors "
            "require investigation, not automatic removal."
        ),
        actual_finding=(
            f"{anomaly['count']:,} visitors are flagged "
            f"({anomaly['rate']:.1%}); the most affected segment is "
            f"{anomaly['key_segment']}."
        ),
        conclusion=(
            "Anomalies represent a small but important operational review queue."
        ),
        recommended_action=(
            "Review severe cases before campaign activation and compare them "
            "with conversion outcomes when labels arrive."
        ),
        limitation=(
            "LOF is sample-relative and does not prove fraud, data error, or "
            "business harm."
        ),
    )

readiness = readiness_table()

st.plotly_chart(
    build_readiness_figure(readiness),
    width="stretch",
    key="exec_readiness_strip",
)
render_interpretation(
    what_it_shows=(
        "Application, scoring, drift, alerting, and delayed-label readiness "
        "based on current project artifacts."
    ),
    how_to_read=(
        "Ready means supporting repository evidence exists. Unavailable means "
        "no matching artifact was found."
    ),
    actual_finding=(
        f"{int(readiness['Status'].eq('Ready').sum())} of "
        f"{len(readiness)} readiness areas currently have evidence."
    ),
    conclusion=(
        "Model quality and operational readiness must be reviewed together "
        "before production activation."
    ),
    recommended_action=(
        "Open the Monitoring and MLOps Architecture pages for detailed checks "
        "and closure actions."
    ),
    limitation=(
        "Artifact presence proves implementation evidence, not live service "
        "availability."
    ),
)

summary = {
    "finding": (
        f"Targeting {target_size:,} current threshold-eligible visitors is "
        f"expected to reach {scenario['expected_buyers']:.1f} buyers when "
        "untouched-holdout precision is used as the planning assumption."
    ),
    "action": (
        f"Prioritise the selected current audience while monitoring a "
        f"{anomaly['rate']:.1%} validation-sample anomaly rate and an "
        f"assumption-based ROI of {scenario['roi']:.1%}."
    ),
    "limitation": (
        "Audience membership comes from cached current scores. Buyer yield and "
        "campaign value remain scenario estimates until observed campaign "
        "outcomes and real transaction values are available."
    ),
}

render_section_header(
    "Executive decision",
    "Finding, action, and honest limitation",
)

render_detail_cards(
    [
        ("Finding", "Current scenario", summary["finding"]),
        ("Action", "Recommended next move", summary["action"]),
        ("Limitation", "What this does not prove", summary["limitation"]),
    ]
)

render_section_header(
    "Investigate further",
    "Move from executive signal to detailed evidence",
)

link_1, link_2, link_3, link_4, link_5 = st.columns(5)

with link_1:
    st.page_link(
        "pages/3_Model_Benchmark_Selection.py",
        label="Model evidence",
        icon="🏁",
    )
with link_2:
    st.page_link(
        "pages/2_Batch_Scoring.py",
        label="Audience scoring",
        icon="📦",
    )
with link_3:
    st.page_link(
        "pages/4_Business_KPI_Forecasting.py",
        label="Forecasts",
        icon="📈",
    )
with link_4:
    st.page_link(
        "pages/5_Anomaly_Outlier.py",
        label="Anomalies",
        icon="🚨",
    )
with link_5:
    st.page_link(
        "pages/6_Monitoring_Drift_Health.py",
        label="Monitoring",
        icon="🩺",
    )

brief = executive_brief_table(
    holdout=holdout,
    scenario=scenario,
    anomaly=anomaly,
    readiness=readiness,
    source_timestamp=evidence.source_timestamp,
)

show_table_with_download(
    label="Filtered executive brief",
    data=brief,
    file_name="executive_decision_brief.csv",
    metadata={
        "evidence_timestamp": evidence.source_timestamp,
        "champion_model": str(holdout["model_name"]),
        "threshold": f"{float(holdout['threshold']):.4f}",
    },
    description=(
        "Download the current decision scenario with its evidence labels "
        "and provenance metadata."
    ),
)

render_source_note(
    source=(
        "Final holdout, ML validation evidence, available forecast summary, "
        "and repository operational artifacts"
    ),
    evidence_type="Validated evidence plus labelled scenario assumptions",
    refreshed_at=evidence.source_timestamp,
    extra=(
        "The brief combines validated model evidence with user-controlled "
        "campaign assumptions. Assumption rows are labelled as Scenario."
    ),
)
