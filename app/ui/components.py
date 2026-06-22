"""Reusable Streamlit components for a consistent product experience.

The functions in this module keep page code small and readable. Each component
explains one job: page identity, section hierarchy, KPI context, visual framing,
interpretation, evidence, or unavailable-data guidance.
"""

from __future__ import annotations

import html
from typing import Iterable, Mapping

import pandas as pd
import streamlit as st

from app.ui.design_system import product_css
from app.ui.plotly_style import get_chart_config


def _escape(value: object) -> str:
    """Escape dynamic content before rendering it inside HTML."""

    return html.escape(str(value))


def inject_product_css() -> None:
    """Apply the shared visual design system to the current page."""

    st.markdown(product_css(), unsafe_allow_html=True)


def render_page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    badges: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Render a polished page header with optional evidence badges.

    Badge tone values:
        - positive
        - info
        - warning
        - neutral
    """

    badge_html = ""

    if badges:
        items = []
        for text, tone in badges:
            safe_tone = tone if tone in {"positive", "info", "warning"} else "neutral"
            items.append(
                f'<span class="eci-badge eci-badge--{safe_tone}">{_escape(text)}</span>'
            )
        badge_html = f'<div class="eci-badge-row">{"".join(items)}</div>'

    st.markdown(
        f"""
        <section class="eci-page-hero">
            <div class="eci-eyebrow">{_escape(eyebrow)}</div>
            <h1 class="eci-page-title">{_escape(title)}</h1>
            <p class="eci-page-subtitle">{_escape(subtitle)}</p>
            {badge_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    kicker: str,
    title: str,
    description: str | None = None,
) -> None:
    """Render one clear section boundary before related content."""

    description_html = ""
    if description:
        description_html = (
            f'<div class="eci-section-description">{_escape(description)}</div>'
        )

    st.markdown(
        f"""
        <header class="eci-section-header">
            <div class="eci-section-kicker">{_escape(kicker)}</div>
            <h2 class="eci-section-title">{_escape(title)}</h2>
            {description_html}
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, context: str) -> None:
    """Render one consistent KPI card with business context."""

    st.markdown(
        f"""
        <div class="eci-kpi-card">
            <div class="eci-kpi-label">{_escape(label)}</div>
            <div class="eci-kpi-value">{_escape(value)}</div>
            <div class="eci-kpi-context">{_escape(context)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_kpi_grid(cards: Iterable[tuple[str, str, str]]) -> None:
    """Render a stable responsive KPI grid without Streamlit column wrapping."""

    card_html = "".join(
        '<article class="eci-kpi-card">'
        f'<div class="eci-kpi-label">{_escape(label)}</div>'
        f'<div class="eci-kpi-value">{_escape(value)}</div>'
        f'<div class="eci-kpi-context">{_escape(context)}</div>'
        "</article>"
        for label, value, context in cards
    )

    st.markdown(
        f'<section class="eci-kpi-grid">{card_html}</section>',
        unsafe_allow_html=True,
    )


def render_detail_cards(cards: Iterable[tuple[str, str, str]]) -> None:
    """Render compact explanatory cards for components, rules, or actions."""

    card_html = "".join(
        '<article class="eci-detail-card">'
        f'<div class="eci-detail-label">{_escape(label)}</div>'
        f'<div class="eci-detail-title">{_escape(title)}</div>'
        f'<div class="eci-detail-text">{_escape(text)}</div>'
        "</article>"
        for label, title, text in cards
    )
    st.markdown(
        f'<section class="eci-detail-grid">{card_html}</section>',
        unsafe_allow_html=True,
    )

def render_chart_header(title: str, subtitle: str) -> None:
    """Render a decision-focused title directly above a major chart."""

    st.markdown(
        f"""
        <div class="eci-chart-shell">
            <div class="eci-chart-title">{_escape(title)}</div>
            <div class="eci-chart-subtitle">{_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_interpretation(
    *,
    what_it_shows: str,
    how_to_read: str,
    actual_finding: str,
    conclusion: str,
    recommended_action: str,
    limitation: str,
) -> None:
    """Explain a visual with six compact, consistently styled insights.

    The HTML is deliberately assembled without leading indentation. Markdown
    treats four-space-indented HTML as a code block, which previously exposed
    raw ``<div>`` tags in the rendered application.
    """

    items = [
        ("What it shows", what_it_shows),
        ("How to read it", how_to_read),
        ("Actual finding", actual_finding),
        ("Business conclusion", conclusion),
        ("Recommended action", recommended_action),
        ("Limitation", limitation),
    ]

    item_html = "".join(
        '<article class="eci-interpretation-item">'
        f'<div class="eci-interpretation-label">{_escape(label)}</div>'
        f'<div class="eci-interpretation-text">{_escape(text)}</div>'
        "</article>"
        for label, text in items
    )

    st.markdown(
        '<section class="eci-interpretation" '
        'aria-label="Chart interpretation">'
        f"{item_html}</section>",
        unsafe_allow_html=True,
    )

def render_source_note(
    *,
    source: str,
    evidence_type: str,
    refreshed_at: str,
    extra: str | None = None,
) -> None:
    """Show the data source and evidence status below a visual."""

    parts = [
        f"Source: {source}",
        f"Evidence: {evidence_type}",
        f"Generated or refreshed: {refreshed_at}",
    ]
    if extra:
        parts.append(extra)

    st.markdown(
        f'<div class="eci-source-note">{_escape(" · ".join(parts))}</div>',
        unsafe_allow_html=True,
    )


def render_evidence_chart(
    *,
    figure: object,
    key: str,
    what_it_shows: str,
    how_to_read: str,
    actual_finding: str,
    conclusion: str,
    recommended_action: str,
    limitation: str,
    source: str,
    evidence_type: str,
    refreshed_at: str,
    source_extra: str | None = None,
) -> bool:
    """Render one chart with its required interpretation and provenance.

    Returning ``False`` lets the caller show a clear unavailable-data state.
    The component enforces the Package 1 rule that no major visual appears
    without an actual-number finding, action, limitation, source, and date.
    """

    if figure is None:
        return False

    st.plotly_chart(
        figure,
        use_container_width=True,
        config=get_chart_config(),
        key=key,
    )
    render_interpretation(
        what_it_shows=what_it_shows,
        how_to_read=how_to_read,
        actual_finding=actual_finding,
        conclusion=conclusion,
        recommended_action=recommended_action,
        limitation=limitation,
    )
    render_source_note(
        source=source,
        evidence_type=evidence_type,
        refreshed_at=refreshed_at,
        extra=source_extra,
    )
    return True


def render_empty_state(title: str, message: str, next_action: str) -> None:
    """Render a helpful unavailable-data state instead of a blank area."""

    st.markdown(
        f"""
        <div class="eci-empty-state">
            <div class="eci-empty-title">{_escape(title)}</div>
            <div class="eci-empty-text">
                {_escape(message)}<br><br>
                <strong>Next action:</strong> {_escape(next_action)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_table_with_download(
    *,
    label: str,
    data: pd.DataFrame,
    file_name: str,
    metadata: Mapping[str, str] | None = None,
    description: str | None = None,
) -> None:
    """Show a styled table and export it with governance metadata."""

    if data.empty:
        render_empty_state(
            title=f"{label} is not available",
            message="The expected evidence table is currently empty or missing.",
            next_action="Regenerate the project evidence outputs, then refresh this page.",
        )
        return

    if description:
        st.caption(description)

    display_data = data.copy()
    st.dataframe(display_data, use_container_width=True, hide_index=True)

    export_data = display_data.copy()
    if metadata:
        for key, value in metadata.items():
            export_data[f"export_{key}"] = value

    st.download_button(
        label=f"Download {label}",
        data=export_data.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )
