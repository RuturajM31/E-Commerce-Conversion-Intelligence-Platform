"""Central design tokens for the Streamlit visual intelligence platform.

Why this file exists:
    Every page should use the same colors, spacing, typography, chart surfaces,
    and semantic states. Central tokens prevent copied-and-pasted styling from
    slowly making the application inconsistent.

Used next:
    - app.ui.components renders shared page and content components.
    - app.ui.plotly_style applies the same visual language to Plotly figures.
"""

from __future__ import annotations

from typing import Final


FONT_FAMILY: Final[str] = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"

COLORS: Final[dict[str, str]] = {
    "app_bg": "#070D18",
    "sidebar_bg": "#111522",
    "surface_1": "#0D1626",
    "surface_2": "#111C2F",
    "surface_3": "#17233A",
    "border": "#263753",
    "border_soft": "rgba(148, 163, 184, 0.18)",
    "text": "#F8FAFC",
    "text_muted": "#B6C3D5",
    "text_subtle": "#8393A9",
    "brand": "#38BDF8",
    "brand_soft": "#7DD3FC",
    "positive": "#34D399",
    "warning": "#FBBF24",
    "critical": "#FB7185",
    "information": "#818CF8",
    "neutral": "#94A3B8",
}

CLUSTER_PALETTE: Final[list[str]] = [
    "#38BDF8",
    "#34D399",
    "#FBBF24",
    "#A78BFA",
    "#FB7185",
    "#22D3EE",
    "#F97316",
    "#60A5FA",
    "#4ADE80",
    "#E879F9",
]

SEMANTIC_PALETTE: Final[dict[str, str]] = {
    "Normal": COLORS["brand"],
    "Watch": COLORS["warning"],
    "High": "#F97316",
    "Critical": COLORS["critical"],
    "Core": COLORS["brand"],
    "Border": COLORS["information"],
    "Noise": COLORS["neutral"],
}

CHART_CONFIG: Final[dict[str, object]] = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "ecommerce_visual_intelligence_chart",
        "height": 900,
        "width": 1600,
        "scale": 2,
    },
}


def product_css() -> str:
    """Return the complete shared product stylesheet.

    The stylesheet preserves the approved dark identity while improving finish,
    spacing, typography, component hierarchy, responsiveness, and visual room.
    """

    return f"""
    <style>
        :root {{
            --eci-bg: {COLORS['app_bg']};
            --eci-sidebar: {COLORS['sidebar_bg']};
            --eci-surface-1: {COLORS['surface_1']};
            --eci-surface-2: {COLORS['surface_2']};
            --eci-surface-3: {COLORS['surface_3']};
            --eci-border: {COLORS['border']};
            --eci-text: {COLORS['text']};
            --eci-muted: {COLORS['text_muted']};
            --eci-subtle: {COLORS['text_subtle']};
            --eci-brand: {COLORS['brand']};
            --eci-positive: {COLORS['positive']};
            --eci-warning: {COLORS['warning']};
            --eci-critical: {COLORS['critical']};
        }}

        html, body, [class*="css"] {{
            font-family: {FONT_FAMILY};
        }}

        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 82% 4%, rgba(56, 189, 248, 0.055), transparent 28%),
                linear-gradient(180deg, #070D18 0%, #090F1B 100%);
        }}

        [data-testid="stMain"] .block-container {{
            max-width: 1580px;
            padding-top: 1.35rem;
            padding-right: clamp(1.35rem, 3vw, 3.2rem);
            padding-bottom: 5rem;
            padding-left: clamp(1.35rem, 3vw, 3.2rem);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #151824 0%, #10131E 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }}

        [data-testid="stSidebarNav"] a {{
            border-radius: 10px;
            margin: 2px 8px;
            transition: background 140ms ease, transform 140ms ease;
        }}

        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(56, 189, 248, 0.08);
            transform: translateX(2px);
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.16), rgba(56, 189, 248, 0.05));
            border: 1px solid rgba(56, 189, 248, 0.22);
        }}

        .eci-page-hero {{
            position: relative;
            overflow: hidden;
            padding: clamp(22px, 3vw, 34px);
            border-radius: 22px;
            background:
                radial-gradient(circle at 8% 14%, rgba(56, 189, 248, 0.20), transparent 34%),
                radial-gradient(circle at 92% 88%, rgba(52, 211, 153, 0.12), transparent 34%),
                linear-gradient(135deg, rgba(13, 22, 38, 0.98), rgba(17, 28, 47, 0.96));
            border: 1px solid rgba(125, 211, 252, 0.20);
            box-shadow: 0 22px 58px rgba(0, 0, 0, 0.30);
            margin-bottom: 18px;
        }}

        .eci-page-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(115deg, rgba(255,255,255,0.025), transparent 34%);
        }}

        .eci-eyebrow {{
            color: var(--eci-brand);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.17em;
            text-transform: uppercase;
            margin-bottom: 11px;
        }}

        .eci-page-title {{
            color: var(--eci-text);
            font-size: clamp(1.85rem, 3.3vw, 2.70rem);
            line-height: 1.04;
            font-weight: 850;
            letter-spacing: -0.035em;
            margin: 0 0 14px;
        }}

        .eci-page-subtitle {{
            color: var(--eci-muted);
            font-size: clamp(0.94rem, 1.25vw, 1.04rem);
            line-height: 1.58;
            max-width: 1080px;
            margin: 0;
        }}

        .eci-badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            margin-top: 15px;
        }}

        .eci-badge {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 760;
            letter-spacing: 0.035em;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(148, 163, 184, 0.08);
            color: var(--eci-muted);
        }}

        .eci-badge--positive {{
            color: #A7F3D0;
            border-color: rgba(52, 211, 153, 0.30);
            background: rgba(52, 211, 153, 0.10);
        }}

        .eci-badge--info {{
            color: #BAE6FD;
            border-color: rgba(56, 189, 248, 0.30);
            background: rgba(56, 189, 248, 0.10);
        }}

        .eci-badge--warning {{
            color: #FDE68A;
            border-color: rgba(251, 191, 36, 0.30);
            background: rgba(251, 191, 36, 0.10);
        }}

        .eci-section-header {{
            margin-top: 34px;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.13);
        }}

        .eci-section-kicker {{
            color: #7DD3FC;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}

        .eci-section-title {{
            color: var(--eci-text);
            font-size: clamp(1.45rem, 2.3vw, 2.0rem);
            line-height: 1.22;
            font-weight: 820;
            letter-spacing: -0.022em;
            margin: 0;
        }}

        .eci-section-description {{
            color: var(--eci-muted);
            font-size: 0.96rem;
            line-height: 1.62;
            max-width: 1050px;
            margin-top: 8px;
        }}

        .eci-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 0 0 20px;
        }}

        .eci-kpi-card {{
            min-height: 118px;
            padding: 17px 18px 16px;
            border-radius: 16px;
            background:
                linear-gradient(145deg, rgba(17, 28, 47, 0.98), rgba(13, 22, 38, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.23);
            transition: transform 150ms ease, border-color 150ms ease;
        }}

        .eci-kpi-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.30);
        }}

        .eci-kpi-label {{
            color: var(--eci-subtle);
            font-size: 0.74rem;
            font-weight: 780;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 9px;
        }}

        .eci-kpi-value {{
            color: var(--eci-text);
            font-size: clamp(1.35rem, 2.2vw, 1.95rem);
            line-height: 1.03;
            font-weight: 850;
            letter-spacing: -0.035em;
            margin-bottom: 10px;
        }}

        .eci-kpi-context {{
            color: var(--eci-muted);
            font-size: 0.86rem;
            line-height: 1.48;
        }}

        .eci-detail-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 22px;
        }}

        .eci-detail-card {{
            min-height: 132px;
            padding: 16px;
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(17, 28, 47, 0.92), rgba(13, 22, 38, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.15);
        }}

        .eci-detail-label {{
            color: var(--eci-brand);
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .eci-detail-title {{
            color: var(--eci-text);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.32;
            margin-bottom: 8px;
        }}

        .eci-detail-text {{
            color: var(--eci-muted);
            font-size: 0.86rem;
            line-height: 1.52;
        }}

        .eci-chart-shell {{
            padding: clamp(16px, 2.1vw, 22px);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(13, 22, 38, 0.96), rgba(9, 16, 29, 0.97));
            border: 1px solid rgba(148, 163, 184, 0.17);
            box-shadow: 0 18px 46px rgba(0, 0, 0, 0.24);
            margin: 8px 0 14px;
        }}

        .eci-chart-title {{
            color: var(--eci-text);
            font-size: 1.25rem;
            line-height: 1.30;
            font-weight: 800;
            margin-bottom: 6px;
        }}

        .eci-chart-subtitle {{
            color: var(--eci-muted);
            font-size: 0.90rem;
            line-height: 1.54;
            max-width: 1100px;
            margin-bottom: 2px;
        }}

        .eci-interpretation {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            padding: 16px;
            border-radius: 18px;
            background: rgba(17, 28, 47, 0.74);
            border: 1px solid rgba(125, 211, 252, 0.16);
            margin: 8px 0 20px;
        }}

        .eci-interpretation-item {{
            padding: 12px 13px;
            border-radius: 14px;
            background: rgba(7, 13, 24, 0.52);
            border: 1px solid rgba(148, 163, 184, 0.10);
        }}

        .eci-interpretation-item:nth-child(3) {{
            border-color: rgba(52, 211, 153, 0.24);
            background: rgba(52, 211, 153, 0.055);
        }}

        .eci-interpretation-item:nth-child(5) {{
            border-color: rgba(56, 189, 248, 0.24);
            background: rgba(56, 189, 248, 0.055);
        }}

        .eci-interpretation-label {{
            color: #7DD3FC;
            font-size: 0.69rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }}

        .eci-interpretation-text {{
            color: var(--eci-muted);
            font-size: 0.89rem;
            line-height: 1.58;
        }}

        .eci-source-note {{
            padding: 12px 14px;
            border-left: 3px solid rgba(56, 189, 248, 0.55);
            border-radius: 0 10px 10px 0;
            background: rgba(56, 189, 248, 0.055);
            color: var(--eci-muted);
            font-size: 0.81rem;
            line-height: 1.50;
            margin: 6px 0 22px;
        }}

        .eci-empty-state {{
            padding: 28px;
            border-radius: 20px;
            border: 1px dashed rgba(148, 163, 184, 0.28);
            background: rgba(17, 28, 47, 0.46);
            margin: 12px 0 22px;
        }}

        .eci-empty-title {{
            color: var(--eci-text);
            font-size: 1.04rem;
            font-weight: 780;
            margin-bottom: 8px;
        }}

        .eci-empty-text {{
            color: var(--eci-muted);
            font-size: 0.90rem;
            line-height: 1.55;
        }}

        [data-testid="stPlotlyChart"] {{
            border-radius: 18px;
            overflow: hidden;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            overflow: hidden;
        }}

        [data-testid="stTabs"] [role="tablist"] {{
            gap: 6px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }}

        [data-testid="stTabs"] button[role="tab"] {{
            border-radius: 10px 10px 0 0;
            padding: 10px 14px;
        }}

        div[data-testid="stHorizontalBlock"] {{
            gap: clamp(0.85rem, 1.8vw, 1.45rem);
        }}

        @media (max-width: 1200px) {{
            .eci-kpi-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            .eci-interpretation,
            .eci-detail-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}

        @media (max-width: 760px) {{
            .eci-kpi-grid,
            .eci-interpretation,
            .eci-detail-grid {{
                grid-template-columns: 1fr;
            }}

            [data-testid="stMain"] .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
        }}
    </style>
    """
