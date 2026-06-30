"""Shared Plotly styling for executive-grade interactive figures."""

from __future__ import annotations

import plotly.graph_objects as go

from app.ui.design_system import CHART_CONFIG, COLORS, FONT_FAMILY



def apply_product_layout(
    figure: go.Figure,
    *,
    height: int = 580,
    legend_title: str | None = None,
    three_d: bool = False,
) -> go.Figure:
    """Apply the shared visual language to a Plotly figure."""

    top_margin = 58 if three_d else 38

    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": COLORS["text_muted"], "size": 13},
        margin={"l": 52, "r": 34, "t": top_margin, "b": 56},
        hoverlabel={
            "bgcolor": COLORS["surface_2"],
            "bordercolor": COLORS["border"],
            "font": {"family": FONT_FAMILY, "color": COLORS["text"]},
        },
        legend={
            "title": {"text": legend_title or ""},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
        uirevision="eci-product-layout",
    )

    if not three_d:
        figure.update_xaxes(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.11)",
            zeroline=False,
            linecolor="rgba(148, 163, 184, 0.16)",
            title_standoff=14,
        )
        figure.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.11)",
            zeroline=False,
            linecolor="rgba(148, 163, 184, 0.16)",
            title_standoff=14,
        )
    else:
        axis_style = {
            "backgroundcolor": "rgba(7, 13, 24, 0.55)",
            "gridcolor": "rgba(148, 163, 184, 0.12)",
            "zerolinecolor": "rgba(148, 163, 184, 0.18)",
            "showbackground": True,
            "color": COLORS["text_muted"],
            "showspikes": False,
        }
        figure.update_layout(
            dragmode="orbit",
            scene={
                "xaxis": axis_style,
                "yaxis": axis_style,
                "zaxis": axis_style,
                "aspectmode": "cube",
                "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.12}},
            },
        )

    return figure

def get_chart_config() -> dict[str, object]:
    """Return a copy of the shared Plotly interaction configuration."""

    return dict(CHART_CONFIG)
