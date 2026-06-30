"""Shared helpers for the final Streamlit visual-intelligence closure sprint.

This module keeps cross-page behaviour consistent without creating a second
visual design system. It reuses Package 1 components and Plotly styling while
adding small helpers for evidence loading, metadata, filtering, and exports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.plotly_style import apply_product_layout


UTC_FORMAT = "%Y-%m-%d %H:%M UTC"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe ratio and use zero when the denominator is unavailable."""

    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def compact_count(value: float | int) -> str:
    """Format a count compactly while keeping small values exact."""

    number = float(value)
    absolute = abs(number)

    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.1f}K"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def percent(value: float, decimals: int = 1) -> str:
    """Format a decimal ratio as a percentage."""

    return f"{float(value):.{decimals}%}"


def currency(value: float) -> str:
    """Format scenario money without assuming a real transaction currency."""

    return f"{float(value):,.2f}"


def file_timestamp(path: Path) -> str:
    """Return a UTC modification timestamp or an explicit unavailable state."""

    if not path.exists():
        return "Unavailable"

    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).strftime(UTC_FORMAT)


def first_existing(paths: Iterable[Path]) -> Path | None:
    """Return the first existing path from an ordered candidate list."""

    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path_text: str) -> pd.DataFrame:
    """Load one CSV with caching and return an empty table when missing."""

    path = Path(path_text)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json_if_exists(path_text: str) -> dict[str, Any]:
    """Load one JSON object with caching and a safe empty fallback."""

    path = Path(path_text)
    if not path.is_file():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def source_hash(path: Path) -> str:
    """Return a short SHA-256 hash for a small evidence file."""

    if not path.is_file():
        return "Unavailable"

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def finish_figure(
    figure: go.Figure,
    *,
    title: str,
    subtitle: str,
    height: int = 520,
    legend_title: str | None = None,
    three_d: bool = False,
) -> go.Figure:
    """Apply Package 1 Plotly styling and a consistent chart heading."""

    apply_product_layout(
        figure,
        height=height,
        legend_title=legend_title,
        three_d=three_d,
    )
    figure.update_layout(
        title={
            "text": (
                f"<b>{title}</b><br>"
                f"<span style='font-size:12px'>{subtitle}</span>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        margin={
            "l": 54,
            "r": 34,
            "t": 98,
            "b": 58,
        },
    )
    return figure


def ensure_shared_filter_state() -> None:
    """Create one small cross-page filter state without changing defaults."""

    defaults: dict[str, Any] = {
        "eci_intent_tiers": [],
        "eci_anomaly_only": False,
        "eci_cluster_filter": [],
        "eci_threshold_override": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_active_filter_summary() -> None:
    """Show active shared filters as a concise caption."""

    ensure_shared_filter_state()
    parts: list[str] = []

    if st.session_state.eci_intent_tiers:
        parts.append(
            "Intent: " + ", ".join(st.session_state.eci_intent_tiers)
        )
    if st.session_state.eci_anomaly_only:
        parts.append("Anomaly only")
    if st.session_state.eci_cluster_filter:
        parts.append(
            "Clusters: "
            + ", ".join(str(value) for value in st.session_state.eci_cluster_filter)
        )
    if st.session_state.eci_threshold_override is not None:
        parts.append(
            f"Scenario threshold: {st.session_state.eci_threshold_override:.3f}"
        )

    if parts:
        st.caption("Active shared filters · " + " · ".join(parts))
    else:
        st.caption("Active shared filters · none")


def export_metadata(
    *,
    source: str,
    evidence_type: str,
    threshold: float | None = None,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build standard metadata columns for downloadable evidence."""

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(UTC_FORMAT),
        "source": source,
        "evidence_type": evidence_type,
    }

    if threshold is not None:
        metadata["threshold"] = f"{float(threshold):.6f}"

    if filters:
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                metadata[f"filter_{key}"] = "|".join(
                    str(item) for item in value
                )
            else:
                metadata[f"filter_{key}"] = str(value)

    return metadata


def add_metadata_columns(
    data: pd.DataFrame,
    metadata: Mapping[str, str],
) -> pd.DataFrame:
    """Add export provenance columns without changing the source DataFrame."""

    output = data.copy()
    for key, value in metadata.items():
        output[f"export_{key}"] = value
    return output


def status_from_exists(path: Path) -> str:
    """Return a simple artifact status for architecture and monitoring pages."""

    return "Ready" if path.exists() else "Unavailable"
