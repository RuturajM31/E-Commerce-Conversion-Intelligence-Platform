"""Business segmentation and customer journey page."""

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


import streamlit as st

from app.app_utils import inject_global_css
from app.ui.components import inject_product_css
from app.ui.customer_segmentation_journey import (
    render_segmentation_and_journey_page,
)


st.set_page_config(
    page_title="Customer Segmentation & Journey | Conversion Intelligence",
    page_icon="🧭",
    layout="wide",
)

inject_global_css()
inject_product_css()
render_segmentation_and_journey_page()
