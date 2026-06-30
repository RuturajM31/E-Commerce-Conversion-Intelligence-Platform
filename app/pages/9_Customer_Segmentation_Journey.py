"""Business segmentation and customer journey page."""

from __future__ import annotations

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
