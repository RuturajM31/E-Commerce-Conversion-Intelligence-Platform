import math

import pytest

from app import app_utils


def test_format_percent_handles_normal_value():
    """format_percent should convert decimals into readable percentages."""

    value = app_utils.format_percent(0.1234)

    assert "12" in value
    assert "%" in value


def test_get_best_threshold_returns_valid_probability():
    """The active model threshold should be between 0 and 1."""

    threshold = app_utils.get_best_threshold()

    assert isinstance(threshold, (int, float))
    assert 0 <= threshold <= 1


def test_get_champion_model_name_returns_text():
    """The dashboard should always show a readable model name."""

    model_name = app_utils.get_champion_model_name()

    assert isinstance(model_name, str)
    assert len(model_name.strip()) > 0


def test_assign_intent_segment_returns_business_label():
    """A score should map to a business-friendly visitor segment."""

    segment = app_utils.assign_intent_segment(0.99)

    assert isinstance(segment, str)
    assert len(segment.strip()) > 0
