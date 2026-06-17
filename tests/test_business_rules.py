from app.app_utils import assign_intent_segment, get_best_threshold


def test_final_threshold_is_valid():
    """The final targeting threshold must be a valid probability."""

    threshold = get_best_threshold()

    assert 0 <= threshold <= 1


def test_high_score_maps_to_high_intent():
    """A very high score should map to the strongest buyer-intent segment."""

    segment = assign_intent_segment(0.99)

    assert isinstance(segment, str)
    assert "High" in segment or "Intent" in segment


def test_low_score_maps_to_low_or_cold_segment():
    """A very low score should not be labelled as a high-intent visitor."""

    segment = assign_intent_segment(0.01)

    assert isinstance(segment, str)
    assert "High" not in segment
