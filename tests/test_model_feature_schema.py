# test_model_feature_schema.py
# Confirm that model training uses the same approved feature schema
# as dataset generation and production scoring.

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.models.model_config import FEATURE_COLUMNS


def test_training_uses_canonical_feature_order():
    """Training must use the shared feature names in the approved order."""

    assert FEATURE_COLUMNS == MODEL_FEATURE_COLUMNS


def test_model_config_keeps_an_independent_list_copy():
    """Changing model config must not mutate the canonical feature schema."""

    assert FEATURE_COLUMNS is not MODEL_FEATURE_COLUMNS
