# Phase 1 — Feature Engineering Parity

## Objective

Remove the training-serving feature mismatch by using one shared feature-engineering module for:

- visitor dataset generation
- Streamlit single prediction
- Streamlit batch scoring

## Canonical formulas

```python
cart_to_view_ratio = addtocart_count / (view_count + 1)
events_per_unique_item = total_events / (unique_items + 1)
```

The existing training formulas were preserved for compatibility with the current trained model.

## Files added

- `src/data/feature_engineering.py`
- `tests/test_feature_engineering.py`
- `tests/test_feature_parity.py`

## Files updated

- `src/data/build_visitor_features.py`
- `app/app_utils.py`

## Comment preservation

- Existing file structure and docstrings were preserved.
- Business explanations for the ratio features were moved into the shared module.
- New comments explain why one shared implementation prevents score differences.
- Duplicate implementation code was removed without removing the learning explanation.

## Validation

Targeted feature tests:

- 9 passed
- 0 failed

Full project suite:

- 39 passed
- 3 intentionally skipped
- 0 failed

## Git evidence

- Shared feature foundation commit: `37b50ea`
- Training and Streamlit parity commit: `2be7862`

## Result

Training, single prediction, and batch scoring now use the same feature formulas and approved feature order.
