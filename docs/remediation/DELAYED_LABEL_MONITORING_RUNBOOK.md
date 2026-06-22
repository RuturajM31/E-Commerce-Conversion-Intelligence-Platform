# Delayed-Label Production Monitoring Runbook

## 1. Purpose

This runbook explains how the E-Commerce Conversion Intelligence Platform evaluates production predictions after their conversion-outcome windows have fully matured.

A prediction is created at scoring time, but the final conversion result may only become available later. The platform therefore stores prediction evidence first and joins the final outcome only after the configured outcome window has ended.

## 2. Truthfulness Rule

The project follows one strict rule:

> No matured production labels means no production-performance claims.

When no valid matured outcomes exist:

- `metrics_available` remains `false`
- `overall_metrics` remains `null`
- dashboards must not display offline validation metrics as live production metrics
- the monitoring output must explain that matured labels are unavailable

This prevents historical validation results from being presented as real post-deployment performance.

## 3. End-to-End Workflow

```text
Production scoring
        |
        v
Prediction ledger
        |
        v
Wait for the complete outcome window
        |
        v
Delayed conversion label arrives
        |
        v
Schema, identity, conflict, and maturity validation
        |
        v
One-to-one prediction and outcome join
        |
        +-----------------------------+
        |                             |
        v                             v
Validation report            Matured outcome cohort
                                      |
                                      v
                         Production performance snapshot
```

The runner rebuilds the joined cohort from the current ledger and delayed-label files each time it runs. This safely includes valid late-arriving labels without accumulating duplicate joined outcomes.

## 4. Standard Project Files

| Purpose | Project path |
|---|---|
| Prediction ledger | `monitoring/prediction_logs/prediction_ledger.csv` |
| Delayed-label input | `monitoring/delayed_labels/delayed_labels.csv` |
| Matured joined outcomes | `monitoring/delayed_labels/matured_prediction_outcomes.csv` |
| Validation report | `monitoring/delayed_labels/delayed_label_validation_report.json` |
| Performance snapshot | `monitoring/delayed_labels/production_performance_snapshot.json` |

Path definitions are maintained in `src/config/paths.py`.

## 5. Prediction-Ledger Contract

### Grain

One row represents one production scoring record.

### Main Fields

| Column | Meaning |
|---|---|
| `prediction_id` | Stable identifier for the scored prediction |
| `visitorid` | Visitor identifier |
| `scoring_time` | UTC time when scoring occurred |
| `outcome_window_days` | Configured observation-window length |
| `outcome_window_end` | Time when the final outcome becomes mature |
| `purchase_intent_score` | Model probability score |
| `production_threshold` | Threshold used for the class decision |
| `predicted_conversion` | Production prediction, `0` or `1` |
| `model_name` | Deployed model name |
| `model_generation` | Model release or lifecycle generation |
| `model_hash` | Deployed model-artifact hash |
| `metadata_hash` | Deployed metadata hash |
| `feature_schema` | Ordered features used during scoring |
| `feature_schema_hash` | Hash of the ordered feature schema |
| `score_source` | Source that produced the score |

### Ledger Protections

The ledger implementation:

- requires timezone-aware timestamps
- normalises timestamps to UTC
- validates scores and thresholds between zero and one
- preserves the ordered feature schema
- generates deterministic prediction IDs
- ignores exact ingestion retries
- rejects conflicting rows using the same prediction ID
- validates the stored CSV header

Implementation: `src/monitoring/prediction_ledger.py`

## 6. Delayed-Label Contract

### Grain

One row represents one final outcome for one production prediction.

### Required Fields

| Column | Meaning |
|---|---|
| `label_schema_version` | Delayed-label schema version |
| `prediction_id` | Matching production prediction |
| `outcome_observed_at` | UTC time when the final outcome was checked |
| `converted` | Final binary result, `0` or `1` |
| `label_source` | Source supplying the outcome |
| `label_recorded_at` | UTC time when monitoring received the label |

### Label Protections

The delayed-label implementation:

- accepts only binary outcomes
- requires the expected `pred_` identifier prefix
- requires timezone-aware timestamps
- rejects labels recorded before they were observed
- ignores exact duplicate ingestion attempts
- rejects conflicting outcomes for one prediction
- validates the stored CSV header and schema version

Implementation: `src/monitoring/delayed_labels.py`

## 7. Initialising Controlled Files

Run from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from src.monitoring.delayed_labels import initialise_delayed_label_input
from src.monitoring.prediction_ledger import initialise_prediction_ledger

print("Prediction ledger:", initialise_prediction_ledger())
print("Delayed-label input:", initialise_delayed_label_input())
PY
```

Existing files are not overwritten.

## 8. Writing a Production Prediction

Production scoring code should build and append a ledger row immediately after scoring.

```python
from src.monitoring.prediction_ledger import (
    append_prediction_ledger_row,
    build_prediction_ledger_row,
)

ledger_row = build_prediction_ledger_row(
    visitorid=12345,
    scoring_time="2015-09-18T02:59:47.788000+00:00",
    purchase_intent_score=0.991,
    production_threshold=0.98,
    model_name="Tuned Random Forest",
    model_generation="final_champion",
    model_hash="model-sha256-example",
    metadata_hash="metadata-sha256-example",
    feature_columns=[
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "cart_to_view_ratio",
        "events_per_unique_item",
    ],
    score_source="production_scoring",
)

print(append_prediction_ledger_row(ledger_row))
```

Possible results:

- `written`: a new prediction was stored
- `duplicate`: the identical prediction already exists
- exception: conflicting provenance was detected

## 9. Writing a Delayed Label

Use the controlled builder and writer rather than manually editing the CSV.

```python
from src.monitoring.delayed_labels import (
    append_delayed_label_row,
    build_delayed_label_row,
)

label_row = build_delayed_label_row(
    prediction_id="pred_85a0c6332a18b5093cd1d137",
    outcome_observed_at="2015-10-02T02:59:47.788000+00:00",
    converted=1,
    label_source="retailrocket_future_events",
    label_recorded_at="2015-10-03T09:00:00+00:00",
)

print(append_delayed_label_row(label_row))
```

Possible results:

- `written`: a new final outcome was stored
- `duplicate`: the identical outcome already exists
- exception: invalid or conflicting outcome data was detected

## 10. Acceptance Rules

A delayed outcome enters the matured evaluation cohort only when:

1. The prediction ledger follows the approved schema.
2. The delayed-label row follows the approved schema.
3. The prediction ID exists in the ledger.
4. The prediction ID appears only once in the ledger.
5. The final outcome is binary.
6. All timestamps are timezone-aware.
7. `outcome_observed_at` is equal to or later than `outcome_window_end`.
8. No conflicting outcome exists for the prediction.
9. The final join remains one-to-one.

Accepted outcomes are written to `monitoring/delayed_labels/matured_prediction_outcomes.csv`.

## 11. Rejection Rules

### Invalid Label

Reason: `invalid_label`

Examples include missing columns, unexpected columns, invalid timestamps, invalid schema versions, and non-binary outcomes.

### Unknown Prediction

Reason: `unknown_prediction_id`

The label cannot be traced to a stored production prediction.

### Premature Label

Reason: `premature_label`

The outcome was observed before the configured outcome window ended.

### Conflicting Labels

Reason: `conflicting_labels`

Different final outcomes or provenance records exist for one prediction ID.

### Exact Duplicates

Exact duplicate labels are counted but accepted only once in the matured cohort.

All rejection evidence is stored in `monitoring/delayed_labels/delayed_label_validation_report.json`.

## 12. Running the Complete Evaluation

Run from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m src.monitoring.run_delayed_label_evaluation
```

The runner:

1. reads the prediction ledger
2. reads the delayed-label input
3. validates both schemas
4. checks one-row-per-prediction grain
5. rejects invalid, unknown, premature, and conflicting outcomes
6. keeps exact duplicate outcomes only once
7. rebuilds the matured outcome cohort
8. writes the validation report
9. calculates performance when valid labels exist
10. writes the JSON performance snapshot

Implementation: `src/monitoring/run_delayed_label_evaluation.py`

## 13. Generated Artifacts

### Matured Outcomes

Path: `monitoring/delayed_labels/matured_prediction_outcomes.csv`

Contains original prediction provenance and the accepted final conversion outcome.

### Validation Report

Path: `monitoring/delayed_labels/delayed_label_validation_report.json`

Important fields include:

- `total_ledger_rows`
- `total_label_rows`
- `accepted_matured_rows`
- `duplicate_label_rows`
- `rejected_invalid_rows`
- `rejected_unknown_prediction_ids`
- `rejected_premature_labels`
- `rejected_conflicting_prediction_ids`
- `rejections`

### Performance Snapshot

Path: `monitoring/delayed_labels/production_performance_snapshot.json`

When valid matured outcomes exist, it contains:

- matured prediction volume
- actual positive volume
- predicted positive volume
- actual conversion rate
- mean prediction score
- PR-AUC
- precision
- recall
- F1 score
- Brier calibration score
- mean-score calibration gap
- metrics grouped by model version

## 14. Metric Interpretation

### Precision

Of predictions classified as conversions, the proportion that actually converted.

### Recall

Of actual converters, the proportion identified by the production threshold.

### F1 Score

A balanced summary of precision and recall.

### PR-AUC

Measures how well probability scores rank actual converters. It remains unavailable when the matured cohort contains only one outcome class.

### Brier Calibration Score

Measures the quality of predicted probabilities.

- lower is better
- zero represents perfect probability predictions

### Mean-Score Calibration Gap

`mean prediction score - actual conversion rate`

A positive value means the average predicted probability is above the observed conversion rate. A negative value means it is below the observed conversion rate.

## 15. Model-Version Grouping

Production metrics are grouped using:

`model_generation:model_hash_prefix`

This prevents predictions created by different deployed artifacts from being silently mixed into one unexplained result.

Each group retains:

- model name
- model generation
- model hash
- metadata hash
- model-specific metrics

## 16. No-Label Behaviour

When no valid matured outcomes exist, the snapshot contains:

```json
{
  "status": "labels_unavailable",
  "metrics_available": false,
  "matured_prediction_volume": 0,
  "overall_metrics": null,
  "by_model_version": []
}
```

This is a valid monitoring result rather than a pipeline failure.

## 17. Late-Arriving Labels

When a valid label arrives after an earlier evaluation:

1. Add it through the controlled delayed-label writer.
2. Run the evaluation runner again.
3. The complete ledger and label files are reread.
4. The matured cohort is rebuilt.
5. The new valid outcome is included.
6. Existing outcomes remain one row each.
7. Metrics are recalculated.

Do not manually modify the generated matured-outcomes CSV.

## 18. Idempotency

The workflow supports safe retries:

- exact prediction retry returns `duplicate`
- exact label retry returns `duplicate`
- repeated evaluation rebuilds the same cohort without accumulating rows

## 19. Current External Data Limitation

The currently available RetailRocket data ends at:

`2015-09-18T02:59:47.788000+00:00`

The current scoring timestamp is also:

`2015-09-18T02:59:47.788000+00:00`

With a 14-day outcome window, valid future outcomes require data through:

`2015-10-02T02:59:47.788000+00:00`

Future source-data days currently available: `0`

Therefore:

- delayed-label infrastructure is implemented
- maturity and validation logic are tested
- synthetic outcomes verify the complete workflow
- genuine post-scoring outcomes are not currently present
- live production metrics remain externally blocked
- the snapshot must report labels unavailable until real future outcomes are supplied

## 20. Testing

Focused delayed-label suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest   tests/test_prediction_ledger.py   tests/test_delayed_labels.py   tests/test_production_performance.py   tests/test_delayed_label_evaluation_runner.py   -q -ra -p no:cacheprovider
```

Complete project suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest   -q -ra -p no:cacheprovider
```

Formatting check:

```bash
git diff --check
```

The complete retraining test remains opt-in through `RUN_FULL_PIPELINE_TESTS=1`.

## 21. Troubleshooting

### Unexpected Prediction-Ledger Schema

Inspect `monitoring/prediction_logs/prediction_ledger.csv`. Do not manually rename or reorder columns.

### Unexpected Delayed-Label Schema

Inspect `monitoring/delayed_labels/delayed_labels.csv`.

### Unknown Prediction ID

Confirm the prediction was written to the ledger before its outcome was ingested.

### Premature Outcome

Confirm:

`outcome_observed_at >= outcome_window_end`

### Conflicting Outcome

Investigate the source system. Do not overwrite generated monitoring artifacts manually.

### Empty Performance Metrics

Inspect `monitoring/delayed_labels/delayed_label_validation_report.json`.

Possible causes include:

- no labels have arrived
- all labels are premature
- all prediction IDs are unknown
- labels failed validation
- conflicting labels were rejected

### Interrupted Output Generation

Generated outputs use temporary files and atomic replacement. Rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m src.monitoring.run_delayed_label_evaluation
```

## 22. Implementation and Git Evidence

| Capability | Main implementation | Commit |
|---|---|---|
| Ledger schema and paths | `src/config/paths.py`, `src/monitoring/prediction_ledger.py` | `94875cc` |
| Safe ledger writing | `src/monitoring/prediction_ledger.py` | `f66a6bf` |
| Delayed-label contract | `src/monitoring/delayed_labels.py` | `f6afa76` |
| Maturity validation | `src/monitoring/delayed_labels.py` | `44f5b54` |
| Formatting cleanup | `tests/test_delayed_labels.py` | `fb3eab5` |
| Production metrics | `src/monitoring/production_performance.py` | `5cd5c1e` |
| End-to-end runner | `src/monitoring/run_delayed_label_evaluation.py` | `d5217e8` |

## 23. Operational Responsibilities

The operator must:

- protect ledger and delayed-label schemas
- ingest outcomes only from approved sources
- investigate conflicting outcomes
- review rejection counts after every run
- confirm metrics use matured outcomes only
- preserve model-version provenance
- retain the no-label limitation
- avoid manually editing generated artifacts
- rerun evaluation when valid late labels arrive
- retain validation reports as audit evidence

## 24. Remaining Related Work

The core delayed-label workflow is implemented and tested.

Separate remaining remediation tasks include:

- adding delayed-label tests explicitly to CI evidence
- updating architecture documentation and diagrams
- connecting snapshot fields to monitoring presentation where required
- reconciling both remediation coverage matrices
- completing final remediation QA
- retaining the external block until genuine future outcomes exist
