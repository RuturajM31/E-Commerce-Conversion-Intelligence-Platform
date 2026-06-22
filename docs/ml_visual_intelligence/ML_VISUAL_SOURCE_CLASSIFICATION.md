# ML Visual Intelligence Source Classification

## Result

The complete 50-item A–J ML Visual Intelligence scope has been audited against the current project artifacts.

| Status | Count |
|---|---:|
| Supported | 27 |
| Conditional | 20 |
| Blocked | 3 |
| **Total** | **50** |

## Main findings

- Champion selection, model ranking, generalisation and champion identity are fully supported.
- Threshold decision intelligence is supported, except economics and capacity capture that need assumptions or row-level outcomes.
- The current holdout artifact is aggregate only. Proper gains, decile lift, PR/ROC curves, score separation, error-zone and calibration visuals require regenerated row-level holdout labels and probabilities.
- The champion is an XGBoost classifier with native booster support and seven canonical features, so the complete explainability package is supported.
- Stability and sensitivity tables support all robustness visuals now.
- Segment score distributions are supported, but segment conversion/error visuals need real outcomes.
- Core feature distribution, correlation and validity visuals are supported. Full split/shift evidence needs the missing training snapshot source.
- MLflow contains 8 runs, 67 metrics, 142 parameters, 15 model versions and 13 aliases. Most experiment-history visuals are supported after meaningful-run filtering.
- Production outcome visuals are blocked only where future matured labels are essential.

## Exact blocked items

- `MLV-J03` Production Performance by Model Version
- `MLV-J06` Performance Before and After Drift
- `MLV-J07` Model Version Comparison in Production

These are not code failures. The raw dataset ends at the scoring timestamp, so no future purchase outcomes exist for the 14-day maturity window.

## Immediate build package

The first implementation wave should cover:

- `MLV-A01`–`MLV-A04`
- `MLV-B01`, `MLV-B04`
- `MLV-E01`–`MLV-E06`
- `MLV-F01`–`MLV-F05`
- `MLV-G02`
- `MLV-H01`, `MLV-H02`, `MLV-H04`
- `MLV-I01`, `MLV-I02`, `MLV-I04`, `MLV-I05`
- `MLV-J04`, `MLV-J08`

Every artifact still requires actual-number interpretation and visual-hygiene QA before it can move from `IMPLEMENTED` to `VALIDATED`.
