# ML Visual Intelligence Coverage Matrix

## Purpose

This matrix is the controlling checklist for all machine-learning visual intelligence in the E-Commerce Conversion Intelligence Platform. It covers the full A–J scope agreed with the user. The scope must not be reduced to a small first package.

## Completion rule

A visual is complete only when all of the following are true:

- The visual is built from real project data or clearly documented assumptions.
- The chart type is appropriate for the business question.
- Titles, subtitles, axes, legends, labels and annotations are readable.
- Nothing is clipped, overlapping, stretched, pixelated, crowded or misaligned.
- The visual renders correctly at its target size and export resolution.
- The visual includes a companion interpretation containing:
  - what it shows
  - how to read it
  - actual-number findings
  - business conclusion
  - recommended action
  - honest limitation
- The artifact is logged to MLflow where appropriate.
- Reuse locations such as Streamlit, reports, presentations, Grafana or Evidently are recorded.
- Tests or validation evidence prove both calculation correctness and visual-layout hygiene.
- Unsupported items remain visible as conditional or blocked with an explicit reason and next action.

## Status values

- `NOT AUDITED` — source files and fields have not yet been verified.
- `SUPPORTED` — required source data exists and implementation can begin.
- `CONDITIONAL` — implementation needs additional genuine runs, labels, joins or assumptions.
- `BLOCKED` — an external dependency prevents completion; next action is documented.
- `IMPLEMENTED` — code and artifacts exist but final QA is still pending.
- `VALIDATED` — calculations, interpretation, export and layout QA all passed.
- `EXCLUDED` — intentionally excluded with a documented reason approved by the user.

## Mandatory companion interpretation schema

Each visual must generate or log a Markdown/HTML interpretation containing these headings:

1. **What this visual shows**
2. **How to read it**
3. **Actual-number findings**
4. **Business conclusion**
5. **Recommended action**
6. **Limitation**

## Mandatory visual-hygiene QA

Every visual must be checked for:

- title and subtitle fit
- axis-label readability
- legend placement
- annotation collisions
- text contrast
- consistent number formatting
- spacing and margins
- panel and card alignment
- aspect ratio
- dark/light compatibility where relevant
- responsive Streamlit behaviour where relevant
- PNG/SVG/HTML export quality
- consistent typography and design language
- no duplicate or low-value visuals

## Coverage summary

| Category | Scope | Visuals |
|---|---|---:|
| A | Champion selection and model comparison | 4 |
| B | Threshold and business-decision intelligence | 4 |
| C | Ranking and conversion-capture performance | 6 |
| D | Probability quality | 3 |
| E | Explainability | 6 |
| F | Robustness and stability | 5 |
| G | Segment and cohort intelligence | 4 |
| H | Data and feature health | 5 |
| I | Experiment-tracking visuals | 5 |
| J | Production and monitoring evidence logged to MLflow | 8 |

**Total ML Visual Intelligence items: 50**

## Source audit result

The source audit was completed against the current remediation branch and local project artifacts.

| Classification | Visuals | Meaning |
|---|---:|---|
| `SUPPORTED` | 27 | Required evidence exists and implementation can begin. Some visuals still require deterministic calculations or filtered samples. |
| `CONDITIONAL` | 20 | Additional genuine row-level evidence, documented assumptions, comparable run coverage or monitoring history is required. |
| `BLOCKED` | 3 | External future-label evidence is unavailable because the source dataset ends at the scoring boundary. |
| **Total** | **50** | All A–J items remain controlled by this matrix. |

### High-confidence implementation sequence

1. Champion selection and identity (`A01`–`A04`)
2. Threshold studio and confusion decision map (`B01`, `B04`)
3. Explainability (`E01`–`E06`)
4. Robustness and sensitivity (`F01`–`F05`)
5. Segment score distribution and core feature health (`G02`, `H01`, `H02`, `H04`)
6. MLflow experiment history after meaningful-run filtering (`I01`, `I02`, `I04`, `I05`)
7. Delayed-label maturity and monitoring freshness (`J04`, `J08`)
8. Conditional items after their evidence contracts are resolved

### Important evidence limitations

- `final_true_champion_holdout.csv` is an aggregate one-row summary, not a row-level holdout prediction file.
- Current production score files include probabilities but no actual outcomes.
- `visitor_training_snapshots.csv` is currently unavailable.
- `segment_summary.csv` has no actual converter counts or conversion rates.
- MLflow contains eight runs and substantial registry history, but test/noise runs must be filtered.
- Production performance visuals requiring matured labels remain blocked by the source-data time boundary.

## A. Champion selection and model comparison

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-A01` | Model Performance Frontier | Which model offers the best precision–recall trade-off? | reports/tables/final_true_champion_comparison.csv<br>reports/tables/manual_model_comparison.csv<br>reports/tables/automl_benchmark_results.csv | Scatter/bubble frontier with champion halo | `model_performance_frontier` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Eight final comparison rows plus broader 5-model and 12-model benchmark tables contain precision, recall, F1, PR-AUC, ROC-AUC, deployability and business score. Implement frontier with champion halo and dominated-model logic. |
| `MLV-A02` | Multi-Metric Model Ranking | How do models rank across all important metrics? | reports/tables/final_true_champion_comparison.csv<br>reports/tables/manual_model_comparison.csv<br>reports/tables/automl_benchmark_results.csv | Cleveland dot plot, lollipop ranking, or compact metric heatmap | `multi_metric_model_ranking` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Required model and metric columns exist. Use a compact metric heatmap or Cleveland ranking; avoid grouped default bars. |
| `MLV-A03` | Validation-to-Holdout Generalisation Slopegraph | Which metrics improved or degraded on untouched holdout data? | models/metadata/final_champion_metadata.json<br>reports/tables/final_true_champion_summary.csv | Metric slopegraph with delta labels | `validation_holdout_slopegraph` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Validation and final-holdout metrics exist in the same metadata contract. Show metric deltas and sample context. |
| `MLV-A04` | Champion Scorecard / Identity Card | What exactly is the production champion and how was it validated? | models/metadata/final_champion_metadata.json<br>models/metadata/mlflow_champion_lineage.json<br>models/metadata/final_champion_score_manifest.json | Executive identity card with model, threshold, hashes and metrics | `champion_identity_card` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Champion name, family, threshold, feature schema, holdout metrics, run ID, registry version and alias exist. Read hashes from the score manifest/model integrity contract during implementation. |

## B. Threshold and business-decision intelligence

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-B01` | Threshold Decision Studio | Which threshold best balances campaign precision, recall and reach? | reports/tables/final_true_champion_thresholds.csv<br>reports/tables/champion_threshold_analysis.csv<br>reports/tables/threshold_analysis.csv | Precision/recall/F1 curves with selected region and audience size | `threshold_decision_studio` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Threshold, precision, recall, F1, positive-rate, audience size and lift fields exist. Highlight production threshold 0.98 and separate validation/legacy evidence clearly. |
| `MLV-B02` | Threshold Economics Curve | Which threshold maximises expected business value under explicit assumptions? | Threshold tables plus a new documented economics-assumptions file | Net-value and cost curves with assumption panel | `threshold_economics_curve` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Technical threshold evidence exists, but contact cost, conversion value, false-positive cost and missed-opportunity value are not audited. Do not calculate economics until assumptions are explicitly documented and approved. |
| `MLV-B03` | Campaign Capacity Simulator | What happens when marketing can contact only the top 1%, 5%, 10% or 20%? | Ranked final-holdout labels and probabilities | Capacity bands with conversions captured, precision and waste | `campaign_capacity_simulator` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Production score files contain probabilities but no actual outcomes. The aggregate holdout file is one row only, and visitor_training_snapshots.csv is currently unavailable. Regenerate a row-level holdout prediction artifact before final capacity/capture calculations. |
| `MLV-B04` | Confusion-Matrix Decision Map | What business outcomes occur at the production threshold? | reports/tables/final_true_champion_holdout.csv<br>models/metadata/final_champion_metadata.json | Annotated decision map with counts, row shares and business meaning | `confusion_matrix_decision_map` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Aggregate holdout counts are sufficient to reconstruct the production-threshold matrix exactly: TP=66, FP=165, FN=252 and TN=248156. Validate integer reconstruction in tests. |

## C. Ranking and conversion-capture performance

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-C01` | Cumulative Gains Curve | How many real conversions are captured as the targeted population expands? | Row-level final-holdout labels and probabilities | Cumulative gains curve against random targeting | `cumulative_gains_curve` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | The one-row holdout summary cannot produce cumulative gains. Regenerate and persist row-level holdout predictions from the chronological modelling source. |
| `MLV-C02` | Lift by Decile | How concentrated are buyers in each score decile? | Row-level final-holdout labels and probabilities | Lift-by-decile profile with baseline reference | `lift_by_decile` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Lift by decile requires actual outcomes within ranked probability bands. Current production score files have no labels. |
| `MLV-C03` | Precision–Recall Curve | How does precision change as recall increases in an imbalanced problem? | Row-level final-holdout labels and probabilities | PR curve with operating point and PR-AUC | `precision_recall_curve` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | The 22-point threshold table can support a threshold trade-off chart, but not a proper sklearn precision-recall curve. Regenerate full row-level holdout predictions. |
| `MLV-C04` | ROC Curve | How well does the model separate classes across thresholds? | Row-level final-holdout labels and probabilities | ROC curve with operating point and ROC-AUC | `roc_curve` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | ROC coordinates require false-positive rates from row-level labels and probabilities. Aggregate ROC-AUC alone is insufficient. |
| `MLV-C05` | Score Separation Ridgeline | How clearly are converters separated from non-converters? | Row-level final-holdout labels and probabilities | Ridgeline or density separation with overlap and threshold | `score_separation_ridgeline` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Converter/non-converter score distributions require actual labels for each holdout row. |
| `MLV-C06` | Error-Zone Distribution | Where do false positives and false negatives occur in score space? | Row-level final-holdout labels, probabilities and thresholded predictions | Four-class density/error-zone distribution | `error_zone_distribution` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | TP/FP/TN/FN score distributions require row-level outcomes; aggregate confusion counts are insufficient for distributions. |

## D. Probability quality

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-D01` | Calibration Intelligence Plot | Do predicted probabilities match observed conversion rates? | Row-level final-holdout labels and probabilities | Calibration curve with band sizes, confidence intervals and Brier score | `calibration_intelligence` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Calibration bins, confidence intervals and Brier score require paired row-level outcomes and probabilities. |
| `MLV-D02` | Calibration Error by Score Band | Where does the model overpredict or underpredict? | Derived probability reliability table | Signed calibration-gap profile by score band | `calibration_error_by_band` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Create only after D03 can be generated from genuine row-level holdout evidence. |
| `MLV-D03` | Probability Reliability Table | What are predicted and observed conversion rates in each score band? | Row-level final-holdout labels and probabilities | Executive table with volume, prediction, actual rate and gap | `probability_reliability_table` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | The current holdout artifact is aggregate only. Persist score-band counts, predicted rate, observed rate and calibration gap after row-level regeneration. |

## E. Explainability

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-E01` | Global SHAP Beeswarm | Which features most influence predictions and in which direction? | models/trained/final_champion_model.joblib<br>data/processed/visitor_features.csv<br>models/metadata/final_champion_metadata.json | Global SHAP beeswarm | `shap_global_beeswarm` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | The champion is an XGBClassifier with get_booster support and seven canonical features. Use a representative, performance-safe production feature sample and native contributions/SHAP. |
| `MLV-E02` | Global Feature-Impact Ranking | Which features have the largest average prediction impact? | Champion XGBoost model plus native contribution values | Mean absolute contribution ranking | `global_feature_impact_ranking` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Compute mean absolute contribution values on the same controlled sample used for E01. |
| `MLV-E03` | SHAP Dependence Plot | How does a feature value change predicted intent? | Champion XGBoost model, native contributions and source features | Dependence plots for top business features | `shap_dependence_plot` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Build dependence views for business-relevant features with enough variation; include density and honest range limits. |
| `MLV-E04` | Visitor-Level Waterfall | Why did one visitor receive this specific score? | Champion XGBoost model plus one scored visitor | Local contribution waterfall from baseline to final score | `visitor_waterfall` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Use native contribution values to reconcile baseline plus feature effects to final model output. Validate numerical reconciliation. |
| `MLV-E05` | Representative Visitor Decision Paths | How do high-, medium- and low-intent visitors reach different predictions? | data/processed/visitor_scores.csv<br>Champion XGBoost contribution values | Comparative decision-path visual | `representative_decision_paths` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Select real high-intent, boundary/uncertain and low-intent visitors using deterministic rules; never hand-pick fabricated examples. |
| `MLV-E06` | Feature Interaction Heatmap | Which feature combinations jointly influence predictions? | Champion XGBoost model plus representative feature sample | Ranked feature-interaction heatmap | `feature_interaction_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | XGBoost native interaction contributions are supported. Use a sampled interaction matrix and validate symmetry/diagonal treatment. |

## F. Robustness and stability

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-F01` | Model Robustness Heatmap | How stable are metrics across seeds, folds or scenarios? | reports/tables/final_true_champion_stability.csv | Scenario-by-metric robustness heatmap | `model_robustness_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Nine seed/model observations contain PR-AUC and ROC-AUC. Build a compact heatmap with mean/range annotations. |
| `MLV-F02` | Metric Variability Distribution | How much do metrics vary across repeated evaluations? | reports/tables/final_true_champion_stability.csv | Box/violin distributions with spread callouts | `metric_variability_distribution` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Three seeds per model provide a small but valid variability view. Clearly disclose the limited repeat count. |
| `MLV-F03` | Champion Stability Radar | Is the champion consistently strong across multiple dimensions? | Derived normalised statistics from final_true_champion_stability.csv | Restrained radar only where readability is proven | `champion_stability_radar` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Use only a restrained set of non-duplicative dimensions such as mean performance, variability and worst-case retention. Radar is optional if another visual communicates better. |
| `MLV-F04` | Sensitivity Tornado | Which assumptions or scenarios most affect performance? | reports/tables/final_true_champion_sensitivity.csv | Ranked tornado chart centred on baseline | `sensitivity_tornado` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | All-validation and non-anomalous scenarios exist for multiple models. Rank absolute PR-AUC/ROC-AUC changes and annotate direction. |
| `MLV-F05` | Performance Degradation Waterfall | How does performance decline under stress scenarios? | reports/tables/final_true_champion_sensitivity.csv | Sequential degradation waterfall | `performance_degradation_waterfall` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Derive baseline-to-stress deltas by model/evaluation group. Use an ordered waterfall only where the scenario relationship is valid. |

## G. Segment and cohort intelligence

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-G01` | Segment-Level Performance Heatmap | How does model quality differ by visitor segment? | Segment labels joined to row-level holdout predictions and outcomes | Segment-by-metric heatmap with volume context | `segment_performance_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | segment_summary.csv has five segments, but actual_converters and conversion_rate are entirely missing. Final segment performance requires genuine labelled holdout rows. |
| `MLV-G02` | Score Distribution by Segment | Which segments dominate high-intent scores? | data/processed/visitor_scores.csv<br>reports/tables/segment_summary.csv | Segment score-density or ridge comparison | `score_distribution_by_segment` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | 79,998 scored visitors contain purchase_intent_score and intent_segment. Build score distributions with volume context and performance-safe sampling. |
| `MLV-G03` | Error Rate by Behaviour Cohort | Which behaviour cohorts create the most prediction errors? | Behaviour-cohort rules plus row-level holdout labels and predictions | Cohort error-rate comparison with volume | `error_rate_by_cohort` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Feature-based cohorts can be defined now, but error rates require actual outcomes per holdout row. |
| `MLV-G04` | Cohort Opportunity Matrix | Which cohorts combine scale, intent and conversion opportunity? | reports/tables/segment_summary.csv plus genuine segment conversion outcomes | Opportunity matrix using volume, intent and outcome | `cohort_opportunity_matrix` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | A predicted-opportunity matrix can be created now from volume and average score, but the final conversion-aware version remains conditional because conversion fields are null. |

## H. Data and feature health

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-H01` | Feature Distribution Profile | What do the production features look like? | data/processed/visitor_features.csv | Compact executive small multiples | `feature_distribution_profile` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Seven canonical model features exist for 78,998 current scoring rows. Use compact distribution small multiples with robust scales for skewed features. |
| `MLV-H02` | Feature Correlation Cluster Map | Which features move together and create redundancy? | data/processed/visitor_features.csv | Hierarchically ordered correlation cluster map | `feature_correlation_cluster_map` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Canonical numeric features support a hierarchically ordered correlation map. Exclude identifier and timestamp fields from feature correlation. |
| `MLV-H03` | Target Imbalance and Sampling Story | How are classes distributed across train, validation and holdout splits? | data/processed/visitor_training_snapshots.csv or equivalent split-count artifact | Sampling-flow story with class balance and counts | `target_imbalance_sampling_story` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Final-holdout counts exist, but the canonical training snapshot file is currently unavailable, so complete train/validation/holdout target and sampling evidence cannot yet be rebuilt. |
| `MLV-H04` | Missingness and Validity Map | Are required features complete and valid? | data/processed/visitor_features.csv<br>existing data-quality tests | Missingness/validity matrix with pass-fail summary | `missingness_validity_map` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Current features are auditable for completeness, range and rule violations. Run checks over the full file, not only the 2,000-row audit sample. |
| `MLV-H05` | Train–Holdout Feature Shift | Did feature distributions change between model development and holdout? | Train and final-holdout feature tables | Distribution-shift profile with effect-size ranking | `train_holdout_feature_shift` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | visitor_training_snapshots.csv is currently unavailable. Feature-shift evidence requires the original chronological train/holdout feature distributions or a regenerated equivalent. |

## I. Experiment-tracking visuals

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-I01` | Experiment Parallel Coordinates | Which parameter combinations produced the strongest runs? | mlflow.db | Parallel-coordinates experiment view | `experiment_parallel_coordinates` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | The local store contains 8 runs, 67 metrics and 142 parameters. Filter out test/noise runs and retain only comparable meaningful experiments before plotting. |
| `MLV-I02` | Run Performance Timeline | How did experiment performance evolve over time? | mlflow.db | Run timeline with champion-change markers | `run_performance_timeline` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Eight timestamped runs support a performance timeline after meaningful-run filtering and metric-name harmonisation. |
| `MLV-I03` | Hyperparameter Response Surface | How do two important parameters affect performance? | mlflow.db with sufficient two-parameter coverage | 2D response surface or contour view | `hyperparameter_response_surface` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Run count alone does not prove a valid response surface. Audit parameter combinations; use a surface only if coverage is dense enough, otherwise retain a parallel-coordinate or partial-dependence alternative. |
| `MLV-I04` | Run Comparison Matrix | How do runs differ in parameters, metrics, size and duration? | mlflow.db | Interactive/static run comparison matrix | `run_comparison_matrix` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | Run metrics, parameters, timestamps and tags support a comparison matrix after filtering incomparable/test runs. |
| `MLV-I05` | Champion-Challenger Evolution | When did the champion change and why? | mlflow.db<br>models/metadata/mlflow_champion_lineage.json | Champion evolution timeline with improvement evidence | `champion_challenger_evolution` | MLflow; Streamlit; reports/presentation where relevant | `SUPPORTED` | The registry contains 15 versions and 13 aliases. Build evolution only from meaningful registered candidates and link each change to comparable metrics; exclude local validation noise. |

## J. Production and monitoring evidence logged to MLflow

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-J01` | Prediction-Score Drift Trend | How is the prediction-score distribution changing over time? | Multiple timestamped score snapshots or prediction-log windows | Score-drift trend with reference bands | `prediction_score_drift_trend` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | Current audited evidence is a single score snapshot plus a small prediction log. A drift trend needs at least two comparable time windows. |
| `MLV-J02` | Feature Drift Severity Matrix | Which features are drifting and how severely? | Evidently drift outputs | Feature-by-period drift severity matrix | `feature_drift_severity_matrix` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | Expected drift-summary JSON was not found by the audit. Locate the actual Evidently artifact path or regenerate a structured summary before building the severity matrix. |
| `MLV-J03` | Production Performance by Model Version | How does real performance compare across deployed versions? | Matured delayed labels joined to model-version lineage | Version-level production performance comparison | `production_performance_by_version` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `BLOCKED` | The raw source ends at the scoring timestamp, leaving zero future outcome days. Real production performance by version cannot be calculated without future labels. |
| `MLV-J04` | Delayed-Label Maturity Funnel | How many predictions are scored, matured, labelled and evaluable? | data/processed/final_champion_visitor_scores.csv<br>monitoring/prediction_logs/prediction_log.csv<br>delayed-label pipeline outputs when generated | Maturity funnel with rejection reasons | `delayed_label_maturity_funnel` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `SUPPORTED` | The maturity funnel can honestly show scored/predicted volume, eligibility and zero matured outcomes. It must explicitly distinguish source-data limitation from pipeline failure. |
| `MLV-J05` | Alert and Incident Timeline | When did alerts occur and what was affected? | Alertmanager/webhook event history | Incident timeline with severity and resolution state | `alert_incident_timeline` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | No genuine incident/event history was audited. Do not fabricate alerts; implement the timeline shell and populate only from real alert records or controlled test evidence clearly labelled as test. |
| `MLV-J06` | Performance Before and After Drift | Did measurable performance change after a drift event? | Drift event history plus matured production outcomes | Before/after performance comparison | `performance_before_after_drift` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `BLOCKED` | Requires both a real drift event boundary and matured labels. Current source-data boundary prevents performance measurement. |
| `MLV-J07` | Model Version Comparison in Production | Which version performs best on matured outcomes? | Matured production outcomes for multiple model versions | Version comparison with confidence and sample size | `model_version_production_comparison` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `BLOCKED` | Registry history exists, but real production outcome comparison is blocked until labelled outcomes mature for each version. |
| `MLV-J08` | Monitoring Freshness and Data-Coverage Card | How current and complete is monitoring evidence? | monitoring/metrics_cache/ecommerce_metrics_snapshot.json<br>monitoring/prediction_logs/prediction_log.csv<br>MLflow lineage and delayed-label validation metadata | Freshness/coverage executive card | `monitoring_freshness_coverage_card` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `SUPPORTED` | Freshness timestamps, current monitoring values and label-coverage state can support an executive card. Show zero coverage honestly and flag stale/missing sources. |

## Per-visual implementation record

For every row moved beyond `NOT AUDITED`, add a detailed implementation record using this template:

```text
Visual ID:
Source files:
Required columns:
Calculation functions:
Interactive output:
Static output:
MLflow artifact path:
Interpretation artifact path:
Reuse surfaces:
Actual-number findings:
Business conclusion:
Recommended action:
Limitation:
Calculation tests:
Visual-layout QA:
Commit evidence:
Final status:
```

## Implementation sequence

The sequence below is for delivery order only. It does not reduce the 50-item scope:

1. Audit source files and columns for all A–J items.
2. Implement supported champion-selection, threshold, ranking and probability-quality visuals.
3. Implement explainability, robustness, cohort and feature-health visuals.
4. Implement experiment-tracking visuals once enough real MLflow runs exist.
5. Implement production/monitoring visuals as delayed labels and version history become available.
6. Log each approved artifact and interpretation to MLflow.
7. Reuse approved calculations and visuals across Streamlit and documentation surfaces.
8. Run calculation QA, export QA and layout/readability QA.
9. Mark each item `VALIDATED`, `CONDITIONAL`, `BLOCKED` or `EXCLUDED` with evidence.

## Sign-off rule

The ML Visual Intelligence category is not complete until every one of the 50 rows has a final evidence-backed status and every implemented visual passes both analytical and visual-hygiene QA.
