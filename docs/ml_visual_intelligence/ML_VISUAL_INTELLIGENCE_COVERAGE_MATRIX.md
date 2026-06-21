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

## A. Champion selection and model comparison

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-A01` | Model Performance Frontier | Which model offers the best precision–recall trade-off? | Model comparison tables and holdout metrics | Scatter/bubble frontier with champion halo | `model_performance_frontier` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-A02` | Multi-Metric Model Ranking | How do models rank across all important metrics? | Model comparison tables | Cleveland dot plot, lollipop ranking, or compact metric heatmap | `multi_metric_model_ranking` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-A03` | Validation-to-Holdout Generalisation Slopegraph | Which metrics improved or degraded on untouched holdout data? | Champion metadata and holdout metrics | Metric slopegraph with delta labels | `validation_holdout_slopegraph` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-A04` | Champion Scorecard / Identity Card | What exactly is the production champion and how was it validated? | Manifest, metadata, MLflow lineage, hashes | Executive identity card with model, threshold, hashes and metrics | `champion_identity_card` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## B. Threshold and business-decision intelligence

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-B01` | Threshold Decision Studio | Which threshold best balances campaign precision, recall and reach? | Threshold-analysis table and holdout predictions | Precision/recall/F1 curves with selected region and audience size | `threshold_decision_studio` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-B02` | Threshold Economics Curve | Which threshold maximises expected business value under explicit assumptions? | Threshold table plus documented cost/value assumptions | Net-value and cost curves with assumption panel | `threshold_economics_curve` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-B03` | Campaign Capacity Simulator | What happens when marketing can contact only the top 1%, 5%, 10% or 20%? | Ranked holdout predictions and actual labels | Capacity bands with conversions captured, precision and waste | `campaign_capacity_simulator` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-B04` | Confusion-Matrix Decision Map | What business outcomes occur at the production threshold? | Holdout labels and thresholded predictions | Annotated decision map with counts, row shares and business meaning | `confusion_matrix_decision_map` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## C. Ranking and conversion-capture performance

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-C01` | Cumulative Gains Curve | How many real conversions are captured as the targeted population expands? | Ranked holdout probabilities and actual labels | Cumulative gains curve against random targeting | `cumulative_gains_curve` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-C02` | Lift by Decile | How concentrated are buyers in each score decile? | Ranked holdout probabilities and actual labels | Lift-by-decile profile with baseline reference | `lift_by_decile` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-C03` | Precision–Recall Curve | How does precision change as recall increases in an imbalanced problem? | Holdout probabilities and actual labels | PR curve with operating point and PR-AUC | `precision_recall_curve` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-C04` | ROC Curve | How well does the model separate classes across thresholds? | Holdout probabilities and actual labels | ROC curve with operating point and ROC-AUC | `roc_curve` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-C05` | Score Separation Ridgeline | How clearly are converters separated from non-converters? | Holdout probabilities and actual labels | Ridgeline or density separation with overlap and threshold | `score_separation_ridgeline` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-C06` | Error-Zone Distribution | Where do false positives and false negatives occur in score space? | Holdout probabilities, labels and thresholded predictions | Four-class density/error-zone distribution | `error_zone_distribution` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## D. Probability quality

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-D01` | Calibration Intelligence Plot | Do predicted probabilities match observed conversion rates? | Holdout probabilities and actual labels | Calibration curve with band sizes, confidence intervals and Brier score | `calibration_intelligence` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-D02` | Calibration Error by Score Band | Where does the model overpredict or underpredict? | Binned probability reliability table | Signed calibration-gap profile by score band | `calibration_error_by_band` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-D03` | Probability Reliability Table | What are predicted and observed conversion rates in each score band? | Holdout probabilities and actual labels | Executive table with volume, prediction, actual rate and gap | `probability_reliability_table` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## E. Explainability

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-E01` | Global SHAP Beeswarm | Which features most influence predictions and in which direction? | Champion model and representative feature sample | Global SHAP beeswarm | `shap_global_beeswarm` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-E02` | Global Feature-Impact Ranking | Which features have the largest average prediction impact? | SHAP/native contribution values | Mean absolute contribution ranking | `global_feature_impact_ranking` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-E03` | SHAP Dependence Plot | How does a feature value change predicted intent? | SHAP/native contribution values and source features | Dependence plots for top business features | `shap_dependence_plot` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-E04` | Visitor-Level Waterfall | Why did one visitor receive this specific score? | Single-row feature values and contribution values | Local contribution waterfall from baseline to final score | `visitor_waterfall` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-E05` | Representative Visitor Decision Paths | How do high-, medium- and low-intent visitors reach different predictions? | Representative visitors and contribution values | Comparative decision-path visual | `representative_decision_paths` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-E06` | Feature Interaction Heatmap | Which feature combinations jointly influence predictions? | SHAP interaction values or supported native interactions | Ranked feature-interaction heatmap | `feature_interaction_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## F. Robustness and stability

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-F01` | Model Robustness Heatmap | How stable are metrics across seeds, folds or scenarios? | Stability results | Scenario-by-metric robustness heatmap | `model_robustness_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-F02` | Metric Variability Distribution | How much do metrics vary across repeated evaluations? | Repeated-fold, seed or sample results | Box/violin distributions with spread callouts | `metric_variability_distribution` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-F03` | Champion Stability Radar | Is the champion consistently strong across multiple dimensions? | Normalised robustness metrics | Restrained radar only where readability is proven | `champion_stability_radar` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-F04` | Sensitivity Tornado | Which assumptions or scenarios most affect performance? | Sensitivity results | Ranked tornado chart centred on baseline | `sensitivity_tornado` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-F05` | Performance Degradation Waterfall | How does performance decline under stress scenarios? | Baseline and stress-test metrics | Sequential degradation waterfall | `performance_degradation_waterfall` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## G. Segment and cohort intelligence

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-G01` | Segment-Level Performance Heatmap | How does model quality differ by visitor segment? | Segment labels joined to holdout predictions | Segment-by-metric heatmap with volume context | `segment_performance_heatmap` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-G02` | Score Distribution by Segment | Which segments dominate high-intent scores? | Segment labels and scored visitors | Segment score-density or ridge comparison | `score_distribution_by_segment` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-G03` | Error Rate by Behaviour Cohort | Which behaviour cohorts create the most prediction errors? | Cohort rules, predictions and actual labels | Cohort error-rate comparison with volume | `error_rate_by_cohort` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-G04` | Cohort Opportunity Matrix | Which cohorts combine scale, intent and conversion opportunity? | Cohort volume, score and conversion data | Opportunity matrix using volume, intent and outcome | `cohort_opportunity_matrix` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## H. Data and feature health

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-H01` | Feature Distribution Profile | What do the production features look like? | Canonical feature table | Compact executive small multiples | `feature_distribution_profile` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-H02` | Feature Correlation Cluster Map | Which features move together and create redundancy? | Canonical feature table | Hierarchically ordered correlation cluster map | `feature_correlation_cluster_map` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-H03` | Target Imbalance and Sampling Story | How are classes distributed across train, validation and holdout splits? | Split metadata and target counts | Sampling-flow story with class balance and counts | `target_imbalance_sampling_story` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-H04` | Missingness and Validity Map | Are required features complete and valid? | Data-quality checks and feature table | Missingness/validity matrix with pass-fail summary | `missingness_validity_map` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-H05` | Train–Holdout Feature Shift | Did feature distributions change between model development and holdout? | Train and holdout feature tables | Distribution-shift profile with effect-size ranking | `train_holdout_feature_shift` | MLflow; Streamlit; reports/presentation where relevant | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

## I. Experiment-tracking visuals

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-I01` | Experiment Parallel Coordinates | Which parameter combinations produced the strongest runs? | Multiple meaningful MLflow runs | Parallel-coordinates experiment view | `experiment_parallel_coordinates` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Requires multiple meaningful MLflow runs; audit current run history first. |
| `MLV-I02` | Run Performance Timeline | How did experiment performance evolve over time? | Multiple meaningful MLflow runs with timestamps | Run timeline with champion-change markers | `run_performance_timeline` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Requires multiple meaningful MLflow runs; audit current run history first. |
| `MLV-I03` | Hyperparameter Response Surface | How do two important parameters affect performance? | Sufficient run coverage across two parameters | 2D response surface or contour view | `hyperparameter_response_surface` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Requires multiple meaningful MLflow runs; audit current run history first. |
| `MLV-I04` | Run Comparison Matrix | How do runs differ in parameters, metrics, size and duration? | Multiple meaningful MLflow runs | Interactive/static run comparison matrix | `run_comparison_matrix` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Requires multiple meaningful MLflow runs; audit current run history first. |
| `MLV-I05` | Champion-Challenger Evolution | When did the champion change and why? | Model registry history and run metrics | Champion evolution timeline with improvement evidence | `champion_challenger_evolution` | MLflow; Streamlit; reports/presentation where relevant | `CONDITIONAL` | Requires multiple meaningful MLflow runs; audit current run history first. |

## J. Production and monitoring evidence logged to MLflow

| ID | Visual | Business question | Primary source to verify | Required design | MLflow artifact stem | Reuse surfaces | Status | Evidence / next action |
|---|---|---|---|---|---|---|---|---|
| `MLV-J01` | Prediction-Score Drift Trend | How is the prediction-score distribution changing over time? | Monitoring score snapshots | Score-drift trend with reference bands | `prediction_score_drift_trend` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-J02` | Feature Drift Severity Matrix | Which features are drifting and how severely? | Evidently drift outputs | Feature-by-period drift severity matrix | `feature_drift_severity_matrix` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-J03` | Production Performance by Model Version | How does real performance compare across deployed versions? | Delayed labels and model-version lineage | Version-level production performance comparison | `production_performance_by_version` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | Requires matured production labels and sufficient model-version evidence. |
| `MLV-J04` | Delayed-Label Maturity Funnel | How many predictions are scored, matured, labelled and evaluable? | Prediction ledger and delayed-label reports | Maturity funnel with rejection reasons | `delayed_label_maturity_funnel` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-J05` | Alert and Incident Timeline | When did alerts occur and what was affected? | Alertmanager/webhook logs and monitoring snapshots | Incident timeline with severity and resolution state | `alert_incident_timeline` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `NOT AUDITED` | Verify real source files, columns and sample sizes. |
| `MLV-J06` | Performance Before and After Drift | Did measurable performance change after a drift event? | Drift events plus matured production metrics | Before/after performance comparison | `performance_before_after_drift` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | Requires matured production labels and sufficient model-version evidence. |
| `MLV-J07` | Model Version Comparison in Production | Which version performs best on matured outcomes? | Model-version production metrics | Version comparison with confidence and sample size | `model_version_production_comparison` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `CONDITIONAL` | Requires matured production labels and sufficient model-version evidence. |
| `MLV-J08` | Monitoring Freshness and Data-Coverage Card | How current and complete is monitoring evidence? | Monitoring timestamps, row counts and label coverage | Freshness/coverage executive card | `monitoring_freshness_coverage_card` | MLflow; Grafana/Evidently; Streamlit monitoring; reports | `NOT AUDITED` | Verify real source files, columns and sample sizes. |

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
