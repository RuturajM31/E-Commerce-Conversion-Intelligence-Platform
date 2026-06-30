# Streamlit Closure Sprint — Packages 3–8 Scope and Current-Visual Audit

## Purpose

This closure sprint finishes the remaining Streamlit intelligence work in one
coordinated implementation wave. It preserves Package 1 and the user-accepted
Package 2 page, reuses the existing design system, and enhances rather than
rebuilds the current application.

## Frozen accepted state

- Package 1 remains closed at commit `82c3f66`.
- Package 2 remains accepted in its current visual state.
- Package 2 investigation links and executive CSV download remain intentionally
  deferred and are not claimed as complete.
- The existing remediation stash remains untouched.

## Existing visual classification

| Page | Existing evidence | Decision | Closure enhancement |
|---|---|---|---|
| Visitor Intent Predictor | Manual/preset score, threshold gauge, business action, champion context | KEEP + ENHANCE | Cached KNN search, local neighbourhood comparison, global native feature impact, local native XGBoost contributions, governed exports |
| Batch Scoring | Upload, validation, model scoring, segment chart, score histogram, top-priority chart, CSV export | KEEP + ENHANCE | Diagnostics, scenario threshold, campaign capacity, strategy comparison, active filters, row detail, anomaly join, campaign-ready export |
| Model Benchmark and Selection | Candidate comparisons, threshold analysis, stability evidence, tables | KEEP + ENHANCE | Interactive evidence map, precision-recall frontier, confusion matrix, validation/holdout separation, campaign economics, registry evidence |
| Business KPI Forecasting | Historical forecast, model error, future targets, actual-vs-predicted, tables | KEEP + ENHANCE | KPI/horizon/scenario controls, empirical uncertainty, residual diagnostics, scenario export |
| Anomaly and Outlier | Rule summary, segment anomaly rate, risk map, ranked visitor table | KEEP + ENHANCE | Filtered investigation, selected visitor versus normal baseline, triggered-rule evidence, review action, governed export |
| Monitoring, Drift and Health | Score buckets, prediction volume, forecast health, source tables | KEEP + ENHANCE | Evidently feature/prediction drift, delayed-label funnel, source freshness, model/score lineage, verified registry evidence |
| MLOps Architecture | Product flow, implementation status, readiness tables | KEEP + ENHANCE | Interactive component inventory, data-to-decision flow, model lifecycle, topology, CI/deployment inventory, export |
| ML Validation & Evidence | PCA, K-Means, DBSCAN, LOF and evidence library | KEEP | Package 1 remains closed; no duplicate proof visuals are added |
| Customer Segmentation & Journey | No dedicated business page | ADD | New page for business PCA segments, profiles, personas, quality evidence, aggregate journey funnel, tier comparison, paths and exports |

## Package mapping

### Package 3 — Batch Scoring & Campaign Intelligence

All ten `BATCH` requirements are implemented through the current scoring flow
plus the campaign-intelligence extension.

### Package 4 — Model Performance & Decision Intelligence

Saved comparison, threshold, confusion, holdout, robustness, and verified
registry evidence are interactive. Full ROC curves and calibration bins remain
conditional because those point-level artifacts were not saved.

### Package 5 — Predictor, KNN & Explainability

The manual predictor remains unchanged. Real saved visitors can be searched by
exact ID using a cached scaled nearest-neighbour index. Global and local native
XGBoost contribution evidence is kept non-causal and computed only on demand.

### Package 6 — Segmentation & Customer Journey

A dedicated page separates business segmentation from course-proof evidence.
PCA, profiles, cluster KPIs, personas, quality evidence, visitor search, and
exports are implemented. UMAP, assignment confidence, segment movement, exact
event transitions, and exact stage-duration evidence remain conditional.

### Package 7 — Forecasting, Anomaly & Monitoring

Forecast scenarios and residuals, visitor-level anomaly investigation, current
Evidently drift, delayed labels, source freshness, lineage, and registry
evidence are implemented. Longitudinal drift/anomaly history remains
conditional until repeated snapshots are stored.

### Package 8 — Architecture, Governance & Evidence Integration

The architecture page gains selectable component evidence, flows, topology,
CI/deployment inventory, and export. Artifact presence is not presented as live
service availability.

## Evidence rules

1. No model is retrained by the Streamlit app.
2. Production defaults are never silently changed by scenario controls.
3. Offline validation, untouched holdout, current snapshot, scenario, and
   matured-production evidence remain separately labelled.
4. Missing evidence produces an unavailable or conditional state.
5. Every new major chart includes finding, conclusion, action, limitation,
   source, evidence type, and refresh time.
6. Expensive KNN and local model explanations run only on demand and use
   Streamlit resource caching.

## Package 9 — Merge and Streamlit Community Cloud readiness

The final automated package validates, commits, pushes, and merges the complete
Streamlit enhancement. It adds an app-local dependency file and a deployment
runbook for branch `main` and entrypoint `app/Executive_Overview.py`. The
external GitHub authorization and Deploy click remain a project-owner action.
