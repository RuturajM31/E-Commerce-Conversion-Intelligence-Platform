# Streamlit Visual Intelligence Enhancement Coverage Matrix

**Project:** E-Commerce Conversion Intelligence Platform
**Planned repository path:** `docs/streamlit/STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.md`
**Companion handoff:** `docs/streamlit/STREAMLIT_VISUAL_ENHANCEMENT_HANDOFF.md`
**Matrix version:** 1.0
**Created:** 21 June 2026

## Controlling decision

The current Streamlit theme and existing app are good and must be preserved. This phase is an **enhancement**, not a redesign or rewrite.

Before any existing chart is changed, it must be classified as **KEEP**, **ENHANCE**, **REPLACE**, **MOVE**, or **RETIRE**, with a documented reason and source comparison.

The matrix is created now for continuity. **Implementation remains inactive until remediation, final remediation QA, user approval, and remediation merge are complete.**

## Current application baseline

| Current page | Current purpose | Enhancement direction |
| --- | --- | --- |
| Executive Overview | High-level story, KPIs, and business value | Preserve theme; enhance decision flow |
| Visitor Intent Predictor | Single visitor scoring and decision support | Add local XAI and similarity |
| Batch Scoring | Campaign-ready batch scoring | Add scenario and audience-quality intelligence |
| Model Benchmark Selection | Model comparison and champion proof | Integrate approved ML visual evidence |
| Business KPI Forecasting | Future KPI outlook | Add uncertainty, backtesting, residuals, and scenarios |
| Anomaly Detection | Unusual behavior and risk patterns | Add profiles, comparisons, and investigation flow |
| Monitoring, Drift and Health | Model, data, and app health | Add history and truthful label readiness |
| MLOps Architecture | System and deployment explanation | Add interactive layered flows |
| MVD Coverage Proof | Assignment and method evidence | Preserve as proof; keep separate from executive story |

## Status definitions

| Status | Meaning |
| --- | --- |
| PLANNED | Agreed requirement; implementation has not started. |
| IN PROGRESS | Active implementation is underway. |
| IMPLEMENTED | Code exists, but full QA and review are incomplete. |
| VERIFIED | Implementation, data, interaction, explanation, performance, and visual QA passed. |
| CONDITIONAL | Implementation or evidence depends on valid data or capability becoming available. |
| BLOCKED | A current external or technical constraint exists; next action must be recorded. |
| EXCLUDED | Intentionally outside scope with an explicit reason. |

## Completion rule

A row is complete only when:

1. the correct data source, grain, keys, calculations, and evidence type are verified;
2. the visual or feature works interactively and renders cleanly;
3. the interpretation contains actual-number findings, business meaning, recommended action, and limitation;
4. performance is safe through caching, compact derived data, or precomputation;
5. tests and visual-review evidence exist;
6. the row is marked `VERIFIED`, `CONDITIONAL`, `BLOCKED`, or `EXCLUDED` with evidence or reason.

## Expert implementation recommendations

- Keep the current theme. Treat it as an approved baseline, not a redesign target.
- Audit and enhance existing charts before creating new ones. New visuals must close a clear decision gap.
- Use one primary interactive chart library after auditing the current implementation; do not rewrite good charts only for uniformity.
- Reuse the approved ML Visual Intelligence calculations and source tables as evidence contracts. Rebuild interactively only when it adds real value.
- Make the Executive Overview a decision cockpit, not a chart gallery.
- Use cross-page filters through session state, but expose only filters valid for each page and dataset.
- Precompute UMAP, SHAP summaries, nearest-neighbor indexes, drift history, and forecast diagnostics.
- Use Parquet or compact derived tables for heavy interactive data and avoid repeated raw CSV scans.
- Generate result-based conclusions from current filtered values. Never ship generic or invented interpretations.
- Treat offline, simulated, current snapshot, and matured production evidence as different evidence types in the UI.
- Use progressive disclosure: executive message first, technical proof in tabs or expanders.
- Finish with one consolidated visual review pack after all automated checks, not repeated manual image reviews after every chart.

## Recommended implementation waves

| Wave | Focus | Deliverables |
| --- | --- | --- |
| 0 | Baseline audit and protection | Inventory current pages/charts; classify KEEP/ENHANCE/REPLACE/MOVE/RETIRE; freeze theme baseline |
| 1 | Shared foundation | Design tokens, global filters, cache strategy, data contracts, reusable cards and interpretation blocks |
| 2 | Executive and campaign value | EXEC + BATCH + key MODEL threshold views |
| 3 | Model understanding | MODEL + XAI + Visitor Predictor integration |
| 4 | Customer intelligence | SEG + KNN + JOURNEY |
| 5 | Operational intelligence | FCST + ANOM + MON |
| 6 | Architecture and proof | ARCH + MVD preservation + documentation |
| 7 | Final QA and sign-off | Automated QA, combined visual review, matrix reconciliation, approval, merge |

## Matrix summary

- Total controlling rows: **204**
- `GOV` — Governance and phase control: **12 rows**
- `AUDIT` — Current app and existing-chart audit: **16 rows**
- `UX` — Theme, design, usability, and explanation standards: **15 rows**
- `SHARED` — Shared filters, data contracts, downloads, and performance: **18 rows**
- `EXEC` — Executive overview: **12 rows**
- `SEG` — Segmentation and clustering: **12 rows**
- `KNN` — KNN similarity and similar-visitor intelligence: **10 rows**
- `BATCH` — Batch scoring and campaign intelligence: **10 rows**
- `MODEL` — Model performance and decision intelligence: **14 rows**
- `XAI` — Explainability: **12 rows**
- `FCST` — Forecasting and scenario intelligence: **12 rows**
- `JOURNEY` — Journey and funnel analysis: **10 rows**
- `ANOM` — Anomaly intelligence: **10 rows**
- `MON` — Monitoring, drift, labels, and production evidence: **14 rows**
- `ARCH` — Interactive architecture and lifecycle: **10 rows**
- `QA` — Testing, visual review, documentation, and sign-off: **17 rows**

- Initial status: **191 PLANNED**, **13 CONDITIONAL**

## GOV — Governance and phase control

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-GOV-01 | Activation gate | Do not begin implementation until remediation, final QA, approval, and merge are complete. | Matrix may exist now for continuity; implementation remains deferred. | Git evidence of remediation merge and dedicated enhancement branch. | P0 | Git and project governance | PLANNED |
| SVE-GOV-02 | Dedicated branch | Create a separate Streamlit enhancement branch after remediation merge. | Recommended branch: feat/streamlit-visual-intelligence-enhancement. | Branch and remote tracking verified. | P0 | Git and project governance | PLANNED |
| SVE-GOV-03 | Controlling checklist | Use this matrix as the authoritative scope for the enhancement phase. | No chart or feature is considered complete outside this matrix. | Every row ends VERIFIED, CONDITIONAL, BLOCKED, or EXCLUDED with reason. | P0 | Git and project governance | PLANNED |
| SVE-GOV-04 | Current theme protection | Preserve the current app theme, typography, navigation, and overall visual identity. | Enhance rather than redesign the app. | Before/after screenshots confirm no regression in the approved theme. | P0 | Git and project governance | PLANNED |
| SVE-GOV-05 | Existing chart protection | Audit every current chart before replacing, removing, or duplicating it. | Classify each as KEEP, ENHANCE, REPLACE, MOVE, or RETIRE. | Baseline inventory records decision and reason for every current chart. | P0 | Git and project governance | PLANNED |
| SVE-GOV-06 | Evidence rule | Use real project data and verified artifacts only. | No invented metrics, demo values, or unsupported production claims. | Source path and calculation recorded for every visual. | P0 | Git and project governance | PLANNED |
| SVE-GOV-07 | Production truthfulness | Do not show offline validation as live production performance. | Matured-cohort performance remains conditional until real labels exist. | Monitoring page exposes label availability and evidence status. | P0 | Git and project governance | PLANNED |
| SVE-GOV-08 | No duplicate storytelling | Avoid multiple charts that communicate the same business message. | Each chart must answer a distinct decision question. | Page-level chart-purpose review completed. | P0 | Git and project governance | PLANNED |
| SVE-GOV-09 | Package-based changes | Prepare complete, reviewable packages for substantive updates. | Do not ask the user to manually edit many project files. | Package manifest, tests, and apply instructions supplied. | P0 | Git and project governance | PLANNED |
| SVE-GOV-10 | New-chat continuity | Keep this matrix and the handoff file physically in the repository. | A new chat must read both files before coding. | Git path verified; latest status and next action recorded. | P0 | Git and project governance | PLANNED |
| SVE-GOV-11 | No premature completion | Do not mark a row complete from code existence alone. | Rendering, data, interaction, explanation, performance, and QA must pass. | Acceptance evidence attached or referenced. | P0 | Git and project governance | PLANNED |
| SVE-GOV-12 | Final approval | Do not merge the enhancement branch without explicit user approval. | Final combined visual review must pass first. | Approval recorded after complete review. | P0 | Git and project governance | PLANNED |

## AUDIT — Current app and existing-chart audit

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-AUD-01 | Executive Overview | Inventory KPI cards, executive charts, conclusions, and navigation. | Preserve strong current theme and business story. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/Executive_Overview.py. | P0 | app/Executive_Overview.py | PLANNED |
| SVE-AUD-02 | Visitor Intent Predictor | Inventory single-visitor score, input controls, decision result, and explanations. | Enhance with local explainability and similar-visitor intelligence. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/1_Visitor_Intent_Predictor.py. | P0 | app/pages/1_Visitor_Intent_Predictor.py | PLANNED |
| SVE-AUD-03 | Batch Scoring | Inventory batch upload, score outputs, targeting charts, and downloads. | Enhance campaign scenarios and audience-quality analysis. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/2_Batch_Scoring.py. | P0 | app/pages/2_Batch_Scoring.py | PLANNED |
| SVE-AUD-04 | Model Benchmark Selection | Inventory benchmark, champion, threshold, calibration, and model evidence. | Reuse approved ML Visual Intelligence data and avoid duplicate charts. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/3_Model_Benchmark_Selection.py. | P0 | app/pages/3_Model_Benchmark_Selection.py | PLANNED |
| SVE-AUD-05 | Business KPI Forecasting | Inventory forecast charts, KPI comparisons, diagnostics, and narrative. | Add uncertainty, backtesting, residuals, and scenarios. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/4_Business_KPI_Forecasting.py. | P0 | app/pages/4_Business_KPI_Forecasting.py | PLANNED |
| SVE-AUD-06 | Anomaly and Outlier | Inventory anomaly rate, risk map, tables, and investigation flow. | Add segment comparisons, profiles, and guided investigation. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/5_Anomaly_Outlier.py. | P0 | app/pages/5_Anomaly_Outlier.py | PLANNED |
| SVE-AUD-07 | Monitoring, Drift and Health | Inventory drift, health, prediction, alert, and label-evidence views. | Add trend history and truthful matured-cohort states. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/6_Monitoring_Drift_Health.py. | P0 | app/pages/6_Monitoring_Drift_Health.py | PLANNED |
| SVE-AUD-08 | MLOps Architecture | Inventory current architecture visuals and explanatory content. | Add interactive layered architecture and lifecycle views. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/7_MLOps_Architecture.py. | P0 | app/pages/7_MLOps_Architecture.py | PLANNED |
| SVE-AUD-09 | MVD Coverage Proof | Inventory course-proof charts and evidence. | Preserve proof; keep it separate from the main executive story. | Page screenshot, chart inventory, source mapping, and KEEP/ENHANCE/REPLACE/MOVE/RETIRE decision recorded for app/pages/8_MVD_Coverage_Proof.py. | P0 | app/pages/8_MVD_Coverage_Proof.py | PLANNED |
| SVE-AUD-10 | Champion PR-AUC comparison | Keep or enhance the existing champion comparison if it remains accurate and readable. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-11 | Threshold trade-off | Keep or enhance the existing precision/recall/F1/target-share trade-off. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-12 | Campaign trade-off | Keep or enhance buyer-capture versus target-quality analysis. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-13 | Forecast visuals | Keep useful future KPI and unique-visitor forecast views; remove duplicates. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-14 | Anomaly visuals | Keep useful anomaly-rate-by-segment and purchase-intent risk map views. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-15 | Clustering proof | Preserve PCA, K-Means, and DBSCAN proof, but separate course proof from business segmentation. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |
| SVE-AUD-16 | Monitoring proof | Preserve useful Grafana and Prometheus evidence without turning Streamlit into a screenshot gallery. | Do not duplicate an existing business message. | Decision, source, and enhancement gap documented. | P0 | Existing verified project outputs | PLANNED |

## UX — Theme, design, usability, and explanation standards

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-UX-01 | Design tokens | Centralize current theme colors, typography, spacing, borders, chart backgrounds, and semantic states. | Preserve current appearance; standardize extensions. | Shared style module and visual consistency review. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-02 | Primary chart library | Choose one primary interactive chart library after auditing current usage. | Do not rewrite good existing charts merely to standardize. | Library decision documented; dependencies justified. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-03 | KPI card system | Create reusable KPI cards with value, context, delta, status, and tooltip. | Use only where a KPI helps a decision. | Cards render consistently across pages. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-04 | Chart header | Give each chart a decision-focused title and concise subtitle. | Avoid generic titles such as 'Chart' or 'Analysis'. | Title explains business question. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-05 | Interpretation block | Every major visual must include what it shows, how to read it, actual-number finding, conclusion, recommended action, and limitation. | No generic boilerplate or invented conclusions. | Interpretation is generated from verified results. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-06 | Progressive disclosure | Use tabs, expanders, and drill-downs to separate executive and technical detail. | Keep first view calm and decision-oriented. | Executive view is readable without scrolling through technical proof. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-07 | Accessibility | Use readable text, color-blind-safe distinctions, non-color cues, keyboard-friendly controls, and clear tooltips. | Do not rely on red/green alone. | Accessibility checklist passed. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-08 | Responsive layout | Support common laptop widths without clipping, overlap, or horizontal overflow. | Avoid fixed-width layouts that break on smaller screens. | Responsive screenshots reviewed. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-09 | Empty states | Provide helpful empty, unavailable, and blocked-data states. | Explain what is missing and the next action. | No blank charts or cryptic exceptions. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-10 | Loading states | Show spinners or status messages for expensive work. | Avoid frozen-looking pages. | User sees progress without technical noise. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-11 | Source notes | Show data source, refresh time, and evidence type where relevant. | Distinguish offline, current, simulated, and production evidence. | Source note visible or available through tooltip/expander. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-12 | Number formatting | Use consistent percentages, compact counts, currency, dates, thresholds, and decimals. | Match business meaning and avoid false precision. | Formatting utility tests pass. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-13 | Visual density | Limit the number of major visuals shown at once. | Prefer one clear message over decorative complexity. | Page review confirms no clutter or repetition. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-14 | Data tables | Use searchable, sortable, filterable tables with meaningful column names and export controls. | Do not show raw technical columns by default. | Table usability test passed. | P0 | Current .streamlit configuration and shared app styles | PLANNED |
| SVE-UX-15 | Theme regression | Prevent visual changes from breaking the approved current theme. | Theme changes require explicit reason and review. | Before/after visual regression evidence. | P0 | Current .streamlit configuration and shared app styles | PLANNED |

## SHARED — Shared filters, data contracts, downloads, and performance

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-SHARED-01 | Global filter state | Create shared cross-page filter state using Streamlit session state. | Filters should persist logically across compatible pages. | Filter-state tests and manual navigation test. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-02 | Date filter | Support date or snapshot-window selection where time exists. | Hide or disable when the source has no valid time grain. | Date-range calculations verified. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-03 | Segment filter | Support business segment or cluster selection. | Use canonical segment identifiers and labels. | Filtered totals reconcile with source. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-04 | Intent tier filter | Support high, medium, and low intent tiers where applicable. | Use the same tier rules across pages. | Cross-page tier counts reconcile. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-05 | Threshold control | Provide controlled threshold simulation without silently changing production defaults. | Display production threshold separately from simulation threshold. | Reset and boundary tests pass. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-06 | Anomaly filter | Support anomaly flag, severity, and investigation status. | Use shared anomaly definitions. | Filtered rows reconcile. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-07 | Cohort filter | Support relevant customer, score, time, or campaign cohorts. | Only expose cohorts supported by data. | Cohort definitions documented. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-08 | Filter summary | Show active-filter chips or a concise summary. | Users must know what data they are viewing. | Filter state visible and resettable. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-09 | Cache data | Use st.cache_data for stable transformed tables and chart inputs. | Cache keys must include relevant filters and source timestamps. | Cache correctness and invalidation tests. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-10 | Cache resources | Use st.cache_resource for models, explainers, nearest-neighbor indexes, and reusable heavy objects. | Avoid repeated model or SHAP initialization. | Resource loads once per session/process. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-11 | Precompute heavy outputs | Precompute UMAP, SHAP summaries, backtests, drift history, and large aggregation tables. | Do not block page loads with avoidable heavy computation. | Precompute scripts and manifests verified. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-12 | Efficient file formats | Prefer Parquet or compact derived tables for large interactive inputs. | Do not repeatedly scan large raw CSV files. | Load-time comparison recorded. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-13 | Sampling policy | Use deterministic sampling for dense scatter plots while preserving totals and distribution context. | Show sample size and full-population totals. | Sampling logic tested. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-14 | Lazy rendering | Render expensive sections only when opened or requested. | Avoid computing every tab at initial load. | Initial page-load timing passes. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-15 | Performance budget | Set page-load and interaction performance targets. | Recommended target: cached initial view <=3 seconds locally; common interactions <=1.5 seconds. | Measured timings recorded; exceptions explained. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-16 | Data contracts | Define source path, grain, key, required columns, refresh rule, and fallback for every visual. | No visual may depend on an undocumented table. | Data-contract table complete. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-17 | Artifact manifests | Record generation time, source hashes, row counts, and parameters for precomputed outputs. | Make stale evidence visible. | Manifest validation passes. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |
| SVE-SHARED-18 | Download governance | Exports must contain active filters, timestamp, threshold, source, and column definitions. | Do not export misleading unlabeled files. | Download test and sample file review. | P0 | Shared Streamlit utilities and verified derived datasets | PLANNED |

## EXEC — Executive overview

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-EXEC-01 | Executive Overview | Enhance current KPI cards with context, evidence type, and concise business meaning. | Preserve strong current cards; improve only where decision value increases. | Values reconcile with approved outputs and render cleanly. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-02 | Executive Overview | Create an interactive total visitors → scored → eligible → targeted → expected buyers funnel. | Use real counts; simulations must be labeled. | Stage counts and retention calculations tested. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-03 | Executive Overview | Simulate target size, contact cost, expected buyers, value, and uplift assumptions. | Keep assumptions editable and explicit. | Formula tests and scenario export. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-04 | Executive Overview | Show audience reduction versus buyer capture and campaign efficiency. | Reuse verified threshold/campaign evidence where possible. | Source and threshold visible. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-05 | Executive Overview | Show composition of the selected target audience by segment, intent tier, and anomaly status. | Avoid pie charts when categories are numerous. | Totals reconcile with filtered audience. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-06 | Executive Overview | Show champion, threshold, validation/holdout evidence, and probability reliability at executive level. | Do not present live production metrics without matured labels. | Evidence type and dates displayed. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-07 | Executive Overview | Show a compact forward KPI outlook with uncertainty and scenario context. | Link to full forecasting page. | Forecast source and horizon verified. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-08 | Executive Overview | Show current anomaly volume, rate, severity, and key affected segment. | Use real anomaly evidence. | Counts reconcile. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-09 | Executive Overview | Show application, scoring, drift, alerts, and label readiness in one compact strip. | Separate local validation from managed production. | Status logic tested. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-10 | Executive Overview | Generate a concise decision summary from active filters and actual results. | No generic static paragraph. | Conclusion contains finding, action, and limitation. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-11 | Executive Overview | Add clear links to investigate model, audience, forecasts, anomalies, and monitoring. | Reduce page-search effort. | All links work. | P1 | Existing verified project outputs | PLANNED |
| SVE-EXEC-12 | Executive Overview | Provide an exportable filtered executive brief or summary table. | Include timestamp, filters, assumptions, and limitations. | Downloaded output reviewed. | P1 | Existing verified project outputs | PLANNED |

## SEG — Segmentation and clustering

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-SEG-01 | Segmentation and Clustering | Interactive PCA map with clusters, segment filters, hover detail, centroids, and explained variance. | Use deterministic sampling for dense data. | PCA inputs, scale, variance, and sample size documented. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-02 | Segmentation and Clustering | Interactive UMAP map for local visitor structure and boundary exploration. | Precompute embedding and seed; do not recompute on every page load. | Embedding manifest and reproducibility check. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-03 | Segmentation and Clustering | Compare normalized feature profiles across clusters. | Use business-friendly labels and explain scaling. | Profile values reconcile with source. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-04 | Segmentation and Clustering | Show size, conversion rate, intent, cart behavior, anomaly rate, and value proxy by cluster. | Only include supported KPIs. | Metric definitions and totals verified. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-05 | Segmentation and Clustering | Compare selected clusters across a controlled set of comparable metrics. | Avoid unreadable many-axis radar charts; use bars when clearer. | Visual choice justified and labels readable. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-06 | Segmentation and Clustering | Create evidence-based cluster personas with behavior, opportunity, risk, and recommended action. | Personas must derive from actual profiles, not invented stereotypes. | Persona generation logic and source values shown. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-07 | Segmentation and Clustering | Show silhouette, Davies-Bouldin, Calinski-Harabasz, size balance, and stability where available. | Explain that quality metrics do not alone prove business usefulness. | Metric calculations tested. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-08 | Segmentation and Clustering | Show boundary or assignment confidence where the method supports it. | Conditional for hard clustering without probabilities. | Honest unavailable state when unsupported. | P2 | Segmentation outputs and precomputed embeddings | CONDITIONAL |
| SVE-SEG-09 | Segmentation and Clustering | Show movement between segment snapshots when comparable time snapshots exist. | Conditional until repeatable segment snapshots exist. | Identity and time-grain validation. | P2 | Segmentation outputs and precomputed embeddings | CONDITIONAL |
| SVE-SEG-10 | Segmentation and Clustering | Provide searchable visitor table for a selected cluster. | Default to useful business columns. | Filtering and export verified. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-11 | Segmentation and Clustering | Map segments to campaign approach, retention action, investigation need, and expected value. | Actions must be rules or recommendations, not causal claims. | Action logic documented. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |
| SVE-SEG-12 | Segmentation and Clustering | Export filtered cluster profiles and visitor assignments. | Include model/version and snapshot metadata. | Download validation. | P2 | Segmentation outputs and precomputed embeddings | PLANNED |

## KNN — KNN similarity and similar-visitor intelligence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-KNN-01 | Visitor Intent Predictor / Similarity | Search by visitor ID or select a visitor from filtered results. | Validate existence and show meaningful errors. | Search and selection tests. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-02 | Visitor Intent Predictor / Similarity | Build or load a cached scaled nearest-neighbor index using approved model features. | Feature order and scaling must match the documented similarity contract. | Index manifest, feature list, and reproducibility test. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-03 | Visitor Intent Predictor / Similarity | Show nearest visitors, distance, intent, segment, anomaly status, and outcome evidence where available. | Do not imply causality from similarity. | Distance ordering and row validation. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-04 | Visitor Intent Predictor / Similarity | Compare selected visitor with neighbors on key features. | Use normalized and raw values with clear labels. | Values reconcile. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-05 | Visitor Intent Predictor / Similarity | Explain which features make two visitors similar or different. | Use transparent standardized-distance contributions. | Contribution sum and sign logic tested. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-06 | Visitor Intent Predictor / Similarity | Show local score, segment, anomaly, and conversion-label composition. | Label availability must be explicit. | Local counts verified. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-07 | Visitor Intent Predictor / Similarity | Suggest business actions based on selected visitor and neighborhood patterns. | Recommendations must be rules-based and explainable. | Rule tests and limitation text. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-08 | Visitor Intent Predictor / Similarity | Allow neighbor count and feature-set selection within safe bounds. | Prevent unsupported or leakage-prone features. | Boundary and contract tests. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-09 | Visitor Intent Predictor / Similarity | Keep index loading and search responsive through caching and precomputation. | No full refit on every interaction. | Search latency measured. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |
| SVE-KNN-10 | Visitor Intent Predictor / Similarity | Export selected visitor and neighbor comparison with metadata. | Include feature definitions and distance method. | Download review. | P2 | Canonical production features and cached nearest-neighbor index | PLANNED |

## BATCH — Batch scoring and campaign intelligence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-BATCH-01 | Batch Scoring | Improve upload diagnostics, schema feedback, row counts, and rejected-row explanations. | Preserve existing scoring workflow. | Valid and invalid upload tests. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-02 | Batch Scoring | Interactive score histogram or density view with threshold and intent bands. | Display sample/full population context. | Band totals and threshold verified. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-03 | Batch Scoring | Show score bands, expected buyer concentration, segment mix, and anomaly mix. | Use real or clearly labeled simulated evidence. | Metrics reconcile. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-04 | Batch Scoring | Show target size, capture, precision proxy, cost, and scenario outcomes by threshold. | Production threshold remains visible and unchanged. | Scenario formulas tested. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-05 | Batch Scoring | Allow campaign-capacity limits and show the highest-priority audience within capacity. | Use stable ranking and tie handling. | Top-N selection tests. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-06 | Batch Scoring | Compare multiple targeting strategies side by side. | Examples: production threshold, fixed budget, top percentile, segment-first. | Strategy definitions documented. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-07 | Batch Scoring | Explain how filters and rules reduce the audience. | Use clear stage counts and reasons. | Stage reconciliation test. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-08 | Batch Scoring | Provide score, tier, key drivers, segment, anomaly, and action for selected rows. | Heavy explanations should be generated on demand. | Spot-check with model outputs. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-09 | Batch Scoring | Export campaign-ready file with rank, score, tier, threshold, segment, explanation, and metadata. | Exclude unnecessary technical columns by default. | CSV schema and metadata test. | P1 | Batch scoring outputs and threshold evidence | PLANNED |
| SVE-BATCH-10 | Batch Scoring | Generate actual-number findings and recommended campaign action. | No generic conclusion. | Result-based narrative review. | P1 | Batch scoring outputs and threshold evidence | PLANNED |

## MODEL — Model performance and decision intelligence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-MODEL-01 | Model Benchmark and Selection | Show champion model, version, threshold, feature schema, data split, and evidence dates. | Reuse verified metadata. | Metadata hash and values verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-02 | Model Benchmark and Selection | Interactive multi-metric model comparison with validation and holdout separation. | Never mix incomparable metrics or datasets. | Model rows and metric sources verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-03 | Model Benchmark and Selection | Show precision-recall or business-score frontier across candidates. | Reuse approved visual logic where appropriate. | Frontier inputs verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-04 | Model Benchmark and Selection | Interactive ROC curve with AUC, operating point, and plain-language explanation. | Use holdout or clearly labeled evaluation split. | Curve and AUC tested. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-05 | Model Benchmark and Selection | Interactive PR curve with baseline prevalence and operating point. | Preferred for rare conversion context. | Curve and PR-AUC tested. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-06 | Model Benchmark and Selection | Interactive confusion matrix with counts, rates, business costs, and threshold. | Labels must be unambiguous. | Counts reconcile with predictions. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-07 | Model Benchmark and Selection | Show precision, recall, F1, target share, and buyer capture across thresholds. | Preserve current strong chart if already correct. | Threshold table and chart reconcile. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-08 | Model Benchmark and Selection | Translate threshold changes into audience, expected buyers, cost, and value. | Assumptions visible. | Formula tests. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-09 | Model Benchmark and Selection | Show predicted probability versus observed frequency with bin counts. | Use available holdout labels; not live production unless matured. | Calibration bins verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-10 | Model Benchmark and Selection | Compare score distributions by actual class or relevant cohort. | Class labels must match evidence source. | Distribution totals verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-11 | Model Benchmark and Selection | Compare validation and untouched holdout performance. | Avoid unsupported multi-run comparisons. | Paired metrics validated. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |
| SVE-MODEL-12 | Model Benchmark and Selection | Show available fold, seed, bootstrap, or sensitivity evidence. | Conditional when repeated-run evidence is unavailable. | Evidence count and limitations shown. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | CONDITIONAL |
| SVE-MODEL-13 | Model Benchmark and Selection | Show genuine ecommerce MLflow evidence only. | If comparable runs are insufficient, show readiness/status rather than demo metrics. | Experiment filters and excluded-run counts verified. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | CONDITIONAL |
| SVE-MODEL-14 | Model Benchmark and Selection | Export model comparison, threshold table, and metadata. | Include evidence source and version. | Download validation. | P1 | Approved model evaluation, threshold, calibration, and MLflow artifacts | PLANNED |

## XAI — Explainability

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-XAI-01 | Visitor Predictor / Explainability | Interactive SHAP global importance using approved feature names. | Do not substitute impurity importance when claiming SHAP. | SHAP artifact and model version verified. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-02 | Visitor Predictor / Explainability | Show feature impact direction and magnitude across visitors. | Use deterministic sample and precompute. | Sample, explainer, and output manifest. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-03 | Visitor Predictor / Explainability | Show effect pattern for selected important features. | Avoid causal language. | Feature values and SHAP values verified. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-04 | Visitor Predictor / Explainability | Explain one visitor score with positive and negative drivers. | Use model-consistent local SHAP values. | Base value plus contributions reconcile. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-05 | Visitor Predictor / Explainability | Show top positive and negative contributions with readable labels. | Limit clutter and group minor drivers. | Contribution arithmetic verified. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-06 | Visitor Predictor / Explainability | Translate local drivers into simple business language. | No invented reason beyond model evidence. | Template rules and examples reviewed. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-07 | Visitor Predictor / Explainability | Show safe, non-causal 'what would change the score' scenarios where technically defensible. | Clearly label as model scenario, not guaranteed outcome. | Conditional implementation and limitations. | P1 | Approved SHAP and feature artifacts | CONDITIONAL |
| SVE-XAI-08 | Visitor Predictor / Explainability | Compare important drivers across segments or score tiers. | Use comparable cohorts and sample sizes. | Cohort definitions verified. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-09 | Visitor Predictor / Explainability | Show data distance or support warning for unusual visitors. | Conditional on a defensible support metric. | Out-of-distribution logic tested. | P1 | Approved SHAP and feature artifacts | CONDITIONAL |
| SVE-XAI-10 | Visitor Predictor / Explainability | Provide business definitions, units, formulas, and source for each feature. | Use canonical feature contract. | Definitions match code. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-11 | Visitor Predictor / Explainability | Cache model and explainer; compute local explanation only on demand. | No full-dataset SHAP during page render. | Load and interaction timings. | P1 | Approved SHAP and feature artifacts | PLANNED |
| SVE-XAI-12 | Visitor Predictor / Explainability | Export visitor score, drivers, model version, and disclaimer. | No sensitive or unnecessary columns. | Download review. | P1 | Approved SHAP and feature artifacts | PLANNED |

## FCST — Forecasting and scenario intelligence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-FCST-01 | Business KPI Forecasting | Allow supported KPI, horizon, and scenario selection. | Restrict controls to validated forecast outputs. | Selector and source mapping tests. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-02 | Business KPI Forecasting | Interactive historical and future forecast with clear boundary. | Preserve current useful forecast charts. | Dates, values, and horizon verified. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-03 | Business KPI Forecasting | Show uncertainty bands and explain their meaning. | Do not show unsupported intervals. | Interval ordering and coverage fields tested. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-04 | Business KPI Forecasting | Show historical actual versus forecast performance. | Use leakage-safe backtest windows. | Backtest inputs and metrics verified. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-05 | Business KPI Forecasting | Show residual over time, distribution, and major outliers. | Use readable, non-repetitive diagnostics. | Residual calculations verified. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-06 | Business KPI Forecasting | Show MAE, RMSE, MAPE or appropriate metrics with plain meaning. | Avoid MAPE when denominator issues make it misleading. | Metric choice documented. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-07 | Business KPI Forecasting | Compare baseline, optimistic, and conservative business scenarios. | Scenarios must use explicit assumptions, not fake predictions. | Assumption table and formulas. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-08 | Business KPI Forecasting | Compare linked operational KPIs where supported. | Avoid implying causality. | Source alignment verified. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-09 | Business KPI Forecasting | Show available feature or decomposition evidence for forecast changes. | Conditional on model support. | Honest unavailable state. | P2 | Forecast outputs, backtests, and diagnostics | CONDITIONAL |
| SVE-FCST-10 | Business KPI Forecasting | Highlight unusual forecast periods or widening uncertainty. | Provide investigation guidance. | Exception rules tested. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-11 | Business KPI Forecasting | Generate actual-number outlook, risk, and recommended planning action. | Include limitation and horizon. | Narrative reviewed. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |
| SVE-FCST-12 | Business KPI Forecasting | Download filtered forecast, intervals, assumptions, and backtest metrics. | Include model/version and generation time. | Download validation. | P2 | Forecast outputs, backtests, and diagnostics | PLANNED |

## JOURNEY — Journey and funnel analysis

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-JOURNEY-01 | Journey and Funnel Analysis | Show view → add-to-cart → transaction counts and rates. | Define grain and deduplication clearly. | Stage logic and counts tested. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-02 | Journey and Funnel Analysis | Compare funnel performance across segments or intent tiers. | Use comparable denominators. | Rates reconcile. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-03 | Journey and Funnel Analysis | Show supported event transitions and major paths. | Pre-aggregate transitions; limit nodes and links. | Transition totals and path definitions verified. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-04 | Journey and Funnel Analysis | Highlight largest stage losses and affected segments. | Explain denominator and event window. | Drop-off calculations tested. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-05 | Journey and Funnel Analysis | Compare frequent converting and non-converting path patterns. | Use leakage-safe historical analysis. | Path cohorts verified. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-06 | Journey and Funnel Analysis | Show time between view, cart, and transaction where timestamps support it. | Use robust percentile summaries. | Timestamp and window logic tested. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-07 | Journey and Funnel Analysis | Support segment, intent tier, anomaly, item, and date filters where available. | Do not expose unsupported product categories. | Filter tests. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-08 | Journey and Funnel Analysis | Provide aggregated path table and selected visitor trace where appropriate. | Protect performance and privacy. | Drill-down validation. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-09 | Journey and Funnel Analysis | Suggest funnel improvement actions based on actual drop-offs. | Recommendations are operational hypotheses, not causal proof. | Rules and limitations reviewed. | P2 | RetailRocket events and precomputed transition tables | PLANNED |
| SVE-JOURNEY-10 | Journey and Funnel Analysis | Export aggregated funnel and transition evidence. | Include grain, filters, and window. | Download validation. | P2 | RetailRocket events and precomputed transition tables | PLANNED |

## ANOM — Anomaly intelligence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-ANOM-01 | Anomaly and Outlier | Show anomaly count, rate, severity, affected segments, and latest evidence time. | Preserve current strong summary if accurate. | Counts reconcile. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-02 | Anomaly and Outlier | Enhance purchase-intent versus anomaly-risk view with filters and hover detail. | Reuse current risk map logic where possible. | Coordinates and categories verified. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-03 | Anomaly and Outlier | Show selected anomalous visitor against normal and segment baselines. | Use robust percentiles and clear units. | Profile values verified. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-04 | Anomaly and Outlier | Compare anomaly rate and severity across business segments. | Preserve existing useful comparison. | Counts and denominators verified. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-05 | Anomaly and Outlier | Show rule, model, or signal contribution to anomaly status. | Use actual implemented rules and scores. | Reason flags reconcile. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-06 | Anomaly and Outlier | Provide sortable table with severity, reason, intent, segment, and recommended review. | Support export and stable ranking. | Queue logic tested. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-07 | Anomaly and Outlier | Show behavior timeline or feature evidence for selected visitor. | Conditional on available event detail. | Honest unavailable state when absent. | P2 | Anomaly outputs and monitoring snapshots | CONDITIONAL |
| SVE-ANOM-08 | Anomaly and Outlier | Show anomaly volume and rate over time where snapshots exist. | Conditional until comparable history exists. | Time-series contract verified. | P2 | Anomaly outputs and monitoring snapshots | CONDITIONAL |
| SVE-ANOM-09 | Anomaly and Outlier | Map anomaly patterns to review, suppress, monitor, or campaign-safe actions. | Rules must be explainable and reversible. | Rule tests. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |
| SVE-ANOM-10 | Anomaly and Outlier | Export filtered investigation table with evidence metadata. | Include generation time and definitions. | Download review. | P2 | Anomaly outputs and monitoring snapshots | PLANNED |

## MON — Monitoring, drift, labels, and production evidence

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-MON-01 | Monitoring, Drift and Health | Show app, exporter, monitoring snapshot, model artifact, and data status. | Do not overstate local checks as managed production. | Status-source mapping tested. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-02 | Monitoring, Drift and Health | Show scoring volume over time when logs provide history. | Conditional if only current snapshot exists. | Time grain and counts verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | CONDITIONAL |
| SVE-MON-03 | Monitoring, Drift and Health | Show score distribution, intent-band proportions, and shifts over time. | Use comparable bins and snapshots. | Distribution and drift inputs verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-04 | Monitoring, Drift and Health | Show feature-level drift history rather than only one report. | Conditional until historical reports exist. | History store and thresholds documented. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | CONDITIONAL |
| SVE-MON-05 | Monitoring, Drift and Health | Show prediction drift score and alert state over time. | Conditional until repeated snapshots exist. | Trend evidence verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | CONDITIONAL |
| SVE-MON-06 | Monitoring, Drift and Health | Provide current-vs-reference feature comparison with effect size and business meaning. | Reuse Evidently outputs without raw report clutter. | Feature mapping and values verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-07 | Monitoring, Drift and Health | Show alert type, severity, first seen, last seen, state, and acknowledgement where available. | Use real Alertmanager or project alert evidence. | Alert records validated. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-08 | Monitoring, Drift and Health | Show prediction windows, matured labels, valid joins, conflicts, and blocked status. | Truthfulness rule is mandatory. | Counts reconcile with delayed-label reports. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-09 | Monitoring, Drift and Health | Show ledger → eligible → labels received → valid joins → evaluated cohort. | Zero-stage retention must display N/A where appropriate. | Stage logic and labels tested. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-10 | Monitoring, Drift and Health | Show real production performance only when valid matured labels exist. | Otherwise show blocked/readiness state. | Conditional gate enforced by tests. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | CONDITIONAL |
| SVE-MON-11 | Monitoring, Drift and Health | Show model hash, metadata hash, feature schema hash, generation, and source. | Use committed provenance evidence. | Hashes and fields verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-12 | Monitoring, Drift and Health | Show genuine ecommerce experiment/registry readiness and latest verified run. | Exclude prompt/demo experiments. | Experiment filters tested. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-13 | Monitoring, Drift and Health | Show report generation time, sample sizes, drift alert, and report links. | Use validated project outputs. | Report paths and version verified. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |
| SVE-MON-14 | Monitoring, Drift and Health | Generate current health summary, required action, and limitations. | No generic or stale status. | Result-based narrative review. | P2 | Monitoring snapshots, MLflow, Evidently, alerts, and delayed labels | PLANNED |

## ARCH — Interactive architecture and lifecycle

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-ARCH-01 | MLOps Architecture | Create an interactive overview of data, ML, app, monitoring, deployment, and governance layers. | Preserve current architecture explanation; improve navigation and clarity. | All components map to real project files. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-02 | MLOps Architecture | Show raw events → features → model → scoring → business outputs. | Use grain, keys, and artifacts accurately. | Flow matches implementation. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-03 | MLOps Architecture | Show training → evaluation → champion → registry → scoring → monitoring → feedback. | Mark future feedback steps honestly. | Lifecycle state mapping verified. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-04 | MLOps Architecture | Explain tracking, artifacts, drift reports, and their separation. | Use actual local architecture. | Paths and services verified. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-05 | MLOps Architecture | Show snapshot → exporter → Prometheus → Grafana → Alertmanager. | Preserve fast monitoring design. | Components and ports verified. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-06 | MLOps Architecture | Show local service topology and health relationships. | Use actual compose services. | Compose config mapping reviewed. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-07 | MLOps Architecture | Show pods, services, config, secrets, probes, and chart packaging. | Use actual manifests and Helm templates. | Manifest mapping reviewed. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-08 | MLOps Architecture | Show push/PR/manual/scheduled checks, security audits, Docker, MLflow, Evidently, and deployment validation. | Do not claim automatic production deployment if absent. | Workflow mapping verified. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-09 | MLOps Architecture | Allow users to select a component and see purpose, inputs, outputs, files, and evidence. | Avoid giant unreadable diagrams. | All links and descriptions work. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |
| SVE-ARCH-10 | MLOps Architecture | Provide downloadable architecture inventory or evidence table. | Include file paths and current status. | Download review. | P3 | Docker, monitoring, Kubernetes, Helm, CI, and documentation | PLANNED |

## QA — Testing, visual review, documentation, and sign-off

| ID | Page / component | Requirement | Baseline / enhancement rule | Acceptance evidence | Priority | Data dependency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVE-QA-01 | Matrix validation | Validate unique IDs, allowed statuses, required fields, and complete category coverage. | Matrix itself must be testable. | Automated matrix test passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-02 | Page smoke tests | Every Streamlit page imports and renders without exception. | Use headless tests where practical. | Smoke suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-03 | Chart coverage tests | Extend chart-coverage tests to all required visuals and states. | Count alone is insufficient; validate identity and source. | Coverage suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-04 | Filter tests | Test global and page-specific filters, reset behavior, and cross-page persistence. | Include empty and boundary states. | Filter suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-05 | Calculation tests | Test KPI, funnel, threshold, scenario, drift, forecast, and export calculations. | Use fixed fixtures and reconciliation checks. | Calculation suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-06 | Data-contract tests | Validate source paths, columns, grain, keys, row counts, and schema hashes. | Fail clearly on stale or incompatible inputs. | Contract suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-07 | Performance tests | Measure cold, warm, and interaction timings for heavy pages. | Record machine/context and thresholds. | Performance report accepted. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-08 | Cache tests | Test cache correctness, invalidation, and no stale cross-filter results. | Avoid hidden state bugs. | Cache suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-09 | Download tests | Validate all exports, metadata, filters, file names, and row counts. | Open sample files during QA. | Download suite passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-10 | Visual hygiene | Check titles, labels, legends, spacing, clipping, overlap, responsive layout, and theme compatibility. | A visual is incomplete until it renders cleanly. | Automated and manual visual QA passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-11 | Accessibility review | Review contrast, color dependence, keyboard use, text alternatives, and table usability. | Fix all material findings. | Accessibility checklist accepted. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-12 | Truthfulness review | Verify evidence labels, simulated values, offline metrics, and production claims. | No unsupported claim may remain. | Truthfulness audit passes. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-13 | Combined visual review | Generate one final combined review PDF or readable review pack for all enhanced pages. | Use one consolidated review round after automated QA. | User approves final review pack. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-14 | Documentation update | Update README, page guide, data contracts, and run instructions. | Keep wording simple and current. | Docs reviewed against implementation. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-15 | Final matrix reconciliation | Reconcile every row with commit, test, visual, or blocked/excluded evidence. | No PLANNED or IN PROGRESS rows at sign-off. | Closure report generated. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-16 | Git diff review | Inspect staged files, generated artifacts, comments, and temporary files before each commit. | Do not stage helper packages, local environments, or caches. | Clean reviewed commit history. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |
| SVE-QA-17 | Final branch approval | Push the enhancement branch and obtain explicit approval before merge. | No automatic merge. | Approval and final status recorded. | P0 | Tests, screenshots, manifests, documentation, and Git evidence | PLANNED |

## Final sign-off gate

The Streamlit enhancement phase cannot be declared complete until:

- no row remains `PLANNED`, `IN PROGRESS`, or merely `IMPLEMENTED`;
- the current theme remains intact or every approved change is documented;
- every current chart has a recorded preservation decision;
- all new visuals use real evidence or are explicitly conditional/blocked;
- all pages pass functional, calculation, performance, accessibility, and visual-hygiene QA;
- one final combined visual review pack has been approved by the user;
- the branch is pushed and the user explicitly approves merge.

## New-chat start instruction

A new chat must first read this matrix and `STREAMLIT_VISUAL_ENHANCEMENT_HANDOFF.md` completely, verify the files exist in Git, confirm the remediation phase is closed, and audit the current Streamlit app before writing enhancement code.
