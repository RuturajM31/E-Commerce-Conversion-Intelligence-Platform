# Package 1 Final Reconciliation

## Why this reconciliation was required

The final 47-segment rendered review confirmed that the visual-finish correction
improved the page, but it also exposed two evidence-quality risks that could not
be left unresolved:

1. the top champion KPI could be sourced from a conflicting summary artifact;
2. historical clustering tables used different samples and could be mistaken for
   the controlling Package 1 calculation.

The review also confirmed that several promised supporting diagnostics needed to
be completed rather than merely mentioned.

## What was corrected

### 1. Champion metric truthfulness

The named champion row in
`reports/tables/final_true_champion_comparison.csv` is now the controlling source
for displayed PR-AUC, precision, recall, F1, and comparison threshold.

`final_true_champion_summary.csv` remains downloadable as an explicitly labelled
fallback/audit artifact. It is not silently used as the primary top KPI when the
controlling row exists.

The model chart now explains that the highest raw PR-AUC candidate may be
non-deployable or may not be the selected champion. Selection is presented as an
operational decision involving deployability, threshold behaviour, and stability,
not as a raw-score beauty contest.

### 2. One coherent unsupervised sample

The evidence builder now uses one deterministic sample for:

- PCA coordinates and loadings;
- K-Means grid and selected labels;
- DBSCAN grid, point types, and selected labels;
- LOF scores, severity, profiles, and investigation rows.

Every controlling method table records the same `sample_rows` value. Historical
course artifacts remain available in separate Evidence Library sections with
clear descriptions of their different row counts and parameter scopes.

### 3. Complete Package 1 analytical scope

The final implementation adds the previously missing supporting evidence:

- PCA density and component explanations;
- complete K-Means `k=1..10` elbow evidence;
- required `k=3/5/7` candidate labels;
- K-Means parallel profiles and observed value/conversion comparison;
- DBSCAN k-distance evidence;
- exact DBSCAN `3×3` parameter grid;
- LOF deviation profile;
- ranked LOF investigation queue; and
- selected-visitor descriptive feature deviations.

### 4. Governed visual explanations

All 19 major visuals use `render_evidence_chart(...)`. This shared component
prevents a chart from appearing without interpretation and provenance.

Each major visual contains:

- what it shows;
- how to read it;
- actual-number finding;
- business conclusion;
- recommended action;
- limitation;
- source;
- evidence type; and
- generation/refreshed date.

### 5. Visual-finish fixes retained

The reconciliation preserves the earlier verified improvements:

- correctly rendered interpretation cards;
- responsive four-card KPI grid;
- reduced hero and vertical spacing;
- readable DBSCAN legend;
- visible K-Means centroids;
- non-distorted LOF marker sizing; and
- log-readable LOF score distribution.

## QA performed before packaging

Assistant-side isolated QA completed:

- 22 focused tests passed;
- Python compilation passed;
- synthetic CLI evidence generation passed with 500 visitors;
- K-Means grid rows: 10;
- DBSCAN grid rows: 9;
- projection rows: 500;
- all expected evidence tables and manifest were written; and
- the complete Streamlit page executed through a controlled runtime stub with all
  diagnostic checkboxes open.

The user environment remains the final source of truth. The package command reruns
all focused tests, evidence generation, compilation, and `git diff --check` before
any Git commit is allowed.

## Honest remaining boundary

Package 1 is offline analytical validation. It does not claim:

- live production performance;
- causal segment effects;
- confirmed fraud/anomaly labels;
- production drift monitoring;
- campaign lift; or
- cloud deployment completion.

Those are controlled by later packages in the 204-row roadmap.

## Final review policy

No additional screenshot loop is required. The existing 47-segment rendered
review is the controlling visual audit. Package 1 closes through calculation,
tests, source review, matrix evidence, and one clean Git commit.
