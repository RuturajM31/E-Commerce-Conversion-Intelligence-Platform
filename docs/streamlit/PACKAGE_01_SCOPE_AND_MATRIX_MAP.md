# Package 1 — Shared Design System and ML Validation Evidence

## Purpose

Package 1 creates the reusable Streamlit product foundation and rebuilds the
former **MVD Coverage Proof** page as **Machine Learning Validation & Evidence**.
The page is now a governed analytical evidence area rather than a collection of
unconnected screenshots and tables.

Historical artifact names such as `mvd_kmeans_summary.csv` are retained only
where renaming would break lineage. They are clearly labelled as historical and
do not control the same-sample Package 1 conclusions.

## Final deliverables

### Shared product foundation

1. Central design tokens and responsive product CSS.
2. Stable page hero, KPI grid, section hierarchy, detail cards, empty states,
   evidence notes, tables, and export components.
3. Shared Plotly layout and export configuration.
4. One governed chart component that always renders:
   - the chart;
   - what it shows;
   - how to read it;
   - the actual finding;
   - the business conclusion;
   - the recommended action;
   - the limitation; and
   - source, evidence type, and generation date.

### Model assurance

- Champion metrics are taken from the named champion row in
  `final_true_champion_comparison.csv`.
- The alternate champion summary remains available for audit but is not silently
  mixed into the top KPI.
- The comparison chart distinguishes selected champion, deployable alternatives,
  experimental/non-deployable candidates, and legacy benchmarks.
- Active threshold wording is used instead of claiming an unsupported production
  threshold.

### PCA evidence

- Interactive PC1–PC2–PC3 view with 2D fallback.
- Density contours in the PC1–PC2 plane.
- Scree and cumulative explained-variance chart.
- Feature-loading heatmap.
- Plain-language component cards based on strongest positive and negative
  loadings.

### K-Means evidence

- Interactive 2D/3D cluster landscape with visible centroids.
- Complete elbow grid for `k=1..10`.
- Required candidate comparison for `k=3, 5, 7`.
- Same-sample selection rule based on silhouette with inertia as supporting
  evidence.
- Cluster profile heatmap.
- Parallel-coordinates profile view.
- Cluster size plus observed value/conversion evidence.
- Transparent, rule-based analytical test suggestions.

### DBSCAN evidence

- Interactive 2D/3D density landscape.
- Core, border, and noise states.
- Readable grouped legend for high-cluster-count outcomes.
- Sorted k-distance diagnostic with selected epsilon.
- Exact same-sample 3×3 grid:
  - `eps = 0.5, 1.0, 2.0`
  - `min_samples = 3, 5, 10`
- Silhouette-after-noise and noise-rate heatmaps.
- Guardrailed parameter selection that avoids isolated fragmented maxima.

### LOF evidence

- Interactive 2D/3D anomaly landscape with capped logarithmic marker sizing.
- LOF score distribution on a log scale with active analytical threshold.
- Average standardized deviation profile for flagged versus normal visitors.
- Ranked descriptive investigation queue.
- Selected-visitor feature deviation chart.
- Explicit statement that feature deviations are descriptive z-scores, not SHAP
  values or causal contributions.

## Controlling data contract

### Source visitor table

- Path: `data/processed/visitor_features.csv`
- Grain: one row per visitor
- Key: `visitorid` when available
- Required columns:
  - `total_events`
  - `view_count`
  - `addtocart_count`
  - `unique_items`
  - `activity_span_ms`

### Same-sample rule

PCA, K-Means, DBSCAN, and LOF are recalculated from one deterministic visitor
sample using random state 42. Historical course tables are never mixed into the
same-sample parameter selection or method summary.

### Generated tables

- `reports/tables/ml_validation_projection_sample.csv`
- `reports/tables/ml_validation_pca_variance.csv`
- `reports/tables/ml_validation_pca_loadings.csv`
- `reports/tables/ml_validation_cluster_profile.csv`
- `reports/tables/ml_validation_cluster_business_summary.csv`
- `reports/tables/ml_validation_kmeans_grid.csv`
- `reports/tables/ml_validation_dbscan_grid.csv`
- `reports/tables/ml_validation_dbscan_k_distance.csv`
- `reports/tables/ml_validation_lof_feature_profile.csv`
- `reports/tables/ml_validation_lof_investigation.csv`
- `reports/tables/ml_validation_method_summary.csv`

### Manifest

- Path: `reports/metadata/ml_validation_manifest.json`
- Records source hash, source rows, sample rows, generation time, feature set,
  selected parameters, selection rules, metric rules, output paths, and honest
  limitations.

## Matrix rows targeted by Package 1

Package 1 supplies implementation evidence for these existing controlling rows.
It does not claim that unrelated rows in the 204-row matrix are complete.

- Governance: `SVE-GOV-03`, `04`, `05`, `06`, `07`, `08`, `09`, `11`
- Audit: `SVE-AUD-09`, `10`, `15`
- UX: `SVE-UX-01` through `SVE-UX-15`
- Shared: `SVE-SHARED-09`, `11`, `13`, `14`, `16`, `17`, `18`
- QA: `SVE-QA-02`, `03`, `05`, `06`, `09`, `10`, `11`, `12`, `16`

The row-by-row implementation evidence is recorded in
`PACKAGE_01_MATRIX_EVIDENCE.csv`.

## Completion gate

Package 1 can close only after:

- evidence generation succeeds;
- K-Means produces 10 grid rows;
- DBSCAN produces exactly 9 grid rows;
- focused tests pass;
- Python compilation passes;
- `git diff --check` passes;
- no major chart bypasses the governed evidence component;
- controlling and historical metrics are visibly separated;
- the final 47-segment rendered review has been audited; and
- Git changes are reviewed before the Package 1 commit.
