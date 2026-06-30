# Package 1 QA Report

## Final package checks

| Check | Expected result | Evidence |
|---|---:|---|
| K-Means grid | 10 rows | `k=1..10` |
| Required K-Means candidates | 3 values | `k=3,5,7` |
| DBSCAN grid | 9 rows | `3 eps × 3 min_samples` |
| Shared sample | One row count | PCA, K-Means, DBSCAN, LOF |
| Governed major visuals | 19 | Unique chart keys |
| Interpretation/source coverage | 100% | `render_evidence_chart(...)` |
| Old page removed | Yes | No `8_MVD_Coverage_Proof.py` |
| Python syntax | Pass | `compileall` |
| Whitespace safety | Pass | `git diff --check` |

## Assistant-side isolated results

- Focused tests: **22 passed**
- Synthetic source rows: **600**
- Synthetic evidence sample: **500**
- K-Means grid rows: **10**
- DBSCAN grid rows: **9**
- Projection rows: **500**
- Synthetic CLI generation: **passed**
- Full page controlled runtime execution: **passed**

## User-environment closure command

The final apply command must regenerate evidence from the real
`visitor_features.csv`, run the focused tests, compile the repository code, and
run `git diff --check`. No staging or commit occurs inside the package installer.
