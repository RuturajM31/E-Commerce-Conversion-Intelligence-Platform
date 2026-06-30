# Package 2 — Executive Overview Intelligence

## Purpose

Package 2 converts the landing page into an executive decision workspace. It
uses validated holdout evidence, Package 1 ML validation outputs, approved
forecast outputs, repository operational artifacts, and clearly labelled
scenario assumptions.

The accepted main-page layout is preserved. The sidebar is intentionally
redesigned as a polished product-navigation and portfolio-ownership area.

## Controlling EXEC coverage

| ID | Final disposition | Implementation |
|---|---|---|
| SVE-EXEC-01 | VERIFIED | Context-rich KPI grid using final holdout evidence |
| SVE-EXEC-02 | VERIFIED | Five-stage visitor-to-buyer funnel |
| SVE-EXEC-03 | VERIFIED | Target size, contact cost, buyer value, and ROI simulator |
| SVE-EXEC-04 | VERIFIED | Audience reduction and buyer-yield comparison |
| SVE-EXEC-05 | VERIFIED | Threshold-selected audience composition with honest unsupported states |
| SVE-EXEC-06 | VERIFIED | Champion metrics, threshold, evidence type, and probability caution |
| SVE-EXEC-07 | VERIFIED | Forward KPI outlook using the approved best conversion forecast |
| SVE-EXEC-08 | VERIFIED | LOF anomaly volume, rate, severity, and affected segment |
| SVE-EXEC-09 | VERIFIED | Application, scoring, drift, alerts, and real label-readiness evidence |
| SVE-EXEC-10 | VERIFIED | Filter-aware finding, action, and limitation |
| SVE-EXEC-11 | EXCLUDED | Visible investigation links deferred with explicit user acceptance |
| SVE-EXEC-12 | EXCLUDED | Visible executive brief download deferred with explicit user acceptance |

## Sidebar scope

The Executive Overview sidebar now provides:

- Conversion Intelligence branding;
- Ruturaj Mokashi — Data Analyst portfolio ownership;
- champion, threshold, evidence track, and validated status;
- evidence freshness; and
- safe single-block Streamlit HTML rendering.

## Evidence rules

1. Final holdout metrics control champion claims.
2. Package 1 validation evidence controls clustering and anomaly summaries.
3. Scenario economics are never presented as observed revenue.
4. Missing or incompatible evidence produces an honest unavailable state.
5. Audience composition must use the selected campaign audience, never the
   complete validation projection by default.
6. Every major chart includes what it shows, how to read it, actual finding,
   business conclusion, recommended action, and limitation.
7. Accepted omissions must be marked `EXCLUDED`, not represented as completed.

## Closure evidence

See:

- `PACKAGE_02_MATRIX_EVIDENCE.csv`
- `PACKAGE_02_QA_PLAN.md`
- `PACKAGE_02_FINAL_RECONCILIATION.md`
- `tests/test_package_02_executive_overview.py`
