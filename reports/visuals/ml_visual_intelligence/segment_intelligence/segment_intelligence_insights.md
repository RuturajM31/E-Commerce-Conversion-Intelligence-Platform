# Segment Intelligence — Actual-Number Findings

## MLV-G02 — Score Distribution by Segment

**What it shows:** Purchase-intent score distributions across **78,998** visitors and **5** business segments.

**Actual finding:** **High Intent** has the highest median score at **0.931**. The largest audience is **Low Intent** with **45,719** visitors, representing **57.9%** of the scored population.

**Business conclusion:** The score bands create a prioritisation ladder, but the largest audience is not automatically the right audience for expensive campaigns.

**Recommended action:** Use segment distributions for campaign sizing and targeting priority, then validate treatment effectiveness when matured outcomes become available.

**Limitation:** The current source has no actual converter counts or conversion rates. G02 explains score composition only; it does not prove observed conversion performance.

## Conditional segment visuals

- **MLV-G01 Segment-Level Performance Heatmap — CONDITIONAL:** requires actual outcomes by segment.
- **MLV-G03 Error Rate by Behaviour Cohort — CONDITIONAL:** requires row-level actual labels and prediction errors.
- **MLV-G04 Cohort Opportunity Matrix — CONDITIONAL:** requires validated conversion or business-value outcomes by cohort.
