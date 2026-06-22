# Explainability Visual Intelligence — Actual-Number Findings

## MLV-E01 — Global SHAP Beeswarm

**What it shows:** The distribution and direction of native XGBoost contributions across **2,504** sampled visitors.

**Actual finding:** **Total events** has the largest global mean absolute contribution at **0.405** log-odds.

**Business conclusion:** This feature has the strongest influence on how the saved champion separates higher- and lower-intent visitors.

**Limitation:** Contributions describe model behaviour and do not prove causality.

## MLV-E02 — Global Feature-Impact Ranking

**Actual finding:** For **Total events**, higher values generally increase model score; its value/contribution correlation is **+0.232**.

**Recommended action:** Prioritise data-quality monitoring and business interpretation for the highest-impact features before expanding the model.

## MLV-E03 — SHAP Dependence Plot

**What it shows:** The model response to **Total events**, coloured by **Activity span**.

**Business conclusion:** Read the binned median line as the model's learned response pattern, not as a causal conversion curve.

## MLV-E04 — Visitor-Level Waterfall

**Actual finding:** The highest-score sampled visitor has probability **99.7%**. Its strongest local driver is **Activity span** with contribution **+1.693** log-odds.

**Recommended action:** Use local explanations to support campaign-review and debugging, not as standalone automated decisions.

## MLV-E05 — Representative Visitor Decision Paths

- **Low intent:** visitor `5121`, score **15.1%**, leading driver **Total events** (-0.682).
- **Medium intent:** visitor `7567`, score **25.1%**, leading driver **Total events** (-0.438).
- **High intent:** visitor `476504`, score **59.0%**, leading driver **Total events** (+0.495).

**Business conclusion:** Different visitors can reach different intent scores through different behavioural paths, even when the same global features dominate overall.

## MLV-E06 — Feature Interaction Heatmap

**Actual finding:** The strongest off-diagonal pair is **Activity span × Total events** with mean absolute interaction **0.098**.

**Recommended action:** Monitor both features together and consider interaction-aware business rules only after confirming stability across time.

## Shared limitation

The visitor feature source does not contain matured production outcomes. These visuals explain the saved model's scoring logic; they do not measure live campaign lift or prove that changing a feature will change conversion.
