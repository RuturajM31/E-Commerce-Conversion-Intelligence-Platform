# Champion Selection Visual Intelligence — Actual-Number Findings

## MLV-A01 — Model Performance Frontier

**What it shows:** Validation precision and recall at each model's selected threshold. Marker size represents PR-AUC, while the connected frontier identifies models that are not dominated on both precision and recall.

**Actual finding:** The saved champion is **Tuned XGBoost**. The validation Pareto frontier contains **Random Forest Class Weight, Tuned XGBoost, Tuned Random Forest, SMOTE Logistic Regression Sample, Existing Champion Model**.

**Business conclusion:** Use the frontier to understand the targeting trade-off instead of selecting a model from one metric alone.

**Recommended action:** Keep the saved champion threshold under controlled monitoring and compare any challenger against the same validation split and business objective.

**Limitation:** This view describes validation behaviour. It does not prove current live-production performance.

## MLV-A02 — Multi-Metric Model Ranking

**What it shows:** Candidate strength across business score, PR-AUC, F1, precision, recall, and ROC-AUC.

**Actual finding:** **Tuned XGBoost** has the highest validation business score at **0.145**. The saved champion is **rank 1 of 8** with a validation business score of **0.145**.

**Business conclusion:** The champion decision is supported by multiple metrics rather than accuracy alone.

**Recommended action:** Continue using business score as the primary selection signal, while retaining PR-AUC and threshold metrics as mandatory supporting evidence.

**Limitation:** Heatmap colour is relative to the candidates in this table and must not be interpreted as an absolute production grade.

## MLV-A03 — Validation-to-Holdout Generalisation

**What it shows:** PR-AUC movement from validation to the untouched final holdout.

**Actual finding:** The champion PR-AUC changed by **+0.089**, from the validation stage to a final holdout PR-AUC of **0.152**.

**Business conclusion:** The final holdout provides a stronger and more honest proof point than validation alone.

**Recommended action:** Preserve the untouched-holdout result as the model-selection evidence and avoid repeatedly tuning against it.

**Limitation:** The final holdout contains only **318 positive rows**, so uncertainty remains material.

## MLV-A04 — Champion Scorecard and Identity

**What it shows:** The selected model, operating threshold, final holdout metrics, evaluation volume, and MLflow registry identity.

**Actual finding:** **Tuned XGBoost** uses threshold **0.98**. On **248,639 holdout rows**, precision is **28.6%**, recall is **20.8%**, F1 is **0.240**, PR-AUC is **0.152**, and ROC-AUC is **0.848**.

**Business conclusion:** The operating point deliberately targets a very small high-intent audience rather than maximising broad prediction volume.

**Recommended action:** Use MLflow registered model **ecommerce-conversion-champion**, version **3**, as the controlled champion reference.

**Limitation:** Real delayed production labels are not yet available beyond the source-data boundary, so live conversion performance must not be invented or claimed.
