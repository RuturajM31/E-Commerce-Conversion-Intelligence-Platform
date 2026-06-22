# Robustness and Stability Visual Intelligence — Actual-Number Findings

## MLV-F01 — Model Robustness Heatmap

**Actual finding:** Across **3 seeds**, **Tuned XGBoost** achieved mean PR-AUC **0.094** and mean ROC-AUC **0.820**.

**Business conclusion:** The champion remains competitive across the tested random seeds, but the evidence is limited to three runs per model.

## MLV-F02 — Metric Variability Distribution

**Actual finding:** Champion PR-AUC ranged from **0.078** to **0.122**; ROC-AUC ranged from **0.794** to **0.841**.

**Recommended action:** Keep seed-level variability visible during future retraining and expand the seed count before making statistical stability claims.

## MLV-F03 — Champion Stability Profile

**Actual finding:** The strongest challenger by mean PR-AUC is **Tuned Random Forest**, with mean PR-AUC **0.093**.

**Limitation:** Profile scores are relative within the current candidate set and are not universal quality grades.

## MLV-F04 — Sensitivity Tornado

**Actual finding:** Removing anomaly-flagged visitors changed champion PR-AUC by **-0.039** and ROC-AUC by **-0.019**.

**Business conclusion:** A material negative delta means anomaly-heavy visitors contribute meaningfully to measured validation performance.

## MLV-F05 — Performance Degradation Waterfall

**Actual finding:** Champion PR-AUC moved from **0.063** to **0.024** after removing **941** anomaly-flagged rows. ROC-AUC moved from **0.803** to **0.784**.

**Recommended action:** Monitor anomaly composition alongside model metrics and avoid treating a single aggregate validation score as universally representative.
