# Threshold Decision Visual Intelligence — Actual-Number Findings

## MLV-B01 — Threshold Decision Studio

**What it shows:** How validation precision, recall, F1, and predicted-positive share change as the decision threshold becomes stricter.

**Actual finding:** The saved champion uses threshold **0.98**. At that validation operating point, precision is **17.5%**, recall is **12.6%**, F1 is **0.146**, and the predicted-positive share is **0.101%**.

**Business conclusion:** The saved threshold deliberately selects a very small audience to prioritise target quality over broad converter capture.

**Recommended action:** Keep the current threshold as the controlled baseline and simulate any future threshold change against campaign capacity and value assumptions before deployment.

**Limitation:** The threshold curves are validation evidence. They do not represent current live-production outcomes.

## MLV-B04 — Confusion-Matrix Decision Map

**What it shows:** Final-holdout decision outcomes at the saved threshold.

**Actual finding:** The model captured **66** converters, generated **165** false alerts, missed **252** converters, and correctly excluded **248,156** non-converters. Precision is **28.6%**, recall is **20.8%**, and F1 is **0.240**.

**Business conclusion:** Approximately **29** of every 100 targeted visitors were converters, but **79.2%** of actual converters were not captured.

**Recommended action:** Use this strict operating point when campaign capacity is limited; evaluate a lower threshold when the cost of missed converters is more important than false alerts.

**Limitation:** These counts come from the untouched final holdout. Real production performance remains blocked until delayed labels mature.
