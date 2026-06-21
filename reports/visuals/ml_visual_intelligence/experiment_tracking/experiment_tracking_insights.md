# Experiment Tracking Visual Intelligence — Actual-Number Findings

## MLV-I01 — Experiment Parallel Coordinates

**What it shows:** **3** comparable MLflow runs across **4** exact shared metric keys.

**Business conclusion:** The visual reveals trade-offs between metrics without collapsing model selection into one number.

## MLV-I02 — Run Performance Timeline

**Actual finding:** The selected shared timeline metric is **Correctness/Mean** using exact key `correctness/mean`. The highest logged value is **0.8571** from **improved-session-evaluation**. The current champion run does not contain this selected shared metric key.

**Recommended action:** Keep validation and holdout metric keys explicitly separated in future MLflow runs.

## MLV-I04 — Run Comparison Matrix

**Actual finding:** The matrix compares only metrics recorded under the same MLflow key across at least two runs. Missing cells remain visibly blank rather than being imputed.

**Business conclusion:** This protects the comparison from silently mixing experiments with different evaluation contracts.

## MLV-I05 — Champion–Challenger Evolution

**Actual finding:** The registry contains **4** registered models, **15** versions, and **13** aliases. The saved production lineage points to **ecommerce-conversion-champion** version **3** with alias **champion**.

## MLV-I03 — Hyperparameter Response Surface

**Status:** CONDITIONAL.

**Observed evidence:** **0** numeric parameters currently have at least three unique values across six or more comparable runs.

**Requirement:** A reliable response surface needs at least two sufficiently varied numeric parameters and enough comparable runs under one consistent evaluation contract. No sparse surface is fabricated.
