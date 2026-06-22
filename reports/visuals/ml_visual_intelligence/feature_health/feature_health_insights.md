# Data and Feature Health Visual Intelligence — Actual-Number Findings

## MLV-H01 — Feature Distribution Profile

**What it shows:** Outlier-safe density profiles for all seven model features across **78,998** processed visitors.

**Actual finding:** Median total events is **2.00** and the 95th percentile is **9.00**. Median add-to-cart events is **0.00**.

**Business conclusion:** Strongly skewed behaviour features justify robust scaling, percentile monitoring, and careful treatment of extreme visitors.

**Limitation:** Count and duration panels use log display for readability, while cards preserve original units.

## MLV-H02 — Feature Correlation Cluster Map

**Actual finding:** The strongest absolute off-diagonal association is **Add-to-cart events × Cart-to-view ratio** with Spearman correlation **+0.998**.

**Business conclusion:** Highly related features may carry overlapping information. Keep this visible during feature review, but do not assume correlation means one behaviour causes another.

## MLV-H04 — Missingness and Validity Map

**Actual finding:** The current feature snapshot contains **267** missing, non-finite, negative, or ratio-above-one rule violations. The highest zero rate is **Add-to-cart events** at **91.13%**.

**Business conclusion:** Zero values must be interpreted feature by feature. Zero add-to-cart activity can be valid behaviour rather than a defect.

**Recommended action:** Re-run these checks during every scoring and retraining pipeline and alert only on rule violations or meaningful distribution shifts.
