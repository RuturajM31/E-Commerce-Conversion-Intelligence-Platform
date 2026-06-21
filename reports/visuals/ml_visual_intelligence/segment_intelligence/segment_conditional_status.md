# Segment and Cohort Conditional Status

| Visual ID | Status | Current blocker | Required next evidence |
|---|---|---|---|
| MLV-G01 | CONDITIONAL | `actual_converters` and `conversion_rate` are null for every segment | Matured row-level outcomes aggregated by `intent_segment` |
| MLV-G03 | CONDITIONAL | No actual labels are present in `visitor_scores.csv` | Visitor-level actual outcome and prediction/error flag |
| MLV-G04 | CONDITIONAL | Opportunity cannot be validated from scores alone | Cohort outcomes plus business value or campaign economics |

These are source-data boundaries, not charting failures. No conversion or error result is fabricated.
