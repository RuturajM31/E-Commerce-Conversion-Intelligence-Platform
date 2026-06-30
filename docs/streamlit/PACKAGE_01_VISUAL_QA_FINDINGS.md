# Package 1 Visual QA Findings and Corrections

## Review source

The review used 57 rendered screenshots covering:

- Overview
- PCA Intelligence
- K-Means
- DBSCAN
- LOF
- Evidence Library and nested evidence tabs

## Blocking issue found

The interpretation component exposed raw HTML tags instead of rendering the
six explanation cards. The component generated indented HTML fragments, which
Markdown treated as a code block.

**Correction:** render compact, non-indented HTML with
`unsafe_allow_html=True`, plus a regression test.

## Layout and finish issues corrected

- Replaced unstable Streamlit KPI columns with a responsive four-card CSS grid.
- Reduced hero and KPI height so analytical content appears sooner.
- Changed interpretation layout to three columns on wide screens, two on
  medium screens, and one on small screens.
- Strengthened source-note contrast and reduced excessive vertical spacing.
- Increased 2D and 3D chart room while preserving responsive width.

## Visual intelligence corrections

### K-Means

- Added visible cluster centroids.
- Highlighted the strongest silhouette candidate.
- Kept inertia as supporting compactness evidence.

### DBSCAN

- Consolidated small clusters into `Other dense clusters`.
- Preserved `Noise` as a separate neutral group.
- Limited the legend to the largest clusters plus grouped remainder.
- Preserved core, border, and noise point symbols inside each trace.

### LOF

- Log-scaled and capped point sizes for readable anomaly landscapes.
- Rebuilt the score distribution on a log scale.
- Normalized distribution density so the smaller outlier group remains visible.

## Honest limitation

This correction improves the current offline validation page. Operational
business segmentation, journey intelligence, production monitoring, and cloud
deployment remain controlled by later packages in the 204-row matrix.
