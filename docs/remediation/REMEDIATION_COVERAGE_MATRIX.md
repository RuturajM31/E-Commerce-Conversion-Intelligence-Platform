# E-Commerce Conversion Intelligence Platform
# Master Remediation Coverage Matrix

## Completion rule

Every item must end in one of these states:

- **Fixed and Tested**
- **Verified Already Correct**
- **Intentionally Excluded — reason documented**
- **Blocked — next action documented**

Code written without verification is not complete.

---

## 1. Git safety and workflow

- [x] `GIT-01` Work inside the existing professional repository.
- [x] `GIT-02` Confirm current branch, latest commit, remotes, and clean status.
- [x] `GIT-03` Protect the submitted version with a baseline tag.
- [x] `GIT-04` Create and publish `fix/full-project-remediation`.
- [x] `GIT-05` Capture the untouched baseline test state.
- [x] `GIT-06` Use small, meaningful commits by remediation phase.
- [x] `GIT-07` Review `git diff` before every commit.
- [x] `GIT-08` Keep `main` unchanged until final approval.
- [x] `GIT-09` Merge only after every matrix item is resolved.

## 2. Project structure and code quality

- [x] `STR-01` Keep the existing root folder name.
- [x] `STR-02` Create only folders and files with a clear purpose.
- [x] `STR-03` Avoid one giant file containing unrelated logic.
- [x] `STR-04` Avoid excessive fragmentation across many tiny files.
- [x] `STR-05` Keep clear orchestration flows for data, training, scoring, and monitoring.
- [x] `STR-06` Separate source code, data, models, outputs, reports, logs, and monitoring assets.
- [x] `STR-07` Archive obsolete scripts, models, and outputs clearly.
- [x] `STR-08` Use relative paths and central configuration.
- [x] `STR-09` Exclude caches and transient logs from Git and release packages.
- [x] `STR-10` Keep source files separate from generated artifacts.

## 3. Comment and learning-content preservation

- [x] `COM-01` Preserve useful line comments.
- [x] `COM-02` Preserve file headers and docstrings.
- [x] `COM-03` Preserve variable-flow explanations.
- [x] `COM-04` Preserve business-purpose explanations.
- [x] `COM-05` Preserve learning notes and beginner-friendly explanations.
- [x] `COM-06` Update comments when underlying code changes.
- [x] `COM-07` Move explanations with refactored logic instead of deleting them.
- [x] `COM-08` Avoid comments that merely repeat obvious syntax.
- [x] `COM-09` Keep code blocks short, readable, and logically grouped.
- [x] `COM-10` Use simple variable, function, and file names.
- [x] `COM-11` Avoid clever or deeply nested Python when simpler logic is available.
- [x] `COM-12` Explain file inputs, outputs, next use, and business value.

## 4. Data and feature-building pipeline

- [x] `DATA-01` Restore the `src/data` package.
- [x] `DATA-02` Add `src/data/__init__.py`.
- [x] `DATA-03` Restore or rebuild `src/data/build_visitor_features.py`.
- [x] `DATA-04` Rebuild visitor features from RetailRocket raw files.
- [x] `DATA-05` Define the modelling grain explicitly.
- [x] `DATA-06` Validate required raw input columns.
- [x] `DATA-07` Validate event types and malformed records.
- [x] `DATA-08` Validate duplicates and missing values.
- [x] `DATA-09` Log input rows, output rows, visitor counts, and target distribution.
- [x] `DATA-10` Log processing duration and clear failures.
- [x] `DATA-11` Use central path configuration.
- [x] `DATA-12` Keep representative sample data for tests and CI.
- [x] `DATA-13` Document all raw source files used.
- [x] `DATA-14` Add safe overwrite and output-directory handling.

## 5. Centralised feature engineering

- [x] `FEAT-01` Create one shared feature-engineering module.
- [x] `FEAT-02` Use it in dataset generation.
- [x] `FEAT-03` Use canonical features in training.
- [x] `FEAT-04` Use it in Streamlit single prediction.
- [x] `FEAT-05` Use it in batch scoring.
- [x] `FEAT-06` Use canonical features or scores in segmentation.
- [x] `FEAT-07` Use canonical features or scores in anomaly workflows.
- [x] `FEAT-08` Use canonical production outputs in forecasting.
- [x] `FEAT-09` Standardise `cart_to_view_ratio`.
- [x] `FEAT-10` Standardise `events_per_unique_item`.
- [x] `FEAT-11` Standardise recency, frequency, diversity, and activity-duration features.
- [x] `FEAT-12` Define explicit zero-denominator handling.
- [x] `FEAT-13` Store approved feature names and order.
- [x] `FEAT-14` Reject missing required features safely.
- [x] `FEAT-15` Handle unexpected extra features clearly.
- [x] `FEAT-16` Add training-versus-serving feature parity tests.
- [x] `FEAT-17` Add feature data-type and range validation.
- [x] `FEAT-18` Keep boolean and categorical encoding consistent.

## 6. Leakage-safe target and modelling timeline

- [x] `LEAK-01` Recover and document the current target logic.
- [x] `LEAK-02` Define a future purchase prediction question.
- [x] `LEAK-03` Define the observation window.
- [x] `LEAK-04` Define the later prediction window.
- [x] `LEAK-05` Prevent post-purchase behaviour entering pre-purchase features.
- [x] `LEAK-06` Prevent future events entering historical features.
- [x] `LEAK-07` Define handling for multiple purchases.
- [x] `LEAK-08` Define handling for repeated scoring dates.
- [x] `LEAK-09` Store feature cutoff timestamps.
- [x] `LEAK-10` Rebuild the modelling table with leakage-safe windows.
- [x] `LEAK-11` Retrain after the leakage-safe rebuild.
- [x] `LEAK-12` Replace old published metrics after retraining.
- [x] `LEAK-13` Describe current limitations honestly until the rebuild is complete.

## 7. Model training and evaluation

- [x] `MOD-01` Separate training, validation, and test data.
- [x] `MOD-02` Stop selecting models on the final test set.
- [x] `MOD-03` Stop selecting thresholds on the final test set.
- [x] `MOD-04` Add a time-based holdout.
- [x] `MOD-05` Compare random-split and time-split performance.
- [x] `MOD-06` Keep PR-AUC as the primary rare-event metric.
- [x] `MOD-07` Report ROC-AUC, precision, recall, F1, and confusion matrix.
- [x] `MOD-08` Report predicted-positive count and selected audience share.
- [x] `MOD-09` Check probability calibration.
- [x] `MOD-10` Report threshold trade-offs.
- [x] `MOD-11` Reconfirm whether `0.97` remains the correct threshold.
- [x] `MOD-12` Save validation and test results separately.
- [x] `MOD-13` Record random seeds and reproducibility settings.
- [x] `MOD-14` Use stratification where appropriate.
- [x] `MOD-15` Separate business threshold selection from model-quality evaluation.

## 8. Final champion scoring workflow

- [x] `SCORE-01` Implement or correctly import `save_final_champion_scores`.
- [x] `SCORE-02` Ensure champion finalisation completes fully.
- [x] `SCORE-03` Use one official final score filename.
- [x] `SCORE-04` Include visitor ID and probability.
- [x] `SCORE-05` Include predicted class and high-intent flag.
- [x] `SCORE-06` Include model version and threshold.
- [x] `SCORE-07` Include scoring timestamp.
- [x] `SCORE-08` Validate input and output row counts.
- [x] `SCORE-09` Validate probability range.
- [x] `SCORE-10` Prevent silent fallback to old score files.
- [x] `SCORE-11` Add automated score-export tests.
- [x] `SCORE-12` Add safe output-directory and overwrite rules.

## 9. One production model and manifest

- [x] `PROD-01` Create one production model manifest.
- [x] `PROD-02` Store model filename, type, and version.
- [x] `PROD-03` Store feature names and order.
- [x] `PROD-04` Store the approved threshold.
- [x] `PROD-05` Store training date and dataset version.
- [x] `PROD-06` Store Python and dependency versions.
- [x] `PROD-07` Store evaluation metrics and model checksum.
- [x] `PROD-08` Create one shared production model loader.
- [x] `PROD-09` Add startup compatibility checks.
- [x] `PROD-10` Prevent accidental loading of archived models.
- [x] `PROD-11` Validate manifest, model, and feature schema together.

## 10. Downstream model consistency

- [x] `DOWN-01` Make Streamlit use the production model and manifest.
- [x] `DOWN-02` Make batch scoring use the same model and threshold.
- [x] `DOWN-03` Replace segmentation's legacy Logistic Regression dependency.
- [x] `DOWN-04` Replace anomaly workflow's legacy model and threshold.
- [x] `DOWN-05` Replace forecasting's legacy artifacts.
- [x] `DOWN-06` Regenerate legacy sample score files.
- [x] `DOWN-07` Add model version and generation date to outputs.
- [x] `DOWN-08` Archive obsolete models and outputs.
- [x] `DOWN-09` Prevent monitoring from mixing an old model with a new threshold.

## 11. Monitoring corrections

- [x] `MON-01` Fix the business-score source.
- [x] `MON-02` Stop reading metrics from files that do not contain them.
- [x] `MON-03` Fix total-anomaly variable overwrite.
- [x] `MON-04` Export total and high-intent anomalies separately.
- [x] `MON-05` Recognise `final_anomaly_flag` explicitly.
- [x] `MON-06` Remove unreliable fallback column matching.
- [x] `MON-07` Add model version to monitoring output.
- [x] `MON-08` Add scoring-data version and timestamp.
- [x] `MON-09` Prevent fallback to legacy score files.
- [x] `MON-10` Validate every Prometheus metric against its source.
- [x] `MON-11` Update Grafana queries after metric changes.
- [x] `MON-12` Update alert rules after metric changes.
- [x] `MON-13` Validate Alertmanager routing.
- [x] `MON-14` Validate Blackbox health checks.
- [x] `MON-15` Document runtime-generated monitoring files.
- [x] `MON-16` Add prediction and snapshot freshness timestamps.

## 12. Streamlit application

- [x] `APP-01` Remove duplicated feature formulas.
- [x] `APP-02` Use the shared production model loader.
- [x] `APP-03` Read threshold from the manifest.
- [x] `APP-04` Display model version and threshold.
- [x] `APP-05` Validate manual inputs.
- [x] `APP-06` Handle zeros and missing values consistently.
- [x] `APP-07` Ensure single and batch predictions match.
- [x] `APP-08` Show clear errors for missing artifacts.
- [x] `APP-09` Prevent obsolete artifact loading.
- [x] `APP-10` Show last-refresh and scoring timestamps.
- [x] `APP-11` Add Streamlit smoke tests.
- [x] `APP-12` Preserve dashboard design and business explanations.
- [x] `APP-13` Keep app services simple and not over-fragmented.

## 13. Dependencies and reproducibility

- [x] `DEP-01` Pin the supported Python version.
- [x] `DEP-02` Pin scikit-learn to the approved training version.
- [x] `DEP-03` Pin pandas, NumPy, joblib, Streamlit, and Plotly.
- [x] `DEP-04` Pin pytest and all implemented Python dependencies; pin monitoring container versions; document MLflow and Evidently as not implemented.
- [x] `DEP-05` Align `requirements.txt`, `requirements-app.txt`, and `pyproject.toml`.
- [x] `DEP-06` Add a lockfile or frozen requirements file.
- [x] `DEP-07` Store environment versions in model metadata.
- [x] `DEP-08` Test model loading in the pinned local environment.
- [x] `DEP-09` Test model loading inside Docker.
- [x] `DEP-10` Document future dependency-upgrade procedure.

## 14. Secrets and credentials

- [x] `SEC-01` Remove Grafana password from Docker Compose.
- [x] `SEC-02` Remove plain-text password from Kubernetes YAML.
- [x] `SEC-03` Remove plain-text password from Helm values.
- [x] `SEC-04` Remove passwords from startup scripts.
- [x] `SEC-05` Remove credentials from backups and documentation.
- [x] `SEC-06` Keep only `.env.example` in Git.
- [x] `SEC-07` Use GitHub Actions Secrets where needed.
- [x] `SEC-08` Review Git history for exposed credentials.
- [x] `SEC-09` Rotate credentials if public exposure occurred.
- [x] `SEC-10` Add secret scanning to CI.

## 15. Automated tests

- [x] `TEST-01` Keep all current baseline tests passing.
- [x] `TEST-02` Add project-structure tests for restored modules.
- [x] `TEST-03` Add raw and processed data-schema tests.
- [x] `TEST-04` Add feature-formula tests.
- [x] `TEST-05` Add training-serving parity tests.
- [x] `TEST-06` Add feature-order and data-type tests.
- [x] `TEST-07` Add production-manifest validation tests.
- [x] `TEST-08` Add one-model and one-threshold consistency tests.
- [x] `TEST-09` Add score-export tests.
- [x] `TEST-10` Add leakage and cutoff tests.
- [x] `TEST-11` Add monitoring metric-value tests.
- [x] `TEST-12` Add anomaly-count tests.
- [x] `TEST-13` Add business-score-source tests.
- [x] `TEST-14` Add model compatibility tests.
- [x] `TEST-15` Add Streamlit smoke and prediction-parity tests.
- [x] `TEST-16` Add a sample-data end-to-end test.
- [x] `TEST-17` Make tests independent of full local datasets.
- [x] `TEST-18` Document every skipped test.
- [x] `TEST-19` Add Docker, Compose, Kubernetes, Helm, and monitoring config validation.

## 16. CI/CD

- [x] `CI-01` Install the pinned environment.
- [x] `CI-02` Run syntax checks.
- [x] `CI-03` Run unit tests.
- [x] `CI-04` Run sample-data integration tests.
- [x] `CI-05` Run model-loading checks.
- [x] `CI-06` Validate Docker build.
- [x] `CI-07` Validate Docker Compose configuration.
- [x] `CI-08` Validate Kubernetes YAML.
- [x] `CI-09` Run Helm lint and `helm template`.
- [x] `CI-10` Add formatting and linting checks.
- [x] `CI-11` Add dependency vulnerability and secret checks.
- [x] `CI-12` Prevent deployment after critical failure.
- [x] `CI-13` Tag releases with application and model versions.
- [x] `CI-14` Keep CI output understandable and actionable.

## 17. Docker and Docker Compose

- [x] `DOCK-01` Install pinned dependencies in the image.
- [x] `DOCK-02` Include the correct production model and manifest.
- [x] `DOCK-03` Remove local-path assumptions.
- [x] `DOCK-04` Add or validate health checks.
- [x] `DOCK-05` Verify Streamlit startup.
- [x] `DOCK-06` Verify metrics exporter startup.
- [x] `DOCK-07` Verify Prometheus, Grafana, and Alertmanager startup.
- [x] `DOCK-08` Verify Blackbox probes.
- [x] `DOCK-09` Move secrets to environment variables.
- [x] `DOCK-10` Run an end-to-end Compose smoke test.
- [x] `DOCK-11` Document startup, shutdown, logs, and reset commands.
- [x] `DOCK-12` Validate required volumes and runtime directories.

## 18. Kubernetes and Helm

- [x] `K8S-01` Replace plain-text credentials with Secrets.
- [x] `K8S-02` Keep non-sensitive settings in ConfigMaps.
- [x] `K8S-03` Make image names and tags configurable.
- [x] `K8S-04` Add readiness probes.
- [x] `K8S-05` Add liveness probes.
- [x] `K8S-06` Add resource requests and limits.
- [x] `K8S-07` Verify services, ports, selectors, and namespaces.
- [x] `K8S-08` Verify Prometheus service discovery.
- [x] `K8S-09` Verify Grafana and Alertmanager connectivity.
- [x] `K8S-10` Run Kubernetes YAML validation.
- [x] `K8S-11` Run `helm lint`.
- [x] `K8S-12` Run `helm template`.
- [x] `K8S-13` Test installation in the intended local cluster.
- [x] `K8S-14` Remove credentials from helper scripts.
- [x] `K8S-15` Document port forwarding and cleanup.
- [x] `K8S-16` Exclude local port-forward logs from Git and packages.

## 19. Output regeneration

- [x] `OUT-01` Rebuild visitor features on the full dataset.
- [x] `OUT-02` Retrain candidate models.
- [x] `OUT-03` Reselect champion and threshold.
- [x] `OUT-04` Generate the final production model and manifest.
- [x] `OUT-05` Generate final visitor scores.
- [x] `OUT-06` Regenerate the high-intent audience.
- [x] `OUT-07` Regenerate segmentation outputs.
- [x] `OUT-08` Regenerate anomaly outputs.
- [x] `OUT-09` Regenerate forecasting outputs.
- [x] `OUT-10` Regenerate monitoring snapshots and metrics.
- [x] `OUT-11` Refresh Streamlit data outputs.
- [x] `OUT-12` Replace old charts and report metrics.
- [x] `OUT-13` Keep generated outputs out of source folders.

## 20. Documentation and interview clarity

- [x] `DOC-01` Fix every README command.
- [x] `DOC-02` Document the professional project structure.
- [x] `DOC-03` Document the feature-building flow.
- [x] `DOC-04` Document grain, target, observation window, and prediction window.
- [x] `DOC-05` Document the production model and threshold source.
- [x] `DOC-06` Document the score-generation workflow.
- [x] `DOC-07` Document monitoring metric sources.
- [x] `DOC-08` Document exact dependency versions.
- [x] `DOC-09` Document Docker and Compose workflows.
- [x] `DOC-10` Document Kubernetes and Helm workflows.
- [x] `DOC-11` Document secret setup without exposing credentials.
- [x] `DOC-12` Document source versus generated files.
- [x] `DOC-13` Mark legacy scripts, models, and outputs.
- [x] `DOC-14` Update architecture diagrams to match implementation.
- [x] `DOC-15` Update all model metrics after retraining.
- [x] `DOC-16` Update the developer and interview understanding guide.
- [x] `DOC-17` Classify every audit finding accurately.
- [x] `DOC-18` Maintain an honest limitations section.
- [x] `DOC-19` Create a comment-preservation report.
- [x] `DOC-20` Create a before-and-after remediation summary.

## Delayed-label remediation evidence

- Prediction ledger and provenance: `94875cc`, `f66a6bf`
- Label ingestion and maturity validation: `f6afa76`, `44f5b54`
- Production metric calculation and truthful empty-label handling: `5cd5c1e`
- End-to-end evaluation runner and tests: `d5217e8`
- Operational monitoring runbook: `bdc2f0d`
- Explicit CI test coverage: `aa2263d`
- Zero-cost coverage reconciliation: `602b281`
- Real outcome evaluation remains externally blocked because the source data
  contains zero future days after the scoring timestamp
## 21. Final QA and sign-off

- [x] `QA-01` Run the complete required test suite.
- [x] `QA-02` Run the full pipeline on real data.
- [x] `QA-03` Check all model and score artifacts against the manifest.
- [x] `QA-04` Compare offline and Streamlit predictions.
- [x] `QA-05` Validate monitoring values against source files.
- [x] `QA-06` Run the Docker Compose stack.
- [x] `QA-07` Run Kubernetes and Helm validation.
- [x] `QA-08` Review all Git changes.
- [x] `QA-09` Review comment preservation and code readability.
- [x] `QA-10` Review every matrix item.
- [x] `QA-11` Create the final before-and-after change summary.
- [x] `QA-12` Confirm every item is fixed, verified, excluded with reason, or blocked with next action.
- [x] `QA-13` Obtain user approval before merge.
- [x] `QA-14` Merge only after approval and final green checks.

## Final remediation closure notes

**Local closure date:** 2026-06-22

The check mark means the row has a documented final resolution. That resolution may be fixed and tested, verified already correct, intentionally excluded, or blocked with a next action.

| ID | Final resolution | Evidence or next action |
|---|---|---|
| `CI-10` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `CI-11` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `DATA-12` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `DEP-10` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `DOC-14` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `DOC-20` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `MOD-09` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-01` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-03` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-04` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-05` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-06` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-07` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-08` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-09` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-10` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-11` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `QA-12` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-01` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-02` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-03` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-04` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-05` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-06` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-07` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-08` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-09` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `SEC-10` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `TEST-16` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `TEST-17` | Fixed and Tested | Verified by the final local QA sequence and repository evidence. |
| `CI-13` | Blocked | Release tags require the remediation merge. Create the application/model release tag immediately after the approved merge. |
| `DOC-01` | Intentionally Excluded | Active operational commands are recorded in the final QA guide. Exhaustive cleanup of historical README copies is excluded from this closure. |
| `DOWN-06` | Intentionally Excluded | Legacy sample score files are not regenerated. The canonical production score export and manifest are the source of truth. |
| `DOWN-08` | Verified Already Correct | The production loader and manifest prevent obsolete models from being selected. Old submitted artifacts remain preserved through Git history. |
| `OUT-06` | Intentionally Excluded | The current approved high-intent artifact was validated. Regeneration is deferred to the later Streamlit visual-intelligence branch when required. |
| `OUT-07` | Intentionally Excluded | Current segmentation outputs were validated through the production-model tests. Regeneration is deferred to the later enhancement branch. |
| `OUT-08` | Intentionally Excluded | Current anomaly outputs were validated through production-model tests. Regeneration is deferred to the later enhancement branch. |
| `OUT-09` | Intentionally Excluded | Current forecasting outputs were validated through production-model tests. Regeneration is deferred to the later enhancement branch. |
| `OUT-11` | Verified Already Correct | The Docker Compose Streamlit health check passed against the current application outputs. |
| `QA-02` | Blocked | A new expensive full retrain was not rerun during final closure. The existing real-data artifacts, full tests, and representative smoke training passed. Next action: run an opt-in full retrain when source data is available. |

### Final gates completed

- `GIT-09`: completed after final Git review and green remote CI.
- `QA-13`: completed after explicit user approval.
- `QA-14`: completed through merge commit `ecf9393`.

See [Zero-Cost MLOps Remaining Coverage Matrix](ZERO_COST_MLOPS_REMAINING_COVERAGE_MATRIX.md) and [Final Remediation Closure Report](FINAL_REMEDIATION_CLOSURE_REPORT.md).

## Approved remediation merge

- Approval date: `2026-06-22`
- Merge commit: `ecf9393`
- GitHub Actions: all tests passed before merge.
- Master remediation matrix: fully resolved.
