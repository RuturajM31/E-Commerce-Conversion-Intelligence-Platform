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
- [ ] `GIT-06` Use small, meaningful commits by remediation phase.
- [ ] `GIT-07` Review `git diff` before every commit.
- [ ] `GIT-08` Keep `main` unchanged until final approval.
- [ ] `GIT-09` Merge only after every matrix item is resolved.

## 2. Project structure and code quality

- [ ] `STR-01` Keep the existing root folder name.
- [ ] `STR-02` Create only folders and files with a clear purpose.
- [ ] `STR-03` Avoid one giant file containing unrelated logic.
- [ ] `STR-04` Avoid excessive fragmentation across many tiny files.
- [ ] `STR-05` Keep clear orchestration flows for data, training, scoring, and monitoring.
- [ ] `STR-06` Separate source code, data, models, outputs, reports, logs, and monitoring assets.
- [ ] `STR-07` Archive obsolete scripts, models, and outputs clearly.
- [ ] `STR-08` Use relative paths and central configuration.
- [ ] `STR-09` Exclude caches and transient logs from Git and release packages.
- [ ] `STR-10` Keep source files separate from generated artifacts.

## 3. Comment and learning-content preservation

- [ ] `COM-01` Preserve useful line comments.
- [ ] `COM-02` Preserve file headers and docstrings.
- [ ] `COM-03` Preserve variable-flow explanations.
- [ ] `COM-04` Preserve business-purpose explanations.
- [ ] `COM-05` Preserve learning notes and beginner-friendly explanations.
- [ ] `COM-06` Update comments when underlying code changes.
- [ ] `COM-07` Move explanations with refactored logic instead of deleting them.
- [ ] `COM-08` Avoid comments that merely repeat obvious syntax.
- [ ] `COM-09` Keep code blocks short, readable, and logically grouped.
- [ ] `COM-10` Use simple variable, function, and file names.
- [ ] `COM-11` Avoid clever or deeply nested Python when simpler logic is available.
- [ ] `COM-12` Explain file inputs, outputs, next use, and business value.

## 4. Data and feature-building pipeline

- [x] `DATA-01` Restore the `src/data` package.
- [x] `DATA-02` Add `src/data/__init__.py`.
- [ ] `DATA-03` Restore or rebuild `src/data/build_visitor_features.py`.
- [ ] `DATA-04` Rebuild visitor features from RetailRocket raw files.
- [ ] `DATA-05` Define the modelling grain explicitly.
- [ ] `DATA-06` Validate required raw input columns.
- [ ] `DATA-07` Validate event types and malformed records.
- [ ] `DATA-08` Validate duplicates and missing values.
- [ ] `DATA-09` Log input rows, output rows, visitor counts, and target distribution.
- [ ] `DATA-10` Log processing duration and clear failures.
- [ ] `DATA-11` Use central path configuration.
- [ ] `DATA-12` Keep representative sample data for tests and CI.
- [ ] `DATA-13` Document all raw source files used.
- [ ] `DATA-14` Add safe overwrite and output-directory handling.

## 5. Centralised feature engineering

- [x] `FEAT-01` Create one shared feature-engineering module.
- [x] `FEAT-02` Use it in dataset generation.
- [x] `FEAT-03` Use canonical features in training.
- [x] `FEAT-04` Use it in Streamlit single prediction.
- [x] `FEAT-05` Use it in batch scoring.
- [ ] `FEAT-06` Use canonical features or scores in segmentation.
- [ ] `FEAT-07` Use canonical features or scores in anomaly workflows.
- [ ] `FEAT-08` Use canonical production outputs in forecasting.
- [x] `FEAT-09` Standardise `cart_to_view_ratio`.
- [x] `FEAT-10` Standardise `events_per_unique_item`.
- [ ] `FEAT-11` Standardise recency, frequency, diversity, and activity-duration features.
- [x] `FEAT-12` Define explicit zero-denominator handling.
- [x] `FEAT-13` Store approved feature names and order.
- [x] `FEAT-14` Reject missing required features safely.
- [ ] `FEAT-15` Handle unexpected extra features clearly.
- [x] `FEAT-16` Add training-versus-serving feature parity tests.
- [x] `FEAT-17` Add feature data-type and range validation.
- [ ] `FEAT-18` Keep boolean and categorical encoding consistent.

## 6. Leakage-safe target and modelling timeline

- [ ] `LEAK-01` Recover and document the current target logic.
- [ ] `LEAK-02` Define a future purchase prediction question.
- [ ] `LEAK-03` Define the observation window.
- [ ] `LEAK-04` Define the later prediction window.
- [ ] `LEAK-05` Prevent post-purchase behaviour entering pre-purchase features.
- [ ] `LEAK-06` Prevent future events entering historical features.
- [ ] `LEAK-07` Define handling for multiple purchases.
- [ ] `LEAK-08` Define handling for repeated scoring dates.
- [ ] `LEAK-09` Store feature cutoff timestamps.
- [ ] `LEAK-10` Rebuild the modelling table with leakage-safe windows.
- [ ] `LEAK-11` Retrain after the leakage-safe rebuild.
- [ ] `LEAK-12` Replace old published metrics after retraining.
- [ ] `LEAK-13` Describe current limitations honestly until the rebuild is complete.

## 7. Model training and evaluation

- [ ] `MOD-01` Separate training, validation, and test data.
- [ ] `MOD-02` Stop selecting models on the final test set.
- [ ] `MOD-03` Stop selecting thresholds on the final test set.
- [ ] `MOD-04` Add a time-based holdout.
- [ ] `MOD-05` Compare random-split and time-split performance.
- [ ] `MOD-06` Keep PR-AUC as the primary rare-event metric.
- [ ] `MOD-07` Report ROC-AUC, precision, recall, F1, and confusion matrix.
- [ ] `MOD-08` Report predicted-positive count and selected audience share.
- [ ] `MOD-09` Check probability calibration.
- [ ] `MOD-10` Report threshold trade-offs.
- [ ] `MOD-11` Reconfirm whether `0.97` remains the correct threshold.
- [ ] `MOD-12` Save validation and test results separately.
- [ ] `MOD-13` Record random seeds and reproducibility settings.
- [ ] `MOD-14` Use stratification where appropriate.
- [ ] `MOD-15` Separate business threshold selection from model-quality evaluation.

## 8. Final champion scoring workflow

- [ ] `SCORE-01` Implement or correctly import `save_final_champion_scores`.
- [ ] `SCORE-02` Ensure champion finalisation completes fully.
- [ ] `SCORE-03` Use one official final score filename.
- [ ] `SCORE-04` Include visitor ID and probability.
- [ ] `SCORE-05` Include predicted class and high-intent flag.
- [ ] `SCORE-06` Include model version and threshold.
- [ ] `SCORE-07` Include scoring timestamp.
- [ ] `SCORE-08` Validate input and output row counts.
- [ ] `SCORE-09` Validate probability range.
- [ ] `SCORE-10` Prevent silent fallback to old score files.
- [ ] `SCORE-11` Add automated score-export tests.
- [ ] `SCORE-12` Add safe output-directory and overwrite rules.

## 9. One production model and manifest

- [ ] `PROD-01` Create one production model manifest.
- [ ] `PROD-02` Store model filename, type, and version.
- [ ] `PROD-03` Store feature names and order.
- [ ] `PROD-04` Store the approved threshold.
- [ ] `PROD-05` Store training date and dataset version.
- [ ] `PROD-06` Store Python and dependency versions.
- [ ] `PROD-07` Store evaluation metrics and model checksum.
- [ ] `PROD-08` Create one shared production model loader.
- [ ] `PROD-09` Add startup compatibility checks.
- [ ] `PROD-10` Prevent accidental loading of archived models.
- [ ] `PROD-11` Validate manifest, model, and feature schema together.

## 10. Downstream model consistency

- [ ] `DOWN-01` Make Streamlit use the production model and manifest.
- [ ] `DOWN-02` Make batch scoring use the same model and threshold.
- [ ] `DOWN-03` Replace segmentation's legacy Logistic Regression dependency.
- [ ] `DOWN-04` Replace anomaly workflow's legacy model and threshold.
- [ ] `DOWN-05` Replace forecasting's legacy artifacts.
- [ ] `DOWN-06` Regenerate legacy sample score files.
- [ ] `DOWN-07` Add model version and generation date to outputs.
- [ ] `DOWN-08` Archive obsolete models and outputs.
- [ ] `DOWN-09` Prevent monitoring from mixing an old model with a new threshold.

## 11. Monitoring corrections

- [ ] `MON-01` Fix the business-score source.
- [ ] `MON-02` Stop reading metrics from files that do not contain them.
- [ ] `MON-03` Fix total-anomaly variable overwrite.
- [ ] `MON-04` Export total and high-intent anomalies separately.
- [ ] `MON-05` Recognise `final_anomaly_flag` explicitly.
- [ ] `MON-06` Remove unreliable fallback column matching.
- [ ] `MON-07` Add model version to monitoring output.
- [ ] `MON-08` Add scoring-data version and timestamp.
- [ ] `MON-09` Prevent fallback to legacy score files.
- [ ] `MON-10` Validate every Prometheus metric against its source.
- [ ] `MON-11` Update Grafana queries after metric changes.
- [ ] `MON-12` Update alert rules after metric changes.
- [ ] `MON-13` Validate Alertmanager routing.
- [ ] `MON-14` Validate Blackbox health checks.
- [ ] `MON-15` Document runtime-generated monitoring files.
- [ ] `MON-16` Add prediction and snapshot freshness timestamps.

## 12. Streamlit application

- [ ] `APP-01` Remove duplicated feature formulas.
- [ ] `APP-02` Use the shared production model loader.
- [ ] `APP-03` Read threshold from the manifest.
- [ ] `APP-04` Display model version and threshold.
- [ ] `APP-05` Validate manual inputs.
- [ ] `APP-06` Handle zeros and missing values consistently.
- [ ] `APP-07` Ensure single and batch predictions match.
- [ ] `APP-08` Show clear errors for missing artifacts.
- [ ] `APP-09` Prevent obsolete artifact loading.
- [ ] `APP-10` Show last-refresh and scoring timestamps.
- [ ] `APP-11` Add Streamlit smoke tests.
- [ ] `APP-12` Preserve dashboard design and business explanations.
- [ ] `APP-13` Keep app services simple and not over-fragmented.

## 13. Dependencies and reproducibility

- [ ] `DEP-01` Pin the supported Python version.
- [ ] `DEP-02` Pin scikit-learn to the approved training version.
- [ ] `DEP-03` Pin pandas, NumPy, joblib, Streamlit, and Plotly.
- [ ] `DEP-04` Pin pytest and all implemented Python dependencies; pin monitoring container versions; document MLflow and Evidently as not implemented.
- [ ] `DEP-05` Align `requirements.txt`, `requirements-app.txt`, and `pyproject.toml`.
- [ ] `DEP-06` Add a lockfile or frozen requirements file.
- [ ] `DEP-07` Store environment versions in model metadata.
- [ ] `DEP-08` Test model loading in the pinned local environment.
- [ ] `DEP-09` Test model loading inside Docker.
- [ ] `DEP-10` Document future dependency-upgrade procedure.

## 14. Secrets and credentials

- [ ] `SEC-01` Remove Grafana password from Docker Compose.
- [ ] `SEC-02` Remove plain-text password from Kubernetes YAML.
- [ ] `SEC-03` Remove plain-text password from Helm values.
- [ ] `SEC-04` Remove passwords from startup scripts.
- [ ] `SEC-05` Remove credentials from backups and documentation.
- [ ] `SEC-06` Keep only `.env.example` in Git.
- [ ] `SEC-07` Use GitHub Actions Secrets where needed.
- [ ] `SEC-08` Review Git history for exposed credentials.
- [ ] `SEC-09` Rotate credentials if public exposure occurred.
- [ ] `SEC-10` Add secret scanning to CI.

## 15. Automated tests

- [x] `TEST-01` Keep all current baseline tests passing.
- [ ] `TEST-02` Add project-structure tests for restored modules.
- [ ] `TEST-03` Add raw and processed data-schema tests.
- [x] `TEST-04` Add feature-formula tests.
- [x] `TEST-05` Add training-serving parity tests.
- [x] `TEST-06` Add feature-order and data-type tests.
- [ ] `TEST-07` Add production-manifest validation tests.
- [ ] `TEST-08` Add one-model and one-threshold consistency tests.
- [ ] `TEST-09` Add score-export tests.
- [ ] `TEST-10` Add leakage and cutoff tests.
- [ ] `TEST-11` Add monitoring metric-value tests.
- [ ] `TEST-12` Add anomaly-count tests.
- [ ] `TEST-13` Add business-score-source tests.
- [ ] `TEST-14` Add model compatibility tests.
- [ ] `TEST-15` Add Streamlit smoke and prediction-parity tests.
- [ ] `TEST-16` Add a sample-data end-to-end test.
- [ ] `TEST-17` Make tests independent of full local datasets.
- [ ] `TEST-18` Document every skipped test.
- [ ] `TEST-19` Add Docker, Compose, Kubernetes, Helm, and monitoring config validation.

## 16. CI/CD

- [ ] `CI-01` Install the pinned environment.
- [ ] `CI-02` Run syntax checks.
- [ ] `CI-03` Run unit tests.
- [ ] `CI-04` Run sample-data integration tests.
- [ ] `CI-05` Run model-loading checks.
- [ ] `CI-06` Validate Docker build.
- [ ] `CI-07` Validate Docker Compose configuration.
- [ ] `CI-08` Validate Kubernetes YAML.
- [ ] `CI-09` Run Helm lint and `helm template`.
- [ ] `CI-10` Add formatting and linting checks.
- [ ] `CI-11` Add dependency vulnerability and secret checks.
- [ ] `CI-12` Prevent deployment after critical failure.
- [ ] `CI-13` Tag releases with application and model versions.
- [ ] `CI-14` Keep CI output understandable and actionable.

## 17. Docker and Docker Compose

- [ ] `DOCK-01` Install pinned dependencies in the image.
- [ ] `DOCK-02` Include the correct production model and manifest.
- [ ] `DOCK-03` Remove local-path assumptions.
- [ ] `DOCK-04` Add or validate health checks.
- [ ] `DOCK-05` Verify Streamlit startup.
- [ ] `DOCK-06` Verify metrics exporter startup.
- [ ] `DOCK-07` Verify Prometheus, Grafana, and Alertmanager startup.
- [ ] `DOCK-08` Verify Blackbox probes.
- [ ] `DOCK-09` Move secrets to environment variables.
- [ ] `DOCK-10` Run an end-to-end Compose smoke test.
- [ ] `DOCK-11` Document startup, shutdown, logs, and reset commands.
- [ ] `DOCK-12` Validate required volumes and runtime directories.

## 18. Kubernetes and Helm

- [ ] `K8S-01` Replace plain-text credentials with Secrets.
- [ ] `K8S-02` Keep non-sensitive settings in ConfigMaps.
- [ ] `K8S-03` Make image names and tags configurable.
- [ ] `K8S-04` Add readiness probes.
- [ ] `K8S-05` Add liveness probes.
- [ ] `K8S-06` Add resource requests and limits.
- [ ] `K8S-07` Verify services, ports, selectors, and namespaces.
- [ ] `K8S-08` Verify Prometheus service discovery.
- [ ] `K8S-09` Verify Grafana and Alertmanager connectivity.
- [ ] `K8S-10` Run Kubernetes YAML validation.
- [ ] `K8S-11` Run `helm lint`.
- [ ] `K8S-12` Run `helm template`.
- [ ] `K8S-13` Test installation in the intended local cluster.
- [ ] `K8S-14` Remove credentials from helper scripts.
- [ ] `K8S-15` Document port forwarding and cleanup.
- [ ] `K8S-16` Exclude local port-forward logs from Git and packages.

## 19. Output regeneration

- [ ] `OUT-01` Rebuild visitor features on the full dataset.
- [ ] `OUT-02` Retrain candidate models.
- [ ] `OUT-03` Reselect champion and threshold.
- [ ] `OUT-04` Generate the final production model and manifest.
- [ ] `OUT-05` Generate final visitor scores.
- [ ] `OUT-06` Regenerate the high-intent audience.
- [ ] `OUT-07` Regenerate segmentation outputs.
- [ ] `OUT-08` Regenerate anomaly outputs.
- [ ] `OUT-09` Regenerate forecasting outputs.
- [ ] `OUT-10` Regenerate monitoring snapshots and metrics.
- [ ] `OUT-11` Refresh Streamlit data outputs.
- [ ] `OUT-12` Replace old charts and report metrics.
- [ ] `OUT-13` Keep generated outputs out of source folders.

## 20. Documentation and interview clarity

- [ ] `DOC-01` Fix every README command.
- [ ] `DOC-02` Document the professional project structure.
- [ ] `DOC-03` Document the feature-building flow.
- [ ] `DOC-04` Document grain, target, observation window, and prediction window.
- [ ] `DOC-05` Document the production model and threshold source.
- [ ] `DOC-06` Document the score-generation workflow.
- [ ] `DOC-07` Document monitoring metric sources.
- [ ] `DOC-08` Document exact dependency versions.
- [ ] `DOC-09` Document Docker and Compose workflows.
- [ ] `DOC-10` Document Kubernetes and Helm workflows.
- [ ] `DOC-11` Document secret setup without exposing credentials.
- [ ] `DOC-12` Document source versus generated files.
- [ ] `DOC-13` Mark legacy scripts, models, and outputs.
- [ ] `DOC-14` Update architecture diagrams to match implementation.
- [ ] `DOC-15` Update all model metrics after retraining.
- [ ] `DOC-16` Update the developer and interview understanding guide.
- [ ] `DOC-17` Classify every audit finding accurately.
- [ ] `DOC-18` Maintain an honest limitations section.
- [ ] `DOC-19` Create a comment-preservation report.
- [ ] `DOC-20` Create a before-and-after remediation summary.

## 21. Final QA and sign-off

- [ ] `QA-01` Run the complete required test suite.
- [ ] `QA-02` Run the full pipeline on real data.
- [ ] `QA-03` Check all model and score artifacts against the manifest.
- [ ] `QA-04` Compare offline and Streamlit predictions.
- [ ] `QA-05` Validate monitoring values against source files.
- [ ] `QA-06` Run the Docker Compose stack.
- [ ] `QA-07` Run Kubernetes and Helm validation.
- [ ] `QA-08` Review all Git changes.
- [ ] `QA-09` Review comment preservation and code readability.
- [ ] `QA-10` Review every matrix item.
- [ ] `QA-11` Create the final before-and-after change summary.
- [ ] `QA-12` Confirm every item is fixed, verified, excluded with reason, or blocked with next action.
- [ ] `QA-13` Obtain user approval before merge.
- [ ] `QA-14` Merge only after approval and final green checks.
