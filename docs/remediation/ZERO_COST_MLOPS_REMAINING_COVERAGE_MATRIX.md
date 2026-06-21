# Zero-Cost MLOps Remaining Coverage Matrix

## Project

**E-Commerce Conversion Intelligence Platform**

## Purpose

This matrix controls the remaining technical and deployment work required to complete the project using free and open-source tools.

It covers:

- MLflow experiment tracking
- Evidently monitoring reports
- delayed production labels
- full-retraining and validation workflows
- free Streamlit deployment
- free Grafana monitoring
- local Kubernetes and Helm
- security, documentation, testing, and final merge

The later Streamlit visual-intelligence enhancement will use a separate matrix.

---

## Status definitions

| Status | Meaning |
|---|---|
| `COMPLETED` | Implemented, tested, and evidence recorded |
| `IN PROGRESS` | Active work has started |
| `NOT STARTED` | Required work has not begun |
| `VERIFIED` | Existing implementation checked and accepted |
| `DEFERRED` | Intentionally scheduled for a later controlled phase |
| `OPTIONAL` | Useful enhancement but not required for remediation closure |
| `BLOCKED` | External dependency or constraint exists; next action documented |
| `EXCLUDED` | Deliberately outside scope, with an honest reason |

---

# 1. Governance and scope

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `GOV-01` | Preserve submitted baseline tag | Remote tag exists and points to submitted commit | `COMPLETED` |
| `GOV-02` | Keep remediation in dedicated branch | `fix/full-project-remediation` exists locally and remotely | `COMPLETED` |
| `GOV-03` | Use this matrix as controlling checklist | File is stored under `docs/remediation/` and used to control closure | `COMPLETED` |
| `GOV-04` | Keep Streamlit visual expansion separate | Separate branch and matrix created after remediation merge | `DEFERRED` |
| `GOV-05` | Require evidence before marking an item complete | Test, file, command output, screenshot, or documented exclusion | `IN PROGRESS` |
| `GOV-06` | Do not claim managed cloud production | Documentation clearly distinguishes local validation from managed cloud | `IN PROGRESS` |
| `GOV-07` | Maintain zero paid-software cost | All required tools are open source or use a free tier | `IN PROGRESS` |
| `GOV-08` | Record external constraints honestly | Cloud billing, future labels, resource limits, and data size documented | `NOT STARTED` |

---

# 2. MLflow experiment tracking

## Goal

Add practical local MLflow tracking without requiring Databricks or paid hosting.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `MLF-01` | Add MLflow as a pinned dependency | `requirements-mlflow.txt` contains compatible pinned versions | `COMPLETED` |
| `MLF-02` | Configure local SQLite backend | Local server uses `sqlite:///mlflow.db` | `COMPLETED` |
| `MLF-03` | Configure local artifact storage | Artifacts are written under `mlflow_artifacts/` | `COMPLETED` |
| `MLF-04` | Add MLflow paths to `.gitignore` | Database, environment, PID, and artifact runtime paths are ignored | `COMPLETED` |
| `MLF-05` | Create experiment naming convention | Experiment name is `ecommerce-conversion-intelligence` | `COMPLETED` |
| `MLF-06` | Log model parameters | Champion run contains model and project parameters | `COMPLETED` |
| `MLF-07` | Log validation metrics | Validation metrics are logged with `validation_` prefixes | `COMPLETED` |
| `MLF-08` | Log holdout metrics separately | Untouched holdout metrics are logged with `holdout_` prefixes | `COMPLETED` |
| `MLF-09` | Log selected threshold | Champion threshold is logged as a run parameter | `COMPLETED` |
| `MLF-10` | Log feature schema | Seven ordered production features are stored in the model signature | `COMPLETED` |
| `MLF-11` | Log environment provenance | Original metadata and environment provenance are attached to the run | `COMPLETED` |
| `MLF-12` | Log model signature and input example | Registered model contains a validated signature and real input example | `COMPLETED` |
| `MLF-13` | Log reports and charts | Evaluation CSV evidence is logged; chart artifacts remain to be added | `IN PROGRESS` |
| `MLF-14` | Log trained champion artifact | Real XGBoost champion is logged and registered | `COMPLETED` |
| `MLF-15` | Record champion run lineage | Generated lineage file stores run ID, version, alias, and model URI without changing hashed production metadata | `COMPLETED` |
| `MLF-16` | Add local MLflow UI startup command | Start and stop scripts manage the local server on port 5000 | `COMPLETED` |
| `MLF-17` | Add optional Docker Compose MLflow service | Service has not yet been added to Compose | `NOT STARTED` |
| `MLF-18` | Test tracking integration | Bridge tests and end-to-end registry validator pass | `COMPLETED` |
| `MLF-19` | Verify no production scoring depends on MLflow availability | Tracking is opt-in and failures are non-fatal | `COMPLETED` |
| `MLF-20` | Document MLflow limitations | Local tracking and registry limitations still require a runbook | `NOT STARTED` |

---

# 3. Evidently monitoring reports

## Goal

Generate practical local data-quality, feature-drift, and prediction-drift reports.

### Implementation evidence

- Evidently implementation commit: `f25e482`
- Focused Evidently validation completed successfully.
- Full project tests remained green after the implementation.
- Dependency compatibility was checked with `pip check`.
- These rows were reconciled because the original matrix-update command failed before saving the status changes.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `EVD-01` | Add Evidently as a pinned dependency | Compatible version recorded in dependency files | `COMPLETED` |
| `EVD-02` | Define reference dataset | Training or validated historical reference cohort documented | `COMPLETED` |
| `EVD-03` | Define current dataset | Latest production scoring cohort documented | `COMPLETED` |
| `EVD-04` | Enforce matching feature schema | Reference and current data use canonical model features | `COMPLETED` |
| `EVD-05` | Generate data-quality report | HTML and machine-readable report generated | `COMPLETED` |
| `EVD-06` | Generate feature-drift report | Per-feature drift results generated | `COMPLETED` |
| `EVD-07` | Generate prediction-drift report | Score-distribution drift generated | `COMPLETED` |
| `EVD-08` | Handle absent outcome labels | Report runs without pretending current targets exist | `COMPLETED` |
| `EVD-09` | Generate report summary table | Compact CSV or JSON summary produced for Streamlit and monitoring | `COMPLETED` |
| `EVD-10` | Save reports under controlled paths | Reports stored in `monitoring/evidently/reports/` | `COMPLETED` |
| `EVD-11` | Add report timestamps and provenance | Data hashes, model version, period, and creation timestamp stored | `COMPLETED` |
| `EVD-12` | Integrate with monitoring refresh | Monitoring build process can regenerate Evidently outputs | `COMPLETED` |
| `EVD-13` | Add Streamlit Evidently summary | Monitoring page links to or summarises current reports | `DEFERRED` |
| `EVD-14` | Add drift threshold interpretation | Stable, warning, and critical states documented | `IN PROGRESS` |
| `EVD-15` | Add tests for report creation | Tests verify expected files and metric structure | `COMPLETED` |
| `EVD-16` | Test missing/extra columns | Clear failure or controlled alignment behaviour verified | `COMPLETED` |
| `EVD-17` | Test empty and small current cohorts | Report generation handles invalid cohorts honestly | `COMPLETED` |
| `EVD-18` | Document Evidently limitations | Statistical drift is not described as causal model failure | `NOT STARTED` |

---

# 4. Delayed-label monitoring workflow

## Goal

Prepare the project to evaluate production predictions once future conversion outcomes become available.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `LBL-01` | Define prediction ledger schema | Visitor, score, model, threshold, timestamp, and outcome-window end recorded | `NOT STARTED` |
| `LBL-02` | Generate stable prediction ID | Every scored record receives a reproducible unique identifier | `NOT STARTED` |
| `LBL-03` | Store model and data provenance | Model hash, metadata hash, feature schema, and score source recorded | `NOT STARTED` |
| `LBL-04` | Define delayed-label input schema | Visitor/prediction ID, label, event timestamp, and source documented | `NOT STARTED` |
| `LBL-05` | Implement label ingestion | Valid label file can be loaded and validated | `NOT STARTED` |
| `LBL-06` | Join labels to predictions safely | One-to-one or documented join logic prevents duplication | `NOT STARTED` |
| `LBL-07` | Reject premature labels | Only matured prediction windows enter production evaluation | `NOT STARTED` |
| `LBL-08` | Handle late-arriving labels | Reprocessing logic updates matured cohorts safely | `NOT STARTED` |
| `LBL-09` | Handle duplicate and conflicting labels | Validation rules and error reporting implemented | `NOT STARTED` |
| `LBL-10` | Calculate production metrics | PR-AUC, precision, recall, F1, calibration, and volume calculated when valid | `NOT STARTED` |
| `LBL-11` | Track performance by model version | Metrics grouped by production model/run ID | `NOT STARTED` |
| `LBL-12` | Export production-performance snapshot | CSV or JSON output generated for monitoring | `NOT STARTED` |
| `LBL-13` | Hide metrics when labels are unavailable | Existing truthfulness behaviour remains enforced | `COMPLETED` |
| `LBL-14` | Add delayed-label tests | Mature, premature, duplicate, missing, and conflicting cases tested | `NOT STARTED` |
| `LBL-15` | Document operational process | Score → wait → ingest → join → evaluate workflow documented | `NOT STARTED` |

---

# 5. Full retraining and CI workflows

## Goal

Keep normal CI fast while providing explicit full-pipeline validation.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `CI-01` | Keep fast test suite on every push and PR | Existing GitHub Actions workflow runs automated checks | `COMPLETED` |
| `CI-02` | Keep full retraining opt-in | Expensive test remains disabled during routine tests | `COMPLETED` |
| `CI-03` | Add manual `workflow_dispatch` pipeline | GitHub Actions manual trigger exists | `NOT STARTED` |
| `CI-04` | Add smoke-retraining mode | Small or generated data can validate the training workflow | `NOT STARTED` |
| `CI-05` | Evaluate full-data GitHub feasibility | Runtime, storage, download, and memory requirements documented | `NOT STARTED` |
| `CI-06` | Avoid uploading full private/local dataset | Workflow obtains data safely or uses smoke mode | `NOT STARTED` |
| `CI-07` | Add scheduled compact validation | Optional schedule validates artifacts and reports without full retraining | `NOT STARTED` |
| `CI-08` | Add scheduled Evidently generation where feasible | Compact reference/current artifacts used | `NOT STARTED` |
| `CI-09` | Add MLflow integration test to CI | Temporary local backend is tested | `NOT STARTED` |
| `CI-10` | Add Evidently generation test to CI | Report creation test runs in CI | `NOT STARTED` |
| `CI-11` | Add delayed-label tests to CI | Production label workflow tests run | `NOT STARTED` |
| `CI-12` | Preserve Docker, Helm, and Kubernetes checks | Existing configuration checks remain green | `COMPLETED` |
| `CI-13` | Upload compact test artifacts on failure | Logs and selected reports available for debugging | `NOT STARTED` |
| `CI-14` | Set artifact-retention limits | CI storage kept within free usage limits | `NOT STARTED` |
| `CI-15` | Document why full retraining is not run per commit | README and developer guide explain cost and runtime trade-off | `NOT STARTED` |

---

# 6. Free Streamlit deployment readiness

## Goal

Publish the enhanced business application without processing the complete local dataset in the hosted environment.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `STDEP-01` | Keep deployment after technical remediation merge | Deployment work starts from clean main branch | `DEFERRED` |
| `STDEP-02` | Keep visual enhancement in separate branch | Dedicated branch created | `DEFERRED` |
| `STDEP-03` | Create compact deployment artifact set | Only required models, tables, and visual outputs included | `NOT STARTED` |
| `STDEP-04` | Exclude 1.7 GB raw/processed dataset | Hosted app does not require full local data | `NOT STARTED` |
| `STDEP-05` | Precompute PCA, UMAP, SHAP, KNN, forecast, and drift outputs | App loads compact artifacts | `DEFERRED` |
| `STDEP-06` | Validate repository and artifact sizes | Git and hosting limits checked | `NOT STARTED` |
| `STDEP-07` | Configure Streamlit secrets safely | No secrets committed to Git | `NOT STARTED` |
| `STDEP-08` | Add hosted-environment dependency file | Free deployment installs compatible pinned packages | `NOT STARTED` |
| `STDEP-09` | Add startup health checks | App fails clearly when critical artifacts are missing | `NOT STARTED` |
| `STDEP-10` | Test app with production-like clean environment | Fresh environment launch succeeds | `NOT STARTED` |
| `STDEP-11` | Deploy to Streamlit Community Cloud | Public app URL works | `DEFERRED` |
| `STDEP-12` | Validate all pages publicly | Navigation, charts, filters, and downloads checked | `DEFERRED` |
| `STDEP-13` | Record resource limitations | Memory, cold starts, and precomputation documented | `NOT STARTED` |
| `STDEP-14` | Avoid exposing private or sensitive data | Deployment sample reviewed and approved | `NOT STARTED` |

---

# 7. Free Grafana monitoring publication

## Goal

Keep the full local monitoring stack while optionally publishing compact metrics through a free Grafana tier.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `GRA-01` | Preserve local Grafana stack | Local Grafana already validated | `COMPLETED` |
| `GRA-02` | Preserve local Prometheus and Alertmanager | Runtime health and alert flow already validated | `COMPLETED` |
| `GRA-03` | Decide whether public Grafana is required | User approval recorded | `NOT STARTED` |
| `GRA-04` | Create free Grafana account if selected | No paid plan or credit card requirement used | `DEFERRED` |
| `GRA-05` | Define secure metric-publishing method | No local credentials or services exposed unsafely | `NOT STARTED` |
| `GRA-06` | Export only compact monitoring metrics | Snapshot-based metrics remain within free allowance | `NOT STARTED` |
| `GRA-07` | Secure API keys as secrets | Keys stored outside Git | `NOT STARTED` |
| `GRA-08` | Import or recreate dashboard | Dashboard panels display expected metrics | `DEFERRED` |
| `GRA-09` | Preserve label-availability truthfulness | Conversion metrics hidden when labels do not exist | `COMPLETED` |
| `GRA-10` | Verify dashboard freshness | Timestamp and snapshot age displayed | `NOT STARTED` |
| `GRA-11` | Configure free alert delivery where feasible | Email or supported free contact point tested | `DEFERRED` |
| `GRA-12` | Document public-versus-local differences | Recruiter view and local engineering view distinguished | `NOT STARTED` |

---

# 8. Kubernetes and Helm at zero cost

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `K8S-01` | Maintain local Kubernetes deployment | Seven deployments and services previously validated | `COMPLETED` |
| `K8S-02` | Maintain Helm chart validation | Lint, render, dry-run, and runtime checks passed | `COMPLETED` |
| `K8S-03` | Keep secret-based Grafana credentials | Secret workflow tested | `COMPLETED` |
| `K8S-04` | Keep alerting resources in Kubernetes | Alertmanager and webhook resources exist | `COMPLETED` |
| `K8S-05` | Do not require paid managed Kubernetes | Local cluster remains primary demonstration | `COMPLETED` |
| `K8S-06` | Document optional free-hosting experiments separately | Oracle/k3s or similar is not required for closure | `NOT STARTED` |
| `K8S-07` | Avoid claiming cloud production deployment | Documentation reviewed | `IN PROGRESS` |
| `K8S-08` | Re-run Helm/Kubernetes validation after new services | MLflow/Evidently additions do not break deployment | `NOT STARTED` |

---

# 9. Security and repository hygiene

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `SEC-01` | Ignore MLflow database and runtime artifacts | MLflow database, artifacts, environment, PID, and logs are protected | `COMPLETED` |
| `SEC-02` | Ignore generated Evidently reports where appropriate | Runtime/large reports excluded; compact examples controlled | `NOT STARTED` |
| `SEC-03` | Keep `.env` excluded | Only `.env.example` committed | `COMPLETED` |
| `SEC-04` | Keep Grafana credentials external | Environment variables and Kubernetes Secrets used | `COMPLETED` |
| `SEC-05` | Protect GitHub and cloud tokens | Repository secrets used | `NOT STARTED` |
| `SEC-06` | Run secret-pattern scan | No committed credentials or private keys found | `NOT STARTED` |
| `SEC-07` | Review generated reports for sensitive visitor data | Public artifacts anonymised or sampled | `NOT STARTED` |
| `SEC-08` | Pin all newly implemented dependencies | MLflow, Evidently, and supporting libraries reproducible | `NOT STARTED` |
| `SEC-09` | Run `pip check` after dependency changes | No incompatible dependency versions | `NOT STARTED` |
| `SEC-10` | Rebuild Docker images after dependency changes | Clean image build and health checks pass | `NOT STARTED` |

---

# 10. Documentation and operational guides

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `DOC-01` | Update main README with MLflow | Architecture, startup, outputs, and limitations documented | `NOT STARTED` |
| `DOC-02` | Update main README with Evidently | Report generation and interpretation documented | `NOT STARTED` |
| `DOC-03` | Add MLflow runbook | Start, stop, inspect, reset, and troubleshoot steps | `NOT STARTED` |
| `DOC-04` | Add Evidently runbook | Generate, locate, interpret, and troubleshoot reports | `NOT STARTED` |
| `DOC-05` | Add delayed-label runbook | Ingestion and matured-cohort process documented | `NOT STARTED` |
| `DOC-06` | Add free-deployment guide | Streamlit and optional Grafana steps documented | `DEFERRED` |
| `DOC-07` | Update architecture diagrams | MLflow, Evidently, and delayed labels included | `NOT STARTED` |
| `DOC-08` | Update limitations section | Real limitations separated from unfinished requirements | `NOT STARTED` |
| `DOC-09` | Create comment-preservation report | Final report committed | `NOT STARTED` |
| `DOC-10` | Create before/after remediation summary | Final report committed | `NOT STARTED` |
| `DOC-11` | Update original remediation matrix | All original findings receive final resolution | `NOT STARTED` |
| `DOC-12` | Link this matrix from original matrix | Cross-reference exists | `NOT STARTED` |
| `DOC-13` | Add interviewer explanation | Simple architecture and business story provided | `NOT STARTED` |
| `DOC-14` | Record exact validation commands | Reproducible final QA checklist documented | `NOT STARTED` |

---

# 11. Final testing and remediation closure

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `QA-01` | MLflow focused tests pass | Bridge tests and end-to-end champion validation passed | `COMPLETED` |
| `QA-02` | Evidently focused tests pass | Quality and drift reports verified | `NOT STARTED` |
| `QA-03` | Delayed-label focused tests pass | Matured-cohort evaluation verified | `NOT STARTED` |
| `QA-04` | Full normal pytest suite passes | Only approved opt-in tests skipped | `NOT STARTED` |
| `QA-05` | Full retraining smoke test passes | Explicit full-pipeline command succeeds | `COMPLETED` |
| `QA-06` | Docker Compose builds and runs | All required services healthy | `NOT STARTED` |
| `QA-07` | MLflow service health verified | Local health endpoint, SQLite backend, artifact storage, registry, and alias were verified | `COMPLETED` |
| `QA-08` | Evidently generation works in clean environment | Reports generated from canonical artifacts | `NOT STARTED` |
| `QA-09` | Monitoring snapshot remains truthful | No fabricated outcome metrics | `COMPLETED` |
| `QA-10` | Helm lint and template pass | Chart validates after changes | `NOT STARTED` |
| `QA-11` | Kubernetes client dry-run passes | Rendered resources accepted | `NOT STARTED` |
| `QA-12` | Local Kubernetes runtime passes where services are added | Pods, services, and health endpoints verified | `NOT STARTED` |
| `QA-13` | CI workflow passes remotely | Latest remediation-branch run is green | `NOT STARTED` |
| `QA-14` | `git diff --check` passes | No whitespace errors | `NOT STARTED` |
| `QA-15` | Secret and large-file audit passes | No accidental credentials or oversized artifacts | `NOT STARTED` |
| `QA-16` | Baseline-to-remediation diff reviewed | No accidental deletions or unrelated files | `NOT STARTED` |
| `QA-17` | Every matrix item resolved | No unexplained `NOT STARTED` or `IN PROGRESS` items | `NOT STARTED` |
| `QA-18` | Final documentation commit pushed | Remote branch synchronized | `NOT STARTED` |
| `QA-19` | User reviews final remediation result | Explicit approval recorded | `NOT STARTED` |
| `QA-20` | Remediation branch merged | Merge occurs only after approval and green checks | `NOT STARTED` |
| `QA-21` | Optional release tag created | Tag created after successful merge | `OPTIONAL` |

---

# 12. Explicit exclusions and external constraints

| ID | Item | Resolution | Status |
|---|---|---|---|
| `EXT-01` | Managed paid Kubernetes platform | Excluded from zero-cost remediation; local Kubernetes is validated | `EXCLUDED` |
| `EXT-02` | Paid Databricks or managed MLflow | Excluded; local open-source MLflow will be used | `EXCLUDED` |
| `EXT-03` | Paid Evidently Cloud | Excluded; open-source Evidently reports will be generated locally | `EXCLUDED` |
| `EXT-04` | Permanent paid production database | Excluded; SQLite/local artifacts are sufficient for portfolio scope | `EXCLUDED` |
| `EXT-05` | Live Kafka/Redpanda event stream | Deferred architecture enhancement, not required for current batch project | `DEFERRED` |
| `EXT-06` | Immediate production outcome labels | Externally unavailable until the future outcome window matures | `BLOCKED` |
| `EXT-07` | Full 1.7 GB data hosted in Streamlit | Excluded; compact precomputed artifacts will be deployed | `EXCLUDED` |
| `EXT-08` | Guaranteed always-on free cloud infrastructure | Not promised because free-tier policies and capacity may change | `EXCLUDED` |

---

# 13. Closure rules

This matrix may be closed only when:

1. MLflow is practically implemented and tested.
2. Evidently is practically implemented and tested.
3. Delayed-label readiness is implemented and tested.
4. Fast CI and opt-in full retraining are documented and validated.
5. Newly added dependencies are pinned and reproducible.
6. Docker, Helm, Kubernetes, monitoring, and Streamlit regressions pass.
7. The original remediation matrix is fully reconciled.
8. Documentation accurately distinguishes:
   - historical labeled evaluation,
   - current unlabeled scoring,
   - snapshot-based monitoring,
   - local deployment,
   - public free-tier deployment,
   - managed production infrastructure.
9. Every row is marked completed, verified, deferred, blocked, excluded, or optional with a reason.
10. The user approves the final remediation before merge.

---

## Immediate execution order

1. Add this matrix to the repository.
2. Implement MLflow tracking.
3. Implement Evidently reports.
4. Implement delayed-label readiness.
5. Add manual and compact scheduled CI workflows.
6. Re-run dependency, Docker, Helm, Kubernetes, and test validation.
7. Finish documentation and reconcile both remediation matrices.
8. Push and confirm remote CI.
9. Obtain user approval.
10. Merge the remediation branch.
11. Start the separate Streamlit visual-intelligence enhancement branch.
