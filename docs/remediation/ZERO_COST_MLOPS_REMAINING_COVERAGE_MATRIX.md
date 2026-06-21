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
| `GOV-05` | Require evidence before marking an item complete | Commit, test, report, and generated-artifact evidence is recorded before closure | `COMPLETED` |
| `GOV-06` | Do not claim managed cloud production | Runbooks and monitoring outputs distinguish local validation from managed production | `COMPLETED` |
| `GOV-07` | Maintain zero paid-software cost | Required remediation tools use local open-source software or GitHub free automation | `COMPLETED` |
| `GOV-08` | Record external constraints honestly | Future labels, large data, free-tier limits, and later cloud work are documented | `COMPLETED` |

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
| `MLF-13` | Log reports and charts | Champion and threshold visual artifacts were logged and verified in MLflow | `COMPLETED` |
| `MLF-14` | Log trained champion artifact | Real XGBoost champion is logged and registered | `COMPLETED` |
| `MLF-15` | Record champion run lineage | Generated lineage file stores run ID, version, alias, and model URI without changing hashed production metadata | `COMPLETED` |
| `MLF-16` | Add local MLflow UI startup command | Start and stop scripts manage the local server on port 5000 | `COMPLETED` |
| `MLF-17` | Add optional Docker Compose MLflow service | Optional Compose service is deferred because local isolated MLflow already satisfies this phase | `DEFERRED` |
| `MLF-18` | Test tracking integration | Bridge tests and end-to-end registry validator pass | `COMPLETED` |
| `MLF-19` | Verify no production scoring depends on MLflow availability | Tracking is opt-in and failures are non-fatal | `COMPLETED` |
| `MLF-20` | Document MLflow limitations | Local MLflow startup, validation, registry, storage, and limitations are documented | `COMPLETED` |

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
| `EVD-01` | Add Evidently as a pinned dependency | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-02` | Define reference dataset | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-03` | Define current dataset | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-04` | Enforce matching feature schema | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-05` | Generate data-quality report | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-06` | Generate feature-drift report | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-07` | Generate prediction-drift report | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-08` | Handle absent outcome labels | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-09` | Generate report summary table | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-10` | Save reports under controlled paths | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-11` | Add report timestamps and provenance | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-12` | Integrate with monitoring refresh | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-13` | Add Streamlit Evidently summary | Streamlit presentation remains scheduled for the dedicated visual-enhancement branch | `DEFERRED` |
| `EVD-14` | Add drift threshold interpretation | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-15` | Add tests for report creation | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-16` | Test missing/extra columns | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-17` | Test empty and small current cohorts | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |
| `EVD-18` | Document Evidently limitations | Implemented and tested in commit f25e482, with later CI validation | `COMPLETED` |

---

# 4. Delayed-label monitoring workflow

## Goal

Prepare the project to evaluate production predictions once future conversion outcomes become available.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `LBL-01` | Define prediction ledger schema | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-02` | Generate stable prediction ID | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-03` | Store model and data provenance | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-04` | Define delayed-label input schema | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-05` | Implement label ingestion | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-06` | Join labels to predictions safely | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-07` | Reject premature labels | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-08` | Handle late-arriving labels | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-09` | Handle duplicate and conflicting labels | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-10` | Calculate production metrics | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-11` | Track performance by model version | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-12` | Export production-performance snapshot | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-13` | Hide metrics when labels are unavailable | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-14` | Add delayed-label tests | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |
| `LBL-15` | Document operational process | Implemented, tested, documented, and connected through the delayed-label runner | `COMPLETED` |

## Delayed-label implementation evidence

- Prediction-ledger schema and provenance: `94875cc`, `f66a6bf`
- Delayed-label input and maturity validation: `f6afa76`, `44f5b54`, `fb3eab5`
- Production metrics and truthful empty-label behaviour: `5cd5c1e`
- End-to-end evaluation runner and focused tests: `d5217e8`
- Operational runbook: `bdc2f0d`
- Explicit delayed-label CI test step: `aa2263d`
- Focused delayed-label suite: 43 tests passed
- Full normal test suite: passed with only the approved opt-in retraining test skipped
- `EXT-06` remains `BLOCKED` because genuine future production outcomes are unavailable
- `DOC-07` remains `NOT STARTED` until architecture diagrams are updated
---

# 5. Full retraining and CI workflows

## Goal

Keep normal CI fast while providing explicit full-pipeline validation.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `CI-01` | Keep fast test suite on every push and PR | Existing GitHub Actions workflow runs automated checks | `COMPLETED` |
| `CI-02` | Keep full retraining opt-in | Expensive test remains disabled during routine tests | `COMPLETED` |
| `CI-03` | Add manual `workflow_dispatch` pipeline | GitHub Actions now includes workflow_dispatch | `COMPLETED` |
| `CI-04` | Add smoke-retraining mode | Small or generated data can validate the training workflow | `COMPLETED` |
| `CI-05` | Evaluate full-data GitHub feasibility | The CI guide records why full-data GitHub retraining is not practical | `COMPLETED` |
| `CI-06` | Avoid uploading full private/local dataset | Normal CI uses repository tests and does not upload the large local dataset | `COMPLETED` |
| `CI-07` | Add scheduled compact validation | GitHub Actions now includes a weekly compact validation schedule | `COMPLETED` |
| `CI-08` | Add scheduled Evidently generation where feasible | Compact reference/current artifacts used | `COMPLETED` |
| `CI-09` | Add MLflow integration test to CI | Manual and scheduled runs execute the isolated MLflow bridge test | `COMPLETED` |
| `CI-10` | Add Evidently generation test to CI | Manual and scheduled runs execute Evidently bridge and drift-summary tests | `COMPLETED` |
| `CI-11` | Add delayed-label tests to CI | Delayed-label tests are included in the normal CI workflow | `COMPLETED` |
| `CI-12` | Preserve Docker, Helm, and Kubernetes checks | Existing configuration checks remain green | `COMPLETED` |
| `CI-13` | Upload compact test artifacts on failure | Failure-only evidence upload is configured | `COMPLETED` |
| `CI-14` | Set artifact-retention limits | Failure evidence is retained for seven days | `COMPLETED` |
| `CI-15` | Document why full retraining is not run per commit | CI_WORKFLOW_GUIDE.md explains the full-retraining cost and runtime trade-off | `COMPLETED` |

---

# 6. Free Streamlit deployment readiness

## Goal

Publish the enhanced business application without processing the complete local dataset in the hosted environment.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `STDEP-01` | Keep deployment after technical remediation merge | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-02` | Keep visual enhancement in separate branch | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-03` | Create compact deployment artifact set | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-04` | Exclude 1.7 GB raw/processed dataset | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-05` | Precompute PCA, UMAP, SHAP, KNN, forecast, and drift outputs | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-06` | Validate repository and artifact sizes | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-07` | Configure Streamlit secrets safely | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-08` | Add hosted-environment dependency file | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-09` | Add startup health checks | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-10` | Test app with production-like clean environment | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-11` | Deploy to Streamlit Community Cloud | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-12` | Validate all pages publicly | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-13` | Record resource limitations | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |
| `STDEP-14` | Avoid exposing private or sensitive data | Deferred to the mandatory post-Streamlit-enhancement cloud deployment phase | `DEFERRED` |

---

# 7. Free Grafana monitoring publication

## Goal

Keep the full local monitoring stack while optionally publishing compact metrics through a free Grafana tier.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `GRA-01` | Preserve local Grafana stack | Local Grafana already validated | `COMPLETED` |
| `GRA-02` | Preserve local Prometheus and Alertmanager | Runtime health and alert flow already validated | `COMPLETED` |
| `GRA-03` | Decide whether public Grafana is required | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-04` | Create free Grafana account if selected | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-05` | Define secure metric-publishing method | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-06` | Export only compact monitoring metrics | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-07` | Secure API keys as secrets | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-08` | Import or recreate dashboard | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-09` | Preserve label-availability truthfulness | Conversion metrics hidden when labels do not exist | `COMPLETED` |
| `GRA-10` | Verify dashboard freshness | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-11` | Configure free alert delivery where feasible | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |
| `GRA-12` | Document public-versus-local differences | Public Grafana publication is deferred to the later zero-cost cloud phase | `DEFERRED` |

---

# 8. Kubernetes and Helm at zero cost

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `K8S-01` | Maintain local Kubernetes deployment | Seven deployments and services previously validated | `COMPLETED` |
| `K8S-02` | Maintain Helm chart validation | Lint, render, dry-run, and runtime checks passed | `COMPLETED` |
| `K8S-03` | Keep secret-based Grafana credentials | Secret workflow tested | `COMPLETED` |
| `K8S-04` | Keep alerting resources in Kubernetes | Alertmanager and webhook resources exist | `COMPLETED` |
| `K8S-05` | Do not require paid managed Kubernetes | Local cluster remains primary demonstration | `COMPLETED` |
| `K8S-06` | Document optional free-hosting experiments separately | Optional free-hosting experiments are not required for local Kubernetes closure | `OPTIONAL` |
| `K8S-07` | Avoid claiming cloud production deployment | Documentation and runbooks avoid claims of managed cloud production | `COMPLETED` |
| `K8S-08` | Re-run Helm/Kubernetes validation after new services | CI continues to validate Helm and Kubernetes after MLflow and Evidently work | `COMPLETED` |

---

# 9. Security and repository hygiene

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `SEC-01` | Ignore MLflow database and runtime artifacts | MLflow database, artifacts, environment, PID, and logs are protected | `COMPLETED` |
| `SEC-02` | Ignore generated Evidently reports where appropriate | Generated Evidently runtime outputs are controlled and repository hygiene is documented | `COMPLETED` |
| `SEC-03` | Keep `.env` excluded | Only `.env.example` committed | `COMPLETED` |
| `SEC-04` | Keep Grafana credentials external | Environment variables and Kubernetes Secrets used | `COMPLETED` |
| `SEC-05` | Protect GitHub and cloud tokens | Cloud-token handling is deferred to the later zero-cost deployment phase | `DEFERRED` |
| `SEC-06` | Run secret-pattern scan | No committed credentials or private keys found | `COMPLETED` |
| `SEC-07` | Review generated reports for sensitive visitor data | Public artifacts anonymised or sampled | `VERIFIED` |
| `SEC-08` | Pin all newly implemented dependencies | MLflow and Evidently dependency files are pinned | `COMPLETED` |
| `SEC-09` | Run `pip check` after dependency changes | No incompatible dependency versions | `COMPLETED` |
| `SEC-10` | Rebuild Docker images after dependency changes | Clean image build and health checks pass | `COMPLETED` |

---

# 10. Documentation and operational guides

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `DOC-01` | Update main README with MLflow | Architecture, startup, outputs, and limitations documented | `VERIFIED` |
| `DOC-02` | Update main README with Evidently | Report generation and interpretation documented | `VERIFIED` |
| `DOC-03` | Add MLflow runbook | MLflow startup, stop, inspect, validation, and local-storage steps are documented | `COMPLETED` |
| `DOC-04` | Add Evidently runbook | Evidently generation, outputs, interpretation, and limitations are documented | `COMPLETED` |
| `DOC-05` | Add delayed-label runbook | The delayed-label monitoring runbook documents the full operational process | `COMPLETED` |
| `DOC-06` | Add free-deployment guide | Streamlit and optional Grafana steps documented | `DEFERRED` |
| `DOC-07` | Update architecture diagrams | MLflow, Evidently, and delayed labels included | `VERIFIED` |
| `DOC-08` | Update limitations section | Limitations are separated from historical validation and current unlabeled scoring | `COMPLETED` |
| `DOC-09` | Create comment-preservation report | Comment-preservation inventory and summary exist | `COMPLETED` |
| `DOC-10` | Create before/after remediation summary | Final report committed | `COMPLETED` |
| `DOC-11` | Update original remediation matrix | All original findings receive final resolution | `COMPLETED` |
| `DOC-12` | Link this matrix from original matrix | Cross-reference exists | `COMPLETED` |
| `DOC-13` | Add interviewer explanation | Architecture and business explanations exist in project guides and visual evidence | `COMPLETED` |
| `DOC-14` | Record exact validation commands | Reproducible final QA checklist documented | `COMPLETED` |

---

# 11. Final testing and remediation closure

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| `QA-01` | MLflow focused tests pass | Bridge tests and end-to-end champion validation passed | `COMPLETED` |
| `QA-02` | Evidently focused tests pass | Focused Evidently bridge, schema, cohort, and drift tests passed during remediation | `COMPLETED` |
| `QA-03` | Delayed-label focused tests pass | Delayed-label contract, maturity, evaluation, and CI tests passed | `COMPLETED` |
| `QA-04` | Full normal pytest suite passes | Only approved opt-in tests skipped | `COMPLETED` |
| `QA-05` | Full retraining smoke test passes | Explicit full-pipeline command succeeds | `COMPLETED` |
| `QA-06` | Docker Compose builds and runs | All required services healthy | `COMPLETED` |
| `QA-07` | MLflow service health verified | Local health endpoint, SQLite backend, artifact storage, registry, and alias were verified | `COMPLETED` |
| `QA-08` | Evidently generation works in clean environment | Evidently reports were generated from canonical project artifacts | `COMPLETED` |
| `QA-09` | Monitoring snapshot remains truthful | No fabricated outcome metrics | `COMPLETED` |
| `QA-10` | Helm lint and template pass | Chart validates after changes | `COMPLETED` |
| `QA-11` | Kubernetes client dry-run passes | Rendered resources accepted | `COMPLETED` |
| `QA-12` | Local Kubernetes runtime passes where services are added | Pods, services, and health endpoints verified | `DEFERRED` |
| `QA-13` | CI workflow passes remotely | Latest remediation-branch run is green | `IN PROGRESS` |
| `QA-14` | `git diff --check` passes | No whitespace errors | `COMPLETED` |
| `QA-15` | Secret and large-file audit passes | No accidental credentials or oversized artifacts | `COMPLETED` |
| `QA-16` | Baseline-to-remediation diff reviewed | No accidental deletions or unrelated files | `COMPLETED` |
| `QA-17` | Every matrix item resolved | No unexplained `NOT STARTED` or `IN PROGRESS` items | `IN PROGRESS` |
| `QA-18` | Final documentation commit pushed | Remote branch synchronized | `IN PROGRESS` |
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

## Final local closure evidence

**Recorded:** 2026-06-22

- Full pytest suite: passed.
- Security and reproducibility audit: passed.
- Representative smoke training: passed.
- Dependency compatibility and vulnerability checks: passed.
- MLflow 3.14 native XGBoost validation: passed.
- Evidently 0.7.21 real-report validation: passed.
- Docker build, Compose services, health checks, and Streamlit endpoint: passed.
- Helm lint/template and Kubernetes client dry-runs: passed.
- Git whitespace review and baseline diff review: passed.
- Local Kubernetes runtime is `DEFERRED` to the later deployment phase; static validation is complete.
- Remote CI, push confirmation, user approval, and merge remain open because they can happen only after this closure commit.
- Immediate production outcome labels remain externally blocked until the future label window matures.
