# Security and Reproducibility Audit

This report is generated from the current repository state.

## Repository environment files

**Status:** `PASS`

Only the safe .env.example file is present.

**Evidence:**

- `.env.example`

## Repository secret-pattern scan

**Status:** `PASS`

No obvious committed credentials were found.

**Evidence:**

- No evidence files found.

## Platform password configuration

**Status:** `PASS`

Deployment passwords use runtime variables or Secret references.

**Evidence:**

- No evidence files found.

## Git history high-risk secret scan

**Status:** `PASS`

No private-key or high-risk token signatures were found.

**Evidence:**

- No evidence files found.

## Retired Grafana demo credential

**Status:** `INFO`

The former demo value is retired and must never be reused. Current configuration requires a runtime secret.

**Evidence:**

- `74380ac 2026-06-20 Secure Kubernetes and Helm alerting`
- `8aa32d2 2026-06-20 Add secure local alert delivery`
- `03e508e 2026-06-18 Initial commit: ecommerce conversion intelligence MLOps platform`

## Dependency health: main

**Status:** `PASS`

Installed packages report compatible dependencies.

**Evidence:**

- `No broken requirements found.`

## Dependency health: MLflow

**Status:** `PASS`

Installed packages report compatible dependencies.

**Evidence:**

- `No broken requirements found.`

## Dependency health: Evidently

**Status:** `PASS`

Installed packages report compatible dependencies.

**Evidence:**

- `No broken requirements found.`

## Representative sample data

**Status:** `PASS`

Small repository data exists for tests and CI.

**Evidence:**

- `reports/qa/smoke_training_result.json`

## Probability calibration evidence

**Status:** `PASS`

Calibration code or evidence exists.

**Evidence:**

- `docs/remediation/REMEDIATION_COVERAGE_MATRIX.md`
- `reports/qa/security_repro_audit.json`
- `reports/qa/security_repro_audit.md`
- `scripts/audit_security_repro.py`
- `src/monitoring/production_performance.py`
- `tests/test_production_performance.py`

## CI quality and security checks

**Status:** `PASS`

CI contains formatting, vulnerability, secret, and sample-training checks.

**Evidence:**

- `formatting_or_lint: FOUND`
- `dependency_scan: FOUND`
- `secret_scan: FOUND`
- `sample_smoke_training: FOUND`
