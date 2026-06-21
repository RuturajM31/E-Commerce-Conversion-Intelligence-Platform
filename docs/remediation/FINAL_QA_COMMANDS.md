# Final QA Commands

This file records the validation families used for remediation closure.

## Python tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -ra -p no:cacheprovider
```

## Security and reproducibility

```bash
python3 scripts/audit_security_repro.py --ci
python3 -m pip_audit -r requirements.txt --progress-spinner off
git diff --check
```

## Docker and Streamlit

```bash
docker build -t ecommerce-conversion-platform-ci .
docker compose config
docker compose up -d --build
curl -fsS http://localhost:8501/_stcore/health
docker compose down
```

## Helm and Kubernetes static validation

```bash
helm lint helm/ecommerce-conversion-platform
helm template ecommerce-conversion-platform   helm/ecommerce-conversion-platform   --namespace ecommerce-conversion-platform
kubectl apply --dry-run=client --validate=false -f k8s/
```

## Service-specific validation

- MLflow 3.14 native XGBoost log/load/predict validation.
- Evidently 0.7.21 feature-drift and prediction-drift report generation.
- Dependency candidate `pip check` and `pip-audit`.
