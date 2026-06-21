# Grafana Credential Rotation Record

## What was found

Two documentation files contained the former local demo password
`Techno#123`.

The value was used only for a local portfolio demonstration. It was not a
managed cloud or production credential.

## Remediation

- The password is removed from the current repository files.
- Docker Compose uses `GRAFANA_ADMIN_PASSWORD` from the local environment.
- Kubernetes and Helm use a Kubernetes Secret reference.
- The startup script requires `GRAFANA_ADMIN_PASSWORD` at runtime.
- The former value is retired and must never be reused.

## Git history

The old value may remain visible in earlier public commits. A destructive
history rewrite is not performed because the value was a demo credential and
is now retired.

Any real credential exposed in Git must be rotated immediately and handled
through the appropriate secret manager.

## Local setup

Create a new local password before starting Grafana:

```bash
export GRAFANA_ADMIN_PASSWORD='use-a-new-strong-local-password'
```

Do not place the value in source code, documentation, Helm values, Kubernetes
manifests, startup scripts, or committed `.env` files.
