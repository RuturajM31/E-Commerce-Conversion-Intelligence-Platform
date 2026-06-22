# CI workflow guide

## What this update adds

The GitHub workflow already runs the main project tests and checks Docker,
Docker Compose, Helm, Kubernetes, and delayed-label logic.

This update adds:

- a manual **Run workflow** button;
- a weekly light MLOps check;
- an MLflow integration test;
- an Evidently integration test;
- small failure evidence kept for seven days.

## Why MLflow and Evidently do not run on every push

Those tools need extra packages and take longer to install.

Normal pushes and pull requests keep using the fast project checks.
MLflow and Evidently run only:

- when the workflow is started manually;
- or during the weekly scheduled check.

This keeps GitHub usage small and free.

## Why full retraining is still not automatic

The complete RetailRocket data is large and is not stored in GitHub.

A full retraining job would take more time, memory, and storage than a normal
code check. The existing full-retraining test therefore remains optional.

A true small-data retraining mode should be added separately before that
matrix item is marked complete.

## What this update does not claim

This workflow validates the project. It does not prove that the platform is
running as a paid or managed cloud production service.
