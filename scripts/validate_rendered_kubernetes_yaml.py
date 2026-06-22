#!/usr/bin/env python3
"""Validate rendered Kubernetes YAML without contacting a cluster.

Input:
    Path to a YAML file created by `helm template`.

Checks:
    - At least one Kubernetes resource exists.
    - Every document is a mapping.
    - Every resource has apiVersion, kind, and metadata.
    - Every resource has metadata.name.

Output:
    A short success message with the number of validated resources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "apiVersion",
    "kind",
    "metadata",
}


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load non-empty Kubernetes YAML documents from one file."""

    if not path.is_file():
        raise FileNotFoundError(f"Rendered YAML file is missing: {path}")

    documents = [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document is not None
    ]

    if not documents:
        raise ValueError("No Kubernetes resources were found.")

    return documents


def validate_document(
    document: dict[str, Any],
    position: int,
) -> None:
    """Validate the minimum structure of one Kubernetes resource."""

    if not isinstance(document, dict):
        raise ValueError(f"Resource {position} is not a YAML mapping.")

    missing = sorted(REQUIRED_KEYS - set(document))

    if missing:
        raise ValueError(f"Resource {position} is missing keys: {missing}")

    metadata = document["metadata"]

    if not isinstance(metadata, dict):
        raise ValueError(f"Resource {position} metadata is not a mapping.")

    if not metadata.get("name"):
        raise ValueError(f"Resource {position} has no metadata.name.")


def main() -> None:
    """Read the rendered file and validate every resource offline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_yaml", type=Path)
    args = parser.parse_args()

    documents = load_documents(args.rendered_yaml)

    for position, document in enumerate(documents, start=1):
        validate_document(document, position)

    print("GOOD: Validated " f"{len(documents)} rendered Kubernetes resources offline")


if __name__ == "__main__":
    main()
