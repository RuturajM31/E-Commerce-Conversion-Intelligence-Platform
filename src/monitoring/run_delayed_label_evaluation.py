# run_delayed_label_evaluation.py
# This module connects the complete delayed-label monitoring workflow.
#
# Simple business flow:
#   1. Read production predictions from the prediction ledger.
#   2. Read conversion outcomes that arrived later.
#   3. Reject unknown, premature, duplicate, conflicting, or invalid labels.
#   4. Join trustworthy matured outcomes to their original predictions.
#   5. Save validation evidence and truthful production metrics.
#
# Late-arriving labels:
#   The runner rebuilds the matured cohort from the complete current input
#   files each time it runs. A newly arrived valid label is therefore included
#   safely on the next run without duplicating previous outcomes.

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.config.paths import (
    DELAYED_LABEL_INPUT_PATH,
    DELAYED_LABEL_VALIDATION_PATH,
    MATURED_OUTCOMES_PATH,
    PREDICTION_LEDGER_PATH,
    PRODUCTION_PERFORMANCE_PATH,
)
from src.monitoring.delayed_labels import (
    MATURED_OUTCOME_COLUMNS,
    read_delayed_labels,
    validate_and_join_matured_outcomes,
)
from src.monitoring.prediction_ledger import (
    read_prediction_ledger,
)
from src.monitoring.production_performance import (
    build_production_performance_snapshot,
    write_production_performance_snapshot,
)


VALIDATION_REPORT_SCHEMA_VERSION = "1.0"


def write_matured_outcomes(
    rows: list[Dict[str, Any]],
    path: Path = MATURED_OUTCOMES_PATH,
) -> Path:
    """Write the current one-to-one matured cohort atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=MATURED_OUTCOME_COLUMNS,
        )
        writer.writeheader()

        for row in rows:
            missing_columns = [
                column
                for column in MATURED_OUTCOME_COLUMNS
                if column not in row
            ]

            extra_columns = [
                column
                for column in row
                if column not in MATURED_OUTCOME_COLUMNS
            ]

            if missing_columns or extra_columns:
                raise ValueError(
                    "Matured outcome row does not match the approved "
                    f"schema. Missing: {missing_columns}. "
                    f"Extra: {extra_columns}."
                )

            writer.writerow(
                {
                    column: row[column]
                    for column in MATURED_OUTCOME_COLUMNS
                }
            )

    temporary_path.replace(path)

    return path


def write_validation_report(
    report: Dict[str, Any],
    path: Path = DELAYED_LABEL_VALIDATION_PATH,
) -> Path:
    """Write label acceptance and rejection evidence atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path


def run_delayed_label_evaluation(
    *,
    ledger_path: Path = PREDICTION_LEDGER_PATH,
    label_path: Path = DELAYED_LABEL_INPUT_PATH,
    matured_outcomes_path: Path = MATURED_OUTCOMES_PATH,
    validation_report_path: Path = DELAYED_LABEL_VALIDATION_PATH,
    performance_snapshot_path: Path = PRODUCTION_PERFORMANCE_PATH,
    created_at_utc: str | None = None,
) -> Dict[str, Any]:
    """Run the complete delayed-label evaluation and return a summary."""

    execution_time = (
        created_at_utc
        if created_at_utc is not None
        else datetime.now(timezone.utc).isoformat()
    )

    # These readers enforce the approved prediction and label schemas.
    ledger_rows = read_prediction_ledger(
        path=ledger_path
    )
    label_rows = read_delayed_labels(
        path=label_path
    )

    # Only known predictions with completed outcome windows are accepted.
    matured_rows, validation_result = (
        validate_and_join_matured_outcomes(
            ledger_rows=ledger_rows,
            label_rows=label_rows,
        )
    )

    validation_report = {
        "validation_report_schema_version": (
            VALIDATION_REPORT_SCHEMA_VERSION
        ),
        "created_at_utc": execution_time,
        **validation_result,
    }

    # The output is rebuilt on every run. This safely incorporates valid
    # late-arriving labels without appending duplicate joined outcomes.
    write_matured_outcomes(
        matured_rows,
        path=matured_outcomes_path,
    )

    write_validation_report(
        validation_report,
        path=validation_report_path,
    )

    performance_snapshot = (
        build_production_performance_snapshot(
            matured_rows,
            created_at_utc=execution_time,
        )
    )

    write_production_performance_snapshot(
        performance_snapshot,
        path=performance_snapshot_path,
    )

    return {
        "created_at_utc": execution_time,
        "ledger_rows_read": len(ledger_rows),
        "label_rows_read": len(label_rows),
        "matured_rows_written": len(matured_rows),
        "metrics_available": performance_snapshot[
            "metrics_available"
        ],
        "matured_outcomes_path": str(
            matured_outcomes_path
        ),
        "validation_report_path": str(
            validation_report_path
        ),
        "performance_snapshot_path": str(
            performance_snapshot_path
        ),
    }


def main() -> None:
    """Run the workflow using the standard project paths."""

    summary = run_delayed_label_evaluation()

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
