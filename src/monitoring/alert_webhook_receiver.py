"""Receive Alertmanager notifications and store them locally.

This gives the project a real, testable alert-delivery path without requiring
Slack, email, Teams, or a paid external service.

Flow:
Prometheus -> Alertmanager -> this webhook -> alerts.jsonl
"""

from __future__ import annotations

import json
import os

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)

ALERT_PORT = int(
    os.getenv(
        "ALERT_WEBHOOK_PORT",
        "5001",
    )
)

ALERT_DIRECTORY = Path(
    os.getenv(
        "ALERT_NOTIFICATION_DIR",
        PROJECT_ROOT
        / "monitoring"
        / "alert_notifications",
    )
)

ALERT_FILE = ALERT_DIRECTORY / "alerts.jsonl"


def save_alert(payload: dict[str, Any]) -> None:
    """Append one Alertmanager notification as a JSON line."""

    ALERT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "received_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "payload": payload,
    }

    with ALERT_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record)
            + "\n"
        )


class AlertWebhookHandler(BaseHTTPRequestHandler):
    """Handle health checks and Alertmanager webhook requests."""

    def send_json(
        self,
        status_code: int,
        payload: dict[str, Any],
    ) -> None:
        """Return a JSON response."""

        body = json.dumps(payload).encode(
            "utf-8"
        )

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Expose a health endpoint for Docker and Kubernetes checks."""

        if self.path != "/health":
            self.send_json(
                404,
                {"status": "not_found"},
            )
            return

        self.send_json(
            200,
            {
                "status": "healthy",
                "alert_file": str(ALERT_FILE),
            },
        )

    def do_POST(self) -> None:
        """Receive and persist an Alertmanager notification."""

        if self.path != "/alerts":
            self.send_json(
                404,
                {"status": "not_found"},
            )
            return

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        raw_body = self.rfile.read(
            content_length
        )

        try:
            payload = json.loads(
                raw_body.decode("utf-8")
            )
        except json.JSONDecodeError:
            self.send_json(
                400,
                {
                    "status": "invalid_json",
                },
            )
            return

        save_alert(payload)

        alert_count = len(
            payload.get(
                "alerts",
                [],
            )
        )

        self.send_json(
            202,
            {
                "status": "accepted",
                "alert_count": alert_count,
            },
        )

    def log_message(
        self,
        format_string: str,
        *args,
    ) -> None:
        """Keep logs readable without the default noisy formatting."""

        print(
            f"Alert webhook: "
            f"{format_string % args}"
        )


def main() -> None:
    """Start the local Alertmanager webhook receiver."""

    ALERT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", ALERT_PORT),
        AlertWebhookHandler,
    )

    print(
        "Alert webhook receiver running on "
        f"port {ALERT_PORT}"
    )

    print(
        f"Alert log file: {ALERT_FILE}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
