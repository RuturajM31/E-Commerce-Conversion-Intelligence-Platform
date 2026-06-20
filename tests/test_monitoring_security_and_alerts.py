from pathlib import Path


def test_grafana_password_is_not_hardcoded():
    compose = Path(
        "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}" in compose

    assert "Techno#123" not in compose


def test_environment_template_documents_grafana_credentials():
    env_example = Path(
        ".env.example"
    ).read_text(encoding="utf-8")

    assert "GRAFANA_ADMIN_USER=" in env_example
    assert "GRAFANA_ADMIN_PASSWORD=" in env_example


def test_alertmanager_sends_to_real_local_receiver():
    config = Path(
        "monitoring/alertmanager/alertmanager.yml"
    ).read_text(encoding="utf-8")

    assert "receiver: local-webhook" in config
    assert "http://alert-webhook:5001/alerts" in config
    assert "send_resolved: true" in config
    assert "local-noop" not in config


def test_docker_compose_contains_alert_webhook_service():
    compose = Path(
        "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert "alert-webhook:" in compose

    assert (
        "src.monitoring.alert_webhook_receiver"
        in compose
    )


def test_alert_receiver_source_exists():
    assert Path(
        "src/monitoring/alert_webhook_receiver.py"
    ).exists()
