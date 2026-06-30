"""Interactive architecture and governance inventory.

The architecture page already tells the project story. This extension adds one
selectable evidence inventory, two interactive flows, explicit local service
and deployment relationships, and a governed export without turning the page
into a screenshot gallery.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.closure_core import (
    compact_count,
    export_metadata,
    file_timestamp,
    finish_figure,
    status_from_exists,
)
from app.ui.components import (
    inject_product_css,
    render_detail_cards,
    render_evidence_chart,
    render_kpi_grid,
    render_section_header,
    render_source_note,
    show_table_with_download,
)


COMPONENTS = [
    {
        "Layer": "Data",
        "Component": "Raw RetailRocket events",
        "Purpose": "Source behaviour events for feature and KPI generation.",
        "Inputs": "data/raw/retailrocket/events.csv",
        "Outputs": "Visitor snapshots and daily KPI evidence",
        "Path": "data/raw/retailrocket/events.csv",
        "Evidence": "src/data/build_visitor_features.py",
    },
    {
        "Layer": "Data",
        "Component": "Leakage-safe visitor features",
        "Purpose": "One scoring row per visitor and historical snapshot rows for training.",
        "Inputs": "Raw events",
        "Outputs": "visitor_features.csv; visitor_training_snapshots.csv",
        "Path": "data/processed/visitor_features.csv",
        "Evidence": "src/data/build_visitor_features.py",
    },
    {
        "Layer": "ML",
        "Component": "Final champion model",
        "Purpose": "Rank visitor purchase intent with a governed threshold.",
        "Inputs": "Approved feature schema",
        "Outputs": "Model probabilities and campaign decisions",
        "Path": "models/trained/final_champion_model.joblib",
        "Evidence": "models/metadata/final_champion_metadata.json",
    },
    {
        "Layer": "Application",
        "Component": "Streamlit intelligence app",
        "Purpose": "Expose scoring, campaign, model, forecast, anomaly, monitoring, and architecture evidence.",
        "Inputs": "Saved models and generated evidence tables",
        "Outputs": "Interactive business decisions and governed exports",
        "Path": "app/Executive_Overview.py",
        "Evidence": "app/pages",
    },
    {
        "Layer": "Monitoring",
        "Component": "Prediction ledger",
        "Purpose": "Record prediction and batch scoring evidence for delayed labels and drift.",
        "Inputs": "App scoring events",
        "Outputs": "Prediction and batch logs",
        "Path": "monitoring/prediction_logs",
        "Evidence": "src/monitoring/prediction_logger.py",
    },
    {
        "Layer": "Monitoring",
        "Component": "Evidently drift reports",
        "Purpose": "Compare current and reference feature and score populations.",
        "Inputs": "Reference and current scored populations",
        "Outputs": "Feature and prediction drift JSON/HTML",
        "Path": "reports/evidently/latest",
        "Evidence": "src/monitoring/evidently_drift.py",
    },
    {
        "Layer": "Observability",
        "Component": "Prometheus and Grafana",
        "Purpose": "Scrape health metrics, visualise operations, and trigger alert rules.",
        "Inputs": "Monitoring exporter and service metrics",
        "Outputs": "Dashboards and alert states",
        "Path": "monitoring/prometheus/prometheus.yml",
        "Evidence": "monitoring/grafana/dashboards",
    },
    {
        "Layer": "Alerting",
        "Component": "Alertmanager",
        "Purpose": "Route verified alert conditions to configured receivers.",
        "Inputs": "Prometheus alert rules",
        "Outputs": "Alert notifications",
        "Path": "monitoring/alertmanager/alertmanager.yml",
        "Evidence": "monitoring/prometheus/alert_rules.yml",
    },
    {
        "Layer": "Experiment tracking",
        "Component": "MLflow lineage",
        "Purpose": "Store verified ecommerce run and registry lineage separately from unrelated evaluation runs.",
        "Inputs": "Model metrics, parameters, and artifacts",
        "Outputs": "Verified run and registry evidence",
        "Path": "models/metadata/mlflow_champion_lineage.json",
        "Evidence": "reports/visuals/ml_visual_intelligence/experiment_tracking",
    },
    {
        "Layer": "Packaging",
        "Component": "Docker image",
        "Purpose": "Package the app and runtime dependencies reproducibly.",
        "Inputs": "Application source and pinned requirements",
        "Outputs": "Container image",
        "Path": "Dockerfile",
        "Evidence": ".github/workflows/ci.yml",
    },
    {
        "Layer": "Deployment",
        "Component": "Kubernetes and Helm",
        "Purpose": "Describe services, probes, configuration, secrets, and chart packaging.",
        "Inputs": "Container image and environment configuration",
        "Outputs": "Deployment, service, config, and chart resources",
        "Path": "k8s",
        "Evidence": "helm",
    },
    {
        "Layer": "Governance",
        "Component": "CI and coverage matrices",
        "Purpose": "Validate security, reproducibility, tests, deployment files, and closure evidence.",
        "Inputs": "Repository changes and scheduled checks",
        "Outputs": "Test evidence and reconciled matrix state",
        "Path": ".github/workflows/ci.yml",
        "Evidence": "docs/streamlit/STREAMLIT_VISUAL_ENHANCEMENT_COVERAGE_MATRIX.csv",
    },
]


def component_inventory() -> pd.DataFrame:
    """Create the current architecture inventory with filesystem status."""

    frame = pd.DataFrame(COMPONENTS)
    frame["Status"] = frame["Path"].apply(lambda value: status_from_exists(Path(value)))
    frame["Refreshed"] = frame["Path"].apply(lambda value: file_timestamp(Path(value)))
    return frame


def build_business_flow_figure() -> go.Figure:
    """Build raw-event to business-output Sankey evidence."""

    labels = [
        "Raw events",
        "Visitor snapshots",
        "Final champion",
        "Single score",
        "Batch campaign",
        "Forecast and anomaly",
        "Business action",
    ]
    source = [0, 1, 2, 2, 1, 3, 4, 5]
    target = [1, 2, 3, 4, 5, 6, 6, 6]
    value = [10, 8, 3, 5, 4, 3, 5, 4]

    figure = go.Figure(
        go.Sankey(
            node={"label": labels, "pad": 22, "thickness": 20},
            link={"source": source, "target": target, "value": value},
        )
    )
    return finish_figure(
        figure,
        title="Data-to-decision architecture",
        subtitle="Raw behaviour becomes leakage-safe features, model scores, intelligence views, and governed actions.",
        height=520,
    )


def build_ml_lifecycle_figure() -> go.Figure:
    """Build training-to-feedback lifecycle evidence."""

    labels = [
        "Feature snapshots",
        "Train and validate",
        "Final holdout",
        "Champion artifact",
        "MLflow lineage",
        "Scoring",
        "Prediction ledger",
        "Drift and labels",
        "Feedback decision",
    ]
    source = [0, 1, 2, 3, 3, 5, 6, 7, 4]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 8]
    value = [10, 8, 6, 3, 6, 6, 5, 4, 2]

    figure = go.Figure(
        go.Sankey(
            node={"label": labels, "pad": 20, "thickness": 19},
            link={"source": source, "target": target, "value": value},
        )
    )
    return finish_figure(
        figure,
        title="Model lifecycle and feedback loop",
        subtitle="Selection, registry evidence, scoring, monitoring, matured labels, and feedback remain distinct stages.",
        height=550,
    )


def render_architecture_governance_intelligence() -> None:
    """Render the final interactive architecture and evidence extension."""

    inject_product_css()

    render_section_header(
        "Interactive architecture",
        "Select a component and inspect purpose, inputs, outputs, files, and evidence",
        "Artifact presence is implementation evidence; it is not a claim that every service is currently running.",
    )

    inventory = component_inventory()
    ready = int(inventory["Status"].eq("Ready").sum())
    layers = int(inventory["Layer"].nunique())

    render_kpi_grid(
        [
            ("Architecture components", compact_count(len(inventory)), "Documented components in the controlled inventory"),
            ("Evidence available", f"{ready}/{len(inventory)}", "Paths currently present in the repository or local project"),
            ("System layers", compact_count(layers), "Data through governance"),
            ("Deployment path", "Local → Docker → K8s", "Packaging and orchestration remain explicitly separated"),
        ]
    )

    flow = build_business_flow_figure()
    render_evidence_chart(
        figure=flow,
        key="closure_arch_business_flow",
        what_it_shows="How raw events become features, scores, intelligence outputs, and business action.",
        how_to_read="Links move left to right through the project layers; widths are explanatory, not production volumes.",
        actual_finding=f"The inventory documents {len(inventory)} components across {layers} layers.",
        conclusion="The portfolio is a connected intelligence platform rather than a standalone notebook or model file.",
        recommended_action="Use the inventory to trace any dashboard metric back to its source and generating code.",
        limitation="Sankey widths communicate architecture relationships and are not measured traffic counts.",
        source="Repository architecture and evidence inventory",
        evidence_type="Static architecture relationship map",
        refreshed_at=file_timestamp(Path("app/Executive_Overview.py")),
    )

    lifecycle = build_ml_lifecycle_figure()
    render_evidence_chart(
        figure=lifecycle,
        key="closure_arch_lifecycle",
        what_it_shows="Training, holdout evaluation, champion packaging, scoring, monitoring, and feedback stages.",
        how_to_read="Each stage has a separate evidence responsibility and should not be collapsed into one performance claim.",
        actual_finding="Final holdout, model artifact, score manifest, drift reports, and delayed-label evidence are stored separately.",
        conclusion="Separation of evidence reduces leakage and prevents offline metrics from being presented as live production results.",
        recommended_action="Require valid matured labels and lineage checks before replacing the champion.",
        limitation="The diagram proves repository design, not continuous cloud availability.",
        source="Model, monitoring, MLflow, and deployment repository artifacts",
        evidence_type="Governed ML lifecycle architecture",
        refreshed_at=file_timestamp(Path("models/metadata/final_champion_metadata.json")),
    )

    layer_filter = st.multiselect(
        "Architecture layers",
        options=sorted(inventory["Layer"].unique().tolist()),
        default=sorted(inventory["Layer"].unique().tolist()),
        key="closure_arch_layer_filter",
    )
    filtered = inventory.loc[inventory["Layer"].isin(layer_filter)].copy()

    if filtered.empty:
        st.warning("Select at least one architecture layer to inspect components.")
        return

    selected_component = st.selectbox(
        "Inspect component",
        options=filtered["Component"].tolist(),
        key="closure_arch_component",
    )
    selected = filtered.loc[filtered["Component"].eq(selected_component)].iloc[0]
    render_detail_cards(
        [
            ("Purpose", selected["Component"], selected["Purpose"]),
            ("Inputs", selected["Layer"], selected["Inputs"]),
            ("Outputs", selected["Status"], selected["Outputs"]),
            ("Files and evidence", selected["Path"], selected["Evidence"]),
        ]
    )

    show_table_with_download(
        label="architecture component inventory",
        data=filtered,
        file_name="architecture_component_inventory.csv",
        metadata=export_metadata(
            source="Repository filesystem and controlled architecture map",
            evidence_type="Architecture inventory",
            filters={"layers": layer_filter},
        ),
    )

    topology = pd.DataFrame(
        [
            ("Streamlit app", "Monitoring exporter", "Application metrics", "Local service"),
            ("Monitoring exporter", "Prometheus", "Scrape endpoint", "Observability"),
            ("Prometheus", "Grafana", "Dashboard queries", "Observability"),
            ("Prometheus", "Alertmanager", "Alert rules", "Alerting"),
            ("Prediction ledger", "Evidently", "Reference/current populations", "ML monitoring"),
            ("Evidently", "Streamlit monitoring", "Saved JSON and HTML reports", "ML monitoring"),
            ("Docker image", "Kubernetes deployment", "Container runtime", "Deployment"),
            ("Kubernetes service", "Streamlit app", "Network route and probes", "Deployment"),
        ],
        columns=["From", "To", "Relationship", "Layer"],
    )
    show_table_with_download(
        label="local service and deployment topology",
        data=topology,
        file_name="service_and_deployment_topology.csv",
        metadata=export_metadata(
            source="Docker, monitoring, Kubernetes, and Helm repository configuration",
            evidence_type="Service topology map",
        ),
    )

    ci_checks = pd.DataFrame(
        [
            ("Push and pull request", "Python tests, security, reproducibility, Docker and configuration checks"),
            ("Manual dispatch", "Controlled full validation when finalising a package"),
            ("Scheduled", "Dependency and security checks where configured"),
            ("Deployment validation", "Docker, Kubernetes, Helm, probes, and offline manifest checks"),
        ],
        columns=["Trigger", "Evidence scope"],
    )
    show_table_with_download(
        label="CI and deployment validation inventory",
        data=ci_checks,
        file_name="ci_deployment_validation_inventory.csv",
        metadata=export_metadata(
            source=".github/workflows and deployment test files",
            evidence_type="CI/CD architecture evidence",
        ),
    )

    render_source_note(
        source="Repository artifact inventory",
        evidence_type="Architecture implementation evidence",
        refreshed_at=file_timestamp(Path(".github/workflows/ci.yml")),
        extra="Service health must still be verified at runtime and again after cloud deployment.",
    )
