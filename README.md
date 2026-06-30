<div align="center">

# E-Commerce Conversion Intelligence Platform

### End-to-end machine learning, business intelligence, and MLOps for purchase-intent prediction

Transforming raw e-commerce behaviour into visitor-level intelligence, campaign-ready audiences, explainable model decisions, monitored production signals, and deployable business applications.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white&style=flat-square)](https://ecommerce-conversion-intelligence.streamlit.app)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/ci.yml?branch=main&label=CI%2FCD&logo=githubactions&logoColor=white&style=flat-square)](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/github/license/RuturajM31/E-Commerce-Conversion-Intelligence-Platform?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/RuturajM31/E-Commerce-Conversion-Intelligence-Platform?style=flat-square)](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/commits/main)
[![Stars](https://img.shields.io/github/stars/RuturajM31/E-Commerce-Conversion-Intelligence-Platform?style=flat)](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/stargazers)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?logo=streamlit&logoColor=white&style=flat-square)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikitlearn&logoColor=white&style=flat-square)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-189AB4?style=flat-square)](https://xgboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14.0-0194E2?logo=mlflow&logoColor=white&style=flat-square)](https://mlflow.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.7.21-6C63FF?style=flat-square)](https://www.evidentlyai.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3F4F75?logo=plotly&logoColor=white&style=flat-square)](https://plotly.com/python/)

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white&style=flat-square)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Validated-326CE5?logo=kubernetes&logoColor=white&style=flat-square)](https://kubernetes.io/)
[![Helm](https://img.shields.io/badge/Helm-Chart-0F1689?logo=helm&logoColor=white&style=flat-square)](https://helm.sh/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white&style=flat-square)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana&logoColor=white&style=flat-square)](https://grafana.com/)
[![pytest](https://img.shields.io/badge/pytest-Automated%20QA-0A9EDC?logo=pytest&logoColor=white&style=flat-square)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Lint%20%26%20Format-D7FF64?logo=ruff&logoColor=black&style=flat-square)](https://docs.astral.sh/ruff/)
[![pip-audit](https://img.shields.io/badge/pip--audit-Dependency%20Security-2E8B57?style=flat-square)](https://pypi.org/project/pip-audit/)

<br/>

[**Launch the app**](https://ecommerce-conversion-intelligence.streamlit.app)
·
[**View CI/CD**](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/actions/workflows/ci.yml)
·
[**Explore the source**](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform)

</div>

---

<div align="center">

**Navigate:** [Overview](#executive-summary) · [Architecture](#end-to-end-platform-architecture) · [Application](#streamlit-business-application) · [MLOps](#mlflow-experiment-tracking-architecture) · [Monitoring](#monitoring-and-observability-stack) · [CI/CD](#cicd-and-quality-gates) · [Run Locally](#quick-start)

</div>

---

## Executive Summary

The **E-Commerce Conversion Intelligence Platform** is a portfolio-grade, end-to-end machine learning and MLOps system built to answer one practical business question:

> **Which visitors are most likely to purchase, and how should the business act on that signal?**

The platform converts raw RetailRocket event data into visitor-level features, benchmarks multiple models, selects and validates a champion, optimizes the decision threshold, scores visitors, creates campaign-ready audiences, explains model decisions, monitors drift and delayed outcomes, and presents the complete workflow through a multi-page Streamlit business application.

This is not only a predictive model. It is a complete analytics product that connects:

- business problem framing;
- event-level data engineering;
- rare-event classification;
- campaign prioritization;
- model explainability;
- experiment tracking;
- drift monitoring;
- delayed-label performance measurement;
- business forecasting;
- anomaly detection;
- customer segmentation and journey intelligence;
- containerization and orchestration;
- CI/CD, security, and reproducibility controls.

---

## Project at a Glance

| Metric | Result | Business meaning |
|---|---:|---|
| Visitors analyzed | **1,407,500** | Full visitor population evaluated |
| Observed conversion rate | **0.827%** | Highly imbalanced purchase problem |
| High-intent audience | **19,588 visitors** | Focused campaign activation group |
| High-intent share | **~1.39%** | Small, prioritized audience |
| Champion model | **Tuned Random Forest** | Final deployable purchase-intent model |
| PR-AUC | **0.4151** | Strong ranking quality for rare conversions |
| ROC-AUC | **0.9696** | Strong class separation |
| Precision | **36.61%** | Selected audience is much cleaner than random targeting |
| Recall | **72.16%** | Captures a large share of actual buyers |
| F1 score | **0.4858** | Balanced precision and recall |
| Decision threshold | **0.97** | Business cutoff for high-intent selection |
| Anomaly rate | **3.66%** | Share of visitors flagged for unusual behaviour |
| Buyer concentration | **~44× baseline** | Model precision versus observed conversion rate |

> **Important:** the ~44× concentration is model-evaluation evidence, not proven campaign uplift. Real marketing impact must still be validated with controlled A/B testing.

---

## Why This Project Matters

Most e-commerce visitors do not purchase. Broad targeting wastes budget on users with little evidence of intent.

This platform changes the decision process from:

| Traditional workflow | Intelligence-driven workflow |
|---|---|
| Target a broad visitor population | Prioritize high-intent visitors |
| Use traffic volume as the main signal | Use visitor-level purchase probability |
| Select campaigns manually | Use model-supported audience ranking |
| Monitor only application uptime | Monitor features, predictions, outcomes, and services |
| Treat ML as a notebook result | Deliver ML through a governed business application |

The final output is not simply a probability. It is a decision system:

> **Decision flow:** raw behaviour becomes an actionable audience, then measured outcomes become new monitoring evidence.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph SIGNAL["1 · Build the signal"]
        direction LR
        A["Raw visitor<br/>events"]:::data --> B["Visitor-level<br/>features"]:::data
        B --> C["Purchase-intent<br/>model"]:::ml
    end

    subgraph DECIDE["2 · Make the decision"]
        direction LR
        D["Calibrated<br/>intent score"]:::ml --> E{"Business<br/>threshold"}:::decision
        E --> F["High-intent<br/>audience"]:::business
    end

    subgraph LEARN["3 · Measure and improve"]
        direction LR
        G["Campaign<br/>prioritization"]:::business --> H["Measured<br/>outcomes"]:::ops
        H --> I["Monitoring and<br/>retraining evidence"]:::monitor
    end

    C --> D
    F --> G

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

---

## Business Capabilities

The platform supports six connected decision areas:

| Capability | Business output |
|---|---|
| Purchase-intent prediction | Probability that a visitor will convert |
| Campaign prioritization | Ranked and capacity-aware high-intent audience |
| Model governance | Champion/challenger comparison, threshold, stability, and economics |
| Customer intelligence | Segments, nearest-neighbour context, and journey patterns |
| Operational intelligence | Forecasts, anomalies, drift, delayed outcomes, and system health |
| Deployment intelligence | Streamlit Cloud, Docker Compose, Kubernetes, Helm, and CI/CD evidence |

---

## End-to-End Platform Architecture

Six connected layers move the project from raw events to a governed business application.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    CI["GitHub Actions<br/>quality · security · delivery"]:::platform

    subgraph DATA["1 · Data foundation"]
        direction LR
        D1["RetailRocket<br/>events"]:::data --> D2["Clean and<br/>validate"]:::data
        D2 --> D3["Visitor-level<br/>feature table"]:::data
    end

    subgraph ML["2 · Machine-learning decision engine"]
        direction LR
        M1["Benchmark<br/>candidates"]:::ml --> M2["Tuned Random<br/>Forest"]:::ml
        M2 --> M3["Threshold and<br/>batch scores"]:::decision
    end

    subgraph INTEL["3 · Business intelligence"]
        direction LR
        B1["Executive and<br/>visitor intelligence"]:::business
        B2["Campaign and<br/>model intelligence"]:::business
        B3["Forecast, anomaly<br/>and journey"]:::business
    end

    subgraph GOVERN["4 · MLOps and governance"]
        direction LR
        O1["MLflow<br/>experiments"]:::ops
        O2["Model metadata<br/>and provenance"]:::ops
        O3["Prediction ledger<br/>and delayed labels"]:::ops
        O4["Evidently<br/>drift analysis"]:::ops
    end

    subgraph OBS["5 · Observability"]
        direction LR
        P1["Monitoring<br/>snapshot"]:::monitor --> P2["Prometheus and<br/>health probes"]:::monitor
        P2 --> P3["Grafana and<br/>Alertmanager"]:::monitor
    end

    subgraph DELIVERY["6 · Delivery"]
        direction LR
        X1["Streamlit<br/>Community Cloud"]:::platform
        X2["Docker<br/>Compose"]:::platform
        X3["Kubernetes<br/>and Helm"]:::platform
    end

    CI -. validates .-> DATA
    D3 --> M1
    M3 --> B1
    M3 --> B2
    M3 --> B3
    M2 --> O1
    M3 --> O2
    M3 --> O3
    D3 --> O4
    O3 --> P1
    O4 --> P1
    B1 --> X1
    P3 --> X2
    X2 --> X3

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

---

## Data Engineering Pipeline

The RetailRocket dataset is event-level: one row represents one visitor action such as a product view, add-to-cart event, or transaction. The business decision is visitor-level, so the project changes the data grain.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph INGEST["1 · Ingest and verify"]
        direction LR
        A["events.csv<br/>one action per row"]:::data --> B["Schema and<br/>quality checks"]:::data
        B --> C["Clean event<br/>records"]:::data
    end

    subgraph AGG["2 · Change the grain"]
        direction LR
        D["Group by<br/>visitor"]:::ml --> E["Create behavioural<br/>features"]:::ml
        E --> F["One row<br/>per visitor"]:::ml
    end

    subgraph READY["3 · Prepare for modelling"]
        direction LR
        G["Training and<br/>scoring split"]:::decision --> H["Model-ready<br/>dataset"]:::business
    end

    C --> D
    F --> G

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Core visitor features

| Feature | Meaning |
|---|---|
| `view_count` | Number of product views |
| `addtocart_count` | Number of add-to-cart actions |
| `unique_items` | Number of distinct items viewed |
| `activity_span_ms` | Duration of observed visitor activity |
| `converted` | Purchase outcome used as the target |

The project also contains modular packages for:

- feature engineering;
- segmentation;
- forecasting;
- anomaly detection;
- model evaluation;
- monitoring and delayed labels;
- visualization and business intelligence.

---

## Machine Learning Strategy

The observed conversion rate is only **0.827%**, so accuracy is not a useful primary metric. A model that predicts “no purchase” for almost everyone can look accurate while being useless for campaign targeting.

The project therefore emphasizes:

| Metric | Why it matters |
|---|---|
| PR-AUC | Evaluates ranking quality under severe class imbalance |
| ROC-AUC | Measures buyer versus non-buyer separation |
| Precision | Measures the quality of the selected campaign audience |
| Recall | Measures how many actual buyers are captured |
| F1 | Balances precision and recall |
| Threshold economics | Converts model scores into a business operating decision |

### Model lifecycle

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph DEVELOP["1 · Develop"]
        direction LR
        A["Visitor feature<br/>table"]:::data --> B["Baseline<br/>model"]:::ml
        A --> C["Candidate<br/>benchmark"]:::ml
        C --> D["Hyperparameter<br/>tuning"]:::ml
    end

    subgraph SELECT["2 · Select and govern"]
        direction LR
        E["Tuned Random<br/>Forest champion"]:::ml --> F["Threshold<br/>analysis"]:::decision
        F --> G["Stability and<br/>sensitivity checks"]:::decision
        G --> H["Champion<br/>metadata"]:::ops
    end

    subgraph OPERATE["3 · Score and learn"]
        direction LR
        I["Batch visitor<br/>scoring"]:::business --> J["Campaign<br/>audience"]:::business
        J --> K["Prediction<br/>ledger"]:::ops
        K --> L["Delayed outcome<br/>evaluation"]:::monitor
    end

    D --> E
    H --> I

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Champion decision

The tuned Random Forest was selected because it combined strong rare-event ranking with a practical campaign operating point:

- **PR-AUC:** 0.4151
- **ROC-AUC:** 0.9696
- **Precision:** 36.61%
- **Recall:** 72.16%
- **F1:** 0.4858
- **Decision threshold:** 0.97

---

## Streamlit Business Application

The Streamlit application is the business-facing layer of the platform.

**Entrypoint:** `app/Executive_Overview.py`

> **Navigation model:** one executive command centre connects decision, understanding, and governance workflows.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    HOME["Executive Overview<br/>portfolio command centre"]:::business

    subgraph DECIDE["Decide"]
        direction LR
        P1["Visitor Intent Predictor<br/>single-visitor support"]:::business
        P2["Batch Scoring<br/>audience and capacity"]:::business
        P3["Model Benchmark Selection<br/>champion and economics"]:::business
    end

    subgraph UNDERSTAND["Understand"]
        direction LR
        P4["Business KPI Forecasting<br/>scenarios and residuals"]:::ml
        P5["Anomaly and Outlier<br/>triage and investigation"]:::ml
        P9["Customer Segmentation<br/>segments and journeys"]:::ml
    end

    subgraph GOVERN["Govern"]
        direction LR
        P6["Monitoring, Drift and Health<br/>data, outcomes and services"]:::monitor
        P7["MLOps Architecture<br/>components and deployment"]:::ops
        P8["ML Validation and Evidence<br/>tests and provenance"]:::ops
    end

    HOME --> DECIDE
    HOME --> UNDERSTAND
    HOME --> GOVERN

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Application pages

| Page | Primary purpose |
|---|---|
| Executive Overview | Headline KPIs, business narrative, risk, and action priorities |
| Visitor Intent Predictor | Score and explain a single visitor scenario |
| Batch Scoring | Score visitors, apply campaign capacity, and create target audiences |
| Model Benchmark Selection | Compare candidates and explain the champion decision |
| Business KPI Forecasting | Forecast business KPIs with scenario and residual diagnostics |
| Anomaly and Outlier | Detect, rank, and investigate unusual behaviour |
| Monitoring, Drift and Health | Review data drift, prediction drift, delayed outcomes, and service status |
| MLOps Architecture | Explain platform components, deployment paths, and operational boundaries |
| ML Validation and Evidence | Surface test, reproducibility, provenance, and model-validation evidence |
| Customer Segmentation and Journey | Explore behavioural segments, similar visitors, and journey patterns |

---

## Campaign Intelligence

Batch scoring is designed as a business workflow, not only a technical inference step.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph INPUT["1 · Prepare the audience"]
        direction LR
        A["Visitor<br/>batch"]:::data --> B["Feature<br/>validation"]:::data
        B --> C["Champion<br/>scoring"]:::ml
    end

    subgraph PRIORITIZE["2 · Prioritize"]
        direction LR
        D["Intent<br/>probability"]:::ml --> E{"Threshold<br/>decision"}:::decision
        E --> F["Capacity-aware<br/>ranking"]:::decision
    end

    subgraph ACTIVATE["3 · Activate"]
        direction LR
        G["Campaign<br/>audience"]:::business --> H["Recommended<br/>action"]:::business
        H --> I["Export and<br/>activation"]:::ops
    end

    C --> D
    F --> G

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

The app supports:

- score distributions;
- campaign capacity scenarios;
- audience-size trade-offs;
- threshold effects;
- high-intent segment composition;
- conversion-risk context;
- action recommendations;
- exportable campaign lists.

---

## Explainability and Similarity Intelligence

The platform combines global model evidence with local visitor-level context.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    A["Scored visitor"]:::business

    subgraph MODEL["Model evidence"]
        direction LR
        B["Feature<br/>values"]:::data --> F["Feature contribution<br/>explanation"]:::ml
        C["Nearest real<br/>visitors"]:::data --> G["Similarity and<br/>difference breakdown"]:::ml
    end

    subgraph CONTEXT["Business context"]
        direction LR
        D["Segment<br/>membership"]:::business --> H["Segment<br/>profile"]:::business
        E["Anomaly<br/>context"]:::monitor --> I["Risk<br/>flags"]:::monitor
    end

    J["Governed visitor explanation<br/>score · evidence · context · caution"]:::decision

    A --> B
    A --> C
    A --> D
    A --> E
    F --> J
    G --> J
    H --> J
    I --> J

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

This supports more responsible business use by showing:

- why a visitor received a score;
- which behaviours are most influential;
- how the visitor compares with similar real visitors;
- whether the visitor belongs to a specific behavioural segment;
- whether anomaly signals require additional review.

---

## MLflow Experiment Tracking Architecture

MLflow is isolated from the lightweight app runtime and uses its own pinned environment.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    A["Training and<br/>benchmark scripts"]:::data --> B["MLflow bridge"]:::ml

    subgraph LOG["1 · Log the experiment"]
        direction LR
        C["Run<br/>parameters"]:::ml
        D["Evaluation<br/>metrics"]:::ml
        E["Model<br/>artifacts"]:::ml
        F["Environment<br/>provenance"]:::ml
    end

    subgraph STORE["2 · Store the evidence"]
        direction LR
        G["MLflow tracking<br/>server"]:::ops --> H["SQLite tracking<br/>backend"]:::ops
        G --> I["Artifact<br/>store"]:::ops
    end

    J["Experiment<br/>comparison"]:::decision --> K["Champion decision<br/>evidence"]:::business

    B --> C
    B --> D
    B --> E
    B --> F
    C --> G
    D --> G
    E --> G
    F --> G
    H --> J
    I --> J

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### MLflow responsibilities

- record model parameters and evaluation metrics;
- preserve experiment and environment provenance;
- validate model serialization and loading;
- support champion/challenger evidence;
- provide an auditable bridge between training and deployment metadata.

**Isolated dependency file:** `requirements-mlflow.txt`

**Validated MLflow version:** `3.14.0`

---

## Evidently Drift and Monitoring Architecture

Evidently is used in an isolated monitoring environment to avoid dependency conflicts with the application runtime.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph INPUTS["1 · Compare two populations"]
        direction LR
        R["Reference<br/>features + predictions"]:::data
        C["Current<br/>features + predictions"]:::data
    end

    E["Evidently<br/>analysis bridge"]:::ml

    subgraph REPORTS["2 · Produce drift evidence"]
        direction LR
        F["Feature drift<br/>report"]:::monitor
        G["Prediction drift<br/>report"]:::monitor
        H["Drift summary<br/>and provenance"]:::monitor
    end

    subgraph ACTIONS["3 · Publish and act"]
        direction LR
        I["Streamlit<br/>monitoring page"]:::business
        J["Monitoring<br/>snapshot"]:::ops
        K["Prometheus, Grafana<br/>and alerts"]:::monitor
    end

    R --> E
    C --> E
    E --> F
    E --> G
    F --> H
    G --> H
    H --> I
    H --> J
    J --> K

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Evidently responsibilities

- compare reference and current feature distributions;
- evaluate prediction drift;
- summarize drifted features and affected populations;
- preserve report provenance;
- feed drift evidence into the monitoring layer and Streamlit application.

**Isolated dependency file:** `requirements-evidently.txt`

**Validated Evidently version:** `0.7.21`

---

## Delayed Labels and Production Performance

Purchase outcomes may arrive after predictions are made. The project therefore separates prediction-time evidence from later outcome evaluation.

```mermaid
%%{init: {"theme":"base","sequence":{"useMaxWidth":true,"wrap":true,"diagramMarginX":30,"diagramMarginY":20,"actorMargin":45,"messageMargin":35},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","actorBkg":"#162033","actorBorder":"#5DADE2","actorTextColor":"#E5EEF8","signalColor":"#9FB3C8","signalTextColor":"#E5EEF8","noteBkgColor":"#3B2F12","noteBorderColor":"#D4A93A","noteTextColor":"#FDE68A"}}}%%
sequenceDiagram
    autonumber
    participant App as Scoring App
    participant Ledger as Prediction Ledger
    participant Outcome as Delayed Outcome
    participant Eval as Production Evaluation
    participant Monitor as Monitoring

    App->>Ledger: Store visitor, score, threshold and model version
    Note over Ledger: Prediction-time evidence is preserved
    Outcome->>Ledger: Attach the later purchase outcome
    Ledger->>Eval: Build the matched evaluation sample
    Eval->>Eval: Calculate precision, recall and conversion evidence
    Eval->>Monitor: Publish production-performance summary
    Monitor-->>App: Display health and performance evidence
```

This prevents a common MLOps mistake: evaluating only training metrics while ignoring real post-deployment outcomes.

---

## Monitoring and Observability Stack

The full local monitoring stack includes application, model, and infrastructure signals.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    subgraph SIGNALS["1 · Operational signals"]
        direction LR
        APP["Streamlit<br/>application"]:::business
        LEDGER["Prediction and<br/>delayed-label outputs"]:::data
        DRIFT["Evidently drift<br/>summaries"]:::data
    end

    subgraph COLLECT["2 · Collect and expose"]
        direction LR
        BLACKBOX["Blackbox Exporter<br/>:9115"]:::ops
        SNAPSHOT["Monitoring<br/>snapshot"]:::ops
        EXPORTER["Metrics Exporter<br/>:8000"]:::ops
    end

    PROM["Prometheus<br/>:9090"]:::monitor

    subgraph ACT["3 · Visualize and respond"]
        direction LR
        GRAFANA["Grafana<br/>:3000"]:::monitor
        ALERT["Alertmanager<br/>:9093"]:::monitor
        WEBHOOK["Alert Webhook<br/>:5001"]:::monitor
    end

    APP --> BLACKBOX
    LEDGER --> SNAPSHOT
    DRIFT --> SNAPSHOT
    SNAPSHOT --> EXPORTER
    BLACKBOX --> PROM
    EXPORTER --> PROM
    PROM --> GRAFANA
    PROM --> ALERT
    ALERT --> WEBHOOK

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Monitored areas

| Area | Examples |
|---|---|
| Application health | Streamlit availability and probe status |
| Model output | Score distribution, high-intent rate, threshold outcomes |
| Data quality | Missingness, invalid values, population changes |
| Drift | Feature drift and prediction drift |
| Production outcomes | Delayed-label precision, recall, and conversion evidence |
| Business health | Campaign audience size, forecast movement, anomaly rate |
| Infrastructure | Metrics endpoint health, alert routing, dashboard availability |

### Monitoring snapshot pattern

Prometheus should scrape lightweight metrics, not repeatedly scan large analytical files. The project therefore uses a cached snapshot:

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    A["Large analytical<br/>CSV and JSON outputs"]:::data --> B["Snapshot<br/>builder"]:::ml
    B --> C["ecommerce_metrics_<br/>snapshot.json"]:::ops
    C --> D["Lightweight metrics<br/>exporter"]:::ops
    D --> E["Prometheus"]:::monitor

    subgraph CONSUMERS["Fast monitoring consumers"]
        direction LR
        F["Grafana<br/>dashboards"]:::monitor
        G["Alertmanager<br/>rules"]:::monitor
    end

    E --> F
    E --> G

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

---

## Docker Compose Architecture

The local platform runs as a multi-service Docker Compose stack.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    USER["Business user"]:::business --> APP["streamlit-app<br/>:8501"]:::business

    subgraph DATA["Read-only analytical assets"]
        direction LR
        D1["Data"]:::data
        D2["Reports"]:::data
        D3["Models"]:::data
    end

    subgraph METRICS["Monitoring services"]
        direction LR
        CACHE["Metrics<br/>cache"]:::ops --> EXPORTER["metrics-exporter<br/>:8000"]:::ops
        BLACKBOX["blackbox-exporter<br/>:9115"]:::ops
        PROM["Prometheus<br/>:9090"]:::monitor
    end

    subgraph RESPONSE["Dashboards and alerts"]
        direction LR
        GRAFANA["Grafana<br/>:3000"]:::monitor
        ALERTMANAGER["Alertmanager<br/>:9093"]:::monitor
        WEBHOOK["alert-webhook<br/>:5001"]:::monitor
    end

    D1 --> APP
    D2 --> APP
    D3 --> APP
    APP --> BLACKBOX
    BLACKBOX --> PROM
    EXPORTER --> PROM
    PROM --> GRAFANA
    PROM --> ALERTMANAGER
    ALERTMANAGER --> WEBHOOK

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Docker services

| Service | Purpose | Local port |
|---|---|---:|
| `streamlit-app` | Business application | `8501` |
| `metrics-exporter` | Prometheus-compatible project metrics | `8000` |
| `prometheus` | Metrics collection and rule evaluation | `9090` |
| `blackbox-exporter` | External application health probes | `9115` |
| `alertmanager` | Alert grouping and routing | `9093` |
| `alert-webhook` | Local alert receiver and evidence writer | `5001` |
| `grafana` | Mission-control dashboards | `3000` |

---

## CI/CD and Quality Gates

The GitHub Actions workflow validates code, security, machine learning integrations, containers, orchestration files, and monitoring configuration.

> **Pipeline design:** quality and security gates run before ML validation and deployment-asset checks.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    A["Push · pull request<br/>manual · weekly"]:::platform --> B["GitHub Actions"]:::platform

    subgraph QUALITY["1 · Code quality and security"]
        direction LR
        Q1["Python 3.10<br/>dependencies"]:::data --> Q2["Imports, structure<br/>and syntax"]:::data
        Q2 --> Q3["Ruff lint<br/>and format"]:::data
        Q3 --> Q4["pip-audit and<br/>secret scan"]:::data
    end

    subgraph MLTEST["2 · Machine-learning validation"]
        direction LR
        M1["Sample-data<br/>smoke training"]:::ml --> M2["pytest<br/>suite"]:::ml
        M2 --> M3["MLflow<br/>integration"]:::ml
        M2 --> M4["Evidently<br/>integration"]:::ml
    end

    subgraph DELIVERY["3 · Delivery validation"]
        direction LR
        D1["Docker image and<br/>Compose"]:::ops --> D2["Helm and<br/>Kubernetes"]:::ops
        D2 --> D3["Prometheus and<br/>Alertmanager"]:::monitor
        D3 --> D4["Grafana JSON<br/>validation"]:::monitor
    end

    U["Green pipeline<br/>or compact failure evidence"]:::business

    B --> Q1
    Q4 --> M1
    M3 --> D1
    M4 --> D1
    D4 --> U

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### CI controls

- Python 3.10 dependency installation and `pip check`;
- project import and folder validation;
- Python and shell syntax checks;
- delayed-label and production-performance tests;
- Ruff linting and format checks;
- dependency vulnerability scanning with `pip-audit`;
- repository secret scanning;
- sample-data smoke training;
- complete automated test suite;
- scheduled/manual MLflow integration validation;
- scheduled/manual Evidently integration validation;
- Docker image and Docker Compose validation;
- Helm linting and rendered Kubernetes validation;
- Prometheus, alert-rule, Alertmanager, and Grafana validation;
- compact failure-evidence uploads.

> The CI workflow validates deployment assets but does not automatically deploy production infrastructure.

---

## Deployment Paths

The repository supports three delivery modes.

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true,"nodeSpacing":32,"rankSpacing":48},"themeVariables":{"background":"transparent","fontFamily":"Inter,Segoe UI,Arial,sans-serif","fontSize":"16px","primaryColor":"#162033","primaryTextColor":"#E5EEF8","primaryBorderColor":"#5DADE2","lineColor":"#7F8EA3","secondaryColor":"#201A33","tertiaryColor":"#10261D","clusterBkg":"#0B1220","clusterBorder":"#334155","edgeLabelBackground":"#0F172A"}}}%%
flowchart TB
    GIT["GitHub<br/>main branch"]:::platform --> CI["GitHub Actions<br/>validation"]:::platform

    subgraph PATHS["Three supported delivery paths"]
        direction LR

        subgraph CLOUDPATH["Portfolio cloud"]
            C1["Streamlit<br/>Community Cloud"]:::business --> C2["Public portfolio<br/>application"]:::business
        end

        subgraph LOCALPATH["Local MLOps stack"]
            L1["Docker<br/>image"]:::ops --> L2["Docker<br/>Compose"]:::ops
            L2 --> L3["App + monitoring<br/>services"]:::ops
        end

        subgraph K8SPATH["Orchestrated demo"]
            K1["Kubernetes<br/>manifests"]:::monitor --> K2["Helm<br/>chart"]:::monitor
            K2 --> K3["Kubernetes<br/>demo release"]:::monitor
        end
    end

    CI --> C1
    CI --> L1
    CI --> K1

classDef data fill:#12263A,stroke:#4DA3D9,color:#D6ECFF,stroke-width:1.5px;
    classDef ml fill:#221C38,stroke:#9A7BFF,color:#EEE6FF,stroke-width:1.5px;
    classDef business fill:#142A1D,stroke:#34D399,color:#DCFCE7,stroke-width:1.5px;
    classDef ops fill:#332012,stroke:#FB923C,color:#FFEDD5,stroke-width:1.5px;
    classDef monitor fill:#341424,stroke:#FB7185,color:#FFE4E6,stroke-width:1.5px;
    classDef platform fill:#1A2435,stroke:#94A3B8,color:#E2E8F0,stroke-width:1.5px;
    classDef decision fill:#3A2D12,stroke:#FBBF24,color:#FEF3C7,stroke-width:1.5px;
    linkStyle default stroke:#71859C,stroke-width:1.5px,color:#E2E8F0;
```

### Streamlit Community Cloud

- **Repository:** `RuturajM31/E-Commerce-Conversion-Intelligence-Platform`
- **Branch:** `main`
- **Entrypoint:** `app/Executive_Overview.py`
- **Validated Python:** `3.10`
- **App dependencies:** `app/requirements.txt`
- **Secrets required:** No
- **Public URL:** `https://ecommerce-conversion-intelligence.streamlit.app`

### Docker Compose

Build the monitoring snapshot before starting the stack:

```bash
python3 -m src.monitoring.build_monitoring_snapshot
docker compose up -d --build
docker compose ps
```

Open:

- Streamlit: `http://localhost:8501`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

### Kubernetes and Helm demo

```bash
helm upgrade --install ecommerce-conversion-platform \
  helm/ecommerce-conversion-platform \
  --namespace ecommerce-mlops \
  --create-namespace
```

The secure wrapper manages the local Kubernetes demo and generates a local Grafana password:

```bash
./E-Commerce-Conversion-Intelligence-Platform start
./E-Commerce-Conversion-Intelligence-Platform password
./E-Commerce-Conversion-Intelligence-Platform stop
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform.git
cd E-Commerce-Conversion-Intelligence-Platform
```

### 2. Create a Python 3.10 environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install the project

```bash
python -m pip install -r requirements.txt
```

### 4. Launch Streamlit

```bash
python -m streamlit run app/Executive_Overview.py
```

### 5. Run tests

```bash
python -m pytest -q
```

---

## Optional Isolated MLOps Environments

The main application remains lightweight. MLflow and Evidently are intentionally isolated.

### MLflow validation environment

```bash
python3.10 -m venv .venv-mlflow
source .venv-mlflow/bin/activate
python -m pip install -r requirements-mlflow.txt
python -m pytest tests/test_mlflow_bridge.py -q
```

### Evidently validation environment

```bash
python3.10 -m venv .venv-evidently
source .venv-evidently/bin/activate
python -m pip install -r requirements-evidently.txt
python -m pytest \
  tests/test_evidently_bridge.py \
  tests/test_drift_summary.py \
  -q
```

---

## Reproducibility and Security

The project uses explicit controls to make results repeatable and repository operations safer.

### Dependency strategy

| File | Purpose |
|---|---|
| `requirements.txt` | Full validated project environment |
| `requirements-app.txt` | Lightweight application runtime |
| `app/requirements.txt` | Streamlit Community Cloud runtime |
| `requirements-mlflow.txt` | Isolated MLflow environment |
| `requirements-evidently.txt` | Isolated Evidently environment |
| `requirements-ci.txt` | CI-only quality and security tools |

### Security and reproducibility controls

- exact dependency pinning;
- `pip check` compatibility validation;
- `pip-audit` vulnerability scanning;
- repository secret scanning;
- CI-only disposable fixtures for large local artifacts;
- smoke training from sample data;
- environment provenance recording;
- model serialization and loading tests;
- read-only mounts for data, reports, and models in Docker Compose;
- generated local Grafana password for the Kubernetes demo;
- no committed production credentials.

---

## Repository Structure

```text
E-Commerce-Conversion-Intelligence-Platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── app/
│   ├── Executive_Overview.py
│   ├── app_utils.py
│   ├── requirements.txt
│   ├── pages/
│   │   ├── 1_Visitor_Intent_Predictor.py
│   │   ├── 2_Batch_Scoring.py
│   │   ├── 3_Model_Benchmark_Selection.py
│   │   ├── 4_Business_KPI_Forecasting.py
│   │   ├── 5_Anomaly_Outlier.py
│   │   ├── 6_Monitoring_Drift_Health.py
│   │   ├── 7_MLOps_Architecture.py
│   │   ├── 8_ML_Validation_Evidence.py
│   │   └── 9_Customer_Segmentation_Journey.py
│   └── ui/
├── src/
│   ├── anomaly/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── forecasting/
│   ├── models/
│   ├── monitoring/
│   ├── segmentation/
│   └── visualization/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── models/
│   ├── trained/
│   └── metadata/
├── reports/
│   ├── figures/
│   ├── final/
│   ├── monitoring/
│   ├── qa/
│   └── tables/
├── monitoring/
│   ├── alertmanager/
│   ├── blackbox/
│   ├── grafana/
│   ├── metrics_cache/
│   ├── prediction_logs/
│   └── prometheus/
├── helm/
│   └── ecommerce-conversion-platform/
├── k8s/
├── scripts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Key Outputs

| Output | Purpose |
|---|---|
| `models/trained/final_champion_model.joblib` | Serialized champion model |
| `models/metadata/final_champion_metadata.json` | Model identity, threshold, metrics, and provenance |
| `data/processed/final_champion_visitor_scores.csv` | Visitor-level scores and campaign decisions |
| `reports/tables/final_true_champion_comparison.csv` | Candidate comparison |
| `reports/tables/final_true_champion_thresholds.csv` | Threshold trade-offs |
| `reports/tables/final_true_champion_stability.csv` | Stability evidence |
| `reports/tables/final_true_champion_sensitivity.csv` | Sensitivity evidence |
| `monitoring/metrics_cache/ecommerce_metrics_snapshot.json` | Lightweight Prometheus source |
| `monitoring/prediction_logs/` | Prediction ledger and delayed-label evidence |
| `monitoring/grafana/dashboards/` | Provisioned Grafana dashboards |

Large raw files and trained binary artifacts may be excluded from GitHub. Reproducible scripts and sample-data smoke workflows are included so important outputs can be regenerated and validated.

---

## Visual Intelligence Governance

The final Streamlit enhancement program is controlled by a **204-row visual intelligence matrix**.

| Status | Count |
|---|---:|
| Verified | **177** |
| Conditional | **25** |
| Excluded | **2** |
| Open | **0** |
| Total | **204** |

The matrix covers:

- executive intelligence;
- batch scoring and campaigns;
- model selection and explainability;
- KNN similarity evidence;
- segmentation and journey intelligence;
- forecasting and anomaly investigation;
- monitoring and drift;
- architecture and governance;
- shared UI and UX quality;
- validation and audit evidence.

This provides explicit evidence that the final application was reviewed as a governed product rather than assembled as an untracked collection of charts.

---

## Portfolio Value

This project demonstrates evidence across the complete analytics lifecycle.

| Skill area | Evidence |
|---|---|
| Business analytics | Conversion framing, KPI interpretation, campaign prioritization |
| Data engineering | Event-to-visitor transformation and reproducible feature generation |
| Machine learning | Rare-event modelling, benchmark selection, tuning, threshold economics |
| Explainable AI | Feature contributions, local similarity, segment and anomaly context |
| Customer analytics | Segmentation, journey intelligence, campaign capacity |
| Forecasting | KPI scenarios, uncertainty, residual diagnostics |
| Monitoring | Evidently drift, prediction ledger, delayed labels, service metrics |
| MLOps | MLflow, provenance, model metadata, monitoring snapshots |
| BI and product design | Multi-page Streamlit executive application |
| DevOps | Docker Compose, Kubernetes, Helm, GitHub Actions |
| Security and quality | Ruff, pytest, pip-audit, secret scan, smoke training |
| Technical communication | Architecture diagrams, documented assumptions, decision evidence |

The project is directly relevant to roles such as:

- Data Analyst;
- BI Analyst;
- Product Analyst;
- Marketing Analyst;
- Analytics Engineer;
- Data Scientist;
- Machine Learning Engineer;
- MLOps Engineer.

---

## Honest Limitations

This is a strong portfolio and local production-style platform, but it is not presented as a fully operated enterprise system.

Current boundaries:

- model results are based on a public historical dataset;
- campaign uplift has not yet been validated through a live randomized experiment;
- Streamlit Community Cloud is a portfolio deployment, not a high-availability enterprise serving layer;
- Docker and Kubernetes configurations demonstrate deployment readiness but are not continuously operated cloud infrastructure;
- monitoring snapshots must be scheduled in a real production environment;
- automated retraining and formal model-promotion approval can be expanded;
- access control, audit identity, and centralized secret management would be required for enterprise use.

---

## Roadmap

- [ ] Add campaign A/B testing and uplift measurement
- [ ] Add scheduled retraining and champion promotion
- [ ] Add automated monitoring snapshot refresh
- [ ] Add production feature-store integration
- [ ] Add event streaming for near-real-time scoring
- [ ] Add role-based access control
- [ ] Add managed cloud storage and model registry
- [ ] Add service-level objectives and incident runbooks
- [ ] Add formal data-contract enforcement

---

## Dataset

The project uses the public **RetailRocket E-Commerce Dataset**:

- visitor events;
- item properties;
- category hierarchy.

Dataset source: [RetailRocket recommender system dataset on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

The original event grain is transformed into a visitor-level decision table suitable for purchase-intent modelling and campaign activation.

---

## Author

**Ruturaj Mokashi**

Data Analyst & BI Specialist focused on transforming complex data into clear, measurable business decisions.

[![GitHub](https://img.shields.io/badge/GitHub-RuturajM31-181717?logo=github&logoColor=white&style=flat-square)](https://github.com/RuturajM31)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ruturaj%20Mokashi-0A66C2?logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/ruturaj-mokashi-09627588)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

### From raw visitor behaviour to monitored, explainable, campaign-ready intelligence

[Launch App](https://ecommerce-conversion-intelligence.streamlit.app)
·
[View Repository](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform)
·
[View CI/CD](https://github.com/RuturajM31/E-Commerce-Conversion-Intelligence-Platform/actions/workflows/ci.yml)

</div>
