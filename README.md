# ML Platform v2.0 - Enterprise-Grade MLOps System

<p align="center">
  <img src="docs/assets/ml-platform-logo.png" alt="ML Platform Logo" width="200"/>
</p>

<p align="center">
  <strong>Production-Ready Machine Learning & MLOps Platform</strong>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#services">Services</a> •
  <a href="#api-reference">API</a> •
  <a href="#monitoring">Monitoring</a>
</p>

---

## Overview

The **ML Platform** is a fully autonomous, enterprise-scale Machine Learning & MLOps system that automates the entire ML lifecycle — from data ingestion to real-time inference, monitoring, retraining, and governance.

Built with the same engineering standards used at companies like **Google**, **Netflix**, and **Amazon**, this platform provides:

- 🔄 **Full ML Lifecycle Automation**
- 📊 **Real-time Monitoring & Drift Detection**
- 🚀 **High-Performance Inference API**
- 🧠 **Intelligent Model Selection**
- 🔍 **Model Explainability with SHAP**
- 📈 **Scalable to 100k+ Predictions/Day**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML PLATFORM v2.0                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Data       │  │   Feature    │  │   Training   │  │   Model      │   │
│  │   Service    │──│   Service    │──│   Service    │──│   Registry   │   │
│  │   (:8001)    │  │   (:8002)    │  │   (:8003)    │  │   (:5000)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                 │                 │                 │             │
│         └─────────────────┴─────────────────┴─────────────────┘             │
│                               │                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   Inference  │  │  Monitoring  │  │  Retraining  │                      │
│  │   API        │──│  & Drift     │──│  Orchestrator│                      │
│  │   (:8000)    │  │  (:8004)     │  │  (Airflow)   │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         INFRASTRUCTURE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  PostgreSQL  │  │   MongoDB    │  │    Redis     │  │    MinIO     │   │
│  │  (Metadata)  │  │   (Logs)     │  │   (Cache)    │  │  (Artifacts) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │    MLflow    │  │  Prometheus  │  │   Grafana    │                      │
│  │  (Tracking)  │  │  (Metrics)   │  │ (Dashboards) │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🔄 Full ML Lifecycle Automation

| Stage | Description | Service |
|-------|-------------|---------|
| **Data Ingestion** | Multi-source ingestion (CSV, DB, API) with schema validation | Data Service |
| **Feature Engineering** | Reusable feature definitions, online/offline feature store | Feature Service |
| **Training** | Multi-model training with Optuna hyperparameter optimization | Training Service |
| **Model Registry** | Version control, staging, production promotion | MLflow |
| **Inference** | High-performance real-time predictions with SHAP explanations | Inference API |
| **Monitoring** | Drift detection, performance tracking, alerting | Monitoring Service |
| **Retraining** | Automated retraining on drift or schedule | Airflow |

### 🧠 Supported Algorithms

- **Random Forest** - Ensemble method with hyperparameter tuning
- **XGBoost** - Gradient boosting with advanced regularization
- **LightGBM** - Fast, distributed gradient boosting
- **Gradient Boosting** - Scikit-learn implementation
- **Logistic Regression** - Linear classifier with regularization
- **Neural Networks** - Multi-layer perceptron

### 📊 Monitoring & Observability

- **Data Drift Detection** - Evidently AI integration
- **Concept Drift Detection** - Target distribution monitoring
- **Prediction Drift** - Output distribution tracking
- **Performance Metrics** - Accuracy, precision, recall, F1, ROC-AUC
- **System Metrics** - CPU, memory, latency, throughput
- **Grafana Dashboards** - Real-time visualization

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 16GB+ RAM recommended
- 50GB+ free disk space

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/ml-platform.git
cd ml-platform

# Copy environment file
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 2. Start the Platform

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready (2-3 minutes)
docker-compose ps
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Inference API | http://localhost:8000 | - |
| MLflow UI | http://localhost:5000 | - |
| Airflow UI | http://localhost:8080 | admin/admin |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin |

### 4. Generate Sample Data

```bash
# Generate churn dataset
python scripts/generate_sample_data.py --samples 10000 --output-dir datasets

# Ingest data
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "churn_data",
    "config": {
      "source_type": "csv",
      "source_path": "datasets/churn_data_v1.csv"
    }
  }'
```

### 5. Train a Model

```bash
# Start training
curl -X POST http://localhost:8003/train \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "churn-prediction-v1",
    "model_name": "churn-predictor",
    "model_type": "xgboost",
    "dataset_name": "churn_data",
    "dataset_version": "v1",
    "target_column": "churn",
    "hyperparameter_tuning": true,
    "n_trials": 50
  }'
```

### 6. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure": 24,
      "monthly_charges": 65.5,
      "contract": "One year"
    },
    "return_explanation": true
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"tenure": 24, "monthly_charges": 65.5},
      {"tenure": 12, "monthly_charges": 45.0}
    ]
  }'
```

---

## Services

### Data Service (:8001)

Multi-source data ingestion with validation and versioning.

**Endpoints:**
- `POST /ingest` - Ingest data from various sources
- `POST /upload` - Upload files directly
- `GET /datasets` - List datasets
- `GET /download/{name}/{version}` - Download dataset

### Feature Service (:8002)

Feature engineering and feature store management.

**Endpoints:**
- `POST /feature-sets` - Register feature definitions
- `POST /engineer` - Apply feature engineering
- `POST /online-features` - Retrieve online features
- `GET /statistics/{dataset}/{version}` - Feature statistics

### Training Service (:8003)

Model training with hyperparameter optimization.

**Endpoints:**
- `POST /train` - Train model (sync)
- `POST /train/async` - Train model (async)
- `GET /train/{run_id}/status` - Check training status
- `POST /compare` - Compare multiple models

### Model Registry (:5000)

MLflow-based model versioning and lifecycle management.

**Endpoints:**
- `POST /register` - Register model
- `POST /transition` - Transition model stage
- `GET /models` - List models
- `GET /models/{name}/production` - Get production model

### Inference API (:8000)

High-performance prediction API with explanations.

**Endpoints:**
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions
- `POST /explain` - SHAP explanations
- `POST /reload` - Reload model

### Monitoring Service (:8004)

Drift detection and performance monitoring.

**Endpoints:**
- `POST /drift/detect` - Detect drift
- `POST /performance/monitor` - Monitor performance
- `GET /dashboard/{model}` - Dashboard data
- `GET /reports` - Drift reports

### Retraining Orchestrator (Airflow)

Automated retraining pipeline.

**DAGs:**
- `model_retraining_pipeline` - Full retraining workflow

---

## API Reference

### Prediction Request

```json
{
  "features": {
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1567.2,
    "contract": "One year",
    "payment_method": "Electronic check",
    "internet_service": "DSL",
    "online_security": "Yes",
    "tech_support": "No"
  },
  "model_name": "churn-predictor",
  "model_version": "v1.2.0",
  "return_explanation": true
}
```

### Prediction Response

```json
{
  "prediction_id": "pred-a1b2c3d4e5f6",
  "model_name": "churn-predictor",
  "model_version": "v1.2.0",
  "prediction": 1,
  "probability": 0.85,
  "probabilities": {
    "0": 0.15,
    "1": 0.85
  },
  "explanation": {
    "baseline_value": 0.27,
    "feature_contributions": [
      {"feature": "contract", "value": "One year", "contribution": -0.15},
      {"feature": "tenure", "value": 24, "contribution": -0.12},
      {"feature": "monthly_charges", "value": 65.5, "contribution": 0.08}
    ]
  },
  "latency_ms": 12.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000

**Available Dashboards:**
- **ML Platform Overview** - Key metrics and predictions
- **Model Performance** - Accuracy, latency, error rates
- **System Resources** - CPU, memory, disk usage
- **Drift Detection** - Data and concept drift scores

### Prometheus Metrics

Access Prometheus at http://localhost:9090

**Key Metrics:**
- `ml_predictions_total` - Total predictions by model
- `ml_prediction_latency_seconds` - Prediction latency
- `ml_drift_score` - Drift detection scores
- `ml_training_duration_seconds` - Training time
- `ml_system_memory_bytes` - Memory usage
- `ml_system_cpu_percent` - CPU usage

### Alerts

Configure alerts in `monitoring-service/prometheus/alerts.yml`:

```yaml
groups:
  - name: ml_platform
    rules:
      - alert: HighErrorRate
        expr: sum(rate(ml_predictions_total{status="error"}[5m])) / sum(rate(ml_predictions_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

---

## Development

### Project Structure

```
ml-platform/
├── data-service/          # Data ingestion microservice
├── feature-service/       # Feature engineering microservice
├── training-service/      # Model training microservice
├── model-registry/        # MLflow model registry
├── inference-api/         # Prediction API
├── monitoring-service/    # Drift detection & monitoring
├── retraining-orchestrator/  # Airflow DAGs
├── shared/                # Shared utilities & models
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── datasets/              # Sample datasets
└── docs/                  # Documentation
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest --cov=. --cov-report=html
```

### Adding a New Model Type

1. Add trainer class in `training-service/main.py`:

```python
class MyModelTrainer(ModelTrainer):
    def create_model(self, params=None):
        return MyModel(**(params or {}))
    
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'param1': trial.suggest_int('param1', 10, 100),
            'param2': trial.suggest_float('param2', 0.01, 1.0, log=True)
        }
```

2. Register in `get_trainer()` function

3. Add tests

---

## Deployment

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Deploy with specific profile
docker-compose --profile production up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress/
```

### AWS Deployment

See [docs/deployment/aws.md](docs/deployment/aws.md) for detailed instructions.

---

## Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Single Prediction Latency** | ~10-20ms | p95, XGBoost model |
| **Batch Prediction (1000)** | ~500ms | Parallel processing |
| **Throughput** | 10,000+ req/s | With 4 inference workers |
| **Training Time (10k samples)** | ~2-5 min | With Optuna tuning |
| **Memory Usage** | ~2GB | Per inference worker |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Python**: Black formatter, 120 char line length
- **Type Hints**: Required for all functions
- **Tests**: 80%+ coverage required
- **Documentation**: Docstrings for all public APIs

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## Acknowledgments

- [MLflow](https://mlflow.org/) - Experiment tracking
- [Evidently AI](https://evidentlyai.com/) - Drift detection
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Apache Airflow](https://airflow.apache.org/) - Orchestration

---

## Support

- 📧 Email: ml-platform@company.com
- 💬 Slack: #ml-platform-support
- 📚 Documentation: https://docs.ml-platform.io

---

<p align="center">
  Built with ❤️ by the ML Platform Team
</p>