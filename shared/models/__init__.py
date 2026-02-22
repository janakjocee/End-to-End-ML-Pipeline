"""Shared Pydantic models for ML Platform."""

from shared.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetadata,
    FeatureSchema,
    DriftReport,
    TrainingConfig,
    EvaluationMetrics,
    HealthCheckResponse,
    ServiceInfo
)

__all__ = [
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ModelMetadata',
    'FeatureSchema',
    'DriftReport',
    'TrainingConfig',
    'EvaluationMetrics',
    'HealthCheckResponse',
    'ServiceInfo'
]