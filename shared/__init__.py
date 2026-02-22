"""
ML Platform Shared Module
=========================

Common utilities, models, and exceptions used across all microservices.
"""

__version__ = "2.0.0"
__author__ = "ML Platform Team"

from shared.utils.logger import get_logger
from shared.utils.database import DatabaseManager
from shared.utils.storage import StorageManager
from shared.utils.cache import CacheManager
from shared.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetadata,
    FeatureSchema,
    DriftReport,
    TrainingConfig,
    EvaluationMetrics
)
from shared.exceptions.custom_exceptions import (
    MLPlatformException,
    DataValidationError,
    ModelNotFoundError,
    FeatureStoreError,
    DriftDetectionError,
    TrainingError
)

__all__ = [
    'get_logger',
    'DatabaseManager',
    'StorageManager',
    'CacheManager',
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ModelMetadata',
    'FeatureSchema',
    'DriftReport',
    'TrainingConfig',
    'EvaluationMetrics',
    'MLPlatformException',
    'DataValidationError',
    'ModelNotFoundError',
    'FeatureStoreError',
    'DriftDetectionError',
    'TrainingError'
]