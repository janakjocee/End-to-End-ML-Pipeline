"""
Pydantic Models for ML Platform
================================

Data validation and serialization models.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DriftType(str, Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


class PredictionRequest(BaseModel):
    """Single prediction request."""
    
    features: Dict[str, Any] = Field(..., description="Input features")
    model_name: Optional[str] = Field(None, description="Model name override")
    model_version: Optional[str] = Field(None, description="Model version override")
    request_id: Optional[str] = Field(None, description="Unique request ID")
    return_explanation: bool = Field(False, description="Return SHAP explanation")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "tenure": 24,
                    "monthly_charges": 65.5,
                    "contract": "One year"
                },
                "return_explanation": True
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    prediction_id: str = Field(..., description="Unique prediction ID")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    prediction: Union[int, str, float] = Field(..., description="Prediction result")
    probability: Optional[float] = Field(None, description="Prediction probability")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP explanation")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "pred-123456",
                "model_name": "churn-predictor",
                "model_version": "v1.2.0",
                "prediction": 1,
                "probability": 0.85,
                "latency_ms": 12.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    records: List[Dict[str, Any]] = Field(..., description="List of input records")
    model_name: Optional[str] = Field(None, description="Model name override")
    model_version: Optional[str] = Field(None, description="Model version override")
    return_explanations: bool = Field(False, description="Return SHAP explanations")
    
    @validator('records')
    def validate_records(cls, v):
        if not v:
            raise ValueError("Records list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Batch size cannot exceed 10000")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "records": [
                    {"tenure": 24, "monthly_charges": 65.5},
                    {"tenure": 12, "monthly_charges": 45.0}
                ],
                "return_explanations": False
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    batch_id: str = Field(..., description="Unique batch ID")
    model_name: str = Field(..., description="Model used")
    model_version: str = Field(..., description="Model version")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_records: int = Field(..., description="Total records processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    total_latency_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelMetadata(BaseModel):
    """Model metadata."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: ModelStage = Field(ModelStage.NONE, description="Model stage")
    description: Optional[str] = Field(None, description="Model description")
    model_type: str = Field(..., description="Type of model")
    framework: str = Field(..., description="ML framework used")
    artifact_uri: Optional[str] = Field(None, description="Path to model artifact")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Model metrics")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    tags: Dict[str, str] = Field(default_factory=dict, description="Model tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "churn-predictor",
                "version": "v1.2.0",
                "stage": "Production",
                "model_type": "XGBoost",
                "framework": "xgboost",
                "metrics": {
                    "accuracy": 0.85,
                    "f1_score": 0.82
                }
            }
        }


class FeatureSchema(BaseModel):
    """Feature schema definition."""
    
    name: str = Field(..., description="Feature name")
    dtype: str = Field(..., description="Data type")
    description: Optional[str] = Field(None, description="Feature description")
    nullable: bool = Field(True, description="Whether null values are allowed")
    default_value: Optional[Any] = Field(None, description="Default value")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    categories: Optional[List[str]] = Field(None, description="Allowed categories")
    importance: Optional[float] = Field(None, description="Feature importance score")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "tenure",
                "dtype": "int64",
                "description": "Number of months as customer",
                "nullable": False,
                "min_value": 0,
                "max_value": 100
            }
        }


class DriftReport(BaseModel):
    """Drift detection report."""
    
    report_id: str = Field(..., description="Unique report ID")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    drift_type: DriftType = Field(..., description="Type of drift")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Overall drift score")
    threshold: float = Field(..., description="Drift threshold used")
    features_analyzed: List[str] = Field(..., description="Features analyzed")
    features_drifted: List[str] = Field(default_factory=list, description="Features with drift")
    feature_scores: Dict[str, float] = Field(default_factory=dict, description="Per-feature scores")
    reference_distribution: Optional[Dict[str, Any]] = Field(None, description="Reference stats")
    current_distribution: Optional[Dict[str, Any]] = Field(None, description="Current stats")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "report_id": "drift-123456",
                "model_name": "churn-predictor",
                "drift_type": "data_drift",
                "drift_detected": True,
                "drift_score": 0.35,
                "threshold": 0.1,
                "features_drifted": ["monthly_charges"]
            }
        }


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    experiment_name: str = Field(..., description="Experiment name")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Type of model to train")
    dataset_path: str = Field(..., description="Path to training data")
    target_column: str = Field(..., description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns")
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Test set proportion")
    validation_size: float = Field(0.1, ge=0.0, le=1.0, description="Validation set proportion")
    random_state: int = Field(42, description="Random seed")
    hyperparameter_tuning: bool = Field(True, description="Enable hyperparameter tuning")
    n_trials: int = Field(100, ge=1, description="Number of Optuna trials")
    cross_validation_folds: int = Field(5, ge=2, description="CV folds")
    optimization_metric: str = Field("f1_score", description="Metric to optimize")
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "churn-experiment-v2",
                "model_name": "churn-predictor",
                "model_type": "XGBoost",
                "dataset_path": "s3://datasets/churn/train.csv",
                "target_column": "churn",
                "hyperparameter_tuning": True,
                "n_trials": 100
            }
        }


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics."""
    
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)
    log_loss: Optional[float] = Field(None, ge=0.0)
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    classification_report: Optional[Dict[str, Any]] = Field(None, description="Detailed report")
    calibration_score: Optional[float] = Field(None, description="Calibration score")
    
    # Additional metrics
    training_time_seconds: Optional[float] = Field(None, description="Training duration")
    inference_time_ms: Optional[float] = Field(None, description="Average inference time")
    model_size_mb: Optional[float] = Field(None, description="Model size in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.82,
                "f1_score": 0.83,
                "roc_auc": 0.91,
                "training_time_seconds": 120.5
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, Any] = Field(default_factory=dict, description="Detailed checks")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "inference-api",
                "version": "2.0.0",
                "checks": {
                    "database": "connected",
                    "model_loaded": True
                }
            }
        }


class ServiceInfo(BaseModel):
    """Service information."""
    
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Service description")
    endpoints: List[Dict[str, str]] = Field(default_factory=list, description="Available endpoints")
    dependencies: List[str] = Field(default_factory=list, description="Service dependencies")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "inference-api",
                "version": "2.0.0",
                "description": "Real-time model inference API",
                "endpoints": [
                    {"path": "/predict", "method": "POST"},
                    {"path": "/health", "method": "GET"}
                ]
            }
        }


class RetrainingTrigger(BaseModel):
    """Retraining trigger event."""
    
    trigger_id: str = Field(..., description="Unique trigger ID")
    trigger_type: str = Field(..., description="Type of trigger (drift, schedule, manual)")
    model_name: str = Field(..., description="Model to retrain")
    reason: str = Field(..., description="Reason for retraining")
    drift_report_id: Optional[str] = Field(None, description="Associated drift report")
    triggered_by: Optional[str] = Field(None, description="User or system that triggered")
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    config: Optional[TrainingConfig] = Field(None, description="Training configuration")


class ExperimentRun(BaseModel):
    """MLflow experiment run."""
    
    run_id: str = Field(..., description="MLflow run ID")
    experiment_id: str = Field(..., description="Experiment ID")
    experiment_name: str = Field(..., description="Experiment name")
    status: str = Field(..., description="Run status")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Run metrics")
    params: Dict[str, str] = Field(default_factory=dict, description="Run parameters")
    artifacts: List[str] = Field(default_factory=list, description="Artifact paths")
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    duration_seconds: Optional[float] = Field(None, description="Run duration")


class FeatureImportance(BaseModel):
    """Feature importance information."""
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    importance_type: str = Field(..., description="Type of importance (shap, permutation, etc.)")
    
    
class ModelExplanation(BaseModel):
    """Model explanation with SHAP values."""
    
    prediction_id: str = Field(..., description="Prediction ID")
    baseline_value: float = Field(..., description="Baseline prediction")
    predicted_value: float = Field(..., description="Actual prediction")
    feature_contributions: List[Dict[str, Any]] = Field(..., description="Per-feature contributions")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP values")
    expected_value: Optional[float] = Field(None, description="Expected value")
    plot_data: Optional[Dict[str, Any]] = Field(None, description="Data for plotting")