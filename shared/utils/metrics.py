"""
Metrics Collection Utilities
=============================

Performance metrics and monitoring utilities.
"""

import time
import functools
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server

from shared.utils.logger import get_logger

logger = get_logger(__name__)


# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model_name', 'model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_INFO = Info(
    'ml_model',
    'Model information'
)

DRIFT_SCORE = Gauge(
    'ml_drift_score',
    'Data drift score',
    ['model_name', 'feature_name', 'drift_type']
)

TRAINING_DURATION = Histogram(
    'ml_training_duration_seconds',
    'Training duration in seconds',
    ['model_name', 'model_type']
)

SYSTEM_MEMORY = Gauge(
    'ml_system_memory_bytes',
    'System memory usage',
    ['type']
)

SYSTEM_CPU = Gauge(
    'ml_system_cpu_percent',
    'System CPU usage percent'
)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""
    prediction_id: str
    model_name: str
    model_version: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_prediction: Any = None
    error: Optional[str] = None


class MetricsCollector:
    """
    Centralized metrics collector for the ML platform.
    """
    
    def __init__(self, service_name: str = "mlplatform"):
        self.service_name = service_name
        self.prediction_buffer: List[PredictionMetrics] = []
        self.buffer_size = 1000
        
    def record_prediction(
        self,
        prediction_id: str,
        model_name: str,
        model_version: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a prediction metric."""
        status = 'success' if success else 'error'
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        PREDICTION_LATENCY.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(latency_ms / 1000)
        
        # Buffer for batch processing
        metric = PredictionMetrics(
            prediction_id=prediction_id,
            model_name=model_name,
            model_version=model_version,
            latency_ms=latency_ms,
            error=error
        )
        self.prediction_buffer.append(metric)
        
        # Flush if buffer is full
        if len(self.prediction_buffer) >= self.buffer_size:
            self.flush_prediction_buffer()
            
    def record_training_metrics(
        self,
        model_name: str,
        model_type: str,
        duration_seconds: float,
        metrics: Dict[str, float]
    ):
        """Record training metrics."""
        TRAINING_DURATION.labels(
            model_name=model_name,
            model_type=model_type
        ).observe(duration_seconds)
        
        logger.info(
            f"Training completed for {model_name} ({model_type}) "
            f"in {duration_seconds:.2f}s with metrics: {metrics}"
        )
        
    def record_drift_score(
        self,
        model_name: str,
        feature_name: str,
        drift_type: str,
        score: float
    ):
        """Record drift detection score."""
        DRIFT_SCORE.labels(
            model_name=model_name,
            feature_name=feature_name,
            drift_type=drift_type
        ).set(score)
        
    def update_system_metrics(self):
        """Update system resource metrics."""
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY.labels(type='used').set(memory.used)
        SYSTEM_MEMORY.labels(type='available').set(memory.available)
        SYSTEM_MEMORY.labels(type='total').set(memory.total)
        SYSTEM_CPU.set(psutil.cpu_percent())
        
    def flush_prediction_buffer(self):
        """Flush prediction buffer to persistent storage."""
        if not self.prediction_buffer:
            return
            
        # Here you would typically write to a database or metrics service
        logger.debug(f"Flushing {len(self.prediction_buffer)} prediction metrics")
        self.prediction_buffer = []
        
    @contextmanager
    def measure_latency(self, operation_name: str):
        """Context manager to measure operation latency."""
        start = time.time()
        try:
            yield
        finally:
            latency = (time.time() - start) * 1000
            logger.debug(f"{operation_name} took {latency:.2f}ms")


def timing_decorator(metric_name: str):
    """Decorator to measure function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                latency = (time.time() - start) * 1000
                logger.debug(
                    f"{metric_name} completed in {latency:.2f}ms (status: {status})"
                )
        return wrapper
    return decorator


def track_predictions(model_name: str, model_version: str):
    """Decorator to track prediction metrics."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            prediction_id = kwargs.get('prediction_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                
                PREDICTION_COUNTER.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status='success'
                ).inc()
                
                PREDICTION_LATENCY.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(latency / 1000)
                
                return result
                
            except Exception as e:
                latency = (time.time() - start) * 1000
                
                PREDICTION_COUNTER.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status='error'
                ).inc()
                
                logger.error(f"Prediction failed: {str(e)}")
                raise
                
        return wrapper
    return decorator


class ModelPerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.predictions: List[Dict] = []
        self.max_buffer_size = 10000
        
    def add_prediction(
        self,
        actual: Any,
        predicted: Any,
        probability: Optional[float] = None,
        features: Optional[Dict] = None
    ):
        """Add a prediction for tracking."""
        self.predictions.append({
            'timestamp': datetime.utcnow(),
            'actual': actual,
            'predicted': predicted,
            'probability': probability,
            'features': features
        })
        
        # Keep buffer size manageable
        if len(self.predictions) > self.max_buffer_size:
            self.predictions = self.predictions[-self.max_buffer_size:]
            
    def calculate_accuracy(self, window: Optional[int] = None) -> float:
        """Calculate accuracy over recent predictions."""
        preds = self.predictions[-window:] if window else self.predictions
        if not preds:
            return 0.0
            
        correct = sum(1 for p in preds if p['actual'] == p['predicted'])
        return correct / len(preds)
        
    def calculate_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        preds = self.predictions[-window:] if window else self.predictions
        if not preds:
            return {}
            
        actual = [p['actual'] for p in preds]
        predicted = [p['predicted'] for p in preds]
        probabilities = [p['probability'] for p in preds if p['probability'] is not None]
        
        metrics = {
            'accuracy': accuracy_score(actual, predicted),
            'precision': precision_score(actual, predicted, average='weighted', zero_division=0),
            'recall': recall_score(actual, predicted, average='weighted', zero_division=0),
            'f1_score': f1_score(actual, predicted, average='weighted', zero_division=0),
            'prediction_count': len(preds)
        }
        
        if probabilities and len(set(actual)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(actual, probabilities)
            except ValueError:
                pass
                
        return metrics