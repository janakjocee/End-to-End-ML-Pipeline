"""
Monitoring & Drift Detection Service
=====================================

Microservice for real-time monitoring and drift detection.
Uses Evidently AI for data drift, concept drift, and performance monitoring.
"""

import os
import uuid
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Evidently imports
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *

from shared.utils.logger import get_logger, StructuredLog
from shared.utils.database import DatabaseManager, MongoManager
from shared.utils.storage import StorageManager
from shared.models.schemas import HealthCheckResponse, DriftReport, DriftType
from shared.exceptions import DriftDetectionError

logger = get_logger("monitoring-service")

# Global state
app_state = {
    'db': None,
    'mongo': None,
    'storage': None,
    'reference_data': {},
    'drift_thresholds': {},
    'start_time': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Monitoring Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize connections
    app_state['db'] = DatabaseManager()
    app_state['mongo'] = MongoManager()
    app_state['storage'] = StorageManager()
    
    # Load default thresholds
    app_state['drift_thresholds'] = {
        'data_drift': float(os.getenv('DATA_DRIFT_THRESHOLD', 0.1)),
        'prediction_drift': float(os.getenv('PREDICTION_DRIFT_THRESHOLD', 0.1)),
        'performance_degradation': float(os.getenv('PERFORMANCE_THRESHOLD', 0.05))
    }
    
    yield
    
    logger.info("Shutting down Monitoring Service...")


app = FastAPI(
    title="Monitoring & Drift Detection Service",
    description="Real-time ML model monitoring and drift detection",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Pydantic Models
# ============================================

class DriftDetectionRequest(BaseModel):
    """Drift detection request."""
    model_name: str
    model_version: str
    reference_dataset: str
    reference_version: str
    current_dataset: Optional[str] = None
    current_version: Optional[str] = None
    drift_type: DriftType = DriftType.DATA_DRIFT
    features: Optional[List[str]] = None
    threshold: Optional[float] = None


class PerformanceMonitorRequest(BaseModel):
    """Performance monitoring request."""
    model_name: str
    model_version: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    window_size: int = Field(1000, ge=100, le=10000)


class AlertConfig(BaseModel):
    """Alert configuration."""
    model_name: str
    drift_type: DriftType
    threshold: float
    webhook_url: Optional[str] = None
    email_recipients: Optional[List[str]] = None
    enabled: bool = True


class MonitoringDashboard(BaseModel):
    """Monitoring dashboard data."""
    model_name: str
    time_range: str
    metrics: Dict[str, Any]
    drift_reports: List[DriftReport]
    alerts: List[Dict[str, Any]]


# ============================================
# Drift Detection Functions
# ============================================

def load_dataset(dataset_name: str, version: str) -> pd.DataFrame:
    """Load dataset from storage."""
    return app_state['storage'].load_dataset("ml-datasets", dataset_name, version)


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: Optional[ColumnMapping] = None
) -> Dict[str, Any]:
    """
    Detect data drift using Evidently.
    
    Returns drift report with scores and drifted features.
    """
    # Create data drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Get report as dictionary
    report_dict = data_drift_report.as_dict()
    
    # Extract drift information
    drift_score = report_dict['metrics'][0]['result']['dataset_drift']
    drifted_features = []
    
    for feature, data in report_dict['metrics'][0]['result'].get('drift_by_columns', {}).items():
        if data.get('drift_detected'):
            drifted_features.append(feature)
    
    return {
        'drift_detected': drift_score > app_state['drift_thresholds']['data_drift'],
        'drift_score': drift_score,
        'drifted_features': drifted_features,
        'report': report_dict
    }


def detect_prediction_drift(
    reference_predictions: pd.Series,
    current_predictions: pd.Series
) -> Dict[str, Any]:
    """Detect prediction distribution drift."""
    # Create comparison dataframe
    reference_df = pd.DataFrame({'prediction': reference_predictions})
    current_df = pd.DataFrame({'prediction': current_predictions})
    
    # Use Kolmogorov-Smirnov test for numerical predictions
    # or Chi-square for categorical
    from scipy import stats
    
    if reference_predictions.dtype in ['int64', 'float64']:
        # KS test for continuous distributions
        statistic, p_value = stats.ks_2samp(
            reference_predictions.dropna(),
            current_predictions.dropna()
        )
        drift_detected = p_value < app_state['drift_thresholds']['prediction_drift']
    else:
        # Chi-square for categorical
        ref_counts = reference_predictions.value_counts()
        cur_counts = current_predictions.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
        
        statistic, p_value = stats.chisquare(cur_aligned, ref_aligned)
        drift_detected = p_value < app_state['drift_thresholds']['prediction_drift']
    
    return {
        'drift_detected': drift_detected,
        'drift_score': 1 - p_value,
        'p_value': p_value,
        'statistic': statistic
    }


def detect_concept_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str
) -> Dict[str, Any]:
    """
    Detect concept drift - relationship between features and target changes.
    """
    # Create target drift report
    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])
    
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    report_dict = target_drift_report.as_dict()
    
    # Extract concept drift information
    target_drift = report_dict['metrics'][0]['result'].get('drift_detected', False)
    drift_score = report_dict['metrics'][0]['result'].get('drift_score', 0)
    
    return {
        'drift_detected': target_drift,
        'drift_score': drift_score,
        'report': report_dict
    }


def monitor_performance(
    predictions: List[Dict],
    window_size: int = 1000
) -> Dict[str, Any]:
    """
    Monitor model performance over time.
    """
    if not predictions:
        return {'error': 'No predictions available'}
    
    df = pd.DataFrame(predictions)
    
    # Calculate rolling metrics
    metrics = {
        'total_predictions': len(df),
        'time_range': {
            'start': df['timestamp'].min() if 'timestamp' in df else None,
            'end': df['timestamp'].max() if 'timestamp' in df else None
        },
        'latency': {
            'mean_ms': df['latency_ms'].mean() if 'latency_ms' in df else None,
            'p95_ms': df['latency_ms'].quantile(0.95) if 'latency_ms' in df else None,
            'p99_ms': df['latency_ms'].quantile(0.99) if 'latency_ms' in df else None
        }
    }
    
    # Calculate accuracy if ground truth available
    if 'actual' in df and 'prediction' in df:
        metrics['accuracy'] = (df['actual'] == df['prediction']).mean()
        
        # Rolling window accuracy
        if len(df) >= window_size:
            recent = df.tail(window_size)
            metrics['recent_accuracy'] = (recent['actual'] == recent['prediction']).mean()
            
            # Detect performance degradation
            older = df.head(window_size)
            older_accuracy = (older['actual'] == older['prediction']).mean()
            
            accuracy_drop = older_accuracy - metrics['recent_accuracy']
            metrics['performance_degradation'] = {
                'detected': accuracy_drop > app_state['drift_thresholds']['performance_degradation'],
                'accuracy_drop': accuracy_drop,
                'older_accuracy': older_accuracy,
                'recent_accuracy': metrics['recent_accuracy']
            }
    
    return metrics


# ============================================
# API Endpoints
# ============================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = None
    if app_state['start_time']:
        uptime = (datetime.utcnow() - app_state['start_time']).total_seconds()
    
    return HealthCheckResponse(
        status="healthy",
        service="monitoring-service",
        version="2.0.0",
        checks={
            'database': 'connected' if app_state['db'] else 'disconnected',
            'evidently': 'available'
        },
        uptime_seconds=uptime
    )


@app.post("/drift/detect")
async def detect_drift(request: DriftDetectionRequest, background_tasks: BackgroundTasks):
    """
    Detect drift between reference and current data.
    
    Supports data drift, prediction drift, and concept drift detection.
    """
    log = StructuredLog(logger, "drift_detection")
    log.add_field("model_name", request.model_name)
    log.add_field("drift_type", request.drift_type)
    
    try:
        # Load reference data
        reference_data = load_dataset(request.reference_dataset, request.reference_version)
        
        # Load current data or use predictions
        if request.current_dataset:
            current_data = load_dataset(request.current_dataset, request.current_version)
        else:
            # Use recent predictions as current data
            predictions = app_state['mongo'].get_predictions(
                model_name=request.model_name,
                limit=1000
            )
            if not predictions:
                raise HTTPException(status_code=400, detail="No current data available")
            current_data = pd.DataFrame([p['features'] for p in predictions])
        
        # Select features if specified
        features = request.features or reference_data.columns.tolist()
        reference_data = reference_data[features]
        current_data = current_data[features]
        
        # Detect drift based on type
        if request.drift_type == DriftType.DATA_DRIFT:
            result = detect_data_drift(reference_data, current_data)
        elif request.drift_type == DriftType.PREDICTION_DRIFT:
            result = detect_prediction_drift(
                reference_data.iloc[:, -1],  # Assuming last column is target
                current_data.iloc[:, -1]
            )
        elif request.drift_type == DriftType.CONCEPT_DRIFT:
            result = detect_concept_drift(reference_data, current_data, features[-1])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported drift type: {request.drift_type}")
        
        # Create drift report
        report_id = f"drift-{uuid.uuid4().hex[:12]}"
        threshold = request.threshold or app_state['drift_thresholds']['data_drift']
        
        drift_report = DriftReport(
            report_id=report_id,
            model_name=request.model_name,
            model_version=request.model_version,
            drift_type=request.drift_type,
            drift_detected=result['drift_detected'],
            drift_score=result['drift_score'],
            threshold=threshold,
            features_analyzed=features,
            features_drifted=result.get('drifted_features', []),
            feature_scores=result.get('feature_scores', {}),
            created_at=datetime.utcnow()
        )
        
        # Store report
        app_state['db'].insert('drift_reports', {
            'report_id': report_id,
            'model_name': request.model_name,
            'model_version': request.model_version,
            'drift_type': request.drift_type.value,
            'drift_detected': result['drift_detected'],
            'drift_score': result['drift_score'],
            'threshold': threshold,
            'features_drifted': result.get('drifted_features', []),
            'report_data': result.get('report', {}),
            'created_at': datetime.utcnow()
        })
        
        log.add_field("drift_detected", result['drift_detected'])
        log.add_field("drift_score", result['drift_score'])
        log.info("Drift detection completed")
        
        # Trigger alert if drift detected
        if result['drift_detected']:
            background_tasks.add_task(
                send_drift_alert,
                request.model_name,
                request.model_version,
                drift_report
            )
        
        return drift_report
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/performance/monitor")
async def monitor_model_performance(request: PerformanceMonitorRequest):
    """
    Monitor model performance over time.
    
    Analyzes prediction logs for performance degradation.
    """
    try:
        # Get predictions from MongoDB
        predictions = app_state['mongo'].get_predictions(
            model_name=request.model_name,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.window_size * 2
        )
        
        if not predictions:
            return {
                'message': 'No predictions found for the specified criteria',
                'model_name': request.model_name
            }
        
        # Monitor performance
        metrics = monitor_performance(predictions, request.window_size)
        
        return {
            'model_name': request.model_name,
            'model_version': request.model_version,
            'metrics': metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/{model_name}")
async def get_dashboard(
    model_name: str,
    days: int = Query(7, ge=1, le=30)
):
    """
    Get monitoring dashboard data for a model.
    
    Returns comprehensive monitoring metrics and drift reports.
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get drift reports
        drift_reports_data = app_state['db'].fetch_many(
            """
            SELECT * FROM drift_reports 
            WHERE model_name = %s AND created_at >= %s
            ORDER BY created_at DESC
            """,
            (model_name, start_date)
        )
        
        drift_reports = []
        for row in drift_reports_data:
            drift_reports.append(DriftReport(
                report_id=row['report_id'],
                model_name=row['model_name'],
                model_version=row['model_version'],
                drift_type=DriftType(row['drift_type']),
                drift_detected=row['drift_detected'],
                drift_score=row['drift_score'],
                threshold=row['threshold'],
                features_analyzed=[],
                features_drifted=row['features_drifted'],
                created_at=row['created_at']
            ))
        
        # Get performance metrics
        predictions = app_state['mongo'].get_predictions(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        performance_metrics = monitor_performance(predictions)
        
        # Get alerts
        alerts = []  # Would fetch from alert system
        
        return {
            'model_name': model_name,
            'time_range': f'{days} days',
            'metrics': performance_metrics,
            'drift_reports': [r.dict() for r in drift_reports],
            'alerts': alerts,
            'summary': {
                'total_predictions': performance_metrics.get('total_predictions', 0),
                'drift_detections': sum(1 for r in drift_reports if r.drift_detected),
                'average_latency_ms': performance_metrics.get('latency', {}).get('mean_ms', 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports")
async def list_reports(
    model_name: Optional[str] = Query(None),
    drift_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List drift detection reports."""
    try:
        query = "SELECT * FROM drift_reports WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)
        
        if drift_type:
            query += " AND drift_type = %s"
            params.append(drift_type)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        results = app_state['db'].fetch_many(query, tuple(params))
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get detailed drift report."""
    try:
        result = app_state['db'].fetch_one(
            "SELECT * FROM drift_reports WHERE report_id = %s",
            (report_id,)
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/configure")
async def configure_alert(config: AlertConfig):
    """Configure drift detection alerts."""
    try:
        record = {
            'model_name': config.model_name,
            'drift_type': config.drift_type.value,
            'threshold': config.threshold,
            'webhook_url': config.webhook_url,
            'email_recipients': config.email_recipients,
            'enabled': config.enabled,
            'created_at': datetime.utcnow()
        }
        
        app_state['db'].insert('alert_configs', record)
        
        return {
            'message': 'Alert configuration saved',
            'config': config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def send_drift_alert(
    model_name: str,
    model_version: str,
    drift_report: DriftReport
):
    """Send drift alert notification."""
    # Get alert configuration
    config = app_state['db'].fetch_one(
        "SELECT * FROM alert_configs WHERE model_name = %s AND enabled = true",
        (model_name,)
    )
    
    if not config:
        return
    
    alert_message = {
        'alert_type': 'drift_detected',
        'model_name': model_name,
        'model_version': model_version,
        'drift_type': drift_report.drift_type.value,
        'drift_score': drift_report.drift_score,
        'threshold': drift_report.threshold,
        'features_drifted': drift_report.features_drifted,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Send webhook
    if config.get('webhook_url'):
        try:
            import requests
            requests.post(config['webhook_url'], json=alert_message)
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    # Send email (would integrate with email service)
    if config.get('email_recipients'):
        logger.info(f"Would send email to {config['email_recipients']}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)