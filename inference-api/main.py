"""
Inference API Service
=====================

High-performance real-time model inference API.
Supports single and batch predictions with SHAP explanations.
"""

import os
import uuid
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import mlflow
from mlflow.tracking import MlflowClient
import shap
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from shared.utils.logger import get_logger, StructuredLog
from shared.utils.database import DatabaseManager, MongoManager, CacheManager
from shared.utils.metrics import MetricsCollector, track_predictions
from shared.models.schemas import (
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelExplanation
)
from shared.exceptions import InferenceError, ModelNotFoundError, RateLimitError

logger = get_logger("inference-api")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global state
app_state = {
    'model': None,
    'model_name': None,
    'model_version': None,
    'db': None,
    'mongo': None,
    'cache': None,
    'metrics': None,
    'explainer': None,
    'feature_columns': None,
    'start_time': None,
    'request_count': 0
}

security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Inference API Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize connections
    app_state['db'] = DatabaseManager()
    app_state['mongo'] = MongoManager()
    app_state['cache'] = CacheManager()
    app_state['metrics'] = MetricsCollector("inference-api")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    # Load model
    await load_model()
    
    yield
    
    logger.info("Shutting down Inference API Service...")


app = FastAPI(
    title="ML Platform Inference API",
    description="Enterprise-grade real-time model inference API",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Model Loading
# ============================================

async def load_model():
    """Load model from MLflow registry."""
    model_name = os.getenv('MODEL_NAME', 'churn-predictor')
    model_stage = os.getenv('MODEL_STAGE', 'Production')
    
    try:
        client = MlflowClient()
        
        # Get latest version in stage
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        
        if not versions:
            logger.warning(f"No {model_stage} version found for {model_name}, trying None stage")
            versions = client.get_latest_versions(model_name, stages=["None"])
        
        if not versions:
            raise ModelNotFoundError(model_name)
        
        version = versions[0]
        
        # Load model
        model_uri = f"models:/{model_name}/{model_stage}"
        app_state['model'] = mlflow.sklearn.load_model(model_uri)
        app_state['model_name'] = model_name
        app_state['model_version'] = version.version
        
        # Load feature columns from run
        run = client.get_run(version.run_id)
        app_state['feature_columns'] = json.loads(
            run.data.params.get('feature_columns', '[]')
        )
        
        # Initialize SHAP explainer
        try:
            app_state['explainer'] = shap.TreeExplainer(app_state['model'])
        except Exception:
            app_state['explainer'] = shap.KernelExplainer(
                app_state['model'].predict_proba,
                shap.sample(pd.DataFrame(), 100)  # Would use background data
            )
        
        logger.info(f"Loaded model {model_name} version {version.version}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Don't raise - service can still function for health checks


# ============================================
# Helper Functions
# ============================================

def preprocess_features(features: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input features."""
    df = pd.DataFrame([features])
    
    # Handle categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Simple label encoding - in production use saved encoders
        df[col] = pd.Categorical(df[col]).codes
    
    # Handle missing values
    df = df.fillna(0)
    
    return df


def generate_prediction_id() -> str:
    """Generate unique prediction ID."""
    return f"pred-{uuid.uuid4().hex[:12]}"


def hash_features(features: Dict[str, Any]) -> str:
    """Hash features for caching."""
    feature_str = json.dumps(features, sort_keys=True)
    return hashlib.md5(feature_str.encode()).hexdigest()


def log_prediction(prediction: PredictionResponse, features: Dict[str, Any]):
    """Log prediction to MongoDB."""
    try:
        log_entry = {
            'prediction_id': prediction.prediction_id,
            'model_name': prediction.model_name,
            'model_version': prediction.model_version,
            'features': features,
            'prediction': prediction.prediction,
            'probability': prediction.probability,
            'latency_ms': prediction.latency_ms,
            'timestamp': prediction.timestamp
        }
        app_state['mongo'].insert_prediction(log_entry)
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


def get_explanation(features: pd.DataFrame, prediction_id: str) -> Optional[Dict]:
    """Generate SHAP explanation."""
    if app_state['explainer'] is None:
        return None
    
    try:
        shap_values = app_state['explainer'].shap_values(features)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Create explanation
        explanation = {
            'prediction_id': prediction_id,
            'baseline_value': float(app_state['explainer'].expected_value)
            if hasattr(app_state['explainer'], 'expected_value') else 0.0,
            'feature_contributions': [
                {
                    'feature': col,
                    'value': float(features[col].values[0]),
                    'contribution': float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i])
                }
                for i, col in enumerate(features.columns)
            ]
        }
        
        # Sort by absolute contribution
        explanation['feature_contributions'].sort(
            key=lambda x: abs(x['contribution']),
            reverse=True
        )
        
        return explanation
        
    except Exception as e:
        logger.warning(f"Failed to generate explanation: {e}")
        return None


# ============================================
# API Endpoints
# ============================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = None
    if app_state['start_time']:
        uptime = (datetime.utcnow() - app_state['start_time']).total_seconds()
    
    checks = {
        'database': 'connected' if app_state['db'] else 'disconnected',
        'model_loaded': app_state['model'] is not None,
        'model_name': app_state['model_name'],
        'model_version': app_state['model_version']
    }
    
    status = 'healthy' if app_state['model'] is not None else 'degraded'
    
    return HealthCheckResponse(
        status=status,
        service="inference-api",
        version="2.0.0",
        checks=checks,
        uptime_seconds=uptime
    )


@app.get("/info")
async def get_info():
    """Get API and model information."""
    return {
        'service': 'inference-api',
        'version': '2.0.0',
        'model_name': app_state['model_name'],
        'model_version': app_state['model_version'],
        'feature_columns': app_state['feature_columns']
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make a single prediction.
    
    Returns prediction with optional SHAP explanation.
    """
    start_time = time.time()
    prediction_id = prediction_request.request_id or generate_prediction_id()
    
    log = StructuredLog(logger, "prediction")
    log.add_field("prediction_id", prediction_id)
    
    try:
        # Check cache
        features_hash = hash_features(prediction_request.features)
        cached = app_state['cache'].get_prediction_cache(
            app_state['model_name'],
            features_hash
        )
        
        if cached:
            log.info("Returning cached prediction")
            return PredictionResponse(**cached)
        
        # Preprocess features
        features_df = preprocess_features(prediction_request.features)
        
        # Ensure correct column order
        if app_state['feature_columns']:
            missing_cols = set(app_state['feature_columns']) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            features_df = features_df[app_state['feature_columns']]
        
        # Make prediction
        prediction = app_state['model'].predict(features_df)[0]
        
        # Get probability if available
        probability = None
        probabilities = None
        if hasattr(app_state['model'], 'predict_proba'):
            proba = app_state['model'].predict_proba(features_df)[0]
            probability = float(proba.max())
            probabilities = {str(i): float(p) for i, p in enumerate(proba)}
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate explanation if requested
        explanation = None
        if prediction_request.return_explanation:
            explanation = get_explanation(features_df, prediction_id)
        
        # Create response
        response = PredictionResponse(
            prediction_id=prediction_id,
            model_name=app_state['model_name'],
            model_version=app_state['model_version'],
            prediction=int(prediction),
            probability=probability,
            probabilities=probabilities,
            explanation=explanation,
            latency_ms=latency_ms
        )
        
        # Cache prediction
        app_state['cache'].set_prediction_cache(
            app_state['model_name'],
            features_hash,
            response.dict(),
            expire=300
        )
        
        # Log prediction (async)
        background_tasks.add_task(log_prediction, response, prediction_request.features)
        
        # Record metrics
        app_state['metrics'].record_prediction(
            prediction_id,
            app_state['model_name'],
            app_state['model_version'],
            latency_ms,
            success=True
        )
        
        log.add_field("latency_ms", latency_ms)
        log.add_field("prediction", prediction)
        log.info("Prediction completed")
        
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        app_state['metrics'].record_prediction(
            prediction_id,
            app_state['model_name'],
            app_state['model_version'],
            latency_ms,
            success=False,
            error=str(e)
        )
        log.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
async def batch_predict(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make batch predictions.
    
    Process multiple records in a single request.
    """
    start_time = time.time()
    batch_id = generate_prediction_id()
    
    log = StructuredLog(logger, "batch_prediction")
    log.add_field("batch_id", batch_id)
    log.add_field("record_count", len(batch_request.records))
    
    try:
        predictions = []
        successful = 0
        failed = 0
        
        for i, record in enumerate(batch_request.records):
            try:
                pred_request = PredictionRequest(
                    features=record,
                    return_explanation=batch_request.return_explanations
                )
                
                # Process individual prediction
                features_df = preprocess_features(record)
                
                if app_state['feature_columns']:
                    missing_cols = set(app_state['feature_columns']) - set(features_df.columns)
                    for col in missing_cols:
                        features_df[col] = 0
                    features_df = features_df[app_state['feature_columns']]
                
                prediction = app_state['model'].predict(features_df)[0]
                
                probability = None
                if hasattr(app_state['model'], 'predict_proba'):
                    proba = app_state['model'].predict_proba(features_df)[0]
                    probability = float(proba.max())
                
                pred_response = PredictionResponse(
                    prediction_id=f"{batch_id}-{i}",
                    model_name=app_state['model_name'],
                    model_version=app_state['model_version'],
                    prediction=int(prediction),
                    probability=probability,
                    latency_ms=0,  # Individual latency not tracked in batch
                    timestamp=datetime.utcnow()
                )
                
                predictions.append(pred_response)
                successful += 1
                
            except Exception as e:
                failed += 1
                predictions.append(PredictionResponse(
                    prediction_id=f"{batch_id}-{i}",
                    model_name=app_state['model_name'],
                    model_version=app_state['model_version'],
                    prediction=-1,
                    error=str(e),
                    latency_ms=0,
                    timestamp=datetime.utcnow()
                ))
        
        total_latency = (time.time() - start_time) * 1000
        
        log.add_field("successful", successful)
        log.add_field("failed", failed)
        log.add_field("total_latency_ms", total_latency)
        log.info("Batch prediction completed")
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            model_name=app_state['model_name'],
            model_version=app_state['model_version'],
            predictions=predictions,
            total_records=len(batch_request.records),
            successful_predictions=successful,
            failed_predictions=failed,
            total_latency_ms=total_latency
        )
        
    except Exception as e:
        log.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(features: Dict[str, Any]):
    """
    Get SHAP explanation for features without making prediction.
    """
    try:
        features_df = preprocess_features(features)
        
        if app_state['feature_columns']:
            missing_cols = set(app_state['feature_columns']) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            features_df = features_df[app_state['feature_columns']]
        
        explanation = get_explanation(features_df, generate_prediction_id())
        
        if explanation is None:
            raise HTTPException(status_code=503, detail="Explanation not available")
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model():
    """Reload model from registry."""
    try:
        await load_model()
        return {
            "message": "Model reloaded successfully",
            "model_name": app_state['model_name'],
            "model_version": app_state['model_version']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        'model_name': app_state['model_name'],
        'model_version': app_state['model_version'],
        'uptime_seconds': (datetime.utcnow() - app_state['start_time']).total_seconds()
        if app_state['start_time'] else 0,
        'request_count': app_state['request_count']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)