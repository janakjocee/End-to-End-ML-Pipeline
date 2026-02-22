"""
Model Registry Service
======================

Microservice for model versioning, staging, and lifecycle management.
Integrates with MLflow for experiment tracking and model storage.
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.utils.logger import get_logger
from shared.utils.database import DatabaseManager
from shared.models.schemas import HealthCheckResponse, ModelMetadata, ModelStage
from shared.exceptions import ModelNotFoundError

logger = get_logger("model-registry")

# Global state
app_state = {
    'db': None,
    'mlflow_client': None,
    'start_time': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Model Registry Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize connections
    app_state['db'] = DatabaseManager()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    app_state['mlflow_client'] = MlflowClient()
    
    yield
    
    logger.info("Shutting down Model Registry Service...")


app = FastAPI(
    title="Model Registry Service",
    description="Enterprise model registry and lifecycle management",
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

class RegisterModelRequest(BaseModel):
    """Request to register a model."""
    model_name: str
    run_id: str
    artifact_path: str = "model"
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class StageTransitionRequest(BaseModel):
    """Request to transition model stage."""
    model_name: str
    model_version: str
    stage: ModelStage
    comment: Optional[str] = None


class ModelVersionInfo(BaseModel):
    """Model version information."""
    name: str
    version: str
    stage: str
    run_id: str
    artifact_uri: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    created_at: datetime
    description: Optional[str] = None


class ModelComparison(BaseModel):
    """Model comparison result."""
    model_name: str
    versions: List[ModelVersionInfo]
    comparison_metrics: Dict[str, List[float]]


# ============================================
# Helper Functions
# ============================================

def get_mlflow_client() -> MlflowClient:
    """Get MLflow client."""
    return app_state['mlflow_client']


def model_version_to_dict(mv) -> Dict:
    """Convert MLflow model version to dictionary."""
    return {
        'name': mv.name,
        'version': mv.version,
        'stage': mv.current_stage,
        'run_id': mv.run_id,
        'artifact_uri': mv.source,
        'metrics': {},  # Would fetch from run
        'parameters': {},  # Would fetch from run
        'tags': dict(mv.tags) if mv.tags else {},
        'created_at': datetime.fromtimestamp(mv.creation_timestamp / 1000) if mv.creation_timestamp else None,
        'description': mv.description
    }


# ============================================
# API Endpoints
# ============================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = None
    if app_state['start_time']:
        uptime = (datetime.utcnow() - app_state['start_time']).total_seconds()
    
    mlflow_status = 'connected'
    try:
        get_mlflow_client().search_registered_models(max_results=1)
    except Exception:
        mlflow_status = 'disconnected'
    
    return HealthCheckResponse(
        status="healthy" if mlflow_status == 'connected' else 'degraded',
        service="model-registry",
        version="2.0.0",
        checks={
            'database': 'connected' if app_state['db'] else 'disconnected',
            'mlflow': mlflow_status
        },
        uptime_seconds=uptime
    )


@app.post("/register")
async def register_model(request: RegisterModelRequest):
    """
    Register a new model from an MLflow run.
    
    Creates a new version in the model registry.
    """
    try:
        client = get_mlflow_client()
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=f"runs:/{request.run_id}/{request.artifact_path}",
            name=request.model_name
        )
        
        # Add tags
        if request.tags:
            for key, value in request.tags.items():
                client.set_model_version_tag(
                    name=request.model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        # Add description
        if request.description:
            client.update_model_version(
                name=request.model_name,
                version=model_version.version,
                description=request.description
            )
        
        # Store in local database
        record = {
            'model_name': request.model_name,
            'model_version': model_version.version,
            'model_stage': 'None',
            'artifact_uri': model_version.source,
            'run_id': request.run_id,
            'tags': request.tags or {},
            'description': request.description,
            'created_at': datetime.utcnow()
        }
        app_state['db'].insert('model_registry', record)
        
        logger.info(f"Registered model {request.model_name} version {model_version.version}")
        
        return {
            "model_name": request.model_name,
            "version": model_version.version,
            "stage": "None",
            "artifact_uri": model_version.source,
            "message": "Model registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transition")
async def transition_stage(request: StageTransitionRequest):
    """
    Transition model to a new stage.
    
    Stages: None -> Staging -> Production -> Archived
    """
    try:
        client = get_mlflow_client()
        
        # Transition stage
        client.transition_model_version_stage(
            name=request.model_name,
            version=request.model_version,
            stage=request.stage.value
        )
        
        # Update database
        app_state['db'].update(
            'model_registry',
            {
                'model_stage': request.stage.value,
                'updated_at': datetime.utcnow()
            },
            'model_name = %s AND model_version = %s',
            (request.model_name, request.model_version)
        )
        
        # Add transition comment as tag
        if request.comment:
            client.set_model_version_tag(
                name=request.model_name,
                version=request.model_version,
                key="transition_comment",
                value=request.comment
            )
        
        logger.info(
            f"Transitioned {request.model_name} v{request.model_version} to {request.stage.value}"
        )
        
        return {
            "model_name": request.model_name,
            "version": request.model_version,
            "new_stage": request.stage.value,
            "message": f"Model transitioned to {request.stage.value}"
        }
        
    except Exception as e:
        logger.error(f"Stage transition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models(
    name_filter: Optional[str] = Query(None),
    stage: Optional[ModelStage] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all registered models."""
    try:
        client = get_mlflow_client()
        
        # Get registered models
        registered_models = client.search_registered_models()
        
        models = []
        for rm in registered_models:
            if name_filter and name_filter.lower() not in rm.name.lower():
                continue
                
            # Get latest versions
            versions = client.get_latest_versions(rm.name)
            
            for v in versions:
                if stage and v.current_stage != stage.value:
                    continue
                    
                models.append(model_version_to_dict(v))
        
        return models[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get model details with all versions."""
    try:
        client = get_mlflow_client()
        
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            raise ModelNotFoundError(model_name)
        
        version_list = [model_version_to_dict(v) for v in versions]
        
        return {
            "model_name": model_name,
            "versions": version_list,
            "version_count": len(version_list)
        }
        
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/versions/{version}")
async def get_model_version(model_name: str, version: str):
    """Get specific model version details."""
    try:
        client = get_mlflow_client()
        
        mv = client.get_model_version(model_name, version)
        
        # Get run metrics
        run = client.get_run(mv.run_id)
        metrics = dict(run.data.metrics)
        params = dict(run.data.params)
        
        result = model_version_to_dict(mv)
        result['metrics'] = metrics
        result['parameters'] = params
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/production")
async def get_production_model(model_name: str):
    """Get the production version of a model."""
    try:
        client = get_mlflow_client()
        
        versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"No production version found for {model_name}"
            )
        
        return model_version_to_dict(versions[0])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/versions/{version}/promote")
async def promote_model(
    model_name: str,
    version: str,
    target_stage: ModelStage,
    comment: Optional[str] = None
):
    """Promote model to a higher stage."""
    return await transition_stage(StageTransitionRequest(
        model_name=model_name,
        model_version=version,
        stage=target_stage,
        comment=comment or f"Promoted to {target_stage.value}"
    ))


@app.delete("/models/{model_name}/versions/{version}")
async def delete_model_version(model_name: str, version: str):
    """Delete a model version."""
    try:
        client = get_mlflow_client()
        
        # Delete version
        client.delete_model_version(model_name, version)
        
        # Update database
        app_state['db'].execute(
            "DELETE FROM model_registry WHERE model_name = %s AND model_version = %s",
            (model_name, version)
        )
        
        return {
            "message": f"Model {model_name} version {version} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
async def compare_models(
    model_names: List[str] = Query(...),
    metric: str = Query("f1_score")
):
    """Compare multiple models by a specific metric."""
    try:
        client = get_mlflow_client()
        
        comparisons = []
        for model_name in model_names:
            versions = client.search_model_versions(f"name='{model_name}'")
            
            version_metrics = []
            for v in versions:
                run = client.get_run(v.run_id)
                metric_value = run.data.metrics.get(metric, 0)
                version_metrics.append({
                    'version': v.version,
                    'stage': v.current_stage,
                    metric: metric_value
                })
            
            comparisons.append({
                'model_name': model_name,
                'versions': version_metrics
            })
        
        return {
            'metric': metric,
            'comparisons': comparisons
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{run_id}/download")
async def download_artifact(run_id: str, artifact_path: str = "model"):
    """Download model artifact."""
    try:
        import tempfile
        
        client = get_mlflow_client()
        
        # Download artifact
        local_path = client.download_artifacts(run_id, artifact_path)
        
        return {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "local_path": local_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)