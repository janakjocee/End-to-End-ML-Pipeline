"""
Training Service
================

Microservice for model training with hyperparameter optimization.
Supports multiple algorithms with Optuna for hyperparameter tuning.
"""

import os
import uuid
import json
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# MLflow
import mlflow
import mlflow.sklearn

from shared.utils.logger import get_logger, StructuredLog
from shared.utils.database import DatabaseManager
from shared.utils.storage import StorageManager
from shared.utils.metrics import MetricsCollector
from shared.models.schemas import (
    HealthCheckResponse, TrainingConfig, EvaluationMetrics,
    ModelMetadata, ExperimentRun
)
from shared.exceptions import TrainingError, DataValidationError

logger = get_logger("training-service")

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global state
app_state = {
    'db': None,
    'storage': None,
    'metrics': None,
    'active_trainings': {},
    'executor': ThreadPoolExecutor(max_workers=4),
    'start_time': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Training Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize connections
    app_state['db'] = DatabaseManager()
    app_state['storage'] = StorageManager()
    app_state['metrics'] = MetricsCollector("training-service")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    yield
    
    # Cleanup
    app_state['executor'].shutdown(wait=True)
    logger.info("Shutting down Training Service...")


app = FastAPI(
    title="Training Service",
    description="Enterprise model training with hyperparameter optimization",
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

class ModelType(str, Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"


class TrainingRequest(BaseModel):
    """Training request."""
    experiment_name: str
    model_name: str
    model_type: ModelType
    dataset_name: str
    dataset_version: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    validation_size: float = Field(0.1, ge=0.05, le=0.3)
    hyperparameter_tuning: bool = True
    n_trials: int = Field(50, ge=1, le=500)
    cross_validation_folds: int = Field(5, ge=2, le=10)
    optimization_metric: str = "f1_score"
    random_state: int = 42
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class TrainingResponse(BaseModel):
    """Training response."""
    run_id: str
    experiment_id: str
    model_name: str
    model_version: str
    status: str
    metrics: EvaluationMetrics
    artifact_uri: str
    training_duration_seconds: float
    best_parameters: Dict[str, Any]


class ModelComparisonRequest(BaseModel):
    """Request to compare multiple models."""
    experiment_name: str
    dataset_name: str
    dataset_version: str
    target_column: str
    model_types: List[ModelType]
    hyperparameter_tuning: bool = True
    n_trials: int = 30


# ============================================
# Model Training Classes
# ============================================

class ModelTrainer:
    """Base model trainer class."""
    
    def __init__(self, model_type: str, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = {}
        
    def create_model(self, params: Optional[Dict] = None):
        """Create model instance."""
        raise NotImplementedError
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        """Define hyperparameter search space."""
        raise NotImplementedError
        
    def train(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Train the model."""
        self.model = self.create_model(params)
        self.model.fit(X_train, y_train)
        return self.model
        
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0),
        }
        
        if probabilities is not None and len(np.unique(y_test)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, probabilities)
                metrics['log_loss'] = log_loss(y_test, probabilities)
            except ValueError:
                pass
                
        return metrics


class RandomForestTrainer(ModelTrainer):
    """Random Forest trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }


class XGBoostTrainer(ModelTrainer):
    """XGBoost trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        }


class LightGBMTrainer(ModelTrainer):
    """LightGBM trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        }


class GradientBoostingTrainer(ModelTrainer):
    """Gradient Boosting trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return GradientBoostingClassifier(
            random_state=self.random_state,
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }


class LogisticRegressionTrainer(ModelTrainer):
    """Logistic Regression trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1,
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        return {
            'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga'
        }


class NeuralNetworkTrainer(ModelTrainer):
    """Neural Network trainer."""
    
    def create_model(self, params=None):
        if params is None:
            params = {}
        return MLPClassifier(
            random_state=self.random_state,
            max_iter=500,
            early_stopping=True,
            **params
        )
        
    def get_param_space(self, trial: optuna.Trial) -> Dict:
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256))
        
        return {
            'hidden_layer_sizes': tuple(hidden_layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        }


def get_trainer(model_type: str, random_state: int = 42) -> ModelTrainer:
    """Get appropriate trainer for model type."""
    trainers = {
        'random_forest': RandomForestTrainer,
        'xgboost': XGBoostTrainer,
        'lightgbm': LightGBMTrainer,
        'gradient_boosting': GradientBoostingTrainer,
        'logistic_regression': LogisticRegressionTrainer,
        'neural_network': NeuralNetworkTrainer
    }
    
    if model_type not in trainers:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return trainers[model_type](model_type, random_state)


# ============================================
# Training Pipeline
# ============================================

def run_hyperparameter_optimization(
    trainer: ModelTrainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    optimization_metric: str,
    cv_folds: int
) -> Tuple[Dict, float]:
    """Run Optuna hyperparameter optimization."""
    
    def objective(trial: optuna.Trial) -> float:
        params = trainer.get_param_space(trial)
        
        # Create and train model
        model = trainer.create_model(params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=optimization_metric)
        
        return scores.mean()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def train_model_pipeline(
    request: TrainingRequest,
    run_id: str
) -> Dict:
    """Execute full training pipeline."""
    start_time = datetime.utcnow()
    log = StructuredLog(logger, "model_training")
    log.add_field("run_id", run_id)
    log.add_field("model_type", request.model_type)
    log.add_field("model_name", request.model_name)
    
    try:
        # Load dataset
        bucket_name = "ml-datasets"
        df = app_state['storage'].load_dataset(
            bucket_name,
            request.dataset_name,
            request.dataset_version
        )
        
        log.add_field("dataset_rows", len(df))
        log.add_field("dataset_columns", len(df.columns))
        
        # Prepare features
        feature_cols = request.feature_columns or [c for c in df.columns if c != request.target_column]
        X = df[feature_cols].copy()
        y = df[request.target_column].copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=request.test_size + request.validation_size,
            random_state=request.random_state,
            stratify=y
        )
        
        val_size = request.validation_size / (request.test_size + request.validation_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - val_size,
            random_state=request.random_state,
            stratify=y_temp
        )
        
        log.add_field("train_size", len(X_train))
        log.add_field("val_size", len(X_val))
        log.add_field("test_size", len(X_test))
        
        # Get trainer
        trainer = get_trainer(request.model_type, request.random_state)
        
        # Hyperparameter optimization
        best_params = {}
        if request.hyperparameter_tuning:
            log.info("Starting hyperparameter optimization")
            best_params, best_score = run_hyperparameter_optimization(
                trainer,
                X_train.values, y_train.values,
                X_val.values, y_val.values,
                request.n_trials,
                request.optimization_metric,
                request.cross_validation_folds
            )
            log.add_field("best_score", best_score)
            log.add_field("best_params", best_params)
        
        # Final training on combined train+val
        X_train_final = pd.concat([X_train, X_val])
        y_train_final = pd.concat([y_train, y_val])
        
        log.info("Training final model")
        trainer.train(X_train_final.values, y_train_final.values, params=best_params)
        
        # Evaluate
        metrics = trainer.evaluate(X_test.values, y_test.values)
        
        # Get confusion matrix
        predictions = trainer.model.predict(X_test.values)
        cm = confusion_matrix(y_test, predictions).tolist()
        
        # Calculate training duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log to MLflow
        with mlflow.start_run(run_name=run_id) as mlflow_run:
            # Log parameters
            mlflow.log_param("model_type", request.model_type)
            mlflow.log_param("model_name", request.model_name)
            mlflow.log_param("dataset", request.dataset_name)
            mlflow.log_param("dataset_version", request.dataset_version)
            mlflow.log_params(best_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("training_duration_seconds", duration)
            
            # Log model
            mlflow.sklearn.log_model(
                trainer.model,
                artifact_path="model",
                registered_model_name=request.model_name
            )
            
            # Get artifact URI
            artifact_uri = mlflow.get_artifact_uri("model")
            
            # Log feature importance if available
            if hasattr(trainer.model, 'feature_importances_'):
                importances = trainer.model.feature_importances_
                feature_importance = dict(zip(feature_cols, importances))
                mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # Store in database
        training_record = {
            'run_id': run_id,
            'experiment_name': request.experiment_name,
            'model_name': request.model_name,
            'model_type': request.model_type,
            'status': 'completed',
            'metrics': metrics,
            'parameters': best_params,
            'artifact_path': artifact_uri,
            'started_at': start_time,
            'completed_at': datetime.utcnow()
        }
        app_state['db'].insert('training_runs', training_record)
        
        log.add_field("metrics", metrics)
        log.add_field("duration_seconds", duration)
        log.info("Training completed successfully")
        
        return {
            'run_id': run_id,
            'experiment_id': mlflow_run.info.experiment_id,
            'model_name': request.model_name,
            'status': 'completed',
            'metrics': metrics,
            'confusion_matrix': cm,
            'artifact_uri': artifact_uri,
            'training_duration_seconds': duration,
            'best_parameters': best_params,
            'feature_importance': feature_importance if hasattr(trainer.model, 'feature_importances_') else None
        }
        
    except Exception as e:
        log.error(f"Training failed: {str(e)}")
        # Update status in database
        app_state['db'].update(
            'training_runs',
            {'status': 'failed', 'error_message': str(e)},
            'run_id = %s',
            (run_id,)
        )
        raise TrainingError(str(e), request.model_name, run_id)


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
        service="training-service",
        version="2.0.0",
        checks={
            'database': 'connected' if app_state['db'] else 'disconnected',
            'mlflow': 'connected'  # Would check actual connection
        },
        uptime_seconds=uptime
    )


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a model with hyperparameter optimization.
    
    Supports Random Forest, XGBoost, LightGBM, Gradient Boosting,
    Logistic Regression, and Neural Networks.
    """
    run_id = str(uuid.uuid4())
    
    # Store training request
    app_state['active_trainings'][run_id] = {
        'status': 'running',
        'request': request,
        'started_at': datetime.utcnow()
    }
    
    try:
        # Run training in executor for non-blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app_state['executor'],
            train_model_pipeline,
            request,
            run_id
        )
        
        app_state['active_trainings'][run_id]['status'] = 'completed'
        
        return TrainingResponse(
            run_id=run_id,
            experiment_id=result['experiment_id'],
            model_name=request.model_name,
            model_version="1",  # Would get from MLflow
            status='completed',
            metrics=EvaluationMetrics(**result['metrics']),
            artifact_uri=result['artifact_uri'],
            training_duration_seconds=result['training_duration_seconds'],
            best_parameters=result['best_parameters']
        )
        
    except Exception as e:
        app_state['active_trainings'][run_id]['status'] = 'failed'
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/async")
async def train_model_async(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start asynchronous model training.
    
    Returns immediately with run_id. Check status with /train/{run_id}/status.
    """
    run_id = str(uuid.uuid4())
    
    app_state['active_trainings'][run_id] = {
        'status': 'running',
        'request': request,
        'started_at': datetime.utcnow()
    }
    
    # Run in background
    def run_training():
        try:
            result = train_model_pipeline(request, run_id)
            app_state['active_trainings'][run_id]['status'] = 'completed'
            app_state['active_trainings'][run_id]['result'] = result
        except Exception as e:
            app_state['active_trainings'][run_id]['status'] = 'failed'
            app_state['active_trainings'][run_id]['error'] = str(e)
    
    background_tasks.add_task(run_training)
    
    return {
        "run_id": run_id,
        "status": "started",
        "message": "Training started. Check status with /train/{run_id}/status"
    }


@app.get("/train/{run_id}/status")
async def get_training_status(run_id: str):
    """Get training job status."""
    if run_id not in app_state['active_trainings']:
        # Check database
        result = app_state['db'].fetch_one(
            "SELECT * FROM training_runs WHERE run_id = %s",
            (run_id,)
        )
        if result:
            return {
                "run_id": run_id,
                "status": result['status'],
                "metrics": result['metrics'],
                "completed_at": result['completed_at']
            }
        raise HTTPException(status_code=404, detail="Training run not found")
    
    training = app_state['active_trainings'][run_id]
    response = {
        "run_id": run_id,
        "status": training['status'],
        "started_at": training['started_at']
    }
    
    if training['status'] == 'completed':
        response['result'] = training.get('result')
    elif training['status'] == 'failed':
        response['error'] = training.get('error')
    
    return response


@app.post("/compare")
async def compare_models(request: ModelComparisonRequest):
    """
    Train and compare multiple model types.
    
    Returns comparison of all models on the same dataset.
    """
    results = []
    
    for model_type in request.model_types:
        train_request = TrainingRequest(
            experiment_name=f"{request.experiment_name}_compare",
            model_name=f"{request.target_column}_{model_type}_compare",
            model_type=model_type,
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            target_column=request.target_column,
            hyperparameter_tuning=request.hyperparameter_tuning,
            n_trials=request.n_trials
        )
        
        try:
            run_id = str(uuid.uuid4())
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                app_state['executor'],
                train_model_pipeline,
                train_request,
                run_id
            )
            results.append({
                'model_type': model_type,
                'status': 'success',
                'metrics': result['metrics'],
                'duration_seconds': result['training_duration_seconds']
            })
        except Exception as e:
            results.append({
                'model_type': model_type,
                'status': 'failed',
                'error': str(e)
            })
    
    # Sort by F1 score
    results.sort(
        key=lambda x: x.get('metrics', {}).get('f1_score', 0),
        reverse=True
    )
    
    return {
        "comparison_id": str(uuid.uuid4()),
        "results": results,
        "best_model": results[0] if results and results[0]['status'] == 'success' else None
    }


@app.get("/experiments")
async def list_experiments(limit: int = Query(100, ge=1, le=1000)):
    """List all training experiments."""
    try:
        results = app_state['db'].fetch_many(
            """
            SELECT run_id, experiment_name, model_name, model_type, 
                   status, metrics, started_at, completed_at
            FROM training_runs 
            ORDER BY started_at DESC 
            LIMIT %s
            """,
            (limit,)
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{run_id}")
async def get_experiment(run_id: str):
    """Get experiment details."""
    try:
        result = app_state['db'].fetch_one(
            "SELECT * FROM training_runs WHERE run_id = %s",
            (run_id,)
        )
        if not result:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)