"""
Feature Engineering Service
===========================

Microservice for feature engineering and feature store management.
Provides reusable feature definitions and online/offline feature retrieval.
"""

import os
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from shared.utils.logger import get_logger, StructuredLog
from shared.utils.database import DatabaseManager, CacheManager
from shared.utils.storage import StorageManager
from shared.models.schemas import HealthCheckResponse, FeatureSchema
from shared.exceptions import FeatureStoreError, DataValidationError

logger = get_logger("feature-service")

# Global state
app_state = {
    'db': None,
    'cache': None,
    'storage': None,
    'feature_registry': {},
    'transformers': {},
    'start_time': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Feature Engineering Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize connections
    app_state['db'] = DatabaseManager()
    app_state['cache'] = CacheManager()
    app_state['storage'] = StorageManager()
    
    # Load feature registry
    await load_feature_registry()
    
    yield
    
    logger.info("Shutting down Feature Engineering Service...")


app = FastAPI(
    title="Feature Engineering Service",
    description="Enterprise feature store and feature engineering service",
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

class FeatureDefinition(BaseModel):
    """Feature definition for registry."""
    name: str
    description: str
    feature_type: str = Field(..., regex="^(numeric|categorical|datetime|derived|embedding)$")
    source_column: Optional[str] = None
    transformation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)


class FeatureSetRequest(BaseModel):
    """Request to create a feature set."""
    feature_set_name: str
    feature_definitions: List[FeatureDefinition]
    base_dataset: str
    base_version: str


class FeatureEngineeringRequest(BaseModel):
    """Feature engineering request."""
    dataset_name: str
    dataset_version: str
    feature_set_name: str
    output_name: str


class OnlineFeaturesRequest(BaseModel):
    """Request for online feature retrieval."""
    feature_set_name: str
    entity_ids: List[str]
    feature_names: Optional[List[str]] = None


class FeatureStatistics(BaseModel):
    """Feature statistics."""
    feature_name: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    null_count: int
    unique_count: int
    sample_values: List[Any]


# ============================================
# Feature Transformations
# ============================================

class FeatureTransformer:
    """Base class for feature transformers."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, column: str):
        """Fit the transformer."""
        raise NotImplementedError
        
    def transform(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Transform the data."""
        raise NotImplementedError
        
    def fit_transform(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Fit and transform."""
        self.fit(df, column)
        return self.transform(df, column)


class NumericScaler(FeatureTransformer):
    """Numeric feature scaler."""
    
    def __init__(self, name: str, method: str = "standard"):
        super().__init__(name)
        self.method = method
        self.scaler = None
        
    def fit(self, df: pd.DataFrame, column: str):
        if self.method == "standard":
            self.scaler = StandardScaler()
        self.scaler.fit(df[[column]].dropna())
        self.fitted = True
        
    def transform(self, df: pd.DataFrame, column: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        values = self.scaler.transform(df[[column]].fillna(df[column].mean()))
        return pd.Series(values.flatten(), index=df.index)


class CategoricalEncoder(FeatureTransformer):
    """Categorical feature encoder."""
    
    def __init__(self, name: str, method: str = "onehot"):
        super().__init__(name)
        self.method = method
        self.encoder = None
        self.categories = None
        
    def fit(self, df: pd.DataFrame, column: str):
        if self.method == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            self.encoder = LabelEncoder()
        self.encoder.fit(df[[column]].fillna('missing'))
        self.fitted = True
        
    def transform(self, df: pd.DataFrame, column: str) -> pd.Series:
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        values = self.encoder.transform(df[[column]].fillna('missing'))
        if self.method == "onehot":
            # Return as list for multi-column encoding
            return pd.Series(list(values), index=df.index)
        return pd.Series(values, index=df.index)


class DateTimeExtractor(FeatureTransformer):
    """Extract features from datetime."""
    
    def __init__(self, name: str, components: List[str] = None):
        super().__init__(name)
        self.components = components or ['year', 'month', 'day', 'dayofweek']
        
    def fit(self, df: pd.DataFrame, column: str):
        self.fitted = True
        
    def transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        dt = pd.to_datetime(df[column])
        features = {}
        for comp in self.components:
            if comp == 'year':
                features[f'{column}_year'] = dt.dt.year
            elif comp == 'month':
                features[f'{column}_month'] = dt.dt.month
            elif comp == 'day':
                features[f'{column}_day'] = dt.dt.day
            elif comp == 'dayofweek':
                features[f'{column}_dayofweek'] = dt.dt.dayofweek
            elif comp == 'hour':
                features[f'{column}_hour'] = dt.dt.hour
        return pd.DataFrame(features, index=df.index)


# ============================================
# Feature Store
# ============================================

class FeatureStore:
    """Feature store for online and offline features."""
    
    def __init__(self, db: DatabaseManager, cache: CacheManager, storage: StorageManager):
        self.db = db
        self.cache = cache
        self.storage = storage
        
    def register_feature_set(
        self,
        name: str,
        features: List[FeatureDefinition],
        base_dataset: str,
        base_version: str
    ) -> str:
        """Register a new feature set."""
        version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
        
        record = {
            'feature_set_name': name,
            'version': version,
            'features': [f.dict() for f in features],
            'base_dataset': base_dataset,
            'base_version': base_version,
            'created_at': datetime.utcnow()
        }
        
        self.db.insert('feature_registry', record)
        
        return version
        
    def get_feature_set(self, name: str, version: Optional[str] = None) -> Dict:
        """Get feature set definition."""
        if version:
            query = """
                SELECT * FROM feature_registry 
                WHERE feature_set_name = %s AND version = %s
                ORDER BY created_at DESC LIMIT 1
            """
            result = self.db.fetch_one(query, (name, version))
        else:
            query = """
                SELECT * FROM feature_registry 
                WHERE feature_set_name = %s
                ORDER BY created_at DESC LIMIT 1
            """
            result = self.db.fetch_one(query, (name,))
            
        return result
        
    def get_online_features(
        self,
        feature_set_name: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get online features for entity IDs."""
        # Try cache first
        cache_key = f"features:{feature_set_name}:{hashlib.md5(str(entity_ids).encode()).hexdigest()}"
        cached = self.cache.get(cache_key)
        
        if cached:
            return pd.read_json(cached)
        
        # Load from feature store
        # In production, this would query a real-time feature store
        query = """
            SELECT * FROM online_features 
            WHERE feature_set_name = %s AND entity_id = ANY(%s)
        """
        results = self.db.fetch_many(query, (feature_set_name, entity_ids))
        
        df = pd.DataFrame(results)
        
        if feature_names:
            df = df[['entity_id'] + feature_names]
        
        # Cache results
        self.cache.set(cache_key, df.to_json(), expire=300)
        
        return df
        
    def materialize_features(
        self,
        feature_set_name: str,
        output_path: str
    ) -> str:
        """Materialize features to storage."""
        # Get feature set definition
        feature_set = self.get_feature_set(feature_set_name)
        if not feature_set:
            raise FeatureStoreError(f"Feature set '{feature_set_name}' not found")
        
        # Load base dataset
        bucket_name = "ml-datasets"
        df = self.storage.load_dataset(
            bucket_name,
            feature_set['base_dataset'],
            feature_set['base_version']
        )
        
        # Apply feature engineering
        df_features = self._apply_transformations(df, feature_set['features'])
        
        # Save to feature store
        version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
        self.storage.save_dataset(
            df_features,
            "ml-features",
            feature_set_name,
            version
        )
        
        return version
        
    def _apply_transformations(
        self,
        df: pd.DataFrame,
        feature_definitions: List[Dict]
    ) -> pd.DataFrame:
        """Apply feature transformations."""
        result_df = df.copy()
        
        for feat_def in feature_definitions:
            name = feat_def['name']
            feat_type = feat_def['feature_type']
            transformation = feat_def.get('transformation')
            source_col = feat_def.get('source_column', name)
            
            if feat_type == 'derived' and transformation:
                # Apply custom transformation
                if transformation == 'log':
                    result_df[name] = np.log1p(result_df[source_col])
                elif transformation == 'sqrt':
                    result_df[name] = np.sqrt(result_df[source_col])
                elif transformation == 'square':
                    result_df[name] = result_df[source_col] ** 2
                elif transformation == 'binning':
                    params = feat_def.get('parameters', {})
                    bins = params.get('bins', 10)
                    result_df[name] = pd.qcut(result_df[source_col], q=bins, labels=False, duplicates='drop')
                    
        return result_df


# ============================================
# Helper Functions
# ============================================

async def load_feature_registry():
    """Load feature registry from database."""
    try:
        results = app_state['db'].fetch_many(
            "SELECT * FROM feature_registry ORDER BY created_at DESC"
        )
        for row in results:
            key = f"{row['feature_set_name']}:{row['version']}"
            app_state['feature_registry'][key] = row
        logger.info(f"Loaded {len(results)} feature sets")
    except Exception as e:
        logger.warning(f"Could not load feature registry: {e}")


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
        service="feature-service",
        version="2.0.0",
        checks={
            'database': 'connected' if app_state['db'] else 'disconnected',
            'cache': 'connected' if app_state['cache'] else 'disconnected'
        },
        uptime_seconds=uptime
    )


@app.post("/feature-sets")
async def create_feature_set(request: FeatureSetRequest):
    """
    Register a new feature set.
    
    Creates reusable feature definitions for consistent feature engineering.
    """
    try:
        store = FeatureStore(app_state['db'], app_state['cache'], app_state['storage'])
        version = store.register_feature_set(
            request.feature_set_name,
            request.feature_definitions,
            request.base_dataset,
            request.base_version
        )
        
        return {
            "feature_set_name": request.feature_set_name,
            "version": version,
            "feature_count": len(request.feature_definitions),
            "message": "Feature set registered successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-sets")
async def list_feature_sets():
    """List all registered feature sets."""
    try:
        results = app_state['db'].fetch_many(
            """
            SELECT feature_set_name, version, base_dataset, created_at,
                   jsonb_array_length(features) as feature_count
            FROM feature_registry 
            ORDER BY created_at DESC
            """
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-sets/{feature_set_name}")
async def get_feature_set(feature_set_name: str, version: Optional[str] = None):
    """Get feature set details."""
    try:
        store = FeatureStore(app_state['db'], app_state['cache'], app_state['storage'])
        feature_set = store.get_feature_set(feature_set_name, version)
        
        if not feature_set:
            raise HTTPException(status_code=404, detail="Feature set not found")
        
        return feature_set
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engineer")
async def engineer_features(request: FeatureEngineeringRequest, background_tasks: BackgroundTasks):
    """
    Apply feature engineering to a dataset.
    
    Creates a new dataset with engineered features.
    """
    log = StructuredLog(logger, "feature_engineering")
    log.add_field("feature_set", request.feature_set_name)
    log.add_field("dataset", request.dataset_name)
    
    try:
        store = FeatureStore(app_state['db'], app_state['cache'], app_state['storage'])
        
        # Get feature set
        feature_set = store.get_feature_set(request.feature_set_name)
        if not feature_set:
            raise HTTPException(status_code=404, detail="Feature set not found")
        
        # Load dataset
        bucket_name = "ml-datasets"
        df = app_state['storage'].load_dataset(
            bucket_name,
            request.dataset_name,
            request.dataset_version
        )
        
        log.add_field("input_rows", len(df))
        log.add_field("input_columns", len(df.columns))
        
        # Apply transformations
        df_engineered = store._apply_transformations(df, feature_set['features'])
        
        # Save engineered dataset
        output_version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
        app_state['storage'].save_dataset(
            df_engineered,
            bucket_name,
            request.output_name,
            output_version
        )
        
        # Store metadata
        metadata = {
            'dataset_name': request.output_name,
            'version': output_version,
            'source_dataset': request.dataset_name,
            'source_version': request.dataset_version,
            'feature_set': request.feature_set_name,
            'row_count': len(df_engineered),
            'column_count': len(df_engineered.columns),
            'columns': list(df_engineered.columns),
            'created_at': datetime.utcnow()
        }
        app_state['db'].insert('engineered_datasets', metadata)
        
        log.add_field("output_rows", len(df_engineered))
        log.add_field("output_columns", len(df_engineered.columns))
        log.info("Feature engineering completed")
        
        return {
            "dataset_name": request.output_name,
            "version": output_version,
            "rows": len(df_engineered),
            "columns": list(df_engineered.columns),
            "feature_count": len(df_engineered.columns) - len(df.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Feature engineering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/online-features")
async def get_online_features(request: OnlineFeaturesRequest):
    """
    Retrieve online features for entity IDs.
    
    Used for real-time inference with low latency.
    """
    try:
        store = FeatureStore(app_state['db'], app_state['cache'], app_state['storage'])
        
        df = store.get_online_features(
            request.feature_set_name,
            request.entity_ids,
            request.feature_names
        )
        
        return {
            "feature_set": request.feature_set_name,
            "entity_count": len(request.entity_ids),
            "features": df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{dataset_name}/{version}")
async def get_feature_statistics(dataset_name: str, version: str):
    """Get statistical summary of features."""
    try:
        # Load dataset
        bucket_name = "ml-datasets"
        df = app_state['storage'].load_dataset(bucket_name, dataset_name, version)
        
        statistics = []
        for col in df.columns:
            stats = {
                'feature_name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                stats['mean'] = float(df[col].mean())
                stats['std'] = float(df[col].std())
                stats['min'] = float(df[col].min())
                stats['max'] = float(df[col].max())
                
            statistics.append(stats)
        
        return {
            "dataset_name": dataset_name,
            "version": version,
            "statistics": statistics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/materialize/{feature_set_name}")
async def materialize_feature_set(feature_set_name: str):
    """Materialize features to offline store."""
    try:
        store = FeatureStore(app_state['db'], app_state['cache'], app_state['storage'])
        version = store.materialize_features(feature_set_name, "")
        
        return {
            "feature_set": feature_set_name,
            "materialized_version": version,
            "message": "Features materialized successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)