"""
Data Ingestion Service
======================

Microservice for data ingestion, validation, and versioning.
Supports multiple data sources: CSV, databases, APIs.
"""

import os
import uuid
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.utils.logger import get_logger, StructuredLog
from shared.utils.database import DatabaseManager, MongoManager
from shared.utils.storage import StorageManager
from shared.utils.validators import DataValidator, ColumnSchema, DataType
from shared.models.schemas import HealthCheckResponse
from shared.exceptions import DataValidationError, StorageError

# Configure logging
logger = get_logger("data-service")

# Global state
app_state = {
    'db': None,
    'mongo': None,
    'storage': None,
    'validator': None,
    'start_time': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Data Ingestion Service...")
    app_state['start_time'] = datetime.utcnow()
    
    # Initialize database connections
    app_state['db'] = DatabaseManager()
    app_state['mongo'] = MongoManager()
    app_state['storage'] = StorageManager()
    app_state['validator'] = DataValidator()
    
    # Create tables
    try:
        app_state['db'].create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Data Ingestion Service...")


app = FastAPI(
    title="Data Ingestion Service",
    description="Enterprise-grade data ingestion and validation service",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
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

class DataSourceConfig(BaseModel):
    """Data source configuration."""
    source_type: str = Field(..., regex="^(csv|database|api|parquet)$")
    source_path: str
    connection_string: Optional[str] = None
    query: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    delimiter: str = ","
    encoding: str = "utf-8"


class IngestionRequest(BaseModel):
    """Data ingestion request."""
    dataset_name: str
    config: DataSourceConfig
    validation_schema: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None


class IngestionResponse(BaseModel):
    """Data ingestion response."""
    ingestion_id: str
    dataset_name: str
    version: str
    rows_ingested: int
    columns: List[str]
    validation_passed: bool
    validation_errors: Optional[List[str]] = None
    storage_path: str
    checksum: str
    timestamp: datetime


class DatasetInfo(BaseModel):
    """Dataset metadata."""
    dataset_name: str
    version: str
    row_count: int
    column_count: int
    columns: List[str]
    size_bytes: int
    created_at: datetime
    tags: Dict[str, str]
    checksum: str


class ValidationSchemaRequest(BaseModel):
    """Validation schema request."""
    schema_name: str
    columns: Dict[str, ColumnSchema]
    strict: bool = True


# ============================================
# Helper Functions
# ============================================

def calculate_checksum(df: pd.DataFrame) -> str:
    """Calculate checksum for a DataFrame."""
    data_string = df.to_csv(index=False)
    return hashlib.sha256(data_string.encode()).hexdigest()[:16]


def generate_version() -> str:
    """Generate dataset version."""
    return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer schema from DataFrame."""
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            data_type = DataType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            data_type = DataType.DATETIME
        else:
            data_type = DataType.CATEGORICAL
            
        schema[col] = {
            'dtype': data_type.value,
            'nullable': df[col].isnull().any(),
            'unique': df[col].nunique() == len(df)
        }
        
        if data_type == DataType.NUMERIC:
            schema[col]['min_value'] = float(df[col].min())
            schema[col]['max_value'] = float(df[col].max())
        elif data_type == DataType.CATEGORICAL:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:
                schema[col]['allowed_values'] = unique_vals.tolist()
                
    return schema


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
        'storage': 'connected' if app_state['storage'] else 'disconnected'
    }
    
    return HealthCheckResponse(
        status="healthy" if all(c == 'connected' for c in checks.values()) else "degraded",
        service="data-service",
        version="2.0.0",
        checks=checks,
        uptime_seconds=uptime
    )


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_data(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Ingest data from various sources.
    
    Supports CSV, Parquet, database, and API sources.
    """
    log = StructuredLog(logger, "data_ingestion")
    log.add_field("dataset_name", request.dataset_name)
    log.add_field("source_type", request.config.source_type)
    
    ingestion_id = str(uuid.uuid4())
    version = generate_version()
    
    try:
        # Load data based on source type
        if request.config.source_type == 'csv':
            df = pd.read_csv(
                request.config.source_path,
                delimiter=request.config.delimiter,
                encoding=request.config.encoding
            )
        elif request.config.source_type == 'parquet':
            df = pd.read_parquet(request.config.source_path)
        elif request.config.source_type == 'database':
            df = pd.read_sql(
                request.config.query,
                request.config.connection_string
            )
        elif request.config.source_type == 'api':
            import requests
            response = requests.get(
                request.config.source_path,
                headers=request.config.headers
            )
            df = pd.DataFrame(response.json())
        else:
            raise ValueError(f"Unsupported source type: {request.config.source_type}")
        
        log.add_field("rows_loaded", len(df))
        log.add_field("columns", list(df.columns))
        
        # Validate data if schema provided
        validation_errors = []
        validation_passed = True
        
        if request.validation_schema:
            schema = {
                name: ColumnSchema(**col_schema)
                for name, col_schema in request.validation_schema.items()
            }
            errors = app_state['validator'].validate_schema(df, schema)
            if errors:
                validation_passed = False
                validation_errors = [f"{col}: {errs}" for col, errs in errors.items()]
                log.add_field("validation_errors", validation_errors)
        
        # Calculate checksum
        checksum = calculate_checksum(df)
        
        # Save to storage
        bucket_name = "ml-datasets"
        storage_path = f"datasets/{request.dataset_name}/{version}/data.parquet"
        
        # Ensure bucket exists
        if not app_state['storage'].bucket_exists(bucket_name):
            app_state['storage'].create_bucket(bucket_name)
        
        # Upload dataset
        app_state['storage'].save_dataset(
            df, bucket_name, request.dataset_name, version, format='parquet'
        )
        
        # Store metadata in database
        metadata = {
            'ingestion_id': ingestion_id,
            'dataset_name': request.dataset_name,
            'version': version,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'size_bytes': df.memory_usage(deep=True).sum(),
            'checksum': checksum,
            'source_type': request.config.source_type,
            'source_path': request.config.source_path,
            'validation_passed': validation_passed,
            'tags': request.tags or {},
            'created_at': datetime.utcnow()
        }
        
        app_state['db'].insert('dataset_metadata', metadata)
        
        # Infer and store schema
        inferred_schema = infer_schema(df)
        schema_record = {
            'dataset_name': request.dataset_name,
            'version': version,
            'schema': inferred_schema,
            'created_at': datetime.utcnow()
        }
        app_state['db'].insert('dataset_schemas', schema_record)
        
        log.info(f"Successfully ingested {len(df)} rows")
        
        return IngestionResponse(
            ingestion_id=ingestion_id,
            dataset_name=request.dataset_name,
            version=version,
            rows_ingested=len(df),
            columns=list(df.columns),
            validation_passed=validation_passed,
            validation_errors=validation_errors if validation_errors else None,
            storage_path=storage_path,
            checksum=checksum,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        log.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    dataset_name: str,
    file: UploadFile = File(...),
    validation_schema: Optional[str] = None
):
    """
    Upload a file directly.
    
    Supports CSV and Parquet formats.
    """
    ingestion_id = str(uuid.uuid4())
    version = generate_version()
    
    try:
        # Read file
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(pd.io.common.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save to storage
        bucket_name = "ml-datasets"
        if not app_state['storage'].bucket_exists(bucket_name):
            app_state['storage'].create_bucket(bucket_name)
        
        app_state['storage'].save_dataset(
            df, bucket_name, dataset_name, version, format='parquet'
        )
        
        checksum = calculate_checksum(df)
        
        return {
            "ingestion_id": ingestion_id,
            "dataset_name": dataset_name,
            "version": version,
            "rows_ingested": len(df),
            "columns": list(df.columns),
            "checksum": checksum
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets(
    dataset_name: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all datasets with metadata."""
    try:
        query = "SELECT * FROM dataset_metadata ORDER BY created_at DESC LIMIT %s"
        params = (limit,)
        
        if dataset_name:
            query = "SELECT * FROM dataset_metadata WHERE dataset_name = %s ORDER BY created_at DESC LIMIT %s"
            params = (dataset_name, limit)
        
        results = app_state['db'].fetch_many(query, params)
        
        datasets = []
        for row in results:
            datasets.append(DatasetInfo(
                dataset_name=row['dataset_name'],
                version=row['version'],
                row_count=row['row_count'],
                column_count=row['column_count'],
                columns=row['columns'],
                size_bytes=row['size_bytes'],
                created_at=row['created_at'],
                tags=row['tags'],
                checksum=row['checksum']
            ))
        
        return datasets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{dataset_name}/{version}")
async def get_dataset(dataset_name: str, version: str):
    """Get dataset metadata and download URL."""
    try:
        query = """
            SELECT * FROM dataset_metadata 
            WHERE dataset_name = %s AND version = %s
        """
        result = app_state['db'].fetch_one(query, (dataset_name, version))
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Generate presigned URL
        bucket_name = "ml-datasets"
        object_key = f"datasets/{dataset_name}/{version}/data.parquet"
        
        return {
            "metadata": result,
            "download_url": f"/download/{dataset_name}/{version}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{dataset_name}/{version}")
async def download_dataset(dataset_name: str, version: str):
    """Download dataset file."""
    try:
        from fastapi.responses import StreamingResponse
        import io
        
        bucket_name = "ml-datasets"
        object_key = f"datasets/{dataset_name}/{version}/data.parquet"
        
        data = app_state['storage'].download_object(bucket_name, object_key)
        
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={dataset_name}_{version}.parquet"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/{dataset_name}/{version}")
async def get_schema(dataset_name: str, version: str):
    """Get dataset schema."""
    try:
        query = """
            SELECT schema FROM dataset_schemas 
            WHERE dataset_name = %s AND version = %s
        """
        result = app_state['db'].fetch_one(query, (dataset_name, version))
        
        if not result:
            raise HTTPException(status_code=404, detail="Schema not found")
        
        return result['schema']
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate_data(
    dataset_name: str,
    version: str,
    validation_rules: Optional[Dict[str, Any]] = None
):
    """
    Validate existing dataset against rules.
    """
    try:
        # Load dataset
        bucket_name = "ml-datasets"
        df = app_state['storage'].load_dataset(bucket_name, dataset_name, version)
        
        # Run validation
        report = app_state['validator'].generate_validation_report(
            df,
            {}  # Would use actual schema
        )
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{dataset_name}/{version}")
async def delete_dataset(dataset_name: str, version: str):
    """Delete a dataset."""
    try:
        # Delete from storage
        bucket_name = "ml-datasets"
        object_key = f"datasets/{dataset_name}/{version}/data.parquet"
        app_state['storage'].delete_object(bucket_name, object_key)
        
        # Delete metadata
        app_state['db'].execute(
            "DELETE FROM dataset_metadata WHERE dataset_name = %s AND version = %s",
            (dataset_name, version)
        )
        
        return {"message": "Dataset deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)