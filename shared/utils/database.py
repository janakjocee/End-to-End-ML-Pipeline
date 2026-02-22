"""
Database Management Utilities
==============================

Unified database interface for PostgreSQL and MongoDB.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import redis
from redis import Redis

from shared.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    PostgreSQL database manager with connection pooling.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL',
            'postgresql://mlplatform:mlplatform_secret@localhost:5432/mlplatform'
        )
        self._connection = None
        
    def connect(self) -> psycopg2.extensions.connection:
        """Create database connection."""
        try:
            self._connection = psycopg2.connect(self.connection_string)
            logger.info("Connected to PostgreSQL database")
            return self._connection
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
            
    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor) -> Generator:
        """Get database cursor with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor(cursor_factory=cursor_factory)
            yield cursor
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                
    def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
            
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a query with multiple parameter sets."""
        with self.get_cursor() as cursor:
            execute_values(cursor, query, params_list)
            return cursor.rowcount
            
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a single record."""
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ', '.join(['%s'] * len(columns))
        
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({placeholders})
            RETURNING id
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            result = cursor.fetchone()
            return result['id'] if result else None
            
    def insert_many(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Insert multiple records."""
        if not records:
            return 0
            
        columns = list(records[0].keys())
        values = [[record.get(col) for col in columns] for record in records]
        
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES %s
        """
        
        with self.get_cursor() as cursor:
            execute_values(cursor, query, values)
            return cursor.rowcount
            
    def update(self, table: str, data: Dict[str, Any], where_clause: str, where_params: tuple) -> int:
        """Update records."""
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        values = list(data.values()) + list(where_params)
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            return cursor.rowcount
            
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """Fetch a single record."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
            
    def fetch_many(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Fetch multiple records."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
            
    def create_tables(self):
        """Create required database tables."""
        ddl_statements = """
        -- Model Registry Table
        CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            model_stage VARCHAR(50) DEFAULT 'None',
            artifact_uri TEXT,
            metrics JSONB,
            parameters JSONB,
            tags JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, model_version)
        );
        
        -- Feature Registry Table
        CREATE TABLE IF NOT EXISTS feature_registry (
            id SERIAL PRIMARY KEY,
            feature_name VARCHAR(255) NOT NULL UNIQUE,
            feature_type VARCHAR(50) NOT NULL,
            feature_description TEXT,
            feature_schema JSONB,
            source_table VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Training Runs Table
        CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(255) NOT NULL UNIQUE,
            experiment_name VARCHAR(255),
            model_name VARCHAR(255),
            model_type VARCHAR(100),
            status VARCHAR(50),
            metrics JSONB,
            parameters JSONB,
            artifact_path TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Drift Reports Table
        CREATE TABLE IF NOT EXISTS drift_reports (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255),
            model_version VARCHAR(50),
            drift_type VARCHAR(50),
            drift_score FLOAT,
            drift_detected BOOLEAN,
            features_drifted JSONB,
            report_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Prediction Logs Table (metadata only, actual logs in MongoDB)
        CREATE TABLE IF NOT EXISTS prediction_metadata (
            id SERIAL PRIMARY KEY,
            prediction_id VARCHAR(255) NOT NULL UNIQUE,
            model_name VARCHAR(255),
            model_version VARCHAR(255),
            prediction_count INTEGER,
            latency_ms FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Audit Logs Table
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            action VARCHAR(100) NOT NULL,
            entity_type VARCHAR(100),
            entity_id VARCHAR(255),
            user_id VARCHAR(255),
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name);
        CREATE INDEX IF NOT EXISTS idx_model_registry_stage ON model_registry(model_stage);
        CREATE INDEX IF NOT EXISTS idx_training_runs_experiment ON training_runs(experiment_name);
        CREATE INDEX IF NOT EXISTS idx_drift_reports_model ON drift_reports(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_drift_reports_created ON drift_reports(created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(ddl_statements)
            logger.info("Database tables created successfully")


class MongoManager:
    """
    MongoDB manager for prediction logs and unstructured data.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            'MONGO_URL',
            'mongodb://mlplatform:mlplatform_secret@localhost:27017'
        )
        self.client = None
        self.db = None
        
    def connect(self, database: str = 'prediction_logs') -> Database:
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[database]
            logger.info(f"Connected to MongoDB database: {database}")
            return self.db
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
            
    def get_collection(self, name: str) -> Collection:
        """Get a collection instance."""
        if self.db is None:
            self.connect()
        return self.db[name]
        
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Insert a prediction log."""
        collection = self.get_collection('predictions')
        prediction_data['created_at'] = datetime.utcnow()
        result = collection.insert_one(prediction_data)
        return str(result.inserted_id)
        
    def insert_predictions_batch(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple prediction logs."""
        collection = self.get_collection('predictions')
        now = datetime.utcnow()
        for pred in predictions:
            pred['created_at'] = now
        result = collection.insert_many(predictions)
        return [str(id) for id in result.inserted_ids]
        
    def get_predictions(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Query prediction logs."""
        collection = self.get_collection('predictions')
        
        query = {}
        if model_name:
            query['model_name'] = model_name
        if start_date or end_date:
            query['created_at'] = {}
            if start_date:
                query['created_at']['$gte'] = start_date
            if end_date:
                query['created_at']['$lte'] = end_date
                
        cursor = collection.find(query).sort('created_at', DESCENDING).limit(limit)
        return list(cursor)
        
    def create_indexes(self):
        """Create MongoDB indexes."""
        predictions = self.get_collection('predictions')
        predictions.create_index([('model_name', ASCENDING)])
        predictions.create_index([('created_at', DESCENDING)])
        predictions.create_index([('prediction_id', ASCENDING)], unique=True)
        
        logger.info("MongoDB indexes created successfully")


class CacheManager:
    """
    Redis cache manager for caching predictions and features.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.client: Optional[Redis] = None
        
    def connect(self) -> Redis:
        """Connect to Redis."""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Connected to Redis cache")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
            
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self.client is None:
            self.connect()
        return self.client.get(key)
        
    def set(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in cache with expiration."""
        if self.client is None:
            self.connect()
        return self.client.setex(key, expire, value)
        
    def delete(self, key: str) -> int:
        """Delete value from cache."""
        if self.client is None:
            self.connect()
        return self.client.delete(key)
        
    def get_prediction_cache(self, model_name: str, features_hash: str) -> Optional[Dict]:
        """Get cached prediction."""
        key = f"pred:{model_name}:{features_hash}"
        cached = self.get(key)
        if cached:
            import json
            return json.loads(cached)
        return None
        
    def set_prediction_cache(
        self,
        model_name: str,
        features_hash: str,
        prediction: Dict,
        expire: int = 300
    ) -> bool:
        """Cache prediction result."""
        import json
        key = f"pred:{model_name}:{features_hash}"
        return self.set(key, json.dumps(prediction), expire)