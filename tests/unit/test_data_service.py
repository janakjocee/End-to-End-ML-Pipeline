"""
Unit Tests for Data Service
============================
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from shared.utils.validators import DataValidator, ColumnSchema, DataType
from shared.utils.database import DatabaseManager


class TestDataValidator:
    """Test data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'score': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
    
    def test_validate_schema_valid(self, validator, sample_df):
        """Test schema validation with valid data."""
        schema = {
            'age': ColumnSchema(name='age', dtype=DataType.NUMERIC, nullable=False, min_value=0, max_value=120),
            'income': ColumnSchema(name='income', dtype=DataType.NUMERIC, nullable=False, min_value=0),
            'category': ColumnSchema(name='category', dtype=DataType.CATEGORICAL, allowed_values=['A', 'B', 'C'])
        }
        
        errors = validator.validate_schema(sample_df, schema)
        assert len(errors) == 0
    
    def test_validate_schema_invalid_values(self, validator):
        """Test schema validation with invalid values."""
        df = pd.DataFrame({
            'age': [-5, 150, 25],  # Out of range
            'category': ['A', 'X', 'B']  # Invalid category
        })
        
        schema = {
            'age': ColumnSchema(name='age', dtype=DataType.NUMERIC, min_value=0, max_value=120),
            'category': ColumnSchema(name='category', dtype=DataType.CATEGORICAL, allowed_values=['A', 'B', 'C'])
        }
        
        errors = validator.validate_schema(df, schema)
        assert 'age' in errors
        assert 'category' in errors
    
    def test_validate_schema_missing_column(self, validator, sample_df):
        """Test schema validation with missing column."""
        schema = {
            'age': ColumnSchema(name='age', dtype=DataType.NUMERIC),
            'missing_col': ColumnSchema(name='missing_col', dtype=DataType.NUMERIC)
        }
        
        errors = validator.validate_schema(sample_df, schema)
        assert '__missing_columns__' in errors
    
    def test_detect_anomalies(self, validator):
        """Test anomaly detection."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        anomalies = validator.detect_anomalies(df, zscore_threshold=2.0)
        
        assert 'value' in anomalies
        assert anomalies['value'].iloc[-1]  # Last value should be flagged
    
    def test_generate_validation_report(self, validator, sample_df):
        """Test validation report generation."""
        schema = {
            'age': ColumnSchema(name='age', dtype=DataType.NUMERIC)
        }
        
        report = validator.generate_validation_report(sample_df, schema)
        
        assert report['valid'] is True
        assert report['total_rows'] == 5
        assert 'age' in [s['feature_name'] for s in report.get('statistics', [])]


class TestDatabaseManager:
    """Test database manager functionality."""
    
    @pytest.fixture
    @patch('shared.utils.database.psycopg2.connect')
    def db_manager(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        return DatabaseManager("postgresql://test:test@localhost/test")
    
    def test_insert(self, db_manager):
        """Test insert operation."""
        with patch.object(db_manager, 'get_cursor') as mock_cursor:
            mock_cursor.return_value.__enter__.return_value.fetchone.return_value = {'id': 1}
            
            result = db_manager.insert('test_table', {'name': 'test', 'value': 123})
            
            assert result == 1


class TestStorageManager:
    """Test storage manager functionality."""
    
    @pytest.fixture
    @patch('shared.utils.storage.boto3.client')
    def storage_manager(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        return StorageManager(
            endpoint_url='http://localhost:9000',
            access_key='test',
            secret_key='test'
        )
    
    def test_bucket_exists(self, storage_manager):
        """Test bucket existence check."""
        storage_manager._get_client().head_bucket.return_value = None
        
        result = storage_manager.bucket_exists('test-bucket')
        
        assert result is True
    
    def test_create_bucket(self, storage_manager):
        """Test bucket creation."""
        result = storage_manager.create_bucket('new-bucket')
        
        assert result is True
        storage_manager._get_client().create_bucket.assert_called_with(Bucket='new-bucket')