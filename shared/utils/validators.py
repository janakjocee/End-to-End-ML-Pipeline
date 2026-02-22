"""
Data Validation Utilities
==========================

Schema validation and data quality checks.
"""

import re
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, Field

from shared.utils.logger import get_logger

logger = get_logger(__name__)


class DataType(str, Enum):
    """Supported data types."""
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    DATETIME = 'datetime'
    TEXT = 'text'
    BOOLEAN = 'boolean'


@dataclass
class ColumnSchema:
    """Schema definition for a column."""
    name: str
    dtype: DataType
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List] = None
    regex_pattern: Optional[str] = None


class DataValidator:
    """
    Comprehensive data validator for ML datasets.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        
    def add_rule(self, column: str, rule: Callable):
        """Add a custom validation rule."""
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        self.validation_rules[column].append(rule)
        
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnSchema]
    ) -> Dict[str, List[str]]:
        """
        Validate DataFrame against schema.
        
        Returns:
            Dictionary of column names to list of error messages
        """
        errors = {}
        
        # Check required columns
        required_columns = set(schema.keys())
        actual_columns = set(df.columns)
        missing_columns = required_columns - actual_columns
        
        if missing_columns:
            errors['__missing_columns__'] = list(missing_columns)
            
        # Validate each column
        for col_name, col_schema in schema.items():
            if col_name not in df.columns:
                continue
                
            col_errors = self._validate_column(df[col_name], col_schema)
            if col_errors:
                errors[col_name] = col_errors
                
        # Apply custom rules
        for col, rules in self.validation_rules.items():
            if col in df.columns:
                for rule in rules:
                    try:
                        rule(df[col])
                    except Exception as e:
                        if col not in errors:
                            errors[col] = []
                        errors[col].append(str(e))
                        
        return errors
        
    def _validate_column(
        self,
        series: pd.Series,
        schema: ColumnSchema
    ) -> List[str]:
        """Validate a single column."""
        errors = []
        
        # Check nullability
        if not schema.nullable and series.isnull().any():
            errors.append(f"Column contains null values but is not nullable")
            
        # Check uniqueness
        if schema.unique and series.nunique() != len(series):
            errors.append(f"Column contains duplicate values but should be unique")
            
        # Check data type
        if schema.dtype == DataType.NUMERIC:
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"Expected numeric type, got {series.dtype}")
            else:
                # Check range
                if schema.min_value is not None and series.min() < schema.min_value:
                    errors.append(f"Values below minimum {schema.min_value}")
                if schema.max_value is not None and series.max() > schema.max_value:
                    errors.append(f"Values above maximum {schema.max_value}")
                    
        elif schema.dtype == DataType.CATEGORICAL:
            if schema.allowed_values is not None:
                invalid_values = set(series.dropna().unique()) - set(schema.allowed_values)
                if invalid_values:
                    errors.append(f"Invalid values: {invalid_values}")
                    
        elif schema.dtype == DataType.DATETIME:
            if not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(f"Expected datetime type, got {series.dtype}")
                
        elif schema.dtype == DataType.TEXT:
            if schema.regex_pattern is not None:
                pattern = re.compile(schema.regex_pattern)
                invalid_mask = series.dropna().apply(lambda x: not pattern.match(str(x)))
                if invalid_mask.any():
                    errors.append(f"Some values don't match pattern: {schema.regex_pattern}")
                    
        return errors
        
    def validate_statistics(
        self,
        df: pd.DataFrame,
        expected_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Validate statistical properties of data.
        
        Args:
            df: DataFrame to validate
            expected_stats: Dictionary of column to expected statistics
                e.g., {'age': {'mean': (20, 80), 'std': (0, 30)}}
        """
        errors = {}
        
        for col, stats in expected_stats.items():
            if col not in df.columns:
                errors[col] = [f"Column not found"]
                continue
                
            col_errors = []
            series = df[col]
            
            for stat_name, expected in stats.items():
                actual = getattr(series, stat_name)()
                
                if isinstance(expected, tuple):
                    min_val, max_val = expected
                    if actual < min_val or actual > max_val:
                        col_errors.append(
                            f"{stat_name}={actual:.2f} not in range [{min_val}, {max_val}]"
                        )
                else:
                    if actual != expected:
                        col_errors.append(
                            f"{stat_name}={actual:.2f} != expected {expected}"
                        )
                        
            if col_errors:
                errors[col] = col_errors
                
        return errors
        
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        zscore_threshold: float = 3.0
    ) -> Dict[str, pd.Series]:
        """
        Detect anomalies using Z-score method.
        
        Returns:
            Dictionary of column names to boolean series indicating anomalies
        """
        anomalies = {}
        
        numeric_cols = numeric_columns or df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                anomalies[col] = z_scores > zscore_threshold
                
        return anomalies
        
    def generate_validation_report(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnSchema]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        schema_errors = self.validate_schema(df, schema)
        
        report = {
            'valid': len(schema_errors) == 0,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'schema_errors': schema_errors,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return report


class PredictionValidator:
    """Validator for prediction inputs and outputs."""
    
    def __init__(self, feature_schema: Dict[str, Any]):
        self.feature_schema = feature_schema
        
    def validate_input(self, features: Dict[str, Any]) -> List[str]:
        """Validate prediction input features."""
        errors = []
        
        # Check required features
        required = set(self.feature_schema.get('required', []))
        provided = set(features.keys())
        missing = required - provided
        
        if missing:
            errors.append(f"Missing required features: {missing}")
            
        # Validate feature types
        for feature, value in features.items():
            if feature in self.feature_schema.get('properties', {}):
                expected_type = self.feature_schema['properties'][feature].get('type')
                
                if expected_type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Feature '{feature}' should be numeric")
                elif expected_type == 'string' and not isinstance(value, str):
                    errors.append(f"Feature '{feature}' should be string")
                elif expected_type == 'integer' and not isinstance(value, int):
                    errors.append(f"Feature '{feature}' should be integer")
                    
        return errors
        
    def validate_output(self, prediction: Dict[str, Any]) -> List[str]:
        """Validate prediction output."""
        errors = []
        
        if 'prediction' not in prediction:
            errors.append("Missing 'prediction' field in output")
            
        if 'probability' in prediction:
            prob = prediction['probability']
            if not (0 <= prob <= 1):
                errors.append(f"Probability {prob} not in valid range [0, 1]")
                
        return errors


# Pydantic models for API validation
class FeatureValidationModel(BaseModel):
    """Base model for feature validation."""
    
    @validator('*', pre=True)
    def check_nulls(cls, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raise ValueError("Null values not allowed")
        return v


class ChurnPredictionFeatures(BaseModel):
    """Example feature schema for churn prediction."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure: int = Field(..., ge=0, le=100, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, le=1000, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract: str = Field(..., regex="^(Month-to-month|One year|Two year)$")
    payment_method: str = Field(..., regex="^(Electronic check|Mailed check|Bank transfer|Credit card)$")
    internet_service: str = Field(..., regex="^(DSL|Fiber optic|No)$")
    online_security: str = Field(..., regex="^(Yes|No|No internet service)$")
    tech_support: str = Field(..., regex="^(Yes|No|No internet service)$")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "tenure": 24,
                "monthly_charges": 65.5,
                "total_charges": 1567.2,
                "contract": "One year",
                "payment_method": "Electronic check",
                "internet_service": "DSL",
                "online_security": "Yes",
                "tech_support": "No"
            }
        }