"""
Custom Exceptions for ML Platform
==================================

Standardized exception hierarchy for the ML platform.
"""

from typing import Any, Dict, Optional


class MLPlatformException(Exception):
    """Base exception for ML Platform."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "MLP-500"
        self.details = details or {}
        self.status_code = status_code
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'status_code': self.status_code
        }


class DataValidationError(MLPlatformException):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, list]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-400-DATA",
            status_code=400,
            details={'validation_errors': validation_errors or {}},
            **kwargs
        )
        self.validation_errors = validation_errors or {}


class ModelNotFoundError(MLPlatformException):
    """Raised when a model is not found."""
    
    def __init__(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        **kwargs
    ):
        message = f"Model '{model_name}'"
        if model_version:
            message += f" version '{model_version}'"
        message += " not found"
        
        super().__init__(
            message=message,
            error_code="MLP-404-MODEL",
            status_code=404,
            details={
                'model_name': model_name,
                'model_version': model_version
            },
            **kwargs
        )
        self.model_name = model_name
        self.model_version = model_version


class FeatureStoreError(MLPlatformException):
    """Raised when feature store operations fail."""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-FEATURE",
            status_code=500,
            details={'feature_name': feature_name},
            **kwargs
        )
        self.feature_name = feature_name


class DriftDetectionError(MLPlatformException):
    """Raised when drift detection fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        drift_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-DRIFT",
            status_code=500,
            details={
                'model_name': model_name,
                'drift_type': drift_type
            },
            **kwargs
        )
        self.model_name = model_name
        self.drift_type = drift_type


class TrainingError(MLPlatformException):
    """Raised when model training fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-TRAIN",
            status_code=500,
            details={
                'model_name': model_name,
                'experiment_id': experiment_id
            },
            **kwargs
        )
        self.model_name = model_name
        self.experiment_id = experiment_id


class InferenceError(MLPlatformException):
    """Raised when model inference fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        prediction_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-INFER",
            status_code=500,
            details={
                'model_name': model_name,
                'prediction_id': prediction_id
            },
            **kwargs
        )
        self.model_name = model_name
        self.prediction_id = prediction_id


class ConfigurationError(MLPlatformException):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-CONFIG",
            status_code=500,
            details={'config_key': config_key},
            **kwargs
        )
        self.config_key = config_key


class StorageError(MLPlatformException):
    """Raised when storage operations fail."""
    
    def __init__(
        self,
        message: str,
        bucket: Optional[str] = None,
        object_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-STORAGE",
            status_code=500,
            details={
                'bucket': bucket,
                'object_key': object_key
            },
            **kwargs
        )
        self.bucket = bucket
        self.object_key = object_key


class DatabaseError(MLPlatformException):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-500-DB",
            status_code=500,
            details={
                'operation': operation,
                'table': table
            },
            **kwargs
        )
        self.operation = operation
        self.table = table


class RateLimitError(MLPlatformException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-429",
            status_code=429,
            details={'retry_after': retry_after},
            **kwargs
        )
        self.retry_after = retry_after


class AuthenticationError(MLPlatformException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="MLP-401",
            status_code=401,
            **kwargs
        )


class AuthorizationError(MLPlatformException):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MLP-403",
            status_code=403,
            details={'required_permission': required_permission},
            **kwargs
        )
        self.required_permission = required_permission


class ServiceUnavailableError(MLPlatformException):
    """Raised when a service is unavailable."""
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        **kwargs
    ):
        message = message or f"Service '{service_name}' is unavailable"
        super().__init__(
            message=message,
            error_code="MLP-503",
            status_code=503,
            details={'service_name': service_name},
            **kwargs
        )
        self.service_name = service_name