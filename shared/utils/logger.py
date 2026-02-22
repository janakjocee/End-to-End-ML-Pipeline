"""
Logging Utility for ML Platform
================================

Centralized logging configuration with structured JSON logging support.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = getattr(record, 'service', 'mlplatform')
        log_record['environment'] = getattr(record, 'environment', 'development')
        
        # Add level
        log_record['level'] = record.levelname
        
        # Add source location
        log_record['source'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName
        }


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    service_name: str = "mlplatform"
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON formatting
        service_name: Name of the service for identification
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if json_format:
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(service_name: str = "mlplatform") -> logging.Logger:
    """
    Get or create a logger instance for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Logger instance
    """
    return logging.getLogger(service_name)


class LoggerContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.original_extra = {}
        
    def __enter__(self):
        for key, value in self.context.items():
            self.original_extra[key] = getattr(self.logger, key, None)
            setattr(self.logger, key, value)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_extra.items():
            if value is None:
                delattr(self.logger, key)
            else:
                setattr(self.logger, key, value)


class StructuredLog:
    """Helper class for structured logging."""
    
    def __init__(self, logger: logging.Logger, event_type: str):
        self.logger = logger
        self.event_type = event_type
        self.start_time = datetime.utcnow()
        self.data = {}
        
    def add_field(self, key: str, value: Any) -> 'StructuredLog':
        """Add a field to the log entry."""
        self.data[key] = value
        return self
        
    def add_fields(self, fields: Dict[str, Any]) -> 'StructuredLog':
        """Add multiple fields to the log entry."""
        self.data.update(fields)
        return self
        
    def info(self, message: str):
        """Log as INFO level."""
        self._log(logging.INFO, message)
        
    def warning(self, message: str):
        """Log as WARNING level."""
        self._log(logging.WARNING, message)
        
    def error(self, message: str):
        """Log as ERROR level."""
        self._log(logging.ERROR, message)
        
    def _log(self, level: int, message: str):
        """Internal log method."""
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        log_entry = {
            'event_type': self.event_type,
            'message': message,
            'duration_seconds': duration,
            'data': self.data
        }
        
        self.logger.log(level, json.dumps(log_entry))