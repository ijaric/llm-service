"""Logging utilities for LLM service."""

import functools
import json
import logging
import time
import typing
import uuid

import pydantic

logger = logging.getLogger(__name__)

def mask_sensitive_data(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Mask sensitive data in logs."""
    SENSITIVE_FIELDS = {'api_key', 'key', 'token', 'password', 'secret'}
    
    def _mask_dict(d: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return {
            k: '***' if k.lower() in SENSITIVE_FIELDS else 
               _mask_dict(v) if isinstance(v, dict) else v
            for k, v in d.items()
        }
    
    return _mask_dict(data)

class RequestLogger:
    """Context manager for logging requests and responses."""
    
    def __init__(
        self,
        provider: str,
        operation: str,
        request_id: str | None = None,
        log_request: bool = True,
        log_response: bool = True,
        mask_sensitive: bool = True
    ):
        self.provider = provider
        self.operation = operation
        self.request_id = request_id or str(uuid.uuid4())
        self.log_request = log_request
        self.log_response = log_response
        self.mask_sensitive = mask_sensitive
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.monotonic()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.monotonic() - self.start_time
        
        if exc_val:
            logger.error(
                "Request failed",
                extra={
                    "request_id": self.request_id,
                    "provider": self.provider,
                    "operation": self.operation,
                    "duration": duration,
                    "error": str(exc_val),
                    "error_type": exc_type.__name__ if exc_type else None
                }
            )
        else:
            logger.info(
                "Request completed",
                extra={
                    "request_id": self.request_id,
                    "provider": self.provider,
                    "operation": self.operation,
                    "duration": duration
                }
            )

def log_request(
    request: typing.Any,
    provider: str,
    operation: str,
    request_id: str | None = None,
    mask_sensitive: bool = True
) -> None:
    """Log request details."""
    if isinstance(request, pydantic.BaseModel):
        request_data = request.model_dump()
    else:
        request_data = request
        
    if mask_sensitive:
        request_data = mask_sensitive_data(request_data)
        
    logger.info(
        "Request details",
        extra={
            "request_id": request_id,
            "provider": provider,
            "operation": operation,
            "request": json.dumps(request_data)
        }
    )

def log_response(
    response: typing.Any,
    provider: str,
    operation: str,
    request_id: str | None = None,
    mask_sensitive: bool = True
) -> None:
    """Log response details."""
    if isinstance(response, pydantic.BaseModel):
        response_data = response.model_dump()
    else:
        response_data = response
        
    if mask_sensitive:
        response_data = mask_sensitive_data(response_data)
        
    logger.info(
        "Response details",
        extra={
            "request_id": request_id,
            "provider": provider,
            "operation": operation,
            "response": json.dumps(response_data)
        }
    )

def log_operation(
    provider: str,
    operation: str,
    log_request: bool = True,
    log_response: bool = True,
    mask_sensitive: bool = True
) -> typing.Callable:
    """Decorator for logging operations."""
    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> typing.Any:
            request_id = str(uuid.uuid4())
            
            if log_request and kwargs.get('request'):
                log_request(
                    kwargs['request'],
                    provider,
                    operation,
                    request_id,
                    mask_sensitive
                )
            
            with RequestLogger(
                provider,
                operation,
                request_id,
                log_request,
                log_response,
                mask_sensitive
            ):
                response = await func(*args, **kwargs)
                
                if log_response and response:
                    log_response(
                        response,
                        provider,
                        operation,
                        request_id,
                        mask_sensitive
                    )
                
                return response
        
        return wrapper
    return decorator