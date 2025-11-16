#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Error Handling System
نظام معالجة الأخطاء المركزي
"""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class NooghException(Exception):
    """Base exception for all Noogh errors"""
    def __init__(self, message: str, code: str = "NOOGH_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(NooghException):
    """Input validation errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class AuthenticationException(NooghException):
    """Authentication errors"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationException(NooghException):
    """Authorization/permission errors"""
    def __init__(self, message: str = "Permission denied", details: Optional[Dict] = None):
        super().__init__(message, "PERMISSION_ERROR", details)


class ResourceNotFoundException(NooghException):
    """Resource not found errors"""
    def __init__(self, resource: str, details: Optional[Dict] = None):
        super().__init__(f"{resource} not found", "NOT_FOUND", details)


class ConfigurationException(NooghException):
    """Configuration errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "CONFIG_ERROR", details)


class ModelException(NooghException):
    """AI model related errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "MODEL_ERROR", details)


def create_error_response(
    error: Exception,
    status_code: int = 500,
    include_details: bool = False
) -> JSONResponse:
    """
    Create standardized error response

    Args:
        error: The exception that occurred
        status_code: HTTP status code
        include_details: Whether to include debug details (only in development)
    """
    error_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    response_data = {
        "error": {
            "id": error_id,
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }

    # Add code if it's a NooghException
    if isinstance(error, NooghException):
        response_data["error"]["code"] = error.code
        if error.details:
            response_data["error"]["details"] = error.details

    # Add debug info in development mode
    if include_details:
        response_data["error"]["traceback"] = traceback.format_exc()

    # Log the error
    logger.error(
        f"Error {error_id}: {type(error).__name__}: {str(error)}",
        exc_info=True
    )

    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def noogh_exception_handler(request: Request, exc: NooghException) -> JSONResponse:
    """Handler for custom Noogh exceptions"""
    status_map = {
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "AUTH_ERROR": status.HTTP_401_UNAUTHORIZED,
        "PERMISSION_ERROR": status.HTTP_403_FORBIDDEN,
        "NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "CONFIG_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "MODEL_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    }

    status_code = status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    return create_error_response(exc, status_code)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler for Pydantic validation errors"""
    details = {
        "errors": exc.errors(),
        "body": exc.body if hasattr(exc, 'body') else None
    }

    validation_exc = ValidationException(
        "Request validation failed",
        details=details
    )

    return create_error_response(validation_exc, status.HTTP_422_UNPROCESSABLE_ENTITY)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected exceptions"""
    logger.critical(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )

    # Don't expose internal errors in production
    safe_message = "An internal error occurred. Please contact support."

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": safe_message,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )


def register_error_handlers(app):
    """Register all error handlers with FastAPI app"""
    app.add_exception_handler(NooghException, noogh_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("✅ Error handlers registered")
