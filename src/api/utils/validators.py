#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Validation Utilities
أدوات التحقق من المدخلات
"""

import re
from typing import Any, Optional
from pathlib import Path
from .error_handlers import ValidationException


def validate_file_path(file_path: str, allowed_dirs: Optional[list] = None) -> Path:
    """
    Validate and sanitize file path to prevent path traversal attacks

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed base directories

    Returns:
        Validated Path object

    Raises:
        ValidationException: If path is invalid or dangerous
    """
    try:
        path = Path(file_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationException(f"Invalid file path: {str(e)}")

    # Check for path traversal attempts
    if ".." in str(path):
        raise ValidationException("Path traversal detected")

    # Check if path is in allowed directories
    if allowed_dirs:
        allowed = False
        for allowed_dir in allowed_dirs:
            try:
                path.relative_to(Path(allowed_dir).resolve())
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValidationException(f"Access to path {path} is not allowed")

    return path


def validate_api_key(api_key: str) -> str:
    """
    Validate API key format

    Args:
        api_key: API key to validate

    Returns:
        Validated API key

    Raises:
        ValidationException: If API key format is invalid
    """
    if not api_key:
        raise ValidationException("API key is required")

    if len(api_key) < 16:
        raise ValidationException("API key too short")

    if len(api_key) > 256:
        raise ValidationException("API key too long")

    # Check for valid characters
    if not re.match(r'^[A-Za-z0-9\-_]+$', api_key):
        raise ValidationException("API key contains invalid characters")

    return api_key


def validate_text_input(text: str, max_length: int = 10000, min_length: int = 1) -> str:
    """
    Validate text input

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        min_length: Minimum required length

    Returns:
        Validated text

    Raises:
        ValidationException: If validation fails
    """
    if not text:
        raise ValidationException("Text input is required")

    text = text.strip()

    if len(text) < min_length:
        raise ValidationException(f"Text too short (minimum {min_length} characters)")

    if len(text) > max_length:
        raise ValidationException(f"Text too long (maximum {max_length} characters)")

    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def validate_model_name(model_name: str) -> str:
    """
    Validate model name

    Args:
        model_name: Model name to validate

    Returns:
        Validated model name

    Raises:
        ValidationException: If validation fails
    """
    if not model_name:
        raise ValidationException("Model name is required")

    # Only allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[A-Za-z0-9\-_.]+$', model_name):
        raise ValidationException("Invalid model name format")

    if len(model_name) > 100:
        raise ValidationException("Model name too long")

    return model_name


def sanitize_sql_input(value: Any) -> str:
    """
    Sanitize input for SQL queries
    Note: This is a backup - always use parameterized queries

    Args:
        value: Value to sanitize

    Returns:
        Sanitized string
    """
    if value is None:
        return ""

    # Convert to string and remove dangerous characters
    sanitized = str(value)
    dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")

    return sanitized[:1000]  # Limit length


def validate_json_size(data: dict, max_size_bytes: int = 1048576) -> dict:
    """
    Validate JSON payload size (default 1MB)

    Args:
        data: Dictionary to validate
        max_size_bytes: Maximum size in bytes

    Returns:
        Validated dictionary

    Raises:
        ValidationException: If payload too large
    """
    import json
    size = len(json.dumps(data).encode('utf-8'))

    if size > max_size_bytes:
        raise ValidationException(
            f"Payload too large ({size} bytes, maximum {max_size_bytes})"
        )

    return data


def validate_email(email: str) -> str:
    """
    Validate email address format

    Args:
        email: Email to validate

    Returns:
        Validated email

    Raises:
        ValidationException: If email invalid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email):
        raise ValidationException("Invalid email format")

    if len(email) > 254:
        raise ValidationException("Email too long")

    return email.lower()


def validate_port(port: int) -> int:
    """
    Validate port number

    Args:
        port: Port number to validate

    Returns:
        Validated port

    Raises:
        ValidationException: If port invalid
    """
    if not isinstance(port, int):
        raise ValidationException("Port must be an integer")

    if port < 1 or port > 65535:
        raise ValidationException("Port must be between 1 and 65535")

    return port
