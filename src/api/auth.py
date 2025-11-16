#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authentication & Authorization
Secure API key management using environment variables
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional, Dict
from enum import Enum
from pydantic import BaseModel
import os
import json
import logging

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Permission levels"""
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    GOVERNMENT_READ = "government:read"
    GOVERNMENT_WRITE = "government:write"
    BRAIN_READ = "brain:read"
    BRAIN_WRITE = "brain:write"
    MONITORING_READ = "monitoring:read"
    MONITORING_WRITE = "monitoring:write"
    TRADING_READ = "trading:read"
    TRADING_WRITE = "trading:write"
    TRADING_EXECUTE = "trading:execute"
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DELETE = "model:delete"
    MODEL_DEPLOY = "model:deploy"


class User(BaseModel):
    """User model"""
    username: str
    permissions: list[Permission] = []
    is_admin: bool = False


# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def load_api_keys() -> Dict[str, User]:
    """
    Load API keys securely from environment variables or config file

    Priority:
    1. Environment variables (NOOGH_API_KEYS_JSON)
    2. Secure config file (api_keys.json)
    3. Default development keys (only if NOOGH_ENV=development)
    """
    api_keys = {}

    # Try environment variable first (JSON format)
    env_keys = os.getenv('NOOGH_API_KEYS_JSON')
    if env_keys:
        try:
            keys_data = json.loads(env_keys)
            for key, user_data in keys_data.items():
                api_keys[key] = User(**user_data)
            logger.info("✅ Loaded API keys from environment variable")
            return api_keys
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse API keys from env: {e}")

    # Try secure config file
    config_path = os.getenv('NOOGH_API_KEYS_FILE', '/etc/noogh/api_keys.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                keys_data = json.load(f)
                for key, user_data in keys_data.items():
                    api_keys[key] = User(**user_data)
            logger.info(f"✅ Loaded API keys from {config_path}")
            return api_keys
        except Exception as e:
            logger.error(f"❌ Failed to load API keys from file: {e}")

    # Development mode only - use default keys with warning
    if os.getenv('NOOGH_ENV') == 'development':
        logger.warning("⚠️  Using default development API keys - NOT FOR PRODUCTION")
        return {
            "dev-test-key": User(
                username="dev_user",
                permissions=[Permission.SYSTEM_READ, Permission.BRAIN_READ]
            ),
        }

    # No keys loaded - system will require configuration
    logger.error("❌ No API keys configured. Set NOOGH_API_KEYS_JSON or NOOGH_API_KEYS_FILE")
    return {}


# Load API keys on module import
VALID_API_KEYS = load_api_keys()

async def get_current_user(api_key: Optional[str] = Depends(api_key_header)) -> User:
    """
    Get current user from API key with proper validation
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return VALID_API_KEYS[api_key]


def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        if user.is_admin or permission in user.permissions:
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required"
        )
    return permission_checker
