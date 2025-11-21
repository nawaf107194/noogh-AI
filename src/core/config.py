#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Configuration - DEPRECATED
================================

This module is deprecated. Please use src.core.settings instead.

Kept for backward compatibility. All values are re-exported from the new
Pydantic Settings implementation.

Migration Guide:
----------------
Old: from src.core.config import API_HOST
New: from src.core.settings import settings
     settings.api_host

OR for backward compatibility:
     from src.core.settings import API_HOST  # Still works!
"""

import warnings

# Show deprecation warning
warnings.warn(
    "src.core.config is deprecated. Use src.core.settings instead. "
    "This module will be removed in version 6.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new settings module for backward compatibility
from .settings import (
    settings,
    API_HOST,
    API_PORT,
    DASHBOARD_PORT,
    LOG_LEVEL,
    LOG_FORMAT,
    USE_GPU,
    GPU_DEVICE,
    DEFAULT_MODEL,
    MAX_LENGTH,
    TEMPERATURE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    BRAIN_CHECKPOINTS_DIR,
    DB_PATH,
    DEEP_COGNITION_DB,
    REFLECTION_DB,
    SELF_AUDIT_DB,
    SUBSYSTEM_INTELLIGENCE_DB,
    SYSTEM_NAME,
    SYSTEM_VERSION,
    COGNITION_LEVEL,
    COGNITION_SCORE,
    NUM_MINISTERS,
    MINISTERS,
)

__all__ = [
    'settings',
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR', 'CHECKPOINTS_DIR', 'BRAIN_CHECKPOINTS_DIR',
    'API_HOST', 'API_PORT', 'DASHBOARD_PORT',
    'USE_GPU', 'GPU_DEVICE',
    'DEFAULT_MODEL', 'MAX_LENGTH', 'TEMPERATURE',
    'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS',
    'DB_PATH', 'DEEP_COGNITION_DB', 'REFLECTION_DB', 'SELF_AUDIT_DB', 'SUBSYSTEM_INTELLIGENCE_DB',
    'LOG_LEVEL', 'LOG_FORMAT',
    'SYSTEM_NAME', 'SYSTEM_VERSION', 'COGNITION_LEVEL', 'COGNITION_SCORE',
    'NUM_MINISTERS', 'MINISTERS'
]
