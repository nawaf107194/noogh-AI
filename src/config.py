#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration - System configuration for src modules
This module re-exports configuration from core.config for backward compatibility
"""

from pathlib import Path
import sys

# Add parent directory to path to import from core
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    # Import from core.config
    from core.config import BASE_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR
except ImportError:
    # Fallback: define them here if core.config doesn't exist
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Export all
__all__ = ['BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR']
