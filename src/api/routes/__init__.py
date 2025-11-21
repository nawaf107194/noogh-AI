#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Routes Package
تجميع المسارات
"""

__all__ = []

# Routes are now imported directly in app.py
# This file serves as a package marker
from pathlib import Path
# Add project root to Python path BEFORE importing routers
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# The original if block was empty and caused a syntax error.
# It has been removed as per the instruction "removing empty if".
# The imports below are now always executed.
from .system import router as system_router
from .brain import router as brain_router
from .models import router as model_router
from .trading import router as trading_router
from .data import router as data_router
from .monitoring import router as monitoring_router

__all__ = [
    'system_router',
    'brain_router',
    'model_router', 
    'trading_router',
    'data_router',
    'monitoring_router'
]
