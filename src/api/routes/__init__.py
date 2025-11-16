#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقاط النهاية لواجهة API
API Endpoints

يحتوي على جميع endpoints المنظمة حسب الوحدات
"""

import sys
from pathlib import Path

# Add project root to Python path BEFORE importing routers
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
