#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Handler
"""

import sys
from pathlib import Path
from enum import Enum

# Add parent to path
PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

class NooghError(Exception):
    pass

class ErrorCategory(Enum):
    SYSTEM = "SYSTEM"
    API = "API"
    BRAIN = "BRAIN"

class ErrorSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class NooghErrorHandler:
    def handle_error(self, error, context=None):
        import logging
        logging.error(f"Error: {error}, Context: {context}")

    def get_error_stats(self):
        return {}
