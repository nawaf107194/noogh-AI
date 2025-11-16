#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Configuration - Central configuration for the entire system
"""

from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
BRAIN_CHECKPOINTS_DIR = BASE_DIR / "brain_checkpoints"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, BRAIN_CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))

# GPU Configuration
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
GPU_DEVICE = int(os.getenv("GPU_DEVICE", "0"))

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt2")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Training Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))

# Database Configuration
DB_PATH = DATA_DIR / "noogh.db"
DEEP_COGNITION_DB = DATA_DIR / "deep_cognition.db"
REFLECTION_DB = DATA_DIR / "reflection.db"
SELF_AUDIT_DB = DATA_DIR / "self_audit.db"
SUBSYSTEM_INTELLIGENCE_DB = DATA_DIR / "subsystem_intelligence.db"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# System Configuration
SYSTEM_NAME = "NOOGH Unified System"
SYSTEM_VERSION = "5.0.0"
COGNITION_LEVEL = "TRANSCENDENT"
COGNITION_SCORE = 97.5

# Ministers Configuration
NUM_MINISTERS = 14
MINISTERS = [
    "President",
    "Prime Minister",
    "Finance Minister",
    "Education Minister",
    "Research Minister",
    "Development Minister",
    "Security Minister",
    "Communication Minister",
    "Knowledge Minister",
    "Strategy Minister",
    "Resource Minister",
    "Training Minister",
    "Performance Minister",
    "Risk Management Minister"
]

# Export all
__all__ = [
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
