#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Unified System - Settings (Pydantic-based Configuration)
===============================================================

Modern configuration management using Pydantic Settings with:
- Type validation
- Environment variable loading
- Default values
- Validation rules
"""

from typing import Optional
from pathlib import Path
from pydantic import Field, field_validator, computed_field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import hashlib


class Settings(BaseSettings):
    """
    Centralized configuration for Noogh Unified System
    
    All settings are loaded from environment variables with fallback defaults.
    Create a .env file in the project root to customize values.
    """
    
    # ============================================================================
    # System Configuration
    # ============================================================================
    
    app_name: str = Field(
        default="Noogh Unified AI System",
        description="Application name"
    )
    
    version: str = Field(
        default="5.0.0",
        description="Application version"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, auto-reload)"
    )
    
    cognition_level: str = Field(
        default="TRANSCENDENT",
        description="System cognition level"
    )
    
    cognition_score: float = Field(
        default=97.5,
        ge=0.0,
        le=100.0,
        description="System cognition score (0-100)"
    )
    
    # ============================================================================
    # Server Configuration
    # ============================================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API host to bind to"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API port to listen on"
    )
    
    dashboard_port: int = Field(
        default=8501,
        ge=1,
        le=65535,
        description="Dashboard port (if using Streamlit)"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated)"
    )
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    
    database_url: str = Field(
        default="sqlite:///./data/noogh.db",
        description="Database connection URL (PostgreSQL, SQLite, etc.)"
    )
    
    db_echo: bool = Field(
        default=False,
        description="Echo SQL queries to console"
    )
    
    db_pool_size: int = Field(
        default=10,
        ge=1,
        description="Database connection pool size"
    )
    
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        description="Maximum overflow connections"
    )
    
    # ============================================================================
    # Security
    # ============================================================================
    
    secret_key: SecretStr = Field(
        default="change-me-in-production-super-secret-key-12345",
        description="Secret key for JWT, sessions, etc. MUST be changed in production!"
    )
    
    access_token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT access token expiration time (minutes)"
    )
    
    # ============================================================================
    # GPU & ML Configuration
    # ============================================================================
    
    use_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    
    gpu_device: int = Field(
        default=0,
        ge=0,
        description="GPU device ID to use"
    )
    
    default_model: str = Field(
        default="gpt2",
        description="Default ML model to use  (legacy)"
    )
    
    # ========================================================================
    # Local Brain Model Settings (Meta-Llama-3-8B on RTX 5070)
    # ========================================================================
    local_model_path: str = Field(default="models/")
    local_model_name: str = Field(default="NousResearch/Meta-Llama-3-8B-Instruct")
    max_tokens: int = Field(default=4096)
    max_length: int = Field(default=4096)
    use_gpu: bool = Field(default=True)
    torch_dtype: str = Field(default="float16")  # FP16 for RTX 5070 optimization
    
    # ========================================================================
    # Trading Configuration (Paper Trading Mode)
    # ========================================================================
    trading_mode: str = Field(default="PAPER")  # PAPER or LIVE
    enable_spot: bool = Field(default=True)
    enable_futures: bool = Field(default=True)
    paper_balance_spot: float = Field(default=10000.0)  # USDT
    paper_balance_futures: float = Field(default=10000.0)  # USDT
    paper_ledger_path: str = Field(default="data/paper_ledger.json")
    
    # ============================================================================
    # Security Configuration
    # ============================================================================

    override_password_hash: str = Field(
        default="",  # Must be set via environment variable
        description="SHA256 hash of override password for OSService (set OVERRIDE_PASSWORD_HASH in .env)"
    )

    @field_validator('override_password_hash')
    @classmethod
    def validate_override_password(cls, v: str) -> str:
        """Validate override password hash is set securely."""
        if not v or v == hashlib.sha256(b"admin").hexdigest():
            # Generate a random secure password and warn user
            import secrets
            import logging
            logger = logging.getLogger(__name__)

            # Generate a strong random password
            random_password = secrets.token_urlsafe(32)
            secure_hash = hashlib.sha256(random_password.encode()).hexdigest()

            logger.critical("=" * 80)
            logger.critical("🔴 SECURITY WARNING: Override password not set in .env!")
            logger.critical("🔐 Generated temporary random password:")
            logger.critical(f"   Password: {random_password}")
            logger.critical(f"   Hash: {secure_hash}")
            logger.critical("")
            logger.critical("📝 To set permanently, add to .env file:")
            logger.critical(f"   OVERRIDE_PASSWORD_HASH={secure_hash}")
            logger.critical("=" * 80)

            return secure_hash

        return v

    default_user_password: SecretStr = Field(
        default="ChangeMe123!",
        description="Default password for generated users (development only)"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for text generation"
    )
    
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Training batch size"
    )
    
    learning_rate: float = Field(
        default=5e-5,
        gt=0.0,
        description="Training learning rate"
    )
    
    num_epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs"
    )
    
    # ============================================================================
    # External APIs (DEPRECATED - Using Local Sovereign AI)
    # ============================================================================
    
    # NOTE: External API keys are NO LONGER USED
    # The system runs 100% locally with on-device inference
    
    # openai_api_key: Optional[str] = Field(
    #     default=None,
    #     description="[DEPRECATED] OpenAI API key - using local models instead"
    # )
    
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Optional HuggingFace token for downloading gated models"
    )
    
    # Trading API Keys
    binance_api_key: Optional[str] = Field(
        default=None,
        description="Binance API key"
    )
    
    binance_api_secret: Optional[SecretStr] = Field(
        default=None,
        description="Binance API secret"
    )
    
    binance_testnet: bool = Field(
        default=True,
        description="Use Binance testnet (recommended for development)"
    )
    
    # ============================================================================
    # Government System
    # ============================================================================
    
    num_ministers: int = Field(
        default=14,
        ge=1,
        description="Number of ministers in the government system"
    )
    
    # ============================================================================
    # Paths (Computed from project root)
    # ============================================================================
    
    @computed_field
    @property
    def base_dir(self) -> Path:
        """Base directory of the project"""
        return Path(__file__).parent.parent
    
    @computed_field
    @property
    def data_dir(self) -> Path:
        """Data directory"""
        path = self.base_dir / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def models_dir(self) -> Path:
        """Models directory"""
        path = self.base_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def logs_dir(self) -> Path:
        """Logs directory"""
        path = self.base_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def checkpoints_dir(self) -> Path:
        """Checkpoints directory"""
        path = self.base_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def brain_checkpoints_dir(self) -> Path:
        """Brain checkpoints directory"""
        path = self.base_dir / "brain_checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # ============================================================================
    # Validators
    # ============================================================================
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Warn if using default secret key"""
        if v == "change-me-in-production-super-secret-key-12345":
            import warnings
            warnings.warn(
                "⚠️  Using default SECRET_KEY! Change this in production!",
                UserWarning
            )
        return v
    
    # ============================================================================
    # Pydantic Settings Configuration
    # ============================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )


# ============================================================================
# Singleton Instance
# ============================================================================

# Create a singleton instance that can be imported anywhere
settings = Settings()


# ============================================================================
# Backward Compatibility Exports
# ============================================================================
# For gradual migration, export individual values as module-level constants

API_HOST = settings.api_host
API_PORT = settings.api_port
DASHBOARD_PORT = settings.dashboard_port
LOG_LEVEL = settings.log_level
USE_GPU = settings.use_gpu
GPU_DEVICE = settings.gpu_device
DEFAULT_MODEL = settings.default_model
MAX_LENGTH = settings.max_length
TEMPERATURE = settings.temperature
BATCH_SIZE = settings.batch_size
LEARNING_RATE = settings.learning_rate
NUM_EPOCHS = settings.num_epochs
SYSTEM_NAME = settings.app_name
SYSTEM_VERSION = settings.version
COGNITION_LEVEL = settings.cognition_level
COGNITION_SCORE = settings.cognition_score
NUM_MINISTERS = settings.num_ministers

# Paths
BASE_DIR = settings.base_dir
DATA_DIR = settings.data_dir
MODELS_DIR = settings.models_dir
LOGS_DIR = settings.logs_dir
CHECKPOINTS_DIR = settings.checkpoints_dir
BRAIN_CHECKPOINTS_DIR = settings.brain_checkpoints_dir

# Database paths (legacy)
DB_PATH = settings.data_dir / "noogh.db"
DEEP_COGNITION_DB = settings.data_dir / "deep_cognition.db"
REFLECTION_DB = settings.data_dir / "reflection.db"
SELF_AUDIT_DB = settings.data_dir / "self_audit.db"
SUBSYSTEM_INTELLIGENCE_DB = settings.data_dir / "subsystem_intelligence.db"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ministers list (for backward compatibility)
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

__all__ = [
    'Settings',
    'settings',
    # Backward compatibility exports
    'API_HOST', 'API_PORT', 'DASHBOARD_PORT', 'LOG_LEVEL',
    'USE_GPU', 'GPU_DEVICE',
    'DEFAULT_MODEL', 'MAX_LENGTH', 'TEMPERATURE',
    'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS',
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR',
    'CHECKPOINTS_DIR', 'BRAIN_CHECKPOINTS_DIR',
    'DB_PATH', 'DEEP_COGNITION_DB', 'REFLECTION_DB',
    'SELF_AUDIT_DB', 'SUBSYSTEM_INTELLIGENCE_DB',
    'LOG_FORMAT', 'SYSTEM_NAME', 'SYSTEM_VERSION',
    'COGNITION_LEVEL', 'COGNITION_SCORE',
    'NUM_MINISTERS', 'MINISTERS'
]


if __name__ == "__main__":
    # Test settings loading
    print("=" * 70)
    print("🔧 Noogh Unified System - Settings")
    print("=" * 70)
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.version}")
    print(f"Debug Mode: {settings.debug_mode}")
    print(f"API: {settings.api_host}:{settings.api_port}")
    print(f"Database: {settings.database_url}")
    print(f"GPU Enabled: {settings.use_gpu}")
    print(f"Log Level: {settings.log_level}")
    print(f"Data Dir: {settings.data_dir}")
    print(f"Ministers: {settings.num_ministers}")
    print("=" * 70)
