#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Configuration Management
إدارة الإعدادات الموحدة

Single source of truth for all configuration
"""

import os
from pathlib import Path
from typing import Optional, List

# Handle Pydantic v2 migration - BaseSettings moved to pydantic-settings
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator
    except ImportError:
        # Fallback: Simple class without validation
        BaseSettings = object
        Field = lambda *args, **kwargs: kwargs.get('default', None)
        def validator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator


from pydantic import model_config

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    model_config = {'extra': 'allow'}

    # Environment
    environment: str = Field(default="development", env="NOOGH_ENV")
    debug: bool = Field(default=False, env="DEBUG")

    # API Server
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Security
    secret_key: str = Field(default="change-this-secret-key", env="SECRET_KEY")
    api_keys_json: Optional[str] = Field(default=None, env="NOOGH_API_KEYS_JSON")
    api_keys_file: str = Field(default="/etc/noogh/api_keys.json", env="NOOGH_API_KEYS_FILE")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        env="CORS_ORIGINS"
    )

    # Removed validator - CORS parsing handled in main.py
    # @validator('cors_origins')
    # def parse_cors_origins(cls, v):
    #     if isinstance(v, str):
    #         return [origin.strip() for origin in v.split(',')]
    #     return v

    # Database
    database_url: str = Field(default="sqlite:///./noogh.db", env="DATABASE_URL")
    db_pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, env="DB_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_enabled: bool = Field(default=True, env="REDIS_ENABLED")

    # Cache
    cache_ttl_default: int = Field(default=3600, env="CACHE_TTL_DEFAULT")
    cache_ttl_knowledge: int = Field(default=7200, env="CACHE_TTL_KNOWLEDGE")
    cache_ttl_models: int = Field(default=86400, env="CACHE_TTL_MODELS")

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Optional[Path] = Field(default=None, env="DATA_DIR")
    models_dir: Optional[Path] = Field(default=None, env="MODELS_DIR")
    logs_dir: Optional[Path] = Field(default=None, env="LOGS_DIR")

    # Removed validator - paths handled in __init__
    # @validator('data_dir', 'models_dir', 'logs_dir', pre=True, always=True)
    # def set_default_paths(cls, v, values, field):
    #     if v is None:
    #         base_dir = values.get('base_dir', Path.cwd())
    #         return base_dir / field.name.replace('_dir', '')
    #     return Path(v)

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

    # PyTorch
    torch_home: str = Field(default="/tmp/torch_cache", env="TORCH_HOME")
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    torch_num_threads: int = Field(default=4, env="TORCH_NUM_THREADS")

    # ALLaM Model
    allam_backend: str = Field(default="production", env="ALLAM_BACKEND")
    allam_api_key: Optional[str] = Field(default=None, env="ALLAM_API_KEY")
    allam_model_path: Optional[Path] = Field(default=None, env="ALLAM_MODEL_PATH")

    # Trading
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    trading_enabled: bool = Field(default=False, env="TRADING_ENABLED")
    trading_mode: str = Field(default="paper", env="TRADING_MODE")  # paper or live

    # Rate Limiting
    rate_limit_requests: int = Field(default=60, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")

    # Model Configuration
    max_context_length: int = Field(default=2048, env="MAX_CONTEXT_LENGTH")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    max_batch_size: int = Field(default=32, env="MAX_BATCH_SIZE")

    # Knowledge Base
    knowledge_chunk_size: int = Field(default=512, env="KNOWLEDGE_CHUNK_SIZE")
    knowledge_overlap: int = Field(default=50, env="KNOWLEDGE_OVERLAP")
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        # Handle fallback mode without Pydantic
        if BaseSettings == object:
            # Initialize with defaults when Pydantic is not available
            self.environment = os.getenv('NOOGH_ENV', 'development')
            self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
            self.api_host = os.getenv('API_HOST', '0.0.0.0')
            self.api_port = int(os.getenv('API_PORT', '8000'))
            self.cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080')
            self.log_level = os.getenv('LOG_LEVEL', 'INFO')
            self.base_dir = Path(__file__).parent.parent.parent
            self.data_dir = Path(os.getenv('DATA_DIR', str(self.base_dir / 'data')))
            self.models_dir = Path(os.getenv('MODELS_DIR', str(self.base_dir / 'models')))
            self.logs_dir = Path(os.getenv('LOGS_DIR', str(self.base_dir / 'logs')))
        else:
            super().__init__(**kwargs)
            # Ensure paths are set if they came back as None
            if self.data_dir is None:
                self.data_dir = self.base_dir / 'data'
            if self.models_dir is None:
                self.models_dir = self.base_dir / 'models'
            if self.logs_dir is None:
                self.logs_dir = self.base_dir / 'logs'

        # Create directories if they don't exist
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"

    def get_log_file(self, name: str = "noogh") -> Path:
        """Get log file path"""
        return self.logs_dir / f"{name}.log"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience function to reload settings
def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings
