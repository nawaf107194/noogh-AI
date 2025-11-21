#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Database Module - SQLAlchemy 2.0 with AsyncEngine
========================================================

Modern async database architecture with:
- AsyncEngine and AsyncSession
- Proper session lifecycle management
- Dependency injection support
- Connection pooling
- Graceful handling of SQLite vs PostgreSQL
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event
import logging

from .settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Database Engine Configuration
# ============================================================================

def get_async_database_url(database_url: str) -> str:
    """
    Convert database URL to async version.
    
    Examples:
        sqlite:///./data/noogh.db -> sqlite+aiosqlite:///./data/noogh.db
        postgresql://user:pass@host/db -> postgresql+asyncpg://user:pass@host/db
        mysql://user:pass@host/db -> mysql+aiomysql://user:pass@host/db
    """
    if database_url.startswith("sqlite"):
        # SQLite: use aiosqlite driver
        if "+aiosqlite" not in database_url:
            return database_url.replace("sqlite://", "sqlite+aiosqlite://")
        return database_url
    
    elif database_url.startswith("postgresql"):
        # PostgreSQL: use asyncpg driver
        if "+asyncpg" not in database_url:
            return database_url.replace("postgresql://", "postgresql+asyncpg://")
        return database_url
    
    elif database_url.startswith("mysql"):
        # MySQL: use aiomysql driver
        if "+aiomysql" not in database_url:
            return database_url.replace("mysql://", "mysql+aiomysql://")
        return database_url
    
    # Already async or unknown - return as is
    return database_url


def create_engine() -> AsyncEngine:
    """
    Create async database engine with appropriate configuration.
    
    Returns:
        Configured AsyncEngine instance
    """
    async_url = get_async_database_url(settings.database_url)
    
    # Determine if we're using SQLite
    is_sqlite = "sqlite" in async_url.lower()
    
    # Engine kwargs
    engine_kwargs = {
        "echo": settings.db_echo,
        "future": True,  # SQLAlchemy 2.0 style
    }
    
    if is_sqlite:
        # SQLite-specific configuration
        engine_kwargs.update({
            "poolclass": NullPool,  # SQLite doesn't need pooling
            "connect_args": {
                "check_same_thread": False,  # Allow multi-threaded access
                "timeout": 30,  # Connection timeout
            }
        })
        logger.info(f"🗄️  Configuring SQLite database: {async_url}")
    else:
        # PostgreSQL/MySQL configuration with connection pooling
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_pre_ping": True,  # Verify connections before using
            "pool_recycle": 3600,  # Recycle connections after 1 hour
        })
        logger.info(f"🗄️  Configuring {async_url.split(':')[0].upper()} database with connection pooling")
    
    engine = create_async_engine(async_url, **engine_kwargs)
    
    # Log engine creation
    if settings.debug_mode:
        logger.debug(f"Database engine created: {async_url}")
    
    return engine


# ============================================================================
# Global Engine and Session Factory
# ============================================================================

# Create global async engine (singleton)
async_engine: AsyncEngine = create_engine()

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autocommit=False,
    autoflush=False,
)


# ============================================================================
# Dependency Injection - FastAPI Compatible
# ============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.
    
    Usage in routes:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    
    The session is automatically:
    - Created before the request
    - Committed if successful
    - Rolled back on exception
    - Closed after the request
    
    Yields:
        AsyncSession instance
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Database Initialization
# ============================================================================

async def init_db():
    """
    Initialize database tables.
    
    Creates all tables defined in models that inherit from Base.
    Should be called once at application startup.
    """
    from .base import Base
    
    async with async_engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified")


async def close_db():
    """
    Close database connections gracefully.
    
    Should be called at application shutdown.
    """
    await async_engine.dispose()
    logger.info("✅ Database connections closed")


# ============================================================================
# Utility Functions
# ============================================================================

async def check_db_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


async def get_db_info() -> dict:
    """
    Get database connection information.
    
    Returns:
        Dict with database details
    """
    async_url = get_async_database_url(settings.database_url)
    is_sqlite = "sqlite" in async_url.lower()
    
    return {
        "database_url": async_url.split("@")[-1] if "@" in async_url else async_url,  # Hide credentials
        "driver": async_url.split("+")[1].split(":")[0] if "+" in async_url else "default",
        "is_sqlite": is_sqlite,
        "pool_size": settings.db_pool_size if not is_sqlite else None,
        "max_overflow": settings.db_max_overflow if not is_sqlite else None,
        "echo": settings.db_echo,
    }


# ============================================================================
# Legacy Sync Session Support (Deprecated)
# ============================================================================

# For backward compatibility with sync code
# TODO: Remove this once all code is migrated to async

from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.orm import sessionmaker, Session

# Create sync engine (for gradual migration)
sync_engine = create_sync_engine(
    settings.database_url.replace("sqlite+aiosqlite", "sqlite"),  # Remove async driver
    echo=settings.db_echo,
)

SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)

def get_db_sync() -> Session:
    """
    DEPRECATED: Use get_db() instead for async routes.
    
    Legacy sync session for backward compatibility.
    Will be removed in version 6.0.0.
    """
    import warnings
    warnings.warn(
        "get_db_sync() is deprecated. Use async get_db() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Async (recommended)
    "async_engine",
    "AsyncSessionLocal",
    "get_db",
    "init_db",
    "close_db",
    "check_db_connection",
    "get_db_info",
    
    # Sync (deprecated)
    "sync_engine",
    "SessionLocal",
    "get_db_sync",
]


if __name__ == "__main__":
    # Test database configuration
    import asyncio
    
    async def test_db():
        print("=" * 70)
        print("🗄️  Database Configuration Test")
        print("=" * 70)
        
        # Show database info
        info = await get_db_info()
        print(f"Database URL: {info['database_url']}")
        print(f"Driver: {info['driver']}")
        print(f"Is SQLite: {info['is_sqlite']}")
        if not info['is_sqlite']:
            print(f"Pool Size: {info['pool_size']}")
            print(f"Max Overflow: {info['max_overflow']}")
        
        # Test connection
        print("\nTesting connection...")
        connected = await check_db_connection()
        print(f"Connection: {'✅ Success' if connected else '❌ Failed'}")
        
        print("=" * 70)
    
    asyncio.run(test_db())
