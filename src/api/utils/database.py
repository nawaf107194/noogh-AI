#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Connection Management with Pooling
إدارة اتصالات قاعدة البيانات مع التجميع
"""

import logging
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# Base class for models
Base = declarative_base()


class DatabaseManager:
    """Database connection manager with pooling support"""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """
        Initialize database manager

        Args:
            database_url: Database connection URL
            pool_size: Number of connections to pool
            max_overflow: Maximum overflow connections
            pool_timeout: Timeout for getting connection
            pool_recycle: Recycle connections after this many seconds
            echo: Echo SQL statements (debug)
        """
        self.database_url = database_url

        # Check if SQLite (doesn't support pooling well)
        if database_url.startswith('sqlite'):
            # Use NullPool for SQLite
            self.engine = create_engine(
                database_url,
                poolclass=NullPool,
                echo=echo,
                connect_args={"check_same_thread": False}  # For SQLite
            )
            logger.info("✅ SQLite database initialized (no pooling)")
        else:
            # Use QueuePool for PostgreSQL, MySQL, etc.
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                echo=echo
            )
            logger.info(f"✅ Database initialized with connection pooling (size={pool_size})")

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Register event listeners
        self._register_events()

    def _register_events(self):
        """Register SQLAlchemy events for monitoring"""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("Database connection opened")

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_conn, connection_record):
            logger.debug("Database connection closed")

    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session with automatic cleanup

        Usage:
            with db_manager.get_session() as session:
                results = session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created")
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise

    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("⚠️  All database tables dropped")
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            raise

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics"""
        if hasattr(self.engine.pool, 'size'):
            return {
                "pool_size": self.engine.pool.size(),
                "checked_in": self.engine.pool.checkedin(),
                "checked_out": self.engine.pool.checkedout(),
                "overflow": self.engine.pool.overflow(),
                "total": self.engine.pool.size() + self.engine.pool.overflow()
            }
        return {"type": "NullPool (SQLite)"}

    def close(self):
        """Close all connections"""
        self.engine.dispose()
        logger.info("✅ Database connections closed")


# Global database manager
_db_manager: Optional[DatabaseManager] = None


def init_database(
    database_url: str,
    pool_size: int = 20,
    max_overflow: int = 10
) -> DatabaseManager:
    """Initialize global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(
        database_url=database_url,
        pool_size=pool_size,
        max_overflow=max_overflow
    )
    return _db_manager


def get_database() -> DatabaseManager:
    """Get global database manager"""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_manager


def get_db_session():
    """Dependency for FastAPI to get database session"""
    db = get_database()
    with db.get_session() as session:
        yield session
