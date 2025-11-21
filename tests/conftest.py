#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Configuration and Fixtures
================================

Provides pytest fixtures for async testing including:
- Event loop management
- Database session with transaction rollback
- Async HTTP client for API testing
"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from httpx import AsyncClient

from src.core.settings import settings
from src.core.base import Base
from src.core.database import get_async_database_url
from src.api.app import create_app


# ============================================================================
# Event Loop Fixture (Session Scope)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the test session.
    
    Scope: session - shared across all tests
    
    This is required for pytest-asyncio to work properly with
    session-scoped async fixtures.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
async def test_engine():
    """
    Create a test database engine.
    
    Uses SQLite in-memory database for fast testing.
    """
    # Use in-memory SQLite for tests
    test_db_url = "sqlite+aiosqlite:///:memory:"
    
    engine = create_async_engine(
        test_db_url,
        echo=False,  # Set to True for SQL debugging
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session for testing.
    
    Each test gets a clean database state:
    1. Begins a transaction
    2. Runs the test
    3. Rolls back the transaction (nothing persisted)
    
    This ensures tests are isolated and don't affect each other.
    
    Usage:
        async def test_create_user(db_session):
            user = User(email="test@example.com")
            db_session.add(user)
            await db_session.flush()
            assert user.id is not None
    """
    # Create session factory
    async_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_factory() as session:
        # Begin a nested transaction
        async with session.begin():
            yield session
            # Rollback happens automatically when context exits
            await session.rollback()


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
async def async_client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP client for API testing.
    
    Overrides the app's get_db dependency to use the test database session.
    
    Usage:
        async def test_health_endpoint(async_client):
            response = await async_client.get("/health")
            assert response.status_code == 200
    """
    from src.core.database import get_db
    
    # Create FastAPI app
    app = create_app()
    
    # Override the get_db dependency to use our test session
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    # Create async client
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Clear overrides
    app.dependency_overrides.clear()


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def sample_user_data():
    """Provide sample user data for tests."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False,
    }


@pytest.fixture
async def create_test_user(db_session):
    """
    Factory fixture to create test users.
    
    Usage:
        async def test_something(create_test_user):
            user = await create_test_user(email="test@example.com")
            assert user.id is not None
    """
    from src.repositories.user_repository import UserRepository
    
    async def _create_user(**kwargs):
        """Create a user with given attributes."""
        repo = UserRepository(db_session)
        
        # Default values
        data = {
            "email": "default@example.com",
            "username": "defaultuser",
            "full_name": "Default User",
            "is_active": True,
            "is_superuser": False,
        }
        data.update(kwargs)
        
        return await repo.create(data)
    
    return _create_user


# ============================================================================
# Markers
# ============================================================================

# Mark tests with @pytest.mark.unit for unit tests
# Mark tests with @pytest.mark.integration for integration tests
# Mark tests with @pytest.mark.slow for slow running tests
