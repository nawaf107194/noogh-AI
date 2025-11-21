#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for API Endpoints
====================================

Tests FastAPI endpoints using the async HTTP client.
"""

import pytest


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    async def test_health_check(self, async_client):
        """Test the health check endpoint."""
        # Act
        response = await async_client.get("/api/routes/example_async_db/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data
        assert "version" in data
    
    async def test_system_stats(self, async_client):
        """Test the system stats endpoint."""
        # Act
        response = await async_client.get("/api/routes/example_async_db/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "database_connected" in data
        assert "ministers" in data
        assert "gpu_enabled" in data


@pytest.mark.integration
class TestUserAPIEndpoints:
    """Test user-related API endpoints (when implemented)."""
    
    async def test_list_users_empty(self, async_client):
        """Test listing users when database is empty."""
        # Act
        response = await async_client.get("/api/routes/example_async_db/items")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] == 0
    
    async def test_create_user_endpoint(self, async_client):
        """Test creating a user via API."""
        # Arrange
        user_data = {
            "name": "Test User",
            "description": "Test Description"
        }
        
        # Act
        response = await async_client.post(
            "/api/routes/example_async_db/items",
            params=user_data
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test User"
        assert data["status"] == "created"


@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""
    
    async def test_404_not_found(self, async_client):
        """Test 404 error for non-existent endpoint."""
        # Act
        response = await async_client.get("/api/nonexistent")
        
        # Assert
        assert response.status_code == 404
    
    async def test_method_not_allowed(self, async_client):
        """Test 405 error for wrong HTTP method."""
        # Act
        response = await async_client.delete("/api/routes/example_async_db/health")
        
        # Assert
        assert response.status_code == 405
