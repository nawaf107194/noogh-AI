#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for UserRepository
==============================

Tests the repository pattern with async database operations.
Each test runs in an isolated transaction that rolls back.
"""

import pytest
from src.repositories.user_repository import UserRepository
from src.models.user import User


@pytest.mark.unit
class TestUserRepository:
    """Test suite for UserRepository CRUD operations."""
    
    async def test_create_user(self, db_session, sample_user_data):
        """Test creating a new user."""
        # Arrange
        repo = UserRepository(db_session)
        
        # Act
        user = await repo.create(sample_user_data)
        
        # Assert
        assert user.id is not None
        assert user.email == sample_user_data["email"]
        assert user.username == sample_user_data["username"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.is_active is True
        assert user.created_at is not None
        assert user.updated_at is not None
    
    async def test_get_user_by_id(self, db_session, create_test_user):
        """Test retrieving a user by ID."""
        # Arrange
        repo = UserRepository(db_session)
        created_user = await create_test_user(email="get@example.com")
        
        # Act
        user = await repo.get(created_user.id)
        
        # Assert
        assert user is not None
        assert user.id == created_user.id
        assert user.email == "get@example.com"
    
    async def test_get_nonexistent_user(self, db_session):
        """Test retrieving a user that doesn't exist."""
        # Arrange
        repo = UserRepository(db_session)
        
        # Act
        user = await repo.get(99999)
        
        # Assert
        assert user is None
    
    async def test_get_by_email(self, db_session, create_test_user):
        """Test retrieving a user by email."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="findme@example.com", username="findme")
        
        # Act
        user = await repo.get_by_email("findme@example.com")
        
        # Assert
        assert user is not None
        assert user.email == "findme@example.com"
        assert user.username == "findme"
    
    async def test_get_by_username(self, db_session, create_test_user):
        """Test retrieving a user by username."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="user@example.com", username="uniqueuser")
        
        # Act
        user = await repo.get_by_username("uniqueuser")
        
        # Assert
        assert user is not None
        assert user.username == "uniqueuser"
    
    async def test_list_users(self, db_session, create_test_user):
        """Test listing users with pagination."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="user1@example.com", username="user1")
        await create_test_user(email="user2@example.com", username="user2")
        await create_test_user(email="user3@example.com", username="user3")
        
        # Act
        users = await repo.list(skip=0, limit=10)
        
        # Assert
        assert len(users) == 3
    
    async def test_list_users_pagination(self, db_session, create_test_user):
        """Test pagination in list users."""
        # Arrange
        repo = UserRepository(db_session)
        for i in range(5):
            await create_test_user(
                email=f"user{i}@example.com",
                username=f"user{i}"
            )
        
        # Act
        first_page = await repo.list(skip=0, limit=2)
        second_page = await repo.list(skip=2, limit=2)
        
        # Assert
        assert len(first_page) == 2
        assert len(second_page) == 2
        assert first_page[0].id != second_page[0].id
    
    async def test_count_users(self, db_session, create_test_user):
        """Test counting users."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="count1@example.com", username="count1")
        await create_test_user(email="count2@example.com", username="count2")
        
        # Act
        count = await repo.count()
        
        # Assert
        assert count == 2
    
    async def test_update_user(self, db_session, create_test_user):
        """Test updating a user."""
        # Arrange
        repo = UserRepository(db_session)
        user = await create_test_user(email="update@example.com", username="oldname")
        
        # Act
        updated_user = await repo.update(user.id, {"full_name": "Updated Name"})
        
        # Assert
        assert updated_user is not None
        assert updated_user.id == user.id
        assert updated_user.full_name == "Updated Name"
        assert updated_user.email == "update@example.com"  # Unchanged
    
    async def test_delete_user(self, db_session, create_test_user):
        """Test deleting a user."""
        # Arrange
        repo = UserRepository(db_session)
        user = await create_test_user(email="delete@example.com", username="deleteme")
        
        # Act
        deleted = await repo.delete(user.id)
        
        # Assert
        assert deleted is True
        
        # Verify user is gone
        user_check = await repo.get(user.id)
        assert user_check is None
    
    async def test_delete_nonexistent_user(self, db_session):
        """Test deleting a user that doesn't exist."""
        # Arrange
        repo = UserRepository(db_session)
        
        # Act
        deleted = await repo.delete(99999)
        
        # Assert
        assert deleted is False
    
    async def test_exists(self, db_session, create_test_user):
        """Test checking if user exists."""
        # Arrange
        repo = UserRepository(db_session)
        user = await create_test_user(email="exists@example.com", username="exists")
        
        # Act
        exists = await repo.exists(user.id)
        not_exists = await repo.exists(99999)
        
        # Assert
        assert exists is True
        assert not_exists is False
    
    async def test_get_active_users(self, db_session, create_test_user):
        """Test filtering active users."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="active@example.com", username="active", is_active=True)
        await create_test_user(email="inactive@example.com", username="inactive", is_active=False)
        
        # Act
        active_users = await repo.get_active_users()
        
        # Assert
        assert len(active_users) == 1
        assert active_users[0].is_active is True
    
    async def test_search_by_name(self, db_session, create_test_user):
        """Test searching users by name."""
        # Arrange
        repo = UserRepository(db_session)
        await create_test_user(email="john@example.com", username="john", full_name="John Doe")
        await create_test_user(email="jane@example.com", username="jane", full_name="Jane Smith")
        await create_test_user(email="bob@example.com", username="bob", full_name="Bob Johnson")
        
        # Act
        john_results = await repo.search_by_name("John")
        smith_results = await repo.search_by_name("Smith")
        
        # Assert
        assert len(john_results) == 2  # John Doe and Bob Johnson
        assert len(smith_results) == 1  # Jane Smith
    
    async def test_deactivate_user(self, db_session, create_test_user):
        """Test deactivating a user."""
        # Arrange
        repo = UserRepository(db_session)
        user = await create_test_user(email="active@example.com", username="tobedeactivated", is_active=True)
        
        # Act
        deactivated = await repo.deactivate_user(user.id)
        
        # Assert
        assert deactivated is not None
        assert deactivated.is_active is False
    
    async def test_activate_user(self, db_session, create_test_user):
        """Test activating a user."""
        # Arrange
        repo = UserRepository(db_session)
        user = await create_test_user(email="inactive@example.com", username="tobeactivated", is_active=False)
        
        # Act
        activated = await repo.activate_user(user.id)
        
        # Assert
        assert activated is not None
        assert activated.is_active is True
