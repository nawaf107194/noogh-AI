#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Repository - Concrete Implementation of BaseRepository
============================================================

Demonstrates how to extend BaseRepository for specific models
with custom query methods.
"""

from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.repository import BaseRepository
from src.models.user import User


class UserRepository(BaseRepository[User]):
    """
    Repository for User model operations.
    
    Inherits standard CRUD from BaseRepository:
    - create(obj_in) -> User
    - get(id) -> Optional[User]
    - list(skip, limit) -> List[User]
    - update(id, obj_in) -> Optional[User]
    - delete(id) -> bool
    - count() -> int
    - exists(id) -> bool
    
    Plus custom methods specific to User.
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize UserRepository with User model."""
        super().__init__(db, User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address to search for
        
        Returns:
            User if found, None otherwise
        
        Example:
            user = await user_repo.get_by_email("john@example.com")
        """
        return await self.get_by(email=email)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username to search for
        
        Returns:
            User if found, None otherwise
        
        Example:
            user = await user_repo.get_by_username("johndoe")
        """
        return await self.get_by(username=username)
    
    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Get all active users.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records
        
        Returns:
            List of active users
        
        Example:
            active_users = await user_repo.get_active_users()
        """
        return await self.list(skip=skip, limit=limit, is_active=True)
    
    async def get_superusers(self) -> List[User]:
        """
        Get all superuser accounts.
        
        Returns:
            List of superusers
        
        Example:
            admins = await user_repo.get_superusers()
        """
        return await self.list(is_superuser=True)
    
    async def search_by_name(
        self,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Search users by full name (case-insensitive).
        
        Args:
            search_term: Search term to match against full_name
            skip: Number of records to skip
            limit: Maximum number of records
        
        Returns:
            List of matching users
        
        Example:
            users = await user_repo.search_by_name("john")
        """
        stmt = (
            select(self.model)
            .where(self.model.full_name.ilike(f"%{search_term}%"))
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def deactivate_user(self, user_id: int) -> Optional[User]:
        """
        Deactivate a user account.
        
        Args:
            user_id: User ID
        
        Returns:
            Updated user or None if not found
        
        Example:
            user = await user_repo.deactivate_user(1)
        """
        return await self.update(user_id, {"is_active": False})
    
    async def activate_user(self, user_id: int) -> Optional[User]:
        """
        Activate a user account.
        
        Args:
            user_id: User ID
        
        Returns:
            Updated user or None if not found
        
        Example:
            user = await user_repo.activate_user(1)
        """
        return await self.update(user_id, {"is_active": True})


# ============================================================================
# Exports
# ============================================================================

__all__ = ["UserRepository"]
