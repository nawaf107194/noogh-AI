#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Service - Business Logic Layer
====================================

Service layer implementing business logic for user management.
Sits between API routes and data repositories.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from ..repositories.user_repository import UserRepository
from ..schemas.user import UserCreate, UserUpdate, UserResponse
from ..models.user import User
from ..core.settings import settings


class UserService:
    """
    Service for user-related business logic.
    
    Responsibilities:
    - Validate business rules
    - Coordinate between multiple repositories if needed
    - Handle complex operations
    - Raise appropriate HTTP exceptions
    
    Usage:
        async def create_user_route(user_data: UserCreate, db: AsyncSession):
            service = UserService(db)
            user = await service.create_user(user_data)
            return user
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize service with database session.
        
        Args:
            db: Async database session
        """
        self.db = db
        self.user_repo = UserRepository(db)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user with validation.
        
        Business Rules:
        - Email must be unique
        - Username must be unique
        - Password must meet security requirements (handled by schema)
        
        Args:
            user_data: User creation data
        
        Returns:
            Created user
        
        Raises:
            HTTPException: 400 if email or username already exists
        
        Example:
            user = await service.create_user(UserCreate(
                email="john@example.com",
                username="johndoe",
                full_name="John Doe",
                password=settings.default_user_password.get_secret_value()
            ))
        """
        # Check if email already exists
        existing_email = await self.user_repo.get_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_data.email}' is already registered"
            )
        
        # Check if username already exists
        existing_username = await self.user_repo.get_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user_data.username}' is already taken"
            )
        
        # TODO: Hash password before storing
        # For now, we'll store it as-is (NOT recommended for production!)
        # In production, use: from passlib.context import CryptContext
        # hashed_password = pwd_context.hash(user_data.password)
        
        # Create user data dict (exclude password from being stored directly)
        user_dict = user_data.model_dump()
        # In production: user_dict["hashed_password"] = hashed_password
        # For now: Remove password field as we don't have hashed_password column
        user_dict.pop("password", None)
        
        # Create user
        user = await self.user_repo.create(user_dict)
        return user
    
    async def get_user(self, user_id: int) -> User:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
        
        Returns:
            User instance
        
        Raises:
            HTTPException: 404 if user not found
        """
        user = await self.user_repo.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            email: Email address
        
        Returns:
            User if found, None otherwise
        """
        return await self.user_repo.get_by_email(email)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
        
        Returns:
            User if found, None otherwise
        """
        return await self.user_repo.get_by_username(username)
    
    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = False
    ) -> tuple[List[User], int]:
        """
        List users with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            active_only: If True, only return active users
        
        Returns:
            Tuple of (users list, total count)
        """
        if active_only:
            users = await self.user_repo.get_active_users(skip=skip, limit=limit)
            total = await self.user_repo.count(is_active=True)
        else:
            users = await self.user_repo.list(skip=skip, limit=limit)
            total = await self.user_repo.count()
        
        return users, total
    
    async def update_user(self, user_id: int, user_data: UserUpdate) -> User:
        """
        Update user information.
        
        Args:
            user_id: User ID
            user_data: Update data
        
        Returns:
            Updated user
        
        Raises:
            HTTPException: 404 if user not found
            HTTPException: 400 if email/username conflict
        """
        # Check user exists
        user = await self.get_user(user_id)
        
        # Check for email conflicts (if email is being updated)
        if user_data.email and user_data.email != user.email:
            existing = await self.user_repo.get_by_email(user_data.email)
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{user_data.email}' is already in use"
                )
        
        # Check for username conflicts (if username is being updated)
        if user_data.username and user_data.username != user.username:
            existing = await self.user_repo.get_by_username(user_data.username)
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Username '{user_data.username}' is already taken"
                )
        
        # Update user
        update_dict = user_data.model_dump(exclude_unset=True)
        # Remove password if present (TODO: handle password hashing)
        update_dict.pop("password", None)
        
        updated_user = await self.user_repo.update(user_id, update_dict)
        return updated_user
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Delete user (hard delete).
        
        Args:
            user_id: User ID
        
        Returns:
            True if deleted
        
        Raises:
            HTTPException: 404 if user not found
        """
        # Check user exists
        await self.get_user(user_id)
        
        # Delete user
        deleted = await self.user_repo.delete(user_id)
        return deleted
    
    async def deactivate_user(self, user_id: int) -> User:
        """
        Deactivate user account (soft delete).
        
        Args:
            user_id: User ID
        
        Returns:
            Deactivated user
        
        Raises:
            HTTPException: 404 if user not found
        """
        user = await self.user_repo.deactivate_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user
    
    async def activate_user(self, user_id: int) -> User:
        """
        Activate user account.
        
        Args:
            user_id: User ID
        
        Returns:
            Activated user
        
        Raises:
            HTTPException: 404 if user not found
        """
        user = await self.user_repo.activate_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/password.
        
        NOTE: This is a DUMMY implementation for demonstration.
        In production, use proper password hashing with passlib/bcrypt.
        
        Args:
            username: Username or email
            password: Plain text password
        
        Returns:
            User if authentication successful, None otherwise
        """
        # Try to find user by username
        user = await self.user_repo.get_by_username(username)
        
        # If not found, try by email
        if not user:
            user = await self.user_repo.get_by_email(username)
        
        # User not found
        if not user:
            return None
        
        # TODO: Verify password hash
        # In production:
        # if not pwd_context.verify(password, user.hashed_password):
        #     return None
        
        # For now, just return the user (INSECURE!)
        # This is only for demonstration purposes
        if not user.is_active:
            return None
        
        return user


# ============================================================================
# Exports
# ============================================================================

__all__ = ["UserService"]
