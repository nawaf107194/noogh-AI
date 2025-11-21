#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User API Router - REST Endpoints for User Management
====================================================

FastAPI router implementing user CRUD endpoints.
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from src.core.database import get_db
from src.services.user_service import UserService
from src.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
)


# Create router
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Register a new user account with email, username, and password"
)
async def create_user(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Create a new user.
    
    - **email**: Valid email address (must be unique)
    - **username**: Username 3-50 characters (must be unique)
    - **full_name**: User's full name
    - **password**: Secure password (min 8 characters)
    - **is_active**: Whether account is active (default: True)
    - **is_superuser**: Whether user is admin (default: False)
    
    Returns the created user (without password).
    """
    service = UserService(db)
    user = await service.create_user(user_data)
    return user


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Retrieve a single user by their ID"
)
async def get_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Get user by ID.
    
    Args:
        user_id: User ID
    
    Returns:
        User information
    
    Raises:
        404: User not found
    """
    service = UserService(db)
    user = await service.get_user(user_id)
    return user


@router.get(
    "/",
    response_model=UserListResponse,
    summary="List users",
    description="Get a paginated list of users"
)
async def list_users(
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Max records to return")] = 10,
    active_only: Annotated[bool, Query(description="Filter active users only")] = False,
    db: Annotated[AsyncSession, Depends(get_db)] = None
):
    """
    List users with pagination.
    
    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum records to return (default: 10, max: 100)
        active_only: If True, only return active users
    
    Returns:
        Paginated list of users with total count
    """
    service = UserService(db)
    users, total = await service.list_users(
        skip=skip,
        limit=limit,
        active_only=active_only
    )
    
    return UserListResponse(
        users=users,
        total=total,
        skip=skip,
        limit=limit
    )


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user information"
)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Update user information.
    
    All fields are optional. Only provided fields will be updated.
    
    Args:
        user_id: User ID
        user_data: Fields to update
    
    Returns:
        Updated user
    
    Raises:
        404: User not found
        400: Email or username conflict
    """
    service = UserService(db)
    user = await service.update_user(user_id, user_data)
    return user


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Permanently delete a user (hard delete)"
)
async def delete_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Delete user (hard delete).
    
    WARNING: This permanently deletes the user from the database.
    For deactivation, use POST /users/{user_id}/deactivate instead.
    
    Args:
        user_id: User ID
    
    Raises:
        404: User not found
    """
    service = UserService(db)
    await service.delete_user(user_id)
    return


@router.post(
    "/{user_id}/deactivate",
    response_model=UserResponse,
    summary="Deactivate user",
    description="Deactivate user account (soft delete)"
)
async def deactivate_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Deactivate user account (soft delete).
    
    The user remains in the database but cannot log in.
    This is preferred over hard delete for data integrity.
    
    Args:
        user_id: User ID
    
    Returns:
        Deactivated user
    
    Raises:
        404: User not found
    """
    service = UserService(db)
    user = await service.deactivate_user(user_id)
    return user


@router.post(
    "/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user",
    description="Reactivate a deactivated user account"
)
async def activate_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Activate user account.
    
    Reactivates a previously deactivated account.
    
    Args:
        user_id: User ID
    
    Returns:
        Activated user
    
    Raises:
        404: User not found
    """
    service = UserService(db)
    user = await service.activate_user(user_id)
    return user


# ============================================================================
# Exports
# ============================================================================

__all__ = ["router"]
