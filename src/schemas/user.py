#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Schemas - Pydantic DTOs for User Domain
=============================================

Data Transfer Objects (DTOs) for API request/response validation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# ============================================================================
# Base Schema - Shared Fields
# ============================================================================

class UserBase(BaseModel):
    """Base user schema with shared fields."""
    
    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["user@example.com"]
    )
    
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Username (alphanumeric, underscore, hyphen only)",
        examples=["john_doe"]
    )
    
    full_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="User's full name",
        examples=["John Doe"]
    )


# ============================================================================
# Create Schema - For User Registration
# ============================================================================

class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="User password (min 8 characters)",
        examples=["SecurePassword123!"]
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether the user account is active"
    )
    
    is_superuser: bool = Field(
        default=False,
        description="Whether the user has admin privileges"
    )


# ============================================================================
# Update Schema - For User Updates
# ============================================================================

class UserUpdate(BaseModel):
    """Schema for updating user information."""
    
    email: Optional[EmailStr] = Field(
        default=None,
        description="New email address"
    )
    
    username: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        description="New username"
    )
    
    full_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New full name"
    )
    
    password: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=100,
        description="New password"
    )
    
    is_active: Optional[bool] = Field(
        default=None,
        description="Update active status"
    )
    
    is_superuser: Optional[bool] = Field(
        default=None,
        description="Update superuser status"
    )


# ============================================================================
# Response Schema - What the API Returns
# ============================================================================

class UserResponse(UserBase):
    """Schema for user responses (what the API returns)."""
    
    id: int = Field(
        ...,
        description="User ID",
        examples=[1]
    )
    
    is_active: bool = Field(
        ...,
        description="Whether the account is active"
    )
    
    is_superuser: bool = Field(
        ...,
        description="Whether the user is an admin"
    )
    
    created_at: datetime = Field(
        ...,
        description="When the user was created"
    )
    
    updated_at: datetime = Field(
        ...,
        description="When the user was last updated"
    )
    
    # Configure Pydantic to work with SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# List Response Schema
# ============================================================================

class UserListResponse(BaseModel):
    """Schema for paginated user list."""
    
    users: list[UserResponse] = Field(
        ...,
        description="List of users"
    )
    
    total: int = Field(
        ...,
        ge=0,
        description="Total number of users"
    )
    
    skip: int = Field(
        ...,
        ge=0,
        description="Number of records skipped"
    )
    
    limit: int = Field(
        ...,
        ge=1,
        description="Maximum records returned"
    )


# ============================================================================
# Authentication Schemas
# ============================================================================

class UserLogin(BaseModel):
    """Schema for user login."""
    
    username: str = Field(
        ...,
        description="Username or email"
    )
    
    password: str = Field(
        ...,
        description="User password"
    )


class Token(BaseModel):
    """Schema for authentication token response."""
    
    access_token: str = Field(
        ...,
        description="JWT access token"
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserListResponse",
    "UserLogin",
    "Token",
]
