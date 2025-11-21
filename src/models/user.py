#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Model - Example SQLAlchemy 2.0 Model
==========================================

Demonstrates best practices for model definition with:
- TimestampMixin for automatic timestamps
- Type-safe field definitions
- Proper indexes
"""

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """
    User model for authentication and profile management.
    
    Automatically includes:
    - id (primary key, auto-increment)
    - created_at (timestamp, auto-set on insert)
    - updated_at (timestamp, auto-set on insert/update)
    """
    
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Required fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        doc="User email address (unique)"
    )
    
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        doc="Username (unique)"
    )
    
    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="User's full name"
    )
    
    # Optional fields
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether the user account is active"
    )
    
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether the user has admin privileges"
    )
    
    # Note: In a real app, you'd store hashed_password here
    # For this example, we'll keep it simple
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


# ============================================================================
# Exports
# ============================================================================

__all__ = ["User"]
