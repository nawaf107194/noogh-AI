#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Models - SQLAlchemy 2.0 Declarative Base and Mixins
=========================================================

Provides:
- DeclarativeBase for all models
- TimestampMixin for created_at/updated_at fields
- Common model utilities
"""

from datetime import datetime
from typing import Any
from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, declared_attr


# ============================================================================
# Declarative Base (SQLAlchemy 2.0 Style)
# ============================================================================

class Base(DeclarativeBase):
    """
    Base class for all database models.
    
    SQLAlchemy 2.0 style with:
    - Type hints (Mapped)
    - Automatic table naming
    - Common utilities
    """
    
    # Automatically generate __tablename__ from class name
    @declared_attr.directive
    @classmethod
    def __tablename__(cls) -> str:
        """Generate table name from class name (snake_case)"""
        import re
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert model to dictionary.
        
        Returns:
            Dictionary representation of the model
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """String representation of the model"""
        # Show primary key columns
        id_str = ""
        for pk_col in self.__table__.primary_key.columns:
            val = getattr(self, pk_col.name, None)
            id_str += f"{pk_col.name}={val}, "
        
        return f"<{self.__class__.__name__}({id_str.rstrip(', ')})>"


# ============================================================================
# Mixins
# ============================================================================

class TimestampMixin:
    """
    Mixin that adds created_at and updated_at timestamp fields.
    
    Usage:
        class User(Base, TimestampMixin):
            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str]
    
    The model will automatically have:
    - created_at: Set on insert
    - updated_at: Set on insert and updated on every update
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when the record was created"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Timestamp when the record was last updated"
    )


class SoftDeleteMixin:
    """
    Mixin that adds soft delete functionality.
    
    Usage:
        class User(Base, TimestampMixin, SoftDeleteMixin):
            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str]
    
    Instead of deleting records, sets deleted_at timestamp.
    """
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="Timestamp when the record was soft deleted"
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted"""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark record as deleted"""
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore soft deleted record"""
        self.deleted_at = None


class UserTrackingMixin:
    """
    Mixin that tracks which user created/modified a record.
    
    Usage:
        class Article(Base, UserTrackingMixin):
            id: Mapped[int] = mapped_column(primary_key=True)
            title: Mapped[str]
    
    Requires manually setting created_by_id and updated_by_id.
    """
    
    created_by_id: Mapped[int | None] = mapped_column(
        nullable=True,
        doc="ID of user who created this record"
    )
    
    updated_by_id: Mapped[int | None] = mapped_column(
        nullable=True,
        doc="ID of user who last updated this record"
    )


# ============================================================================
# Example Model (for reference)
# ============================================================================

# Uncomment to use as template:
"""
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column

class Example(Base, TimestampMixin):
    '''Example model showing best practices'''
    
    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Required fields
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    
    # Optional fields
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Automatically gets: created_at, updated_at from TimestampMixin
    
    def __repr__(self) -> str:
        return f"<Example(id={self.id}, name='{self.name}')>"
"""


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Base",
    "TimestampMixin",
    "SoftDeleteMixin",
    "UserTrackingMixin",
]


if __name__ == "__main__":
    # Show mixin usage example
    print("=" * 70)
    print("📦 Base Models & Mixins")
    print("=" * 70)
    print("\nAvailable Mixins:")
    print("  1. TimestampMixin - Auto created_at/updated_at")
    print("  2. SoftDeleteMixin - Soft delete with deleted_at")
    print("  3. UserTrackingMixin - Track created_by/updated_by")
    print("\nExample Usage:")
    print("""
    class User(Base, TimestampMixin):
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))
        email: Mapped[str] = mapped_column(String(255), unique=True)
    """)
    print("=" * 70)
