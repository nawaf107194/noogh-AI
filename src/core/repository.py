#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Repository Pattern - Base Repository for Data Access
=============================================================

Implements a generic repository pattern with async SQLAlchemy for clean
separation of data access logic from business logic.

Benefits:
- DRY: Don't repeat CRUD operations
- Testability: Easy to mock repositories
- Maintainability: Database logic in one place
- Type Safety: Generic types for compile-time checking
"""

from typing import Generic, TypeVar, Type, Optional, List, Any, Dict
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from .base import Base

# Generic type for SQLAlchemy model
ModelType = TypeVar("ModelType", bound=Base)
# Generic type for Pydantic schema
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType]):
    """
    Generic async repository for database operations.
    
    Usage:
        class UserRepository(BaseRepository[User]):
            pass
        
        # In route
        async def get_users(db: AsyncSession = Depends(get_db)):
            repo = UserRepository(db, User)
            users = await repo.list()
            return users
    """
    
    def __init__(self, db: AsyncSession, model: Type[ModelType]):
        """
        Initialize repository.
        
        Args:
            db: Async database session
            model: SQLAlchemy model class
        """
        self.db = db
        self.model = model
    
    async def create(self, obj_in: Dict[str, Any] | BaseModel) -> ModelType:
        """
        Create a new record.
        
        Args:
            obj_in: Pydantic model or dict with data
        
        Returns:
            Created model instance
        
        Example:
            user = await repo.create({"name": "John", "email": "john@example.com"})
        """
        # Convert Pydantic model to dict if needed
        if isinstance(obj_in, BaseModel):
            obj_data = obj_in.model_dump(exclude_unset=True)
        else:
            obj_data = obj_in
        
        db_obj = self.model(**obj_data)
        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj
    
    async def get(self, id: Any) -> Optional[ModelType]:
        """
        Get a single record by ID.
        
        Args:
            id: Primary key value
        
        Returns:
            Model instance or None if not found
        
        Example:
            user = await repo.get(1)
            if user:
                print(user.name)
        """
        stmt = select(self.model).where(self.model.id == id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by(self, **filters) -> Optional[ModelType]:
        """
        Get a single record by custom filters.
        
        Args:
            **filters: Column name and value pairs
        
        Returns:
            Model instance or None
        
        Example:
            user = await repo.get_by(email="john@example.com")
        """
        stmt = select(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        **filters
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filters.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            **filters: Optional filters (column=value)
        
        Returns:
            List of model instances
        
        Example:
            users = await repo.list(skip=0, limit=10, is_active=True)
        """
        stmt = select(self.model)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def count(self, **filters) -> int:
        """
        Count records matching filters.
        
        Args:
            **filters: Optional filters
        
        Returns:
            Number of matching records
        
        Example:
            total_users = await repo.count(is_active=True)
        """
        stmt = select(func.count()).select_from(self.model)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)
        
        result = await self.db.execute(stmt)
        return result.scalar() or 0
    
    async def update(
        self,
        id: Any,
        obj_in: Dict[str, Any] | BaseModel
    ) -> Optional[ModelType]:
        """
        Update an existing record.
        
        Args:
            id: Primary key value
            obj_in: Pydantic model or dict with update data
        
        Returns:
            Updated model instance or None if not found
        
        Example:
            user = await repo.update(1, {"name": "Jane"})
        """
        # Get existing object
        db_obj = await self.get(id)
        if not db_obj:
            return None
        
        # Convert Pydantic model to dict if needed
        if isinstance(obj_in, BaseModel):
            update_data = obj_in.model_dump(exclude_unset=True)
        else:
            update_data = obj_in
        
        # Update fields
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj
    
    async def delete(self, id: Any) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Primary key value
        
        Returns:
            True if deleted, False if not found
        
        Example:
            deleted = await repo.delete(1)
            if deleted:
                print("User deleted")
        """
        db_obj = await self.get(id)
        if not db_obj:
            return False
        
        await self.db.delete(db_obj)
        await self.db.flush()
        return True
    
    async def exists(self, id: Any) -> bool:
        """
        Check if a record exists.
        
        Args:
            id: Primary key value
        
        Returns:
            True if exists, False otherwise
        
        Example:
            if await repo.exists(1):
                print("User exists")
        """
        stmt = select(func.count()).select_from(self.model).where(self.model.id == id)
        result = await self.db.execute(stmt)
        count = result.scalar()
        return count > 0


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BaseRepository",
    "ModelType",
    "CreateSchemaType",
    "UpdateSchemaType",
]
