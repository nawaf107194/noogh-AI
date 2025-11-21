#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Modern Async Route with Database Session
==================================================

This demonstrates the proper pattern for using async database sessions
in FastAPI routes with dependency injection.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import logging

from src.core.database import get_db
from src.core.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Example: Simple Status Route (No Database)
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Simple health check - no database needed.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.version,
        "database": settings.database_url.split("@")[-1] if "@" in settings.database_url else settings.database_url,
    }


# ============================================================================
# Example: Route with Database Session
# ============================================================================

@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    """
    Get system statistics with database access.
    
    Args:
        db: Async database session (injected by FastAPI)
    
    Returns:
        System statistics
    
    The database session is:
    - ✅ Created automatically before the request
    - ✅ Passed to the function
    - ✅ Committed if successful
    - ✅ Rolled back on error
    - ✅ Closed automatically after the request
    """
    try:
        # Example: Query database
        # result = await db.execute(select(SomeModel))
        # items = result.scalars().all()
        
        # For now, just return mock stats
        return {
            "status": "operational",
            "database_connected": True,
            "ministers": settings.num_ministers,
            "gpu_enabled": settings.use_gpu,
            "debug_mode": settings.debug_mode,
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Example: Create Operation with Database
# ============================================================================

@router.post("/items")
async def create_item(
    name: str,
    description: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new item in the database.
    
    Args:
        name: Item name
        description: Item description (optional)
        db: Async database session
    
    Returns:
        Created item
    """
    try:
        # Example: Create and save object
        # new_item = Item(name=name, description=description)
        # db.add(new_item)
        # await db.flush()  # Get the ID without committing
        # await db.refresh(new_item)  # Refresh to get generated fields
        
        # Mock response
        return {
            "id": 1,
            "name": name,
            "description": description,
            "status": "created"
        }
    
    except Exception as e:
        logger.error(f"Error creating item: {e}")
        # Session will automatically rollback!
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Example: Query with Filtering
# ============================================================================

@router.get("/items")
async def list_items(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    List items with pagination.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
        db: Async database session
    
    Returns:
        List of items
    """
    try:
        # Example: Query with pagination
        # stmt = select(Item).offset(skip).limit(limit)
        # result = await db.execute(stmt)
        # items = result.scalars().all()
        
        # Mock response
        return {
            "items": [],
            "total": 0,
            "skip": skip,
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"Error listing items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Example: Update Operation
# ============================================================================

@router.put("/items/{item_id}")
async def update_item(
    item_id: int,
    name: str = None,
    description: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing item.
    
    Args:
        item_id: Item ID
        name: New name (optional)
        description: New description (optional)
        db: Async database session
    
    Returns:
        Updated item
    """
    try:
        # Example: Get and update
        # stmt = select(Item).where(Item.id == item_id)
        # result = await db.execute(stmt)
        # item = result.scalar_one_or_none()
        #
        # if not item:
        #     raise HTTPException(status_code=404, detail="Item not found")
        #
        # if name:
        #     item.name = name
        # if description:
        #     item.description = description
        #
        # await db.flush()
        # await db.refresh(item)
        
        # Mock response
        return {
            "id": item_id,
            "name": name or "original name",
            "description": description,
            "status": "updated"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Example: Delete Operation
# ============================================================================

@router.delete("/items/{item_id}")
async def delete_item(
    item_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an item.
    
    Args:
        item_id: Item ID
        db: Async database session
    
    Returns:
        Success message
    """
    try:
        # Example: Delete
        # stmt = select(Item).where(Item.id == item_id)
        # result = await db.execute(stmt)
        # item = result.scalar_one_or_none()
        #
        # if not item:
        #     raise HTTPException(status_code=404, detail="Item not found")
        #
        # await db.delete(item)
        
        # Mock response
        return {
            "id": item_id,
            "status": "deleted"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Anti-Pattern Examples (DO NOT USE)
# ============================================================================

"""
❌ WRONG - Don't create sessions manually:

@router.get("/bad-example")
async def bad_example():
    db = SessionLocal()  # WRONG! No cleanup
    try:
        result = await db.execute(select(Item))
        return result.scalars().all()
    finally:
        await db.close()  # Easy to forget!


❌ WRONG - Don't use global sessions:

db = SessionLocal()  # WRONG! Shared across requests

@router.get("/another-bad-example")
async def another_bad_example():
    result = await db.execute(select(Item))
    return result.scalars().all()


✅ CORRECT - Use dependency injection:

@router.get("/good-example")
async def good_example(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Item))
    return result.scalars().all()
"""
