#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Repository - Data Access for Conversations
========================================================

Repository for managing conversation history with the President.
"""

from typing import List
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.repository import BaseRepository
from src.models.conversation import Conversation


class ConversationRepository(BaseRepository[Conversation]):
    """
    Repository for Conversation model operations.
    
    Inherits standard CRUD from BaseRepository:
    - create(obj_in) -> Conversation
    - get(id) -> Optional[Conversation]
    - list(skip, limit) -> List[Conversation]
    - update(id, obj_in) -> Optional[Conversation]
    - delete(id) -> bool
    
    Plus custom methods for conversation history.
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize ConversationRepository with Conversation model."""
        super().__init__(db, Conversation)
    
    async def get_recent_chats(self, limit: int = 10) -> List[Conversation]:
        """
        Get most recent conversations.
        
        Args:
            limit: Maximum number of conversations to return (default: 10)
        
        Returns:
            List of recent conversations, newest first
        
        Example:
            recent = await conv_repo.get_recent_chats(limit=20)
            for chat in recent:
                print(f"User: {chat.user_input}")
                print(f"AI: {chat.ai_response}")
        """
        stmt = (
            select(self.model)
            .order_by(desc(self.model.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_by_minister(
        self,
        minister_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """
        Get conversations handled by a specific minister.
        
        Args:
            minister_name: Name of the minister
            skip: Number of records to skip
            limit: Maximum records to return
        
        Returns:
            List of conversations by minister
        
        Example:
            education_chats = await conv_repo.get_by_minister("education")
        """
        return await self.list(
            skip=skip,
            limit=limit,
            minister_name=minister_name
        )
    
    async def get_by_intent(
        self,
        intent: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """
        Get conversations by detected intent.
        
        Args:
            intent: Intent type
            skip: Number of records to skip
            limit: Maximum records to return
        
        Returns:
            List of conversations with matching intent
        
        Example:
            questions = await conv_repo.get_by_intent("question_kb")
        """
        return await self.list(
            skip=skip,
            limit=limit,
            intent=intent
        )
    
    async def get_failed_conversations(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """
        Get conversations that failed to complete.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
        
        Returns:
            List of failed conversations
        """
        stmt = (
            select(self.model)
            .where(self.model.status != "completed")
            .order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


# ============================================================================
# Exports
# ============================================================================

__all__ = ["ConversationRepository"]
