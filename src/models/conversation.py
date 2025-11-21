#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Model - Database Model for President Interactions
==============================================================

Stores all conversations with the Government System (President).
"""

from sqlalchemy import String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base import Base, TimestampMixin


class Conversation(Base, TimestampMixin):
    """
    Model for storing conversations with the President.
    
    Automatically includes:
    - id (primary key)
    - created_at (timestamp)
    - updated_at (timestamp)
    """
    
    __tablename__ = "conversations"
    
    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Conversation content
    user_input: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="User's input message to the President"
    )
    
    ai_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="President's response"
    )
    
    # Metadata
    minister_name: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Which minister handled the request"
    )
    
    intent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Detected intent from the message"
    )
    
    status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        doc="Task execution status (completed, failed, etc.)"
    )
    
    execution_time_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Execution time in milliseconds"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, minister='{self.minister_name}', intent='{self.intent}')>"


# ============================================================================
# Exports
# ============================================================================

__all__ = ["Conversation"]
