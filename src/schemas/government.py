#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government Schemas - Pydantic DTOs for Government Domain
=========================================================

Data Transfer Objects for President and Cabinet interactions.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Chat Schemas
# ============================================================================

class GovernmentChatRequest(BaseModel):
    """Schema for sending a message to the government system."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to process",
        examples=["What is the system status?", "Analyze the trading market"]
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the request (conversation history, user info, etc.)"
    )
    
    priority: str = Field(
        default="medium",
        pattern="^(low|medium|high|critical)$",
        description="Task priority level",
        examples=["medium", "high"]
    )


class GovernmentChatResponse(BaseModel):
    """Schema for government system response."""
    
    success: bool = Field(
        ...,
        description="Whether the request was processed successfully"
    )
    
    response: str = Field(
        ...,
        description="The President's response message"
    )
    
    minister: Optional[str] = Field(
        default=None,
        description="Which minister handled the request"
    )
    
    intent: Optional[str] = Field(
        default=None,
        description="Detected intent from the message"
    )
    
    task_id: Optional[str] = Field(
        default=None,
        description="Unique task identifier"
    )
    
    status: Optional[str] = Field(
        default=None,
        description="Task execution status"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata from processing"
    )


# ============================================================================
# Status Schemas
# ============================================================================

class GovernmentStatusResponse(BaseModel):
    """Schema for cabinet status information."""
    
    total_ministers: int = Field(
        ...,
        ge=0,
        description="Total number of ministers in the cabinet"
    )
    
    active_ministers: int = Field(
        ...,
        ge=0,
        description="Number of currently active ministers"
    )
    
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total requests processed"
    )
    
    successful_requests: int = Field(
        ...,
        ge=0,
        description="Number of successful requests"
    )
    
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Success rate (0.0 to 1.0)"
    )
    
    ministers: List[str] = Field(
        ...,
        description="List of minister names"
    )
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "GovernmentChatRequest",
    "GovernmentChatResponse",
    "GovernmentStatusResponse",
]
