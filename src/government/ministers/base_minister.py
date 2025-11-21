#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Minister - Abstract Base Class for AI-Powered Ministers
=============================================================

All ministers inherit from this and use LocalBrainService for intelligence.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseMinister(ABC):
    """
    Abstract base class for all government ministers.
    
    Each minister has access to the LocalBrainService (Meta-Llama-3-8B)
    and implements domain-specific logic via execute_task().
    
    Attributes:
        name: Minister's name/title
        description: Minister's role and responsibilities
        brain: LocalBrainService instance for AI inference
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        brain: Optional[Any] = None
    ):
        """
        Initialize a minister.
        
        Args:
            name: Minister's name (e.g., "Education Minister")
            description: Role description
            brain: LocalBrainService instance (injected)
        """
        self.name = name
        self.description = description
        self.brain = brain
        
        # Statistics
        self.tasks_processed = 0
        self.tasks_successful = 0
        
        logger.info(f"✅ {self.name} initialized")
    
    @abstractmethod
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task specific to this minister's domain.
        
        This method MUST be implemented by all ministers.
        
        Args:
            task: Task description/user request
            context: Optional context (history, user info, etc.)
        
        Returns:
            Dictionary with:
                - success: bool
                - response: str (the answer/result)
                - minister: str (name)
                - metadata: dict (optional extra info)
        
        Example:
            result = await minister.execute_task("Explain Python loops")
            print(result["response"])
        """
        pass
    
    async def _think_with_prompt(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512
    ) -> str:
        """
        Helper to use brain with a system prompt.
        
        Args:
            system_prompt: Domain-specific instructions for the AI
            user_message: The actual user query
            max_tokens: Max tokens to generate
        
        Returns:
            AI response
        """
        if self.brain is None:
            return "Brain not available. Please initialize LocalBrainService."
        
        # Combine system prompt + user message
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        try:
            response = await self.brain.think(full_prompt, max_tokens=max_tokens)
            return response
        except Exception as e:
            logger.error(f"Error in {self.name} brain inference: {e}")
            return f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get minister's performance statistics."""
        success_rate = (
            self.tasks_successful / self.tasks_processed
            if self.tasks_processed > 0
            else 0.0
        )
        
        return {
            "name": self.name,
            "tasks_processed": self.tasks_processed,
            "tasks_successful": self.tasks_successful,
            "success_rate": success_rate
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["BaseMinister"]
