#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Communication Minister - Press Secretary & Public Relations
============================================================

AI-powered public relations, announcements, and communications.
"""

from typing import Optional, Dict, Any
import logging

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class CommunicationMinister(BaseMinister):
    """
    Minister of Communication - Press secretary and public relations.
    
    Capabilities:
    - Draft announcements
    - Social media posts
    - Public statements
    - Crisis communication
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Communication Minister."""
        super().__init__(
            name="Communication Minister (Voice)",
            description="Press secretary and public relations specialist.",
            brain=brain
        )
        
        self.system_prompt = """You are the Chief Press Secretary and Communications Director.
Draft professional, clear, and engaging communications for:
- Public announcements
- Social media posts (Twitter/X, LinkedIn)
- Press releases
- Crisis communications

Tone: Professional, reassuring, and transparent.
Be concise and impactful."""
    
    async def draft_announcement(
        self,
        topic: str,
        data: Optional[Dict[str, Any]] = None,
        format_type: str = "general"
    ) -> str:
        """
        Draft an announcement.
        
        Args:
            topic: Announcement topic
            data: Optional data to include
            format_type: Type (tweet, press_release, general)
        
        Returns:
            Drafted announcement
        """
        if format_type == "tweet":
            prompt = f"""Draft a professional Twitter/X post (max 280 chars) about:
{topic}

Data: {data if data else 'None'}

Make it engaging and informative."""
        
        elif format_type == "press_release":
            prompt = f"""Draft a formal press release about:
{topic}

Data: {data if data else 'None'}

Include:
- Headline
- Date
- Body (3-4 paragraphs)
- Contact info placeholder"""
        
        else:
            prompt = f"""Draft a professional announcement about:
{topic}

Data: {data if data else 'None'}

Be clear, professional, and reassuring."""
        
        announcement = await self._think_with_prompt(
            system_prompt=self.system_prompt,
            user_message=prompt,
            max_tokens=400
        )
        
        return announcement
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute communication task.
        
        Args:
            task: Task description
            context: Optional context with data/format
        
        Returns:
            Communication draft
        """
        self.tasks_processed += 1
        
        try:
            # Extract format from context
            format_type = context.get("format", "general") if context else "general"
            data = context.get("data") if context else None
            
            # Draft announcement
            announcement = await self.draft_announcement(
                topic=task,
                data=data,
                format_type=format_type
            )
            
            self.tasks_successful += 1
            
            return {
                "success": True,
                "response": announcement,
                "minister": self.name,
                "domain": "communication",
                "metadata": {
                    "format": format_type,
                    "topic": task
                }
            }
        
        except Exception as e:
            logger.error(f"Communication Minister error: {e}")
            return {
                "success": False,
                "response": f"Communication draft failed: {str(e)}",
                "minister": self.name,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["CommunicationMinister"]
