#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government Service - Business Logic for President & Cabinet
============================================================

Service layer managing the Government System (President) with singleton pattern.
The President maintains state across requests for learning and memory.
"""

import logging
from typing import Optional, Dict, Any
from threading import Lock

from ..government.president import President
from ..schemas.government import GovernmentChatResponse, GovernmentStatusResponse

logger = logging.getLogger(__name__)


class GovernmentService:
    """
    Service managing the Government System.
    
    Uses singleton pattern to maintain a single President instance
    across all requests, enabling:
    - Persistent learning and memory
    - Conversation context across requests
    - Consistent cabinet state
    
    Thread-safe singleton implementation.
    """
    
    _president_instance: Optional[President] = None
    _lock: Lock = Lock()
    
    @classmethod
    def get_president(cls) -> President:
        """
        Get or create the singleton President instance.
        
        Thread-safe lazy initialization.
        
        Returns:
            The President instance
        """
        if cls._president_instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._president_instance is None:
                    logger.info("🏛️ Initializing Government System (President)")
                    cls._president_instance = President(verbose=True)
                    logger.info("✅ Government System ready")
        
        return cls._president_instance
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        db: Optional[Any] = None  # AsyncSession, optional for backward compatibility
    ) -> GovernmentChatResponse:
        """
        Process a user message through the government system.
        
        If db session is provided, automatically saves the conversation to database.
        
        Args:
            message: User message/request
            context: Optional context (conversation history, user info, etc.)
            priority: Task priority (low, medium, high, critical)
            db: Optional database session for conversation persistence
        
        Returns:
            Government chat response with President's answer
        
        Example:
            service = GovernmentService()
            response = await service.process_message(
                "What is the system status?",
                priority="high",
                db=db_session
            )
        """
        import time
        start_time = time.time()
        
        president = self.get_president()
        
        try:
            # Process request through the President
            result = await president.process_request(
                user_input=message,
                context=context or {},
                priority=priority
            )
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Transform result to response schema
            # The President returns a dict from minister_result.to_dict()
            if isinstance(result, dict):
                response = GovernmentChatResponse(
                    success=result.get("status") == "completed",
                    response=self._extract_response_message(result),
                    minister=result.get("minister"),
                    intent=result.get("task_type"),
                    task_id=result.get("task_id"),
                    status=result.get("status"),
                    metadata=result.get("result", {})
                )
            else:
                # Fallback for unexpected result format
                response = GovernmentChatResponse(
                    success=True,
                    response=str(result),
                    minister="unknown",
                    intent="unknown"
                )
            
            # Save to database if session provided
            if db is not None:
                try:
                    await self._save_conversation(
                        db=db,
                        user_input=message,
                        ai_response=response.response,
                        minister_name=response.minister,
                        intent=response.intent,
                        status=response.status or "unknown",
                        execution_time_ms=execution_time_ms
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save conversation: {save_error}", exc_info=True)
                    # Don't fail the request if saving fails
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing message through government: {e}", exc_info=True)
            
            # Still try to save failed conversation if db provided
            execution_time_ms = (time.time() - start_time) * 1000
            if db is not None:
                try:
                    await self._save_conversation(
                        db=db,
                        user_input=message,
                        ai_response=f"Error: {str(e)}",
                        minister_name=None,
                        intent=None,
                        status="failed",
                        execution_time_ms=execution_time_ms
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save error conversation: {save_error}")
            
            return GovernmentChatResponse(
                success=False,
                response=f"Error processing request: {str(e)}",
                minister=None,
                intent=None
            )
    
    def _extract_response_message(self, result: Dict[str, Any]) -> str:
        """
        Extract a user-friendly message from the minister result.
        
        Args:
            result: Result dict from minister execution
        
        Returns:
            User-friendly response message
        """
        # Try different paths to extract the message
        if "result" in result and isinstance(result["result"], dict):
            result_data = result["result"]
            
            # Check for message field
            if "message" in result_data:
                return result_data["message"]
            
            # Check for response field
            if "response" in result_data:
                return result_data["response"]
            
            # Check for answer field
            if "answer" in result_data:
                return result_data["answer"]
        
        # Fallback: stringify the result
        if "result" in result:
            return str(result["result"])
        
        return f"Task {result.get('status', 'completed')} by {result.get('minister', 'minister')}"
    
    def get_cabinet_status(self) -> GovernmentStatusResponse:
        """
        Get the current status of the government cabinet.
        
        Returns:
            Cabinet status information
        """
        president = self.get_president()
        status = president.get_cabinet_status()
        
        return GovernmentStatusResponse(
            total_ministers=status["total_ministers"],
            active_ministers=status["active_ministers"],
            total_requests=status["total_requests"],
            successful_requests=status["successful_requests"],
            success_rate=status["success_rate"],
            ministers=status["ministers"]
        )
    
    @classmethod
    def reset_president(cls):
        """
        Reset the President instance (useful for testing).
        
        Warning: This will clear all learned memories and reset statistics.
        """
        with cls._lock:
            if cls._president_instance is not None:
                logger.warning("🔄 Resetting Government System")
                cls._president_instance = None
    
    async def _save_conversation(
        self,
        db: Any,  # AsyncSession
        user_input: str,
        ai_response: str,
        minister_name: Optional[str],
        intent: Optional[str],
        status: str,
        execution_time_ms: float
    ):
        """
        Save conversation to database.
        
        Args:
            db: AsyncSession for database access
            user_input: User's message
            ai_response: AI's response
            minister_name: Which minister handled it
            intent: Detected intent
            status: Execution status
            execution_time_ms: Time taken to process
        """
        from ..repositories.conversation_repository import ConversationRepository
        
        repo = ConversationRepository(db)
        await repo.create({
            "user_input": user_input,
            "ai_response": ai_response,
            "minister_name": minister_name,
            "intent": intent,
            "status": status,
            "execution_time_ms": execution_time_ms
        })
        logger.info(f"💾 Saved conversation to database (minister: {minister_name}, intent: {intent})")


# ============================================================================
# Exports
# ============================================================================

__all__ = ["GovernmentService"]
