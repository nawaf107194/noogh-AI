"""
Government API Router V2 - Modernized President & Cabinet Interface
====================================================================

Modern async API exposing the Government System with the new architecture.
Uses singleton pattern for President instance management.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from src.core.database import get_db
from src.services.government_service import GovernmentService
from src.schemas.government import (
    GovernmentChatRequest,
    GovernmentChatResponse,
    GovernmentStatusResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/government",
    tags=["🏛️ Government V2"],
    responses={500: {"description": "Government system error"}},
)


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/chat",
    response_model=GovernmentChatResponse,
    summary="Chat with the President",
    description="Send a message to the Government System and get a response from the President"
)
async def chat_with_government(
    request: GovernmentChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the Government System (President).
    
    The President will:
    1. Analyze the message and determine intent
    2. Dispatch to the appropriate minister
    3. Learn from the interaction
    4. **Save conversation to database** (NEW!)
    5. Return a response
    
    - **message**: Your message/request (1-2000 characters)
    - **context**: Optional context (conversation history, etc.)
    - **priority**: Task priority (low, medium, high, critical)
    
    Returns the President's response with minister and intent information.
    Automatically saves all interactions to the database for history and analytics.
    """
    try:
        service = GovernmentService()
        response = await service.process_message(
            message=request.message,
            context=request.context,
            priority=request.priority,
            db=db  # Pass database session for conversation persistence
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in government chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Government system error: {str(e)}"
        )


@router.get(
    "/status",
    response_model=GovernmentStatusResponse,
    summary="Get Cabinet Status",
    description="Get the current status of the government cabinet"
)
async def get_cabinet_status():
    """
    Get the status of the government cabinet.
    
    Returns:
    - Total and active ministers
    - Request statistics
    - Success rate
    - List of all ministers
    """
    try:
        service = GovernmentService()
        status = service.get_cabinet_status()
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting cabinet status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cabinet status: {str(e)}"
        )


@router.get(
    "/health",
    summary="Government System Health Check",
    description="Check if the government system is operational"
)
async def government_health_check():
    """
    Quick health check for the government system.
    
    Returns:
        Health status
    """
    try:
        service = GovernmentService()
        president = service.get_president()
        
        return {
            "status": "healthy",
            "system": "government_v2",
            "president_initialized": president is not None,
            "ministers_count": len(president.cabinet) if president else 0
        }
    
    except Exception as e:
        logger.error(f"Government health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Government system unhealthy"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = ["router"]
