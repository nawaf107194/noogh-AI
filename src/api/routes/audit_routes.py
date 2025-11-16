from fastapi import APIRouter
from ..models import APIResponse
from api.utils.logger import get_logger, LogCategory

router = APIRouter()
logger = get_logger("audit_api")

@router.get("/audit/status", response_model=APIResponse)
async def get_audit_status():
    """
    Get the status of the audit system.
    """
    return APIResponse(
        success=True,
        data={
            "status": "operational",
            "last_audit": "2025-11-15T20:00:00Z"
        }
    )
