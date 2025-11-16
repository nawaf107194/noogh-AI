from fastapi import APIRouter
from ..models import APIResponse
from api.utils.logger import get_logger, LogCategory

router = APIRouter()
logger = get_logger("brain_api")

@router.get("/brain/status", response_model=APIResponse)
async def get_brain_status():
    """
    Get the status of the brain.
    """
    return APIResponse(
        success=True,
        data={
            "status": "operational",
            "neurons": 326,
            "type": "unified_system"
        }
    )
