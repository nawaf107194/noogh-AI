"""
⚙️ Training Auto-Rotation API - Scheduled Training & Model Management
REST API for automatic training, model versioning, and deployment

Endpoints:
- GET  /api/training/status          → Scheduler and models status
- GET  /api/training/jobs            → Recent training jobs
- GET  /api/training/jobs/{job_id}   → Specific job details
- POST /api/training/start           → Start training job
- POST /api/training/scheduler/start → Start scheduler
- POST /api/training/scheduler/stop  → Stop scheduler
- GET  /api/training/models/versions → All model versions
- GET  /api/training/models/active   → Active model version
- POST /api/training/models/rollback → Rollback to previous version
- GET  /api/training/models/compare  → Compare model versions
"""

import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autonomy.training_scheduler import get_training_scheduler
from autonomy.model_manager import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/training",
    tags=["⚙️ Training Auto-Rotation", "Model Versioning", "Scheduled Training"],
)

# Models
class TrainingRequest(BaseModel):
    """Request to start training"""
    training_params: Optional[dict] = None


@router.get("/status")
async def get_training_status():
    """
    Get training scheduler and models status

    Returns:
        Comprehensive status including:
        - Scheduler running status
        - Current job
        - Recent jobs
        - Model statistics
        - Next training time
    """
    try:
        scheduler = get_training_scheduler()
        status = scheduler.get_status()

        return JSONResponse(
            content={
                "success": True,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/jobs")
async def get_training_jobs(limit: int = 10):
    """
    Get recent training jobs

    Args:
        limit: Number of jobs to return (default 10, max 50)

    Returns:
        List of recent training jobs
    """
    try:
        limit = min(max(limit, 1), 50)

        scheduler = get_training_scheduler()
        jobs = scheduler.get_recent_jobs(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(jobs),
                "limit": limit,
                "jobs": [j.to_dict() for j in jobs]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get jobs: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    """
    Get specific training job details

    Args:
        job_id: Training job ID

    Returns:
        Job details including status, metrics, comparison
    """
    try:
        scheduler = get_training_scheduler()
        job = scheduler.get_job(job_id)

        if not job:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Job not found: {job_id}"
                },
                status_code=404
            )

        return JSONResponse(
            content={
                "success": True,
                "job": job.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get job: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/start")
async def start_training(request: TrainingRequest):
    """
    Start a new training job

    Body:
        training_params: Optional training parameters

    Returns:
        Created job details
    """
    try:
        logger.info("🚀 Manual training job requested")

        scheduler = get_training_scheduler()

        # Schedule the job
        job = scheduler.schedule_training(
            training_params=request.training_params
        )

        # Execute asynchronously
        import asyncio
        asyncio.create_task(scheduler.execute_training_job(job))

        return JSONResponse(
            content={
                "success": True,
                "message": "Training job started",
                "job": job.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/scheduler/start")
async def start_scheduler():
    """
    Start the training scheduler

    Returns:
        Confirmation of scheduler start
    """
    try:
        logger.info("▶️  Starting training scheduler via API")

        scheduler = get_training_scheduler()
        scheduler.start()

        stats = scheduler.get_statistics()

        return JSONResponse(
            content={
                "success": True,
                "message": "Training scheduler started",
                "statistics": stats
            }
        )

    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/scheduler/stop")
async def stop_scheduler():
    """
    Stop the training scheduler

    Returns:
        Confirmation of scheduler stop
    """
    try:
        logger.info("⏸️  Stopping training scheduler via API")

        scheduler = get_training_scheduler()
        scheduler.stop()

        stats = scheduler.get_statistics()

        return JSONResponse(
            content={
                "success": True,
                "message": "Training scheduler stopped",
                "statistics": stats
            }
        )

    except Exception as e:
        logger.error(f"Failed to stop scheduler: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/models/versions")
async def get_model_versions(limit: int = 20):
    """
    Get all model versions

    Args:
        limit: Number of versions to return (default 20, max 100)

    Returns:
        List of model versions with performance metrics
    """
    try:
        limit = min(max(limit, 1), 100)

        model_manager = get_model_manager()
        all_versions = model_manager.get_all_versions()

        versions = all_versions[:limit]

        return JSONResponse(
            content={
                "success": True,
                "count": len(versions),
                "total_versions": len(all_versions),
                "limit": limit,
                "versions": [v.to_dict() for v in versions]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get versions: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/models/active")
async def get_active_model():
    """
    Get active model version

    Returns:
        Active model details and performance metrics
    """
    try:
        model_manager = get_model_manager()
        active_version = model_manager.active_version

        if not active_version:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No active model version"
                },
                status_code=404
            )

        return JSONResponse(
            content={
                "success": True,
                "active_version": active_version.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get active model: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/models/rollback")
async def rollback_model():
    """
    Rollback to previous model version

    Returns:
        Previous version details after rollback
    """
    try:
        logger.warning("⚠️ Manual rollback requested via API")

        model_manager = get_model_manager()
        previous_version = model_manager.rollback_to_previous()

        if not previous_version:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No previous version available for rollback"
                },
                status_code=400
            )

        return JSONResponse(
            content={
                "success": True,
                "message": f"Rolled back to version: {previous_version.version_id}",
                "version": previous_version.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/models/compare")
async def compare_model_versions(version1: str, version2: str):
    """
    Compare two model versions

    Args:
        version1: First version ID
        version2: Second version ID

    Returns:
        Comparison results with performance differences
    """
    try:
        model_manager = get_model_manager()

        v1 = model_manager.get_version(version1)
        v2 = model_manager.get_version(version2)

        if not v1:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Version not found: {version1}"
                },
                status_code=404
            )

        if not v2:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Version not found: {version2}"
                },
                status_code=404
            )

        # Extract metrics
        m1 = v1.performance_metrics
        m2 = v2.performance_metrics

        # Calculate differences
        comparison = {
            "version1": {
                "version_id": v1.version_id,
                "cognition_score": m1.get('cognition_score', 0),
                "accuracy": m1.get('accuracy', 0),
                "avg_response_time_ms": m1.get('avg_response_time_ms', 0),
            },
            "version2": {
                "version_id": v2.version_id,
                "cognition_score": m2.get('cognition_score', 0),
                "accuracy": m2.get('accuracy', 0),
                "avg_response_time_ms": m2.get('avg_response_time_ms', 0),
            },
            "differences": {
                "cognition_diff": m2.get('cognition_score', 0) - m1.get('cognition_score', 0),
                "accuracy_diff": m2.get('accuracy', 0) - m1.get('accuracy', 0),
                "speed_ratio": m1.get('avg_response_time_ms', 1) / max(m2.get('avg_response_time_ms', 1), 1),
            }
        }

        return JSONResponse(
            content={
                "success": True,
                "comparison": comparison
            }
        )

    except Exception as e:
        logger.error(f"Failed to compare versions: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/statistics")
async def get_training_statistics():
    """
    Get training and model statistics

    Returns:
        - Total training jobs
        - Success/failure rates
        - Model version statistics
        - Performance trends
    """
    try:
        scheduler = get_training_scheduler()
        model_manager = get_model_manager()

        scheduler_stats = scheduler.get_statistics()
        model_stats = model_manager.get_statistics()

        return JSONResponse(
            content={
                "success": True,
                "statistics": {
                    "scheduler": scheduler_stats,
                    "models": model_stats
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/summary")
async def get_training_summary():
    """
    Get comprehensive training system summary

    Returns:
        - Scheduler status
        - Current job
        - Recent jobs
        - Model versions
        - Active model
        - Statistics
    """
    try:
        scheduler = get_training_scheduler()
        model_manager = get_model_manager()

        status = scheduler.get_status()
        stats = scheduler.get_statistics()
        model_stats = model_manager.get_statistics()
        active_version = model_manager.active_version

        return JSONResponse(
            content={
                "success": True,
                "summary": {
                    "scheduler_running": stats['running'],
                    "current_job": status['current_job'],
                    "recent_jobs_count": len(status['recent_jobs']),
                    "total_jobs": stats['total_jobs'],
                    "success_rate": stats['success_rate'],
                    "total_model_versions": model_stats.get('total_versions', 0),
                    "active_model": active_version.version_id if active_version else None,
                    "next_training_in_hours": stats['next_training_in_hours'],
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )
