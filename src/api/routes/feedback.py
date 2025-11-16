"""
🔄 Self-Feedback Loop API - Minister Feedback & Brain Adjustments
REST API for collecting minister feedback and managing brain adjustments

Endpoints:
- POST /api/feedback/submit           → Submit minister feedback
- GET  /api/feedback/ministers/stats  → Get all minister statistics
- GET  /api/feedback/ministers/{name} → Get specific minister stats
- GET  /api/feedback/insights         → Get overall insights
- GET  /api/feedback/recent           → Get recent feedback entries
- GET  /api/feedback/statistics       → Get feedback collector stats
- POST /api/feedback/recommend        → Generate adjustment recommendations
- GET  /api/feedback/recommendations  → Get recent recommendations
- POST /api/feedback/adjust           → Apply an adjustment
- GET  /api/feedback/adjustments      → Get recent adjustments
- POST /api/feedback/evaluate/{id}    → Evaluate adjustment effectiveness
- POST /api/feedback/rollback/{id}    → Rollback an adjustment
- GET  /api/feedback/summary          → Comprehensive feedback summary
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

from autonomy.feedback_collector import get_feedback_collector
from autonomy.brain_adjuster import get_brain_adjuster

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/feedback",
    tags=["🔄 Self-Feedback Loop", "Minister Feedback", "Brain Adjustments", "Self-Learning"],
)

# Models
class FeedbackSubmission(BaseModel):
    """Request to submit feedback"""
    minister_name: str
    task_type: str
    outcome: str  # "success", "partial_success", "failure", "error"
    confidence_score: float
    execution_time_ms: float
    context: Optional[dict] = None
    learned_insights: Optional[list] = None
    error_details: Optional[str] = None


class AdjustmentRequest(BaseModel):
    """Request to apply adjustment"""
    recommendation_id: str
    force: bool = False


@router.post("/submit")
async def submit_feedback(submission: FeedbackSubmission):
    """
    Submit feedback from a minister

    Body:
        minister_name: Name of the minister
        task_type: Type of task performed
        outcome: Outcome (success/partial_success/failure/error)
        confidence_score: Confidence level (0.0-1.0)
        execution_time_ms: Execution time in milliseconds
        context: Task context (optional)
        learned_insights: Insights learned (optional)
        error_details: Error details if failed (optional)

    Returns:
        Submitted feedback entry
    """
    try:
        logger.info(f"📝 Feedback submission from {submission.minister_name}")

        collector = get_feedback_collector()

        feedback = collector.submit_feedback(
            minister_name=submission.minister_name,
            task_type=submission.task_type,
            outcome=submission.outcome,
            confidence_score=submission.confidence_score,
            execution_time_ms=submission.execution_time_ms,
            context=submission.context,
            learned_insights=submission.learned_insights,
            error_details=submission.error_details
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "Feedback submitted successfully",
                "feedback": feedback.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/ministers/stats")
async def get_all_minister_stats():
    """
    Get statistics for all ministers

    Returns:
        List of minister statistics with performance metrics
    """
    try:
        collector = get_feedback_collector()
        all_stats = collector.get_all_minister_stats()

        return JSONResponse(
            content={
                "success": True,
                "count": len(all_stats),
                "ministers": [asdict(s) for s in all_stats]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get minister stats: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/ministers/{minister_name}")
async def get_minister_stats(minister_name: str):
    """
    Get statistics for a specific minister

    Args:
        minister_name: Name of the minister

    Returns:
        Minister statistics with performance metrics
    """
    try:
        collector = get_feedback_collector()
        stats = collector.get_minister_stats(minister_name)

        if not stats:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"No feedback data for minister: {minister_name}"
                },
                status_code=404
            )

        return JSONResponse(
            content={
                "success": True,
                "minister": asdict(stats)
            }
        )

    except Exception as e:
        logger.error(f"Failed to get minister stats: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/insights")
async def get_insights():
    """
    Get overall insights from all feedback

    Returns:
        System-wide insights and recommendations
    """
    try:
        collector = get_feedback_collector()
        insights = collector.get_insights()

        return JSONResponse(
            content={
                "success": True,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/recent")
async def get_recent_feedback(limit: int = 20):
    """
    Get recent feedback entries

    Args:
        limit: Number of entries to return (default 20, max 100)

    Returns:
        List of recent feedback entries
    """
    try:
        limit = min(max(limit, 1), 100)

        collector = get_feedback_collector()
        recent = collector.get_recent_feedback(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(recent),
                "limit": limit,
                "feedback": [f.to_dict() for f in recent]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get recent feedback: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/statistics")
async def get_feedback_statistics():
    """
    Get feedback collector statistics

    Returns:
        Overall feedback statistics
    """
    try:
        collector = get_feedback_collector()
        stats = collector.get_statistics()

        return JSONResponse(
            content={
                "success": True,
                "statistics": stats,
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


@router.post("/recommend")
async def generate_recommendations():
    """
    Generate adjustment recommendations based on feedback

    Returns:
        List of recommended adjustments
    """
    try:
        logger.info("🔍 Generating adjustment recommendations...")

        adjuster = get_brain_adjuster()
        recommendations = adjuster.analyze_and_recommend()

        return JSONResponse(
            content={
                "success": True,
                "count": len(recommendations),
                "recommendations": [r.to_dict() for r in recommendations],
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/recommendations")
async def get_recommendations(limit: int = 10):
    """
    Get recent recommendations

    Args:
        limit: Number of recommendations to return (default 10, max 50)

    Returns:
        List of recent recommendations
    """
    try:
        limit = min(max(limit, 1), 50)

        adjuster = get_brain_adjuster()
        recommendations = adjuster.get_recent_recommendations(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(recommendations),
                "limit": limit,
                "recommendations": [r.to_dict() for r in recommendations]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/adjust")
async def apply_adjustment(request: AdjustmentRequest):
    """
    Apply an adjustment based on recommendation

    Body:
        recommendation_id: ID of recommendation to apply
        force: Force application even if risky (default false)

    Returns:
        Applied adjustment details
    """
    try:
        logger.info(f"⚙️  Applying adjustment for recommendation: {request.recommendation_id}")

        adjuster = get_brain_adjuster()

        # Find the recommendation
        recommendations = adjuster.get_recent_recommendations(limit=100)
        recommendation = next(
            (r for r in recommendations if r.recommendation_id == request.recommendation_id),
            None
        )

        if not recommendation:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Recommendation not found: {request.recommendation_id}"
                },
                status_code=404
            )

        # Apply adjustment
        adjustment = adjuster.apply_adjustment(
            recommendation=recommendation,
            force=request.force
        )

        if not adjustment:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Adjustment requires manual approval (high risk)",
                    "recommendation": recommendation.to_dict()
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "message": "Adjustment applied successfully",
                "adjustment": adjustment.to_dict()
            }
        )

    except Exception as e:
        logger.error(f"Failed to apply adjustment: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/adjustments")
async def get_adjustments(limit: int = 10):
    """
    Get recent adjustments

    Args:
        limit: Number of adjustments to return (default 10, max 50)

    Returns:
        List of recent adjustments
    """
    try:
        limit = min(max(limit, 1), 50)

        adjuster = get_brain_adjuster()
        adjustments = adjuster.get_recent_adjustments(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(adjustments),
                "limit": limit,
                "adjustments": [a.to_dict() for a in adjustments]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get adjustments: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/evaluate/{adjustment_id}")
async def evaluate_adjustment(adjustment_id: str, test_duration_hours: float = 1.0):
    """
    Evaluate adjustment effectiveness

    Args:
        adjustment_id: ID of adjustment to evaluate
        test_duration_hours: Test duration in hours (default 1.0)

    Returns:
        Evaluation results
    """
    try:
        logger.info(f"📊 Evaluating adjustment: {adjustment_id}")

        adjuster = get_brain_adjuster()

        is_effective, score, reason = adjuster.evaluate_adjustment_effectiveness(
            adjustment_id=adjustment_id,
            test_duration_hours=test_duration_hours
        )

        return JSONResponse(
            content={
                "success": True,
                "adjustment_id": adjustment_id,
                "is_effective": is_effective,
                "effectiveness_score": score,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to evaluate adjustment: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/rollback/{adjustment_id}")
async def rollback_adjustment(adjustment_id: str):
    """
    Rollback an adjustment

    Args:
        adjustment_id: ID of adjustment to rollback

    Returns:
        Rollback confirmation
    """
    try:
        logger.warning(f"⏪ Rolling back adjustment: {adjustment_id}")

        adjuster = get_brain_adjuster()

        success = adjuster.rollback_adjustment(adjustment_id)

        if not success:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Adjustment not found or already rolled back: {adjustment_id}"
                },
                status_code=404
            )

        return JSONResponse(
            content={
                "success": True,
                "message": f"Adjustment rolled back successfully: {adjustment_id}",
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to rollback adjustment: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/summary")
async def get_feedback_summary():
    """
    Get comprehensive feedback loop summary

    Returns:
        - Feedback statistics
        - Minister statistics
        - Recent recommendations
        - Active adjustments
        - Overall insights
    """
    try:
        collector = get_feedback_collector()
        adjuster = get_brain_adjuster()

        feedback_stats = collector.get_statistics()
        adjuster_stats = adjuster.get_statistics()
        insights = collector.get_insights()
        all_ministers = collector.get_all_minister_stats()
        recent_recommendations = adjuster.get_recent_recommendations(limit=5)
        recent_adjustments = adjuster.get_recent_adjustments(limit=5)

        return JSONResponse(
            content={
                "success": True,
                "summary": {
                    "feedback_statistics": feedback_stats,
                    "adjuster_statistics": adjuster_stats,
                    "overall_insights": insights,
                    "total_ministers": len(all_ministers),
                    "recent_recommendations_count": len(recent_recommendations),
                    "recent_adjustments_count": len(recent_adjustments),
                    "top_performers": insights.get('top_performers', []),
                    "needs_support": insights.get('needs_support', []),
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


# Helper function to convert dataclass to dict
from dataclasses import asdict
