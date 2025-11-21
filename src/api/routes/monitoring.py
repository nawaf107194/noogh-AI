"""
📊 Monitoring API - System Health & Alerts
REST API for monitoring system health, alerts, and statistics

Endpoints:
- GET  /api/monitoring/health         → Current health status
- GET  /api/monitoring/history        → Health check history
- GET  /api/monitoring/alerts         → Recent alerts
- GET  /api/monitoring/alerts/unack   → Unacknowledged alerts
- POST /api/monitoring/alerts/ack     → Acknowledge alerts
- GET  /api/monitoring/stats          → Service statistics
- GET  /api/monitoring/service/status → Service status
- POST /api/monitoring/check          → Trigger manual check
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging
# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
from autonomy.self_monitor import get_monitor
from autonomy.monitor_service import get_monitor_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/monitoring",
    tags=["📊 Monitoring", "Health Checks", "Alerts", "System Status"],
)

# Models
class AcknowledgeRequest(BaseModel):
    """Request to acknowledge alerts"""
    acknowledge_all: bool = False
    alert_indices: Optional[list[int]] = None


@router.get("/health")
async def get_current_health():
    """
    Get current system health status (cached for performance)

    Returns:
        Current health check results including:
        - Overall status (healthy/warning/critical/emergency)
        - Brain Hub status
        - Cognition score
        - Ministers status
        - API responsiveness
        - Issues and warnings
    """
    try:
        monitor = get_monitor()
        
        # Try to get recent health check from history first (for performance)
        history = monitor.get_health_history(limit=1)
        if history:
            # Use cached result if less than 30 seconds old
            latest = history[0]
            if hasattr(latest, 'timestamp'):
                try:
                    # Handle both datetime objects and strings
                    if isinstance(latest.timestamp, str):
                        from dateutil import parser
                        latest_time = parser.parse(latest.timestamp)
                    else:
                        latest_time = latest.timestamp
                    
                    age = (datetime.now() - latest_time).total_seconds()
                    if age < 30:
                        return JSONResponse(
                            content={
                                "success": True,
                                "health": latest.to_dict(),
                                "cached": True,
                                "age_seconds": age,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                except Exception as e:
                    logger.warning(f"Could not parse timestamp: {e}")
                    pass
        
        # Perform new health check with timeout protection
        import asyncio
        try:
            health_status = await asyncio.wait_for(
                asyncio.to_thread(monitor.perform_health_check),
                timeout=3.0  # 3 second timeout
            )
            
            return JSONResponse(
                content={
                    "success": True,
                    "health": health_status.to_dict(),
                    "cached": False,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except asyncio.TimeoutError:
            # Return basic health status on timeout
            return JSONResponse(
                content={
                    "success": True,
                    "health": {
                        "status": "unknown",
                        "message": "Health check timed out, system may be under load"
                    },
                    "timeout": True,
                    "timestamp": datetime.now().isoformat()
                }
            )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/history")
async def get_health_history(limit: int = 20):
    """
    Get health check history

    Args:
        limit: Number of recent checks to return (default 20, max 100)

    Returns:
        List of recent health check results
    """
    try:
        limit = min(max(limit, 1), 100)  # Clamp between 1-100

        monitor = get_monitor()
        history = monitor.get_health_history(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(history),
                "limit": limit,
                "history": [h.to_dict() for h in history]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/alerts")
async def get_recent_alerts(limit: int = 10):
    """
    Get recent alerts

    Args:
        limit: Number of recent alerts to return (default 10, max 100)

    Returns:
        List of recent alerts
    """
    try:
        limit = min(max(limit, 1), 100)

        service = get_monitor_service()
        alerts = service.get_recent_alerts(limit=limit)

        return JSONResponse(
            content={
                "success": True,
                "count": len(alerts),
                "limit": limit,
                "alerts": [a.to_dict() for a in alerts]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/alerts/unacknowledged")
async def get_unacknowledged_alerts():
    """
    Get all unacknowledged alerts

    Returns:
        List of alerts that need attention
    """
    try:
        service = get_monitor_service()
        alerts = service.get_unacknowledged_alerts()

        return JSONResponse(
            content={
                "success": True,
                "count": len(alerts),
                "alerts": [a.to_dict() for a in alerts]
            }
        )

    except Exception as e:
        logger.error(f"Failed to get unacknowledged alerts: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/alerts/acknowledge")
async def acknowledge_alerts(request: AcknowledgeRequest):
    """
    Acknowledge alerts

    Body:
        - acknowledge_all: Set to true to acknowledge all alerts
        - alert_indices: List of alert indices to acknowledge (optional)

    Returns:
        Confirmation of acknowledged alerts
    """
    try:
        service = get_monitor_service()

        if request.acknowledge_all:
            service.acknowledge_all_alerts()
            return JSONResponse(
                content={
                    "success": True,
                    "message": "All alerts acknowledged",
                    "acknowledged_count": len(service.unacknowledged_alerts)
                }
            )

        elif request.alert_indices:
            alerts = service.get_recent_alerts(limit=100)
            acknowledged = []

            for idx in request.alert_indices:
                if 0 <= idx < len(alerts):
                    alert = alerts[idx]
                    service.acknowledge_alert(alert)
                    acknowledged.append(idx)

            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Acknowledged {len(acknowledged)} alerts",
                    "acknowledged_indices": acknowledged
                }
            )

        else:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Must specify acknowledge_all or alert_indices"
                },
                status_code=400
            )

    except Exception as e:
        logger.error(f"Failed to acknowledge alerts: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/stats")
async def get_statistics():
    """
    Get monitoring service statistics

    Returns:
        - Total checks performed
        - Total alerts generated
        - Service uptime
        - Health summary
    """
    try:
        service = get_monitor_service()
        monitor = get_monitor()

        stats = service.get_statistics()
        health_summary = monitor.get_health_summary()

        return JSONResponse(
            content={
                "success": True,
                "statistics": stats,
                "health_summary": health_summary,
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


@router.get("/service/status")
async def get_service_status():
    """
    Get monitoring service status

    Returns:
        - Service running status
        - Check interval
        - Statistics
        - Health summary
    """
    try:
        service = get_monitor_service()
        status = service.get_status()

        return JSONResponse(
            content={
                "success": True,
                "service": status,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/check")
async def trigger_manual_check():
    """
    Trigger a manual health check

    Returns:
        Health check results
    """
    try:
        logger.info("🔍 Manual health check triggered via API")

        monitor = get_monitor()
        health_status = monitor.perform_health_check()

        return JSONResponse(
            content={
                "success": True,
                "message": "Manual health check completed",
                "health": health_status.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Manual check failed: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.get("/summary")
async def get_monitoring_summary():
    """
    Get comprehensive monitoring summary

    Returns:
        - Current health
        - Recent alerts
        - Statistics
        - Service status
    """
    try:
        monitor = get_monitor()
        service = get_monitor_service()

        # Get latest health check or perform new one
        history = monitor.get_health_history(limit=1)
        if history:
            current_health = history[0]
        else:
            current_health = monitor.perform_health_check()

        # Get recent alerts
        recent_alerts = service.get_recent_alerts(limit=5)
        unack_alerts = service.get_unacknowledged_alerts()

        # Get stats
        stats = service.get_statistics()
        health_summary = monitor.get_health_summary()

        return JSONResponse(
            content={
                "success": True,
                "summary": {
                    "current_health": current_health.to_dict(),
                    "recent_alerts_count": len(recent_alerts),
                    "unacknowledged_alerts_count": len(unack_alerts),
                    "statistics": stats,
                    "health_summary": health_summary,
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

# Cache monitoring endpoint
@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache performance statistics
    
    Returns:
        Cache statistics including:
        - hits/misses counts
        - hit rate percentage
        - backend type (redis/memory)
        - keys count (if available)
    """
    try:
        from api.utils.cache_manager import get_cache
        
        cache = get_cache()
        stats = cache.get_stats()
        
        return JSONResponse(
            content={
                "success": True,
                "cache_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except ImportError:
        return JSONResponse(
            content={
                "success": False,
                "error": "Cache manager not available"
            },
            status_code=503
        )
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cache entries
    
    Returns:
        Success status
    """
    try:
        from api.utils.cache_manager import get_cache
        
        cache = get_cache()
        cache.clear()
        
        logger.info("✅ Cache cleared successfully")
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except ImportError:
        return JSONResponse(
            content={
                "success": False,
                "error": "Cache manager not available"
            },
            status_code=503
        )
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )
