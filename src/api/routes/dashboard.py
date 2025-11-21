"""
📊 Dashboard API Routes
Public endpoints for the Noogh Unified System Dashboard
No authentication required - designed for the React dashboard frontend
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from pathlib import Path
from datetime import datetime
import time
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
from api.services.system_status import (
    get_system_overview,
    get_ministers_status,
    get_brain_status,
    get_training_summary,
    get_logs_summary,
    get_knowledge_stats,
    get_mcp_status,
    get_cron_status
)

router = APIRouter(prefix="/api/system", tags=["dashboard"])
logger = logging.getLogger(__name__)


@router.get("/overview")
async def system_overview() -> Dict[str, Any]:
    """
    🎯 Main endpoint - aggregate all subsystem data

    Returns complete system state including:
    - MCP Server status
    - Brain v4.0 metrics
    - Knowledge Index stats
    - Training pipeline status
    - Cron automation status
    - Ministers count and list
    - Overall health percentage

    Used by: Dashboard home page
    Auto-refresh: 10-15 seconds
    """
    try:
        overview = get_system_overview()
        return {
            "success": True,
            "data": overview
        }
    except Exception as e:
        logger.error(f"Error in system_overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ministers")
async def get_ministers() -> Dict[str, Any]:
    """
    👥 Get structured minister information

    Returns list of all 14 ministers with:
    - ID, name, status, domain
    - Activity metrics
    - Connection status to unified brain

    Used by: Ministers page
    Auto-refresh: 15 seconds
    """
    try:
        ministers = get_ministers_status()

        # Calculate summary stats
        total = len(ministers)
        active = sum(1 for m in ministers if m.get("status") == "active")
        inactive = total - active

        return {
            "success": True,
            "summary": {
                "total": total,
                "active": active,
                "inactive": inactive,
                "health_percent": (active / total * 100) if total > 0 else 0
            },
            "ministers": ministers
        }
    except Exception as e:
        logger.error(f"Error in get_ministers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brain")
async def get_brain() -> Dict[str, Any]:
    """
    🧠 Get Brain v4.0 status and recent memories

    Returns:
    - Current memory count
    - Capacity and usage percentage
    - Latest interaction timestamp
    - Features list

    Used by: Dashboard overview, Automation page
    Auto-refresh: 10 seconds
    """
    try:
        brain = get_brain_status()
        return {
            "success": True,
            "data": brain
        }
    except Exception as e:
        logger.error(f"Error in get_brain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/history")
async def get_training_history() -> Dict[str, Any]:
    """
    📊 Get daily training reports and historical summary

    Returns:
    - Latest run timestamp
    - Success rate
    - Total runs count
    - Pipeline stages
    - Tasks completed

    Used by: Automation page, Reports page
    Auto-refresh: 15 seconds
    """
    try:
        training = get_training_summary()
        cron = get_cron_status()

        return {
            "success": True,
            "data": {
                **training,
                "automation": {
                    "cron_active": cron["cron_active"],
                    "schedule": cron["schedule_human"],
                    "next_run": "2:00 AM tomorrow" if cron["cron_active"] else "Not scheduled"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in get_training_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/summary")
async def get_logs() -> Dict[str, Any]:
    """
    📝 Tail of recent log files

    Returns last 50 lines from:
    - API logs
    - Training logs
    - MCP server logs
    - Brain logs

    Used by: Reports page, debugging
    Auto-refresh: 20 seconds
    """
    try:
        logs = get_logs_summary()
        return {
            "success": True,
            "data": logs
        }
    except Exception as e:
        logger.error(f"Error in get_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge")
async def get_knowledge() -> Dict[str, Any]:
    """
    📚 Knowledge Index statistics

    Returns:
    - Total chunks count
    - Categories list
    - Progress towards 100 chunks goal
    - Category breakdown

    Used by: Dashboard overview, Automation page
    Auto-refresh: 15 seconds
    """
    try:
        knowledge = get_knowledge_stats()
        return {
            "success": True,
            "data": knowledge
        }
    except Exception as e:
        logger.error(f"Error in get_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp")
async def get_mcp() -> Dict[str, Any]:
    """
    🔧 MCP Server status

    Returns:
    - Port and version
    - Tools and resources count
    - Uptime and request statistics
    - Features list

    Used by: Automation page
    Auto-refresh: 10 seconds
    """
    try:
        mcp = get_mcp_status()
        return {
            "success": True,
            "data": mcp
        }
    except Exception as e:
        logger.error(f"Error in get_mcp: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def system_health() -> Dict[str, Any]:
    """
    ✅ Quick health check

    Returns simplified health status for monitoring
    Used by: Dashboard header, monitoring systems
    Auto-refresh: 5 seconds
    """
    try:
        overview = get_system_overview()
        return {
            "status": "healthy" if overview["health_percent"] >= 80 else "degraded",
            "health_percent": overview["health_percent"],
            "overall_status": overview["overall_status"],
            "timestamp": overview["timestamp"]
        }
    except Exception as e:
        logger.error(f"Error in system_health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# 📊 GRAFANA-OPTIMIZED ENDPOINTS
# ============================================================================

@router.get("/metrics")
async def get_metrics() -> List[Dict[str, Any]]:
    """
    📈 Prometheus-style metrics for Grafana

    Returns current metrics in time-series format:
    - metric: metric name
    - value: current value
    - timestamp: Unix timestamp (ms)
    - labels: optional key-value pairs

    Optimized for Grafana Infinity datasource with JSON parser
    Auto-refresh: 5 seconds
    """
    try:
        overview = get_system_overview()
        timestamp = int(time.time() * 1000)  # Unix timestamp in milliseconds

        # Extract all metrics from overview
        metrics = [
            {
                "metric": "system_health_percent",
                "value": overview["health_percent"],
                "timestamp": timestamp,
                "labels": {"component": "system"}
            },
            {
                "metric": "active_components",
                "value": overview["active_components"],
                "timestamp": timestamp,
                "labels": {"component": "system"}
            },
            {
                "metric": "total_components",
                "value": overview["total_components"],
                "timestamp": timestamp,
                "labels": {"component": "system"}
            },

            # MCP Server
            {
                "metric": "mcp_active",
                "value": 1 if overview["mcp_server"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {"component": "mcp", "version": overview["mcp_server"]["version"]}
            },
            {
                "metric": "mcp_tools_count",
                "value": overview["mcp_server"]["tools"],
                "timestamp": timestamp,
                "labels": {"component": "mcp"}
            },
            {
                "metric": "mcp_resources_count",
                "value": overview["mcp_server"]["resources"],
                "timestamp": timestamp,
                "labels": {"component": "mcp"}
            },
            {
                "metric": "mcp_uptime_seconds",
                "value": overview["mcp_server"]["uptime_seconds"],
                "timestamp": timestamp,
                "labels": {"component": "mcp"}
            },
            {
                "metric": "mcp_total_requests",
                "value": overview["mcp_server"]["total_requests"],
                "timestamp": timestamp,
                "labels": {"component": "mcp"}
            },

            # Brain v4.0
            {
                "metric": "brain_active",
                "value": 1 if overview["brain_v4"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {"component": "brain", "version": overview["brain_v4"]["version"]}
            },
            {
                "metric": "brain_memories_count",
                "value": overview["brain_v4"]["session_memories"],
                "timestamp": timestamp,
                "labels": {"component": "brain"}
            },
            {
                "metric": "brain_capacity",
                "value": overview["brain_v4"]["capacity"],
                "timestamp": timestamp,
                "labels": {"component": "brain"}
            },
            {
                "metric": "brain_usage_percent",
                "value": overview["brain_v4"]["usage_percent"],
                "timestamp": timestamp,
                "labels": {"component": "brain"}
            },

            # Knowledge Index
            {
                "metric": "knowledge_active",
                "value": 1 if overview["knowledge_index"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {"component": "knowledge"}
            },
            {
                "metric": "knowledge_chunks_total",
                "value": overview["knowledge_index"]["total_chunks"],
                "timestamp": timestamp,
                "labels": {"component": "knowledge"}
            },
            {
                "metric": "knowledge_progress_percent",
                "value": overview["knowledge_index"]["progress_percent"],
                "timestamp": timestamp,
                "labels": {"component": "knowledge"}
            },

            # Daily Training
            {
                "metric": "training_success",
                "value": 1 if overview["daily_training"]["status"] == "success" else 0,
                "timestamp": timestamp,
                "labels": {"component": "training"}
            },
            {
                "metric": "training_tasks_completed",
                "value": overview["daily_training"]["tasks_completed"],
                "timestamp": timestamp,
                "labels": {"component": "training"}
            },
            {
                "metric": "training_total_tasks",
                "value": overview["daily_training"]["total_tasks"],
                "timestamp": timestamp,
                "labels": {"component": "training"}
            },
            {
                "metric": "training_total_runs",
                "value": overview["daily_training"]["total_runs"],
                "timestamp": timestamp,
                "labels": {"component": "training"}
            },
            {
                "metric": "training_successful_runs",
                "value": overview["daily_training"]["successful_runs"],
                "timestamp": timestamp,
                "labels": {"component": "training"}
            },

            # Cron Automation
            {
                "metric": "cron_active",
                "value": 1 if overview["cron_automation"]["cron_active"] else 0,
                "timestamp": timestamp,
                "labels": {"component": "automation", "schedule": overview["cron_automation"]["schedule"]}
            },

            # Ministers
            {
                "metric": "ministers_total",
                "value": overview["ministers"]["total"],
                "timestamp": timestamp,
                "labels": {"component": "ministers"}
            },
            {
                "metric": "ministers_active",
                "value": overview["ministers"]["active"],
                "timestamp": timestamp,
                "labels": {"component": "ministers"}
            },
        ]

        return metrics

    except Exception as e:
        logger.error(f"Error in get_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/timeseries")
async def get_timeseries_metrics() -> List[Dict[str, Any]]:
    """
    📈 Time-series data for Grafana graphs

    Returns data points over time (simulated with current values)
    For real time-series, connect to Prometheus or InfluxDB

    Format optimized for Grafana time-series panels
    """
    try:
        overview = get_system_overview()
        current_time = int(time.time() * 1000)

        # Generate data points (current value, can be extended with historical data)
        datapoints = []

        # System Health over time (simulated - replace with real DB queries)
        datapoints.append({
            "target": "System Health",
            "datapoints": [
                [overview["health_percent"], current_time]
            ]
        })

        datapoints.append({
            "target": "Knowledge Chunks",
            "datapoints": [
                [overview["knowledge_index"]["total_chunks"], current_time]
            ]
        })

        datapoints.append({
            "target": "Brain Memories",
            "datapoints": [
                [overview["brain_v4"]["session_memories"], current_time]
            ]
        })

        datapoints.append({
            "target": "Active Ministers",
            "datapoints": [
                [overview["ministers"]["active"], current_time]
            ]
        })

        return datapoints

    except Exception as e:
        logger.error(f"Error in get_timeseries_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ministers/metrics")
async def get_ministers_metrics() -> List[Dict[str, Any]]:
    """
    👥 Individual minister metrics for Grafana table/bar charts

    Returns each minister with status as individual row
    Optimized for Grafana table panel or bar chart
    """
    try:
        ministers_data = get_ministers_status()
        timestamp = int(time.time() * 1000)

        metrics = []
        for minister in ministers_data:
            metrics.append({
                "minister_id": minister["id"],
                "minister_name": minister["name"],
                "domain": minister["domain"],
                "status": minister["status"],
                "status_numeric": 1 if minister["status"] == "active" else 0,
                "timestamp": timestamp
            })

        return metrics

    except Exception as e:
        logger.error(f"Error in get_ministers_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs() -> List[Dict[str, Any]]:
    """
    📝 Recent logs in structured format for Grafana logs panel

    Returns last 100 log entries with:
    - timestamp
    - level (INFO, ERROR, etc.)
    - message
    - source (component name)
    """
    try:
        logs_data = get_logs_summary()
        current_time = datetime.now()

        structured_logs = []

        # Process each log source
        for source, log_lines in logs_data.items():
            if isinstance(log_lines, list):
                for idx, line in enumerate(log_lines[-50:]):  # Last 50 from each source
                    # Simple parsing (enhance based on actual log format)
                    level = "INFO"
                    if "ERROR" in line or "error" in line:
                        level = "ERROR"
                    elif "WARNING" in line or "warning" in line:
                        level = "WARNING"
                    elif "SUCCESS" in line or "success" in line or "✅" in line:
                        level = "SUCCESS"

                    structured_logs.append({
                        "timestamp": int(time.time() * 1000) - (50 - idx) * 1000,  # Simulate timestamps
                        "level": level,
                        "source": source,
                        "message": line.strip()
                    })

        # Sort by timestamp descending
        structured_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return structured_logs[:100]  # Return last 100

    except Exception as e:
        logger.error(f"Error in get_recent_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
