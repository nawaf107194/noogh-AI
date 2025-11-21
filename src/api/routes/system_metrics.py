"""
📊 System Metrics API Routes - Grafana Optimized
Full integration with Grafana Infinity datasource

All endpoints return data in formats optimized for Grafana visualization:
- Prometheus-style metrics
- Time-series data
- Structured logs
- Ministers data for tables/charts

No authentication required - designed for monitoring dashboards
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.services.system_status import (
    get_system_overview,
    get_ministers_status,
    get_brain_status,
    get_knowledge_stats,
    get_mcp_status,
    get_training_summary,
    get_logs_summary,
    get_cron_status
)

router = APIRouter(prefix="/api/system", tags=["metrics", "grafana"])
logger = logging.getLogger(__name__)


# ============================================================================
# 📈 GRAFANA-OPTIMIZED ENDPOINTS
# ============================================================================

@router.get("/metrics")
async def get_prometheus_metrics() -> List[Dict[str, Any]]:
    """
    📈 Prometheus-style metrics for Grafana

    Returns 23 metrics covering all system components:
    - System health, active components
    - MCP server status, tools, resources, uptime
    - Brain v4.0 status, memories, capacity, usage
    - Knowledge index status, chunks, progress
    - Training status, tasks, runs, success rate
    - Cron automation status
    - Ministers count (total, active)

    Format: [{"metric": "name", "value": number, "timestamp": ms, "labels": {...}}, ...]

    Optimized for: Grafana Infinity datasource with JSON parser
    Panel types: Gauge, Stat, Graph, Time series
    Auto-refresh: 5 seconds recommended
    """
    try:
        overview = get_system_overview()
        timestamp = int(time.time() * 1000)  # Unix timestamp in milliseconds

        metrics = [
            # System Overview
            {
                "metric": "system_health_percent",
                "value": overview["health_percent"],
                "timestamp": timestamp,
                "labels": {"component": "system", "unit": "percent"}
            },
            {
                "metric": "active_components",
                "value": overview["active_components"],
                "timestamp": timestamp,
                "labels": {"component": "system", "unit": "count"}
            },
            {
                "metric": "total_components",
                "value": overview["total_components"],
                "timestamp": timestamp,
                "labels": {"component": "system", "unit": "count"}
            },

            # MCP Server
            {
                "metric": "mcp_active",
                "value": 1 if overview["mcp_server"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {
                    "component": "mcp",
                    "version": overview["mcp_server"]["version"],
                    "unit": "boolean"
                }
            },
            {
                "metric": "mcp_tools_count",
                "value": overview["mcp_server"]["tools"],
                "timestamp": timestamp,
                "labels": {"component": "mcp", "unit": "count"}
            },
            {
                "metric": "mcp_resources_count",
                "value": overview["mcp_server"]["resources"],
                "timestamp": timestamp,
                "labels": {"component": "mcp", "unit": "count"}
            },
            {
                "metric": "mcp_uptime_seconds",
                "value": overview["mcp_server"]["uptime_seconds"],
                "timestamp": timestamp,
                "labels": {"component": "mcp", "unit": "seconds"}
            },
            {
                "metric": "mcp_total_requests",
                "value": overview["mcp_server"]["total_requests"],
                "timestamp": timestamp,
                "labels": {"component": "mcp", "unit": "count"}
            },

            # Brain v4.0
            {
                "metric": "brain_active",
                "value": 1 if overview["brain_v4"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {
                    "component": "brain",
                    "version": overview["brain_v4"]["version"],
                    "unit": "boolean"
                }
            },
            {
                "metric": "brain_memories_count",
                "value": overview["brain_v4"]["session_memories"],
                "timestamp": timestamp,
                "labels": {"component": "brain", "unit": "count"}
            },
            {
                "metric": "brain_capacity",
                "value": overview["brain_v4"]["capacity"],
                "timestamp": timestamp,
                "labels": {"component": "brain", "unit": "count"}
            },
            {
                "metric": "brain_usage_percent",
                "value": overview["brain_v4"]["usage_percent"],
                "timestamp": timestamp,
                "labels": {"component": "brain", "unit": "percent"}
            },

            # Knowledge Index
            {
                "metric": "knowledge_active",
                "value": 1 if overview["knowledge_index"]["status"] == "active" else 0,
                "timestamp": timestamp,
                "labels": {"component": "knowledge", "unit": "boolean"}
            },
            {
                "metric": "knowledge_chunks_total",
                "value": overview["knowledge_index"]["total_chunks"],
                "timestamp": timestamp,
                "labels": {"component": "knowledge", "unit": "count"}
            },
            {
                "metric": "knowledge_progress_percent",
                "value": overview["knowledge_index"]["progress_percent"],
                "timestamp": timestamp,
                "labels": {"component": "knowledge", "unit": "percent", "target": "100"}
            },

            # Daily Training
            {
                "metric": "training_success",
                "value": 1 if overview["daily_training"]["status"] == "success" else 0,
                "timestamp": timestamp,
                "labels": {"component": "training", "unit": "boolean"}
            },
            {
                "metric": "training_tasks_completed",
                "value": overview["daily_training"]["tasks_completed"],
                "timestamp": timestamp,
                "labels": {"component": "training", "unit": "count"}
            },
            {
                "metric": "training_total_tasks",
                "value": overview["daily_training"]["total_tasks"],
                "timestamp": timestamp,
                "labels": {"component": "training", "unit": "count"}
            },
            {
                "metric": "training_total_runs",
                "value": overview["daily_training"]["total_runs"],
                "timestamp": timestamp,
                "labels": {"component": "training", "unit": "count"}
            },
            {
                "metric": "training_successful_runs",
                "value": overview["daily_training"]["successful_runs"],
                "timestamp": timestamp,
                "labels": {"component": "training", "unit": "count"}
            },

            # Cron Automation
            {
                "metric": "cron_active",
                "value": 1 if overview["cron_automation"]["cron_active"] else 0,
                "timestamp": timestamp,
                "labels": {
                    "component": "automation",
                    "schedule": overview["cron_automation"]["schedule"],
                    "unit": "boolean"
                }
            },

            # Ministers
            {
                "metric": "ministers_total",
                "value": overview["ministers"]["total"],
                "timestamp": timestamp,
                "labels": {"component": "ministers", "unit": "count"}
            },
            {
                "metric": "ministers_active",
                "value": overview["ministers"]["active"],
                "timestamp": timestamp,
                "labels": {"component": "ministers", "unit": "count"}
            },
        ]

        return metrics

    except Exception as e:
        logger.error(f"Error in get_prometheus_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ministers/metrics")
async def get_ministers_metrics() -> List[Dict[str, Any]]:
    """
    👥 Individual minister metrics for Grafana tables and charts

    Returns all 14 ministers with detailed status information.

    Fields per minister:
    - minister_id: Unique ID (1-14)
    - minister_name: Arabic name (e.g., "وزير التعليم")
    - domain: English domain (e.g., "education")
    - status: "active" or "inactive"
    - status_numeric: 1 for active, 0 for inactive
    - timestamp: Current timestamp in milliseconds

    Optimized for: Grafana table panel, bar charts
    Panel transformations: Group by domain, filter by status
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


@router.get("/overview")
async def get_system_overview_metrics() -> Dict[str, Any]:
    """
    🎯 Complete system overview in a single response

    Returns comprehensive system state including:
    - system_health: Overall health percentage (0-100)
    - active_components: Number of active components
    - knowledge_progress: Knowledge chunks progress
    - brain_memory: Brain memory usage
    - ministers_count: Total and active ministers
    - training_status: Latest training run status
    - automation_status: Cron job status

    Optimized for: Dashboard summary panels, overview gauges
    Use case: Single panel showing multiple metrics
    """
    try:
        overview = get_system_overview()

        return {
            "timestamp": int(time.time() * 1000),
            "system_health": overview["health_percent"],
            "active_components": overview["active_components"],
            "total_components": overview["total_components"],
            "knowledge_progress": overview["knowledge_index"]["progress_percent"],
            "knowledge_chunks": overview["knowledge_index"]["total_chunks"],
            "brain_memory": overview["brain_v4"]["session_memories"],
            "brain_capacity": overview["brain_v4"]["capacity"],
            "brain_usage_percent": overview["brain_v4"]["usage_percent"],
            "ministers_total": overview["ministers"]["total"],
            "ministers_active": overview["ministers"]["active"],
            "training_status": overview["daily_training"]["status"],
            "training_tasks_completed": overview["daily_training"]["tasks_completed"],
            "cron_active": overview["cron_automation"]["cron_active"],
            "overall_status": overview["overall_status"]
        }

    except Exception as e:
        logger.error(f"Error in get_system_overview_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs() -> List[Dict[str, Any]]:
    """
    📝 Recent system logs in structured format

    Returns last 100 log entries from all components with:
    - timestamp: Unix timestamp in milliseconds
    - level: Log level (INFO, WARNING, ERROR, SUCCESS)
    - source: Component name (api, training, mcp, brain, etc.)
    - message: Log message text

    Note: If no logs are available, returns empty array []

    Optimized for: Grafana logs panel, table panel
    Panel settings:
    - Sort by timestamp descending
    - Color code by level (green=SUCCESS, yellow=WARNING, red=ERROR)
    """
    try:
        logs_data = get_logs_summary()
        structured_logs = []

        # Process each log source
        for source, log_lines in logs_data.items():
            if isinstance(log_lines, list) and len(log_lines) > 0:
                for idx, line in enumerate(log_lines[-50:]):  # Last 50 from each source
                    # Parse log level
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
        # Return empty array instead of error for logs
        return []


@router.get("/metrics/timeseries")
async def get_timeseries_data() -> List[Dict[str, Any]]:
    """
    📈 Time-series data for Grafana graphs

    Returns data points over time for key metrics:
    - System Health (%)
    - Knowledge Chunks (count)
    - Brain Memories (count)
    - Active Ministers (count)

    Format: [{"target": "metric_name", "datapoints": [[value, timestamp], ...]}, ...]

    Note: Currently returns single point (current value).
    For historical data, integrate with Prometheus or InfluxDB.

    Optimized for: Grafana time-series panels, graph panels
    """
    try:
        overview = get_system_overview()
        current_time = int(time.time() * 1000)

        datapoints = [
            {
                "target": "System Health",
                "datapoints": [[overview["health_percent"], current_time]]
            },
            {
                "target": "Knowledge Chunks",
                "datapoints": [[overview["knowledge_index"]["total_chunks"], current_time]]
            },
            {
                "target": "Brain Memories",
                "datapoints": [[overview["brain_v4"]["session_memories"], current_time]]
            },
            {
                "target": "Active Ministers",
                "datapoints": [[overview["ministers"]["active"], current_time]]
            }
        ]

        return datapoints

    except Exception as e:
        logger.error(f"Error in get_timeseries_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 🧪 TEST & HEALTH ENDPOINTS
# ============================================================================

@router.get("/metrics/test")
async def test_metrics_endpoint() -> Dict[str, Any]:
    """
    🧪 Test endpoint to verify API is responding

    Returns basic connectivity and API status.
    Use this to test Grafana data source connection.
    """
    return {
        "status": "ok",
        "message": "Noogh System Metrics API is operational",
        "timestamp": int(time.time() * 1000),
        "endpoints": {
            "/api/system/metrics": "Prometheus-style metrics (23 metrics)",
            "/api/system/ministers/metrics": "Ministers data (14 ministers)",
            "/api/system/overview": "Complete system overview",
            "/api/system/logs/recent": "Recent system logs (last 100)",
            "/api/system/metrics/timeseries": "Time-series data (4 series)"
        }
    }
