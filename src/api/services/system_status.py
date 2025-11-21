"""
🔍 System Status Service
Safely aggregates data from all subsystems without crashing the API
"""

import json
import logging
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


def get_mcp_status() -> Dict[str, Any]:
    """Check if MCP server is running on port 8001"""
    try:
        # Check if port 8001 is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 8001))
        sock.close()

        is_running = result == 0

        # Try to read MCP stats if available
        mcp_stats_file = DATA_DIR / "mcp_stats.json"
        stats = {}
        if mcp_stats_file.exists():
            with open(mcp_stats_file, 'r') as f:
                stats = json.load(f)

        return {
            "status": "active" if is_running else "inactive",
            "port": 8001,
            "version": stats.get("version", "2.0"),
            "tools": stats.get("tools", 8),
            "resources": stats.get("resources", 4),
            "uptime_seconds": stats.get("uptime_seconds", 0),
            "total_requests": stats.get("total_requests", 0),
            "features": [
                "execute_command",
                "read_file",
                "write_file",
                "search_files",
                "get_system_info",
                "manage_processes",
                "http_request",
                "run_python"
            ]
        }
    except Exception as e:
        logger.error(f"Error checking MCP status: {e}")
        return {
            "status": "unknown",
            "port": 8001,
            "version": "2.0",
            "tools": 8,
            "resources": 4,
            "error": str(e)
        }


def get_cron_status() -> Dict[str, Any]:
    """Parse crontab and check last run time from logs"""
    try:
        # Read crontab
        result = subprocess.run(
            ['crontab', '-l'],
            capture_output=True,
            text=True,
            timeout=5
        )

        cron_active = False
        schedule = "Not configured"

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'train_daily.py' in line and not line.strip().startswith('#'):
                    cron_active = True
                    # Extract schedule (first 5 fields)
                    parts = line.split()
                    if len(parts) >= 5:
                        schedule = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                    break

        # Check for latest training log
        latest_run = None
        training_dir = DATA_DIR / "training"
        if training_dir.exists():
            reports = sorted(training_dir.glob("daily_report_*.json"), reverse=True)
            if reports:
                latest_report = reports[0]
                with open(latest_report, 'r') as f:
                    data = json.load(f)
                    latest_run = data.get("timestamp")

        return {
            "cron_active": cron_active,
            "schedule": schedule,
            "schedule_human": "Daily at 2:00 AM" if schedule.startswith("0 2") else schedule,
            "latest_run": latest_run,
            "status": "active" if cron_active else "inactive"
        }
    except Exception as e:
        logger.error(f"Error checking cron status: {e}")
        return {
            "cron_active": False,
            "schedule": "Unknown",
            "status": "unknown",
            "error": str(e)
        }


def get_brain_status() -> Dict[str, Any]:
    """Read Brain v4.0 status from database"""
    try:
        from src.core.database import SessionLocal
        from src.core.models import Memory
        
        session = SessionLocal()
        try:
            session_memories = session.query(Memory).count()
            latest_memory = session.query(Memory).order_by(Memory.timestamp.desc()).first()
            
            return {
                "status": "active",
                "version": "4.0",
                "session_memories": session_memories,
                "capacity": 100,
                "usage_percent": min(100, (session_memories / 100) * 100),
                "latest_interaction": latest_memory.timestamp.isoformat() if latest_memory else None,
                "features": [
                    "Contextual Thinking",
                    "Session Memory",
                    "Pattern Detection",
                    "Confidence Scoring",
                    "Unified Database"
                ]
            }
        except Exception as e:
            logger.error(f"Error querying brain status: {e}")
            return {
                "status": "error",
                "version": "4.0",
                "session_memories": 0,
                "capacity": 100,
                "error": str(e)
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error checking brain status: {e}")
        return {
            "status": "error",
            "version": "4.0",
            "session_memories": 0,
            "capacity": 100,
            "error": str(e)
        }


def get_knowledge_stats() -> Dict[str, Any]:
    """Read knowledge index statistics"""
    try:
        index_file = DATA_DIR / "simple_index.json"

        if not index_file.exists():
            return {
                "status": "not_initialized",
                "total_chunks": 0,
                "categories": [],
                "version": "1.0"
            }

        with open(index_file, 'r') as f:
            index_data = json.load(f)

        metadata = index_data.get("metadata", {})
        total_chunks = metadata.get("total_chunks", 0)
        categories = metadata.get("categories", [])

        # Calculate progress towards 100 chunks goal
        target = 100
        progress_percent = min(100, (total_chunks / target) * 100)

        return {
            "status": "active",
            "version": metadata.get("version", "1.0"),
            "total_chunks": total_chunks,
            "categories": categories,
            "target_chunks": target,
            "progress_percent": progress_percent,
            "target_achieved": f"{total_chunks}/{target}",
            "category_breakdown": metadata.get("category_counts", {})
        }
    except Exception as e:
        logger.error(f"Error reading knowledge stats: {e}")
        return {
            "status": "error",
            "total_chunks": 0,
            "categories": [],
            "error": str(e)
        }


def get_training_summary() -> Dict[str, Any]:
    """Get latest training report and historical summary"""
    try:
        training_dir = DATA_DIR / "training"

        if not training_dir.exists():
            return {
                "status": "no_data",
                "tasks_completed": 0,
                "latest_run": None,
                "success_rate": "0%"
            }

        # Get all training reports sorted by date
        reports = sorted(training_dir.glob("daily_report_*.json"), reverse=True)

        if not reports:
            return {
                "status": "no_runs",
                "tasks_completed": 0,
                "latest_run": None,
                "success_rate": "0%"
            }

        # Read latest report
        latest_report_path = reports[0]
        with open(latest_report_path, 'r') as f:
            latest = json.load(f)

        # Calculate historical stats
        total_runs = len(reports)
        successful_runs = 0
        total_tasks = 0

        for report_path in reports[:30]:  # Last 30 reports
            try:
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    if data.get("status") == "success":
                        successful_runs += 1
                    total_tasks += data.get("tasks_completed", 0)
            except:
                pass

        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

        # Infer status from latest report
        tasks_completed_list = latest.get("tasks_completed", [])
        tasks_failed_list = latest.get("tasks_failed", [])

        if isinstance(tasks_completed_list, list):
            tasks_completed_count = len(tasks_completed_list)
        else:
            tasks_completed_count = latest.get("tasks_completed", 0)

        # Determine status
        if latest.get("status"):
            status = latest.get("status")
        elif tasks_failed_list:
            status = "failed"
        elif tasks_completed_count > 0:
            status = "success"
        else:
            status = "unknown"

        return {
            "status": status,
            "latest_run": latest.get("end_time") or latest.get("timestamp") or latest.get("start_time"),
            "tasks_completed": tasks_completed_count,
            "total_tasks": latest.get("total_tasks", tasks_completed_count),
            "success_rate": f"{success_rate:.0f}%",
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "pipeline_stages": latest.get("stages", []),
            "latest_metrics": latest.get("metrics", {})
        }
    except Exception as e:
        logger.error(f"Error reading training summary: {e}")
        return {
            "status": "error",
            "tasks_completed": 0,
            "latest_run": None,
            "success_rate": "0%",
            "error": str(e)
        }


def get_ministers_status() -> List[Dict[str, Any]]:
    """Get status of all 14 ministers from government system"""
    try:
        # Read ministers configuration
        ministers_file = PROJECT_ROOT / "src" / "government" / "ministers_config.json"

        if not ministers_file.exists():
            # Return default minister list
            return [
                {"id": 1, "name": "وزير التعليم", "status": "active", "domain": "education"},
                {"id": 2, "name": "وزير الصحة", "status": "active", "domain": "healthcare"},
                {"id": 3, "name": "وزير التجارة", "status": "active", "domain": "commerce"},
                {"id": 4, "name": "وزير التكنولوجيا", "status": "active", "domain": "technology"},
                {"id": 5, "name": "وزير الأمن", "status": "active", "domain": "security"},
                {"id": 6, "name": "وزير المالية", "status": "active", "domain": "finance"},
                {"id": 7, "name": "وزير البنية التحتية", "status": "active", "domain": "infrastructure"},
                {"id": 8, "name": "وزير البحث العلمي", "status": "active", "domain": "research"},
                {"id": 9, "name": "وزير الإعلام", "status": "active", "domain": "media"},
                {"id": 10, "name": "وزير الزراعة", "status": "active", "domain": "agriculture"},
                {"id": 11, "name": "وزير الطاقة", "status": "active", "domain": "energy"},
                {"id": 12, "name": "وزير النقل", "status": "active", "domain": "transportation"},
                {"id": 13, "name": "وزير البيئة", "status": "active", "domain": "environment"},
                {"id": 14, "name": "وزير التخطيط", "status": "active", "domain": "planning"}
            ]

        with open(ministers_file, 'r', encoding='utf-8') as f:
            ministers = json.load(f)

        return ministers
    except Exception as e:
        logger.error(f"Error reading ministers status: {e}")
        return []


def get_logs_summary() -> Dict[str, Any]:
    """Tail recent logs from various subsystems"""
    try:
        logs = {}

        # Define log files to check
        log_files = {
            "api": LOGS_DIR / "api.log",
            "training": LOGS_DIR / "training.log",
            "mcp": LOGS_DIR / "mcp_server.log",
            "brain": LOGS_DIR / "brain_v4.log"
        }

        for name, log_path in log_files.items():
            if log_path.exists():
                # Read last 50 lines
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    logs[name] = {
                        "total_lines": len(lines),
                        "last_50": [line.strip() for line in lines[-50:] if line.strip()],
                        "last_modified": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
                    }
            else:
                logs[name] = {
                    "total_lines": 0,
                    "last_50": [],
                    "status": "file_not_found"
                }

        return logs
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return {"error": str(e)}


def get_system_overview() -> Dict[str, Any]:
    """Aggregate all subsystem data into single overview"""
    try:
        mcp = get_mcp_status()
        brain = get_brain_status()
        knowledge = get_knowledge_stats()
        training = get_training_summary()
        cron = get_cron_status()
        ministers = get_ministers_status()

        # Calculate overall health
        active_components = sum([
            mcp["status"] == "active",
            brain["status"] == "active",
            knowledge["status"] == "active",
            training["status"] == "success",
            cron["cron_active"]
        ])

        total_components = 5
        health_percent = (active_components / total_components) * 100

        if health_percent == 100:
            overall_status = "🟢 100% OPERATIONAL"
        elif health_percent >= 80:
            overall_status = "🟡 MOSTLY OPERATIONAL"
        elif health_percent >= 50:
            overall_status = "🟠 DEGRADED"
        else:
            overall_status = "🔴 CRITICAL"

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_percent": health_percent,
            "active_components": active_components,
            "total_components": total_components,
            "mcp_server": mcp,
            "brain_v4": brain,
            "knowledge_index": knowledge,
            "daily_training": training,
            "cron_automation": cron,
            "ministers": {
                "total": len(ministers),
                "active": sum(1 for m in ministers if m.get("status") == "active"),
                "list": ministers
            },
            "automation_level": "100% AUTONOMOUS" if cron["cron_active"] else "MANUAL",
            "manual_intervention_required": "None" if health_percent == 100 else "Check failed components"
        }
    except Exception as e:
        logger.error(f"Error generating system overview: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "🔴 ERROR",
            "error": str(e)
        }
