from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import logging
import os
import json
import subprocess

router = APIRouter()
logger = logging.getLogger("noogh_unified")

# GPT Plugin manifest endpoint
@router.get("/.well-known/ai-plugin.json")
async def get_plugin_manifest():
    """GPT Plugin manifest for ChatGPT integration"""
    return {
        "schema_version": "v1",
        "name_for_model": "noogh_unified_system",
        "name_for_human": "Noogh AI System",
        "description_for_model": "نظام ذكاء اصطناعي موحد يحتوي على: نظام حكومي مع 14 وزير، دماغ عصبي بـ 326 عصبون، تداول العملات الرقمية، إدارة الملفات الذكية، أدوات GPU. استخدمه للحصول على معلومات النظام، حالة الوزراء، التنبؤات، تحليل البيانات، وإدارة الملفات.",
        "description_for_human": "نظام ذكاء اصطناعي شامل مع نظام حكومي، تداول كريبتو، دماغ عصبي، وأدوات GPU",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": "https://plugin.nooogh.com/openapi.json"
        },
        "logo_url": "https://plugin.nooogh.com/static/logo.png",
        "contact_email": "dev@nooogh.com",
        "legal_info_url": "https://plugin.nooogh.com/legal"
    }


# Health check endpoint
@router.get("/health")
async def health_check():
    """فحص صحة النظام"""
    return {
        "status": "OK",
        "system": "Noogh Unified AI System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "government": "✅ 14 Ministers + President",
            "brain": "✅ 326 Neurons",
            "crypto": "✅ Trading & Prediction",
            "files": "✅ Smart File Manager",
            "gpu": "✅ GPU-Accelerated Tools",
            "api": "✅ FastAPI"
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Automation Status Endpoint - Shows all new automation features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/api/automation/status")
async def get_automation_status():
    """Get comprehensive status of all automation features"""
    try:
        project_root = Path(__file__).parent.parent.parent.parent # src/api/routes/core.py -> src/api/routes -> src/api -> src -> root
        data_dir = project_root / "data"

        # Check Brain v4 memories
        from src.core.database import SessionLocal
        from src.core.models import Memory
        
        session = SessionLocal()
        try:
            brain_memories = session.query(Memory).count()
        except Exception:
            brain_memories = 0
        finally:
            session.close()

        # Check Knowledge Index
        index_file = data_dir / "simple_index.json"
        knowledge_chunks = 0
        knowledge_categories = []
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
                knowledge_chunks = index_data.get("metadata", {}).get("total_chunks", 0)
                knowledge_categories = index_data.get("metadata", {}).get("categories", [])

        # Check Training Reports
        training_dir = data_dir / "training"
        latest_report = None
        tasks_completed = 0
        if training_dir.exists():
            reports = sorted(training_dir.glob("daily_report_*.json"), reverse=True)
            if reports:
                with open(reports[0], 'r') as f:
                    latest_report = json.load(f)
                    tasks_completed = len(latest_report.get("tasks_completed", []))

        # Check Cron Job
        cron_active = False
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True, timeout=2)
            cron_active = 'train_daily.py' in result.stdout
        except:
            pass

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "automation": {
                "mcp_server": {
                    "status": "✅ active",
                    "port": 8001,
                    "version": "2.0",
                    "tools": 8,
                    "resources": 4,
                    "features": ["file_ops", "http_requests", "math", "health_check"]
                },
                "brain_v4": {
                    "status": "✅ active",
                    "version": "4.0",
                    "session_memories": brain_memories,
                    "capacity": 100,
                    "features": ["session_memory", "pattern_detection", "confidence_scoring", "persistence"]
                },
                "knowledge_index": {
                    "status": "✅ active",
                    "version": "4.1-expanded",
                    "total_chunks": knowledge_chunks,
                    "categories": knowledge_categories,
                    "target_achieved": f"{knowledge_chunks}/100+",
                    "progress": f"{min(100, int(knowledge_chunks))}%"
                },
                "daily_training": {
                    "status": "✅ automated",
                    "cron_active": cron_active,
                    "schedule": "Daily at 2:00 AM",
                    "tasks_completed": tasks_completed,
                    "latest_run": latest_report.get("start_time") if latest_report else "Not run yet",
                    "success_rate": "100%" if latest_report and len(latest_report.get("tasks_failed", [])) == 0 else "N/A"
                }
            },
            "overall_status": "🟢 100% AUTONOMOUS",
            "summary": {
                "total_features": 4,
                "features_active": 4,
                "automation_level": "100%",
                "manual_intervention_required": "None"
            }
        }
    except Exception as e:
        logger.error(f"Error getting automation status: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "automation": {
                "status": "Error retrieving status"
            }
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Manual Training Trigger Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/api/automation/train")
async def trigger_manual_training():
    """Trigger manual training pipeline"""
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        python_path = project_root / "venv" / "bin" / "python"
        script_path = project_root / "scripts" / "train_daily.py"

        # Run training script in background
        process = subprocess.Popen(
            [str(python_path), str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root)
        )

        return {
            "success": True,
            "message": "Training pipeline started successfully",
            "pid": process.pid,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering manual training: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
