#!/usr/bin/env python3
"""
Noogh Agent API - واجهة برمجية للوكيل
- إصلاحات أساسية:
  * حراسة الوصول إلى كائن agent والتحقق من التهيئة.
  * معالج أخطاء موحّد.
  * نقطة تشغيل مباشرة via uvicorn.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Add parent to path (project root)
try:
    from agent.brain import get_agent  # type: ignore
except Exception as e:
    raise RuntimeError(f"Failed to import src.agent.brain: {e}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("noogh.agent.api")

app = FastAPI(title="Noogh Agent API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None  # will be set on startup


class TaskRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None


class ToolRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    global agent
    model_path = Path("/home/noogh/models/noogh_arabic_real_data/final_model")
    try:
        agent = get_agent(model_path=str(model_path) if model_path.exists() else None)
        log.info("✅ Agent initialized (model=%s)", model_path.exists())
    except Exception as e:
        log.error("❌ Agent init failed: %s", e)
        agent = None


@app.get("/")
async def root():
    return {
        "name": "Noogh Agent API",
        "version": "1.0.1",
        "status": "running",
        "description": "نظام وكيل ذكي مستقل للتنفيذ التلقائي للمهام"
    }


@app.post("/agent/execute")
async def execute_task(request: TaskRequest):
    """
    تنفيذ مهمة من المستخدم

    يحلل الطلب، يخطط الخطوات، وينفذها تلقائياً
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        log.info("📥 Request: %s", request.user_input)
        result = agent.process_request(request.user_input)
        return {"success": True, "result": result}
    except Exception as e:
        log.exception("Error executing task")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/tool")
async def execute_tool(request: ToolRequest):
    """
    تنفيذ أداة مباشرة (بدون تخطيط)

    للاستخدام المتقدم - تنفيذ أداة محددة مباشرة
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        log.info("🔧 Tool: %s", request.tool_name)
        res = agent.tool_registry.execute(request.tool_name, **request.params)
        return {"success": res.success, "output": res.output, "error": res.error}
    except Exception as e:
        log.exception("Tool execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/tools")
async def list_tools():
    """عرض الأدوات المتاحة"""
    if agent is None:
        return {"success": True, "tools_count": 0, "tools": []}
    tools = agent.tool_registry.list_tools()
    return {"success": True, "tools_count": len(tools), "tools": tools}


@app.get("/agent/history")
async def get_history(limit: int = 10):
    """الحصول على سجل المهام المنفذة"""
    if agent is None:
        return {"success": True, "tasks_count": 0, "tasks": []}
    try:
        history = getattr(agent, 'get_task_history', lambda limit: [])(limit)
        return {
            "success": True,
            "tasks_count": len(history),
            "tasks": history
        }
    except Exception as e:
        log.exception("Error getting history")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/status")
async def get_status():
    """حالة Agent الحالية"""
    if agent is None:
        return {"success": True, "status": {"running": False, "model_loaded": False}}

    cur = getattr(agent, "current_task", None)
    return {
        "success": True,
        "status": {
            "running": True,
            "total_tasks": len(getattr(agent, "tasks", [])),
            "current_task": {
                "id": getattr(cur, "id", None),
                "description": getattr(cur, "description", None),
                "status": getattr(cur, "status", None),
                "steps_completed": len([s for s in getattr(cur, "steps", []) if s.get("status") == "completed"]) if cur else 0,
                "steps_total": len(getattr(cur, "steps", [])) if cur else 0
            } if cur else None,
            "model_loaded": getattr(agent, "model", None) is not None
        }
    }


@app.get("/agent/examples")
async def get_examples():
    """أمثلة على الأوامر المدعومة"""
    return {
        "success": True,
        "categories": {
            "file_operations": [
                "اقرأ ملف /etc/hostname",
                "اكتب ملف test.txt مع محتوى 'Hello'",
                "اعرض محتويات المجلد /tmp"
            ],
            "search": [
                "ابحث عن كلمة 'error' في المجلد الحالي",
                "جد الملفات التي تنتهي بـ .py",
                "ابحث عن 'function' في ملف script.py"
            ],
            "bash_commands": [
                "شغل أمر 'ls -lh'",
                "نفذ 'nvidia-smi'",
                "اعرض العمليات التي تعمل"
            ],
            "system_info": [
                "معلومات GPU",
                "استخدام القرص",
                "تحقق من العملية python"
            ],
            "git": [
                "عرض حالة git",
                "اعرض آخر التغييرات",
                "سجل git الأخير"
            ],
            "python": [
                "شغل كود Python: ```python\\nprint('Hello')\\n```",
                "نفذ سكريبت بايثون"
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Noogh Agent API...")
    print("📍 API will be available at: http://0.0.0.0:8500")
    print("📖 Docs: http://0.0.0.0:8500/docs")

    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="info")
