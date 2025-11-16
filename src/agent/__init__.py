"""
Noogh Agent System - نظام الوكيل الذكي
نظام تنفيذ تلقائي للمهام بدون تدخل خارجي
"""

from .tools import tool_registry, ToolResult
from .brain import AgentBrain, get_agent, Task

__all__ = [
    "tool_registry",
    "ToolResult",
    "AgentBrain",
    "get_agent",
    "Task"
]

__version__ = "1.0.0"
