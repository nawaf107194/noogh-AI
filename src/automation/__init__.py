#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automation Module - Fallback implementation for automation capabilities
"""

from pathlib import Path
from datetime import datetime
import sys
# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
# Provide fallback implementation
from enum import Enum

class AutomationStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class AutomationTask:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.status = AutomationStatus.IDLE
        self.created_at = datetime.now()
        self.completed_at = None
    
    def start(self):
        self.status = AutomationStatus.RUNNING
        return {"status": "started", "fallback": True}
    
    def complete(self):
        self.status = AutomationStatus.COMPLETED
        self.completed_at = datetime.now()
        return {"status": "completed", "fallback": True}
    
    def fail(self, error):
        self.status = AutomationStatus.FAILED
        return {"status": "failed", "error": str(error), "fallback": True}

class AutomationEngine:
    def __init__(self):
        self.tasks = []
        self.running_tasks = []
    
    def create_task(self, name, description=""):
        """Create a new automation task"""
        task = AutomationTask(name, description)
        self.tasks.append(task)
        return task
    
    def run_task(self, task):
        """Run an automation task"""
        if task not in self.running_tasks:
            self.running_tasks.append(task)
        return task.start()
    
    def get_task_status(self, task_name):
        """Get status of a task"""
        for task in self.tasks:
            if task.name == task_name:
                return {
                    "name": task.name,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat(),
                    "fallback": True
                }
        return None
    
    def get_all_tasks(self):
        """Get all tasks"""
        return [
            {
                "name": task.name,
                "status": task.status.value,
                "description": task.description
            }
            for task in self.tasks
        ]

class AutomationScheduler:
    def __init__(self):
        self.engine = AutomationEngine()
        self.scheduled_tasks = []
    
    def schedule_task(self, task, schedule):
        """Schedule a task"""
        self.scheduled_tasks.append({
            "task": task,
            "schedule": schedule
        })
        return {"status": "scheduled", "fallback": True}
    
    def get_scheduled_tasks(self):
        """Get all scheduled tasks"""
        return self.scheduled_tasks

# Singleton instances
_automation_engine = None
_automation_scheduler = None

def get_automation_engine():
    """Get or create automation engine singleton"""
    global _automation_engine
    if _automation_engine is None:
        _automation_engine = AutomationEngine()
    return _automation_engine

def get_automation_scheduler():
    """Get or create automation scheduler singleton"""
    global _automation_scheduler
    if _automation_scheduler is None:
        _automation_scheduler = AutomationScheduler()
    return _automation_scheduler

# Export all
__all__ = [
    'AutomationStatus', 'AutomationTask', 'AutomationEngine', 'AutomationScheduler',
    'get_automation_engine', 'get_automation_scheduler'
]
