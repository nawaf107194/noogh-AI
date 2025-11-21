from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
# Add project root to path
from autonomy import (
    ActionExecutor, ActionType, Severity, ApprovalQueue,
    GoalTracker, GoalPriority, DecisionLoop, DecisionLoopConfig
)
import time, uuid

router = APIRouter(prefix="/autonomy", tags=["autonomy"])

# مفردات مشتركة (Singletons مبسطة)
executor = ActionExecutor(mode="B")
approvals = ApprovalQueue()
goals = GoalTracker()
loop_cfg = DecisionLoopConfig(interval_sec=30, mode="B")
loop = DecisionLoop(loop_cfg, executor, approvals)
loop.start()

class NewGoal(BaseModel):
    name: str
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM

@router.get("/status")
def status():
    return {
        "mode": executor.mode,
        "cycles": loop.cycles,
        "history_len": len(executor.history),
        "pending_approvals": len(approvals.list_pending()),
        "goals": len(goals.list())
    }

@router.get("/actions")
def actions():
    # Return last 20 actions
    recent = executor.history[-20:] if len(executor.history) > 20 else executor.history
    return [{
        "id": a.id,
        "type": a.type.name,
        "payload": a.payload,
        "severity": a.severity.value,
        "status": a.status,
        "created_at": a.created_at,
        "executed_at": a.executed_at,
        "result": a.result
    } for a in recent]

@router.get("/approvals")
def approvals_list():
    return approvals.list_pending()

@router.post("/approve/{action_id}")
def approve(action_id: str):
    return approvals.decide(action_id, approve=True, reason="approved via API")

@router.post("/reject/{action_id}")
def reject(action_id: str, reason: Optional[str] = "rejected via API"):
    return approvals.decide(action_id, approve=False, reason=reason or "")

@router.get("/goals")
def list_goals():
    return goals.list()

@router.post("/goals")
def create_goal(g: NewGoal):
    return goals.create(g.name, g.description, g.priority)

# مثال يدوي لطلب إجراء حساس (يروح للموافقة)
@router.post("/request/training_start")
def request_training_start():
    aid = "train_" + str(uuid.uuid4())
    action = executor.submit(aid, ActionType.TRAINING_START, {"note":"requested via API"}, approvals)
    return {
        "requested": action.id,
        "status": action.status,
        "severity": action.severity.value,
        "type": action.type.name
    }

# Phase 4.5: Awareness endpoints
@router.get("/awareness/health")
def awareness_health():
    """Get real-time system health snapshot"""
    return loop.monitor.get_health_report()

@router.get("/awareness/rules")
def awareness_rules():
    """Get awareness-to-action mapping rules"""
    return loop.mapper.get_rule_stats()

@router.get("/awareness/report")
def awareness_report():
    """Get comprehensive awareness report"""
    health_report = loop.monitor.get_health_report()
    rule_stats = loop.mapper.get_rule_stats()

    # Get current health snapshot for evaluation
    health = loop.monitor.capture_snapshot()
    current_triggers = loop.mapper.evaluate(health)

    return {
        "timestamp": time.time(),
        "cycles_completed": loop.cycles,
        "health": health_report,
        "rules": rule_stats,
        "current_triggers": len(current_triggers),
        "triggered_rules": [t["rule"] for t in current_triggers]
    }
