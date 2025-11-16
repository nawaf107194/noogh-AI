from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import json, time, os
from pathlib import Path

LOG_FILE = Path("/home/noogh/noogh_unified_system/logs/autonomy.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

class ActionType(Enum):
    LOG_WRITE = auto()
    REPORT_GENERATE = auto()
    TIMELINE_UPDATE = auto()
    PERFORMANCE_SNAPSHOT = auto()
    ALERT_NOTIFY = auto()
    STATS_UPDATE = auto()
    CACHE_CLEAR = auto()
    # Critical
    TRAINING_START = auto()
    TRAINING_STOP = auto()
    SERVICE_RESTART = auto()
    FILE_MODIFY = auto()
    HYPERPARAMETER_CHANGE = auto()
    MODEL_DELETE = auto()
    CONFIG_UPDATE = auto()

class Severity(Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

SAFE_SET = {
    ActionType.LOG_WRITE, ActionType.REPORT_GENERATE, ActionType.TIMELINE_UPDATE,
    ActionType.PERFORMANCE_SNAPSHOT, ActionType.ALERT_NOTIFY, ActionType.STATS_UPDATE,
    ActionType.CACHE_CLEAR
}
CRITICAL_SET = {
    ActionType.TRAINING_START, ActionType.TRAINING_STOP, ActionType.SERVICE_RESTART,
    ActionType.FILE_MODIFY, ActionType.HYPERPARAMETER_CHANGE, ActionType.MODEL_DELETE,
    ActionType.CONFIG_UPDATE
}

@dataclass
class Action:
    id: str
    type: ActionType
    payload: Dict[str, Any]
    severity: Severity
    status: str = "PENDING"
    created_at: float = None
    executed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class ActionExecutor:
    def __init__(self, mode: str = "B"):
        self.mode = mode
        self.history: List[Action] = []

    def classify(self, action_type: ActionType) -> Severity:
        if action_type in SAFE_SET: return Severity.SAFE
        if action_type in CRITICAL_SET: return Severity.CRITICAL
        return Severity.MEDIUM

    def _log(self, msg: str):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def execute_safe(self, action: Action) -> Action:
        action.status = "EXECUTED"
        action.executed_at = time.time()

        # تنفيذات مبسطة قابلة للتوسّع
        if action.type == ActionType.LOG_WRITE:
            self._log(action.payload.get("message","(empty)"))
        elif action.type == ActionType.REPORT_GENERATE:
            Path("/home/noogh/noogh_unified_system/reports").mkdir(exist_ok=True)
            name = action.payload.get("name","auto_report")
            p = Path(f"/home/noogh/noogh_unified_system/reports/{name}_{int(time.time())}.json")
            with open(p,"w",encoding="utf-8") as f:
                json.dump(action.payload, f, ensure_ascii=False, indent=2)
            action.result = {"report_path": str(p)}
        elif action.type == ActionType.PERFORMANCE_SNAPSHOT:
            # placeholder snapshot
            action.result = {"cpu":"n/a","ram":"n/a","gpu":"n/a"}
        elif action.type == ActionType.ALERT_NOTIFY:
            self._log(f"ALERT: {action.payload}")

        action.result = action.result or {"ok": True}
        self.history.append(action)
        return action

    def request_approval(self, action: Action, approval_queue) -> Action:
        action.status = "PENDING_APPROVAL"
        approval_queue.enqueue(action)
        self._log(f"Approval requested for {action.id} ({action.type.name})")
        self.history.append(action)
        return action

    def submit(self, action_id: str, action_type: ActionType, payload: Dict[str,Any], approval_queue=None) -> Action:
        severity = self.classify(action_type)
        action = Action(id=action_id, type=action_type, payload=payload, severity=severity)

        if severity == Severity.SAFE and self.mode in ("B","C"):
            return self.execute_safe(action)

        # كل ما عدا ذلك يذهب للموافقة
        if approval_queue is None:
            raise RuntimeError("ApprovalQueue required for non-safe actions")
        return self.request_approval(action, approval_queue)
