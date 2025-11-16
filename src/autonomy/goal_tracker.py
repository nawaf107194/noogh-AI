from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path
import json, time, uuid

STORE = Path("/home/noogh/noogh_unified_system/data/goals.json")
STORE.parent.mkdir(parents=True, exist_ok=True)

class GoalPriority(str, Enum):
    LOW="LOW"
    MEDIUM="MEDIUM"
    HIGH="HIGH"
    CRITICAL="CRITICAL"

class GoalState(str, Enum):
    PENDING="PENDING"
    IN_PROGRESS="IN_PROGRESS"
    COMPLETED="COMPLETED"
    FAILED="FAILED"
    BLOCKED="BLOCKED"
    CANCELLED="CANCELLED"

@dataclass
class Goal:
    id: str
    name: str
    description: str
    priority: GoalPriority
    progress: int = 0
    state: GoalState = GoalState.PENDING
    deadline: Optional[float] = None
    created_at: float = None
    updated_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

class GoalTracker:
    def __init__(self):
        self._load()

    def _load(self):
        self.goals: Dict[str, dict] = {}
        if STORE.exists():
            try:
                self.goals = json.loads(STORE.read_text(encoding="utf-8"))
            except:
                self.goals = {}

    def _save(self):
        STORE.write_text(json.dumps(self.goals, ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> List[dict]:
        return list(self.goals.values())

    def create(self, name: str, description: str, priority: GoalPriority) -> dict:
        g = Goal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            priority=priority
        )
        self.goals[g.id] = asdict(g)
        self._save()
        return self.goals[g.id]

    def update_progress(self, goal_id: str, progress: int, state: Optional[GoalState] = None) -> dict:
        g = self.goals.get(goal_id)
        if not g:
            raise KeyError("goal not found")
        g["progress"] = max(0, min(100, progress))
        if state:
            g["state"] = state
        g["updated_at"] = time.time()
        self._save()
        return g
