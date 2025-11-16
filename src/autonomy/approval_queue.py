from typing import Dict, List
from pathlib import Path
import json, time

STORE = Path("/home/noogh/noogh_unified_system/data/approval_queue.json")
STORE.parent.mkdir(parents=True, exist_ok=True)

TIMEOUTS = {
    "MEDIUM": 60*60,
    "HIGH":   30*60,
    "CRITICAL": 10*60,
}

class ApprovalQueue:
    def __init__(self):
        self._load()

    def _load(self):
        self.items: Dict[str, dict] = {}
        if STORE.exists():
            try:
                self.items = json.loads(STORE.read_text(encoding="utf-8"))
            except:
                self.items = {}

    def _save(self):
        STORE.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

    def enqueue(self, action):
        timeout = TIMEOUTS.get(action.severity.name, 0)
        self.items[action.id] = {
            "id": action.id,
            "type": action.type.name,
            "payload": action.payload,
            "severity": action.severity.name,
            "status": "PENDING",
            "created_at": action.created_at,
            "expires_at": action.created_at + timeout if timeout else None
        }
        self._save()

    def list_pending(self) -> List[dict]:
        now = time.time()
        out = []
        for a in self.items.values():
            if a["status"]=="PENDING":
                if a["expires_at"] and now > a["expires_at"]:
                    a["status"]="EXPIRED"
                out.append(a)
        self._save()
        return out

    def decide(self, action_id: str, approve: bool, reason: str = "") -> dict:
        a = self.items.get(action_id)
        if not a:
            raise KeyError("action not found")
        if a["status"]!="PENDING":
            return a
        a["status"] = "APPROVED" if approve else "REJECTED"
        a["decision_reason"] = reason
        a["decided_at"] = time.time()
        self._save()
        return a
