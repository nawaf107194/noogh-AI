#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔔 Awareness Notifier (v1.0)
يراقب تغيرات الحالة الإدراكية ويطلق تنبيه صوتي + يسجل التنبيه.
"""
import os
import json
import time
import logging
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("/home/noogh/noogh_unified_system/core/brain/conscious_state.json")
LOG_FILE = Path("/home/noogh/noogh_unified_system/core/brain/awareness_alerts.log")
CACHE_FILE = Path("/home/noogh/noogh_unified_system/core/brain/.last_alert_cache.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("notifier")

def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def load_cache():
    if not CACHE_FILE.exists():
        return {"status": None, "action": None}
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {"status": None, "action": None}

def save_cache(current):
    try:
        CACHE_FILE.write_text(json.dumps(current))
    except Exception:
        pass

def notify(message):
    log.info(f"🔔 Alert: {message}")
    # سجل التنبيه
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] {message}\n")
    # إشعار صوتي - استخدام shutil.which و subprocess الآمن
    try:
        if shutil.which("espeak"):
            subprocess.run(["espeak", message], check=False, capture_output=True)
        elif shutil.which("aplay"):
            subprocess.run(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                         check=False, capture_output=True)
    except Exception as e:
        log.debug(f"Audio notification failed: {e}")

def run():
    state = load_state()
    cache = load_cache()

    current_status = state.get("self_evaluation", {}).get("status")
    current_action = state.get("auto_regulation", {}).get("action")

    if current_status != cache.get("status") or current_action != cache.get("action"):
        msg = f"System status changed → Status: {current_status}, Action: {current_action}"
        notify(msg)

        # تحديث داخل conscious_state
        state["last_alert"] = {
            "status": current_status,
            "trigger": msg,
            "action_taken": current_action,
            "notified_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        except Exception as e:
            log.warning(f"Failed to update conscious_state.json: {e}")

        save_cache({"status": current_status, "action": current_action})

if __name__ == "__main__":
    run()