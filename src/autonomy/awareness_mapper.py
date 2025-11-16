"""
Awareness-to-Action Mapper
Links system awareness to preventive/corrective actions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from .action_executor import ActionType, Action, Severity
from .system_monitor import SystemHealth


@dataclass
class AwarenessRule:
    """Rule that maps awareness state to action"""
    name: str
    condition: callable  # Function that checks if rule applies
    action_type: ActionType
    severity: Severity
    payload_generator: callable  # Generates action payload
    description: str
    cooldown_seconds: int = 300  # Don't trigger same rule within 5 min


class AwarenessMapper:
    """
    Maps system awareness to actions

    Example flow:
    1. Detect: GPU temp > 85°C
    2. Map to: ALERT_NOTIFY (safe) + suggest TRAINING_STOP (critical)
    3. Execute safe action immediately
    4. Queue critical action for approval
    """

    def __init__(self):
        self.rules: List[AwarenessRule] = []
        self.last_triggered: Dict[str, float] = {}  # rule_name -> timestamp
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default awareness-to-action mappings"""

        # Rule 1: GPU Overheating (CRITICAL)
        self.rules.append(AwarenessRule(
            name="gpu_overheat_critical",
            condition=lambda h: h.gpu_available and h.gpu_temp and h.gpu_temp > 85,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "CRITICAL",
                "message": f"GPU overheating: {h.gpu_temp}°C (threshold: 85°C)",
                "metric": "gpu_temp",
                "value": h.gpu_temp,
                "threshold": 85
            },
            description="Alert when GPU temperature exceeds 85°C"
        ))

        # Rule 2: GPU Running Hot (WARNING)
        self.rules.append(AwarenessRule(
            name="gpu_hot_warning",
            condition=lambda h: h.gpu_available and h.gpu_temp and 80 < h.gpu_temp <= 85,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "WARNING",
                "message": f"GPU running hot: {h.gpu_temp}°C",
                "metric": "gpu_temp",
                "value": h.gpu_temp,
                "threshold": 80
            },
            description="Alert when GPU temperature between 80-85°C",
            cooldown_seconds=600  # 10 min cooldown for warnings
        ))

        # Rule 3: VRAM Critical
        self.rules.append(AwarenessRule(
            name="vram_critical",
            condition=lambda h: h.gpu_available and h.vram_percent and h.vram_percent > 90,
            action_type=ActionType.CACHE_CLEAR,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "reason": f"VRAM at {h.vram_percent:.1f}% (critical)",
                "target": "gpu_cache"
            },
            description="Clear cache when VRAM exceeds 90%"
        ))

        # Rule 4: VRAM High
        self.rules.append(AwarenessRule(
            name="vram_high_alert",
            condition=lambda h: h.gpu_available and h.vram_percent and 80 < h.vram_percent <= 90,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "WARNING",
                "message": f"VRAM usage high: {h.vram_percent:.1f}%",
                "metric": "vram_percent",
                "value": h.vram_percent,
                "threshold": 80
            },
            description="Alert when VRAM usage between 80-90%",
            cooldown_seconds=600
        ))

        # Rule 5: CPU Critical
        self.rules.append(AwarenessRule(
            name="cpu_critical",
            condition=lambda h: h.cpu_percent > 90,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "CRITICAL",
                "message": f"CPU usage critical: {h.cpu_percent:.1f}%",
                "metric": "cpu_percent",
                "value": h.cpu_percent,
                "threshold": 90
            },
            description="Alert when CPU usage exceeds 90%"
        ))

        # Rule 6: RAM Critical
        self.rules.append(AwarenessRule(
            name="ram_critical",
            condition=lambda h: h.ram_percent > 90,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "CRITICAL",
                "message": f"RAM usage critical: {h.ram_percent:.1f}% ({h.ram_used_gb:.1f}/{h.ram_total_gb:.1f} GB)",
                "metric": "ram_percent",
                "value": h.ram_percent,
                "threshold": 90
            },
            description="Alert when RAM usage exceeds 90%"
        ))

        # Rule 7: Disk Almost Full
        self.rules.append(AwarenessRule(
            name="disk_full_warning",
            condition=lambda h: h.disk_percent > 90,
            action_type=ActionType.ALERT_NOTIFY,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "level": "WARNING",
                "message": f"Disk almost full: {h.disk_percent:.1f}% ({h.disk_used_gb:.1f}/{h.disk_total_gb:.1f} GB)",
                "metric": "disk_percent",
                "value": h.disk_percent,
                "threshold": 90
            },
            description="Alert when disk usage exceeds 90%",
            cooldown_seconds=1800  # 30 min cooldown
        ))

        # Rule 8: System Healthy - Performance Snapshot
        self.rules.append(AwarenessRule(
            name="healthy_snapshot",
            condition=lambda h: h.is_healthy(),
            action_type=ActionType.PERFORMANCE_SNAPSHOT,
            severity=Severity.SAFE,
            payload_generator=lambda h: {
                "status": "healthy",
                "cpu": h.cpu_percent,
                "ram": h.ram_percent,
                "gpu_temp": h.gpu_temp if h.gpu_available else None,
                "vram": h.vram_percent if h.gpu_available else None
            },
            description="Take snapshot when system is healthy",
            cooldown_seconds=60  # Every minute when healthy
        ))

    def _is_on_cooldown(self, rule_name: str, cooldown: int) -> bool:
        """Check if rule is on cooldown"""
        if rule_name not in self.last_triggered:
            return False

        elapsed = time.time() - self.last_triggered[rule_name]
        return elapsed < cooldown

    def evaluate(self, health: SystemHealth) -> List[Dict]:
        """
        Evaluate all rules against current health state
        Returns list of actions to be taken
        """
        triggered_actions = []

        for rule in self.rules:
            # Check cooldown
            if self._is_on_cooldown(rule.name, rule.cooldown_seconds):
                continue

            # Check condition
            try:
                if rule.condition(health):
                    action_data = {
                        "id": f"{rule.name}_{int(time.time())}",
                        "type": rule.action_type,
                        "severity": rule.severity,
                        "payload": rule.payload_generator(health),
                        "rule": rule.name,
                        "description": rule.description
                    }

                    triggered_actions.append(action_data)
                    self.last_triggered[rule.name] = time.time()

            except Exception as e:
                # Rule evaluation failed, skip
                pass

        return triggered_actions

    def get_rule_stats(self) -> Dict:
        """Get statistics about rules"""
        return {
            "total_rules": len(self.rules),
            "triggered_count": len(self.last_triggered),
            "rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "action_type": rule.action_type.name,
                    "severity": rule.severity.value,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "last_triggered": self.last_triggered.get(rule.name)
                }
                for rule in self.rules
            ]
        }


# Test
if __name__ == "__main__":
    from .system_monitor import SystemMonitor

    print("=" * 70)
    print("🧠 Awareness-to-Action Mapper Test")
    print("=" * 70)

    monitor = SystemMonitor()
    mapper = AwarenessMapper()

    # Get current health
    health = monitor.capture_snapshot()

    print(f"\n📊 Current System State:")
    print(f"   CPU: {health.cpu_percent:.1f}%")
    print(f"   RAM: {health.ram_percent:.1f}%")
    if health.gpu_available:
        print(f"   GPU Temp: {health.gpu_temp}°C")
        print(f"   VRAM: {health.vram_percent:.1f}%")

    # Evaluate rules
    actions = mapper.evaluate(health)

    print(f"\n🎯 Triggered Actions: {len(actions)}")
    for action in actions:
        print(f"\n   Rule: {action['rule']}")
        print(f"   Type: {action['type'].name}")
        print(f"   Severity: {action['severity'].value}")
        print(f"   Description: {action['description']}")
        print(f"   Payload: {action['payload']}")

    # Rule stats
    print(f"\n📊 Rule Statistics:")
    stats = mapper.get_rule_stats()
    print(f"   Total rules: {stats['total_rules']}")
    print(f"   Rules triggered at least once: {stats['triggered_count']}")

    print("\n" + "=" * 70)
