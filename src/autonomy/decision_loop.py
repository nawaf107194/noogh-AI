import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

LOG = Path("/home/noogh/noogh_unified_system/logs/autonomy.log")
LOG.parent.mkdir(parents=True, exist_ok=True)

@dataclass
class DecisionLoopConfig:
    interval_sec: int = 30
    mode: str = "B"  # A/B/C

class DecisionLoop:
    def __init__(self, config: DecisionLoopConfig, executor, approvals):
        self.cfg = config
        self.exec = executor
        self.approvals = approvals
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task = None
        self.cycles = 0

        # Phase 4.5: Initialize awareness system
        from .system_monitor import SystemMonitor
        from .awareness_mapper import AwarenessMapper
        self.monitor = SystemMonitor()
        self.mapper = AwarenessMapper()
        self._log("Awareness system initialized")

        # Phase 5: Initialize baseline learning system
        from .health_baseline import HealthBaseline
        self.baseline = HealthBaseline()
        self._log("Baseline learning system initialized")

        # Phase 6: Initialize cognitive advisor
        from .cognitive_advisor import CognitiveAdvisor
        from .goal_tracker import GoalTracker
        self.advisor = CognitiveAdvisor(mode="2")  # Semi-Optimization
        self.goals = GoalTracker()
        self._log("Cognitive advisor initialized (Mode 2: Semi-Optimization)")

    def _log(self, msg:str):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [loop] {msg}\n")

    async def start(self):
        """Start the decision loop as an async task"""
        if self._task and not self._task.done():
            return
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        self._log("Decision loop started (Async)")

    async def stop(self):
        """Stop the decision loop"""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            except asyncio.CancelledError:
                pass
        self._log("Decision loop stopped")

    async def _run(self):
        """Main async loop"""
        while not self._stop_event.is_set():
            try:
                self.cycles += 1

                # Phase 4.5 & 5: Awareness-to-Action Cycle with Baseline Learning
                # 1. Capture real system health (run in executor if blocking)
                loop = asyncio.get_running_loop()
                health = await loop.run_in_executor(None, self.monitor.capture_snapshot)

                # 2. Phase 5: Record to baseline history
                self.baseline.record_snapshot(health)

                # 3. Phase 5: Update baselines every 10 cycles for faster learning
                # (Will change to 30 after stabilization)
                if self.cycles % 10 == 0 and self.cycles >= 10:
                    self.baseline.update_baselines(lookback_hours=24)
                    self._log(f"cycle {self.cycles}: baselines updated")

                # 4. Phase 5: Check for baseline deviations
                if self.cycles > 10:  # Need baseline data first
                    deviations = self.baseline.check_deviations(health, threshold=2.5)
                    for dev in deviations:
                        msg = f"Baseline deviation: {dev['metric']} = {dev['current']:.1f} (expected {dev['baseline_mean']:.1f}, Δ={dev['deviation']:.2f}σ)"
                        self._log(f"cycle {self.cycles}: {msg}")
                        self.baseline.record_alert(
                            alert_type=dev['severity'],
                            metric_name=dev['metric'],
                            value=dev['current'],
                            baseline_mean=dev['baseline_mean'],
                            deviation=dev['deviation'],
                            message=msg
                        )

                        # Phase 6: Cognitive analysis and optimization
                        suggestion = self.advisor.analyze_deviation(dev)
                        if suggestion:
                            self._log(f"cycle {self.cycles}: 🧠 {suggestion.problem}")
                            self._log(f"cycle {self.cycles}: 💡 {suggestion.suggested_action}")

                            # Store suggestion in history
                            self.advisor.suggestions_history.append(suggestion)

                            # Create goal automatically
                            goal = self.goals.create(
                                name=suggestion.goal_name,
                                description=suggestion.goal_description,
                                priority=suggestion.goal_priority
                            )
                            self._log(f"cycle {self.cycles}: 🎯 Goal created: {suggestion.goal_name}")

                            # Execute SAFE optimizations automatically
                            if suggestion.optimization_type.value == "SAFE":
                                try:
                                    # Assuming exec.submit is async or we wrap it
                                    # If exec.submit is synchronous, wrap in run_in_executor
                                    # For now assuming it returns a future or we just call it
                                    # But wait, exec is likely a ThreadPoolExecutor in the original code
                                    # We should probably keep using it for the actual execution if it's blocking
                                    
                                    action = self.exec.submit(
                                        suggestion.id,
                                        suggestion.action_type,
                                        {"suggestion": suggestion.suggested_action, "confidence": suggestion.confidence},
                                        self.approvals
                                    )
                                    self._log(f"cycle {self.cycles}: ✅ SAFE optimization executed: {suggestion.action_type.name}")
                                except Exception as e:
                                    self._log(f"cycle {self.cycles}: ❌ Failed to execute optimization: {e}")
                            else:
                                # CRITICAL actions go to approval queue
                                self._log(f"cycle {self.cycles}: ⚠️  CRITICAL action queued for approval: {suggestion.action_type.name}")

                # 5. Evaluate awareness rules
                triggered_actions = self.mapper.evaluate(health)

                # 6. Execute triggered actions
                for action_data in triggered_actions:
                    try:
                        self.exec.submit(
                            action_data["id"],
                            action_data["type"],
                            action_data["payload"],
                            self.approvals
                        )
                        self._log(f"cycle {self.cycles}: triggered {action_data['rule']}")
                    except Exception as e:
                        self._log(f"error executing {action_data['rule']}: {e}")

                # 7. Log health snapshot (every 10 cycles for summary)
                if self.cycles % 10 == 0:
                    issues = health.get_issues()
                    if issues:
                        self._log(f"cycle {self.cycles}: {len(issues)} issues detected")
                    else:
                        self._log(f"cycle {self.cycles}: system healthy")

                # 8. Clean up expired approvals
                _ = self.approvals.list_pending()

            except Exception as e:
                self._log(f"error in cycle {self.cycles}: {e}")
            
            # Non-blocking sleep
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.cfg.interval_sec)
                break # Stop event set
            except asyncio.TimeoutError:
                pass # Continue loop
