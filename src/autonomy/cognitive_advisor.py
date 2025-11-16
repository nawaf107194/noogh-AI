"""
Phase 6: Cognitive Advisor - Self-Optimization Intelligence

Mode 2: Semi-Optimization
- Analyzes deviations and suggests solutions
- Creates goals automatically
- Executes SAFE optimizations automatically
- Requires approval for CRITICAL actions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import time

from .action_executor import ActionType, Severity
from .goal_tracker import GoalPriority


class OptimizationType(Enum):
    """Types of optimization actions"""
    SAFE = "SAFE"          # Execute automatically
    CRITICAL = "CRITICAL"  # Require approval


@dataclass
class Suggestion:
    """Optimization suggestion from cognitive analysis"""
    id: str
    deviation_metric: str
    current_value: float
    baseline_value: float
    deviation_sigma: float

    # Suggestion details
    problem: str
    root_cause: str
    suggested_action: str
    action_type: ActionType
    optimization_type: OptimizationType

    # Goal creation
    goal_name: str
    goal_description: str
    goal_priority: GoalPriority

    # Metadata
    created_at: float
    confidence: float  # 0.0 - 1.0


class CognitiveAdvisor:
    """
    Analyzes system deviations and suggests intelligent optimizations.
    Mode 2: Semi-Optimization (safe actions auto, critical need approval)
    """

    def __init__(self, mode: str = "2"):
        self.mode = mode
        self.suggestions_history: List[Suggestion] = []

    def analyze_deviation(self, deviation: Dict) -> Optional[Suggestion]:
        """
        Analyze a deviation and generate intelligent suggestion.

        Args:
            deviation: {
                'metric': str,
                'current': float,
                'baseline_mean': float,
                'deviation': float (sigma),
                'severity': str
            }

        Returns:
            Suggestion object or None
        """
        metric = deviation['metric']
        current = deviation['current']
        baseline = deviation['baseline_mean']
        sigma = deviation['deviation']

        # Route to appropriate analyzer
        if metric == 'cpu_percent':
            return self._analyze_cpu_deviation(current, baseline, sigma)
        elif metric == 'ram_percent':
            return self._analyze_ram_deviation(current, baseline, sigma)
        elif metric == 'gpu_temp':
            return self._analyze_gpu_temp_deviation(current, baseline, sigma)
        elif metric == 'gpu_utilization':
            return self._analyze_gpu_util_deviation(current, baseline, sigma)
        elif metric == 'vram_percent':
            return self._analyze_vram_deviation(current, baseline, sigma)
        elif metric == 'disk_percent':
            return self._analyze_disk_deviation(current, baseline, sigma)

        return None

    def _analyze_cpu_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze CPU usage deviation"""

        # High CPU (> baseline)
        if current > baseline and sigma > 2.5:
            if current > 90:
                # Critical CPU spike
                return Suggestion(
                    id=f"cpu_critical_{int(time.time())}",
                    deviation_metric="cpu_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"Critical CPU spike: {current:.1f}% (expected {baseline:.1f}%)",
                    root_cause="High computational load - possibly training, inference, or runaway process",
                    suggested_action="Reduce concurrent processes or check for resource leaks",
                    action_type=ActionType.ALERT_NOTIFY,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Optimize CPU usage",
                    goal_description=f"Reduce CPU from {current:.1f}% to ~{baseline:.1f}%",
                    goal_priority=GoalPriority.HIGH,
                    created_at=time.time(),
                    confidence=0.9
                )
            elif current > baseline * 1.5:
                # Moderate CPU increase
                return Suggestion(
                    id=f"cpu_moderate_{int(time.time())}",
                    deviation_metric="cpu_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"CPU usage elevated: {current:.1f}% (expected {baseline:.1f}%)",
                    root_cause="Increased workload or inefficient processing",
                    suggested_action="Profile CPU-intensive operations and optimize batch sizes",
                    action_type=ActionType.PERFORMANCE_SNAPSHOT,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Investigate CPU increase",
                    goal_description=f"Analyze and optimize CPU from {current:.1f}% to baseline",
                    goal_priority=GoalPriority.MEDIUM,
                    created_at=time.time(),
                    confidence=0.8
                )

        return None

    def _analyze_ram_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze RAM usage deviation"""

        if current > baseline and sigma > 2.5:
            if current > 85:
                # Critical RAM
                return Suggestion(
                    id=f"ram_critical_{int(time.time())}",
                    deviation_metric="ram_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"Critical RAM usage: {current:.1f}% (expected {baseline:.1f}%)",
                    root_cause="Possible memory leak or large model loaded",
                    suggested_action="Clear caches and investigate memory leaks",
                    action_type=ActionType.CACHE_CLEAR,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Fix memory issue",
                    goal_description=f"Reduce RAM from {current:.1f}% to ~{baseline:.1f}%",
                    goal_priority=GoalPriority.HIGH,
                    created_at=time.time(),
                    confidence=0.85
                )
            elif current > baseline * 1.3:
                return Suggestion(
                    id=f"ram_moderate_{int(time.time())}",
                    deviation_metric="ram_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"RAM usage elevated: {current:.1f}% (expected {baseline:.1f}%)",
                    root_cause="Growing memory footprint",
                    suggested_action="Monitor for gradual memory growth (potential leak)",
                    action_type=ActionType.PERFORMANCE_SNAPSHOT,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Monitor memory trend",
                    goal_description="Track RAM usage over next 24h for leak detection",
                    goal_priority=GoalPriority.MEDIUM,
                    created_at=time.time(),
                    confidence=0.75
                )

        return None

    def _analyze_gpu_temp_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze GPU temperature deviation"""

        if current > baseline and sigma > 2.5:
            if current > 80:
                # GPU getting hot
                return Suggestion(
                    id=f"gpu_temp_high_{int(time.time())}",
                    deviation_metric="gpu_temp",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"GPU temperature elevated: {current:.0f}°C (expected {baseline:.0f}°C)",
                    root_cause="High GPU workload or insufficient cooling",
                    suggested_action="Consider reducing training batch size or inference concurrency",
                    action_type=ActionType.ALERT_NOTIFY,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Cool down GPU",
                    goal_description=f"Reduce GPU temp from {current:.0f}°C to <75°C",
                    goal_priority=GoalPriority.HIGH,
                    created_at=time.time(),
                    confidence=0.9
                )

        return None

    def _analyze_gpu_util_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze GPU utilization deviation"""

        # High GPU util - flag for testing (lowered threshold)
        if current > baseline and sigma > 3.0 and current > 5:
            return Suggestion(
                id=f"gpu_util_high_{int(time.time())}",
                deviation_metric="gpu_utilization",
                current_value=current,
                baseline_value=baseline,
                deviation_sigma=sigma,
                problem=f"GPU utilization at maximum: {current:.0f}%",
                root_cause="Heavy GPU workload",
                suggested_action="Normal during training, but monitor temperature",
                action_type=ActionType.PERFORMANCE_SNAPSHOT,
                optimization_type=OptimizationType.SAFE,
                goal_name="Monitor GPU workload",
                goal_description="Ensure GPU workload is intentional and sustainable",
                goal_priority=GoalPriority.LOW,
                created_at=time.time(),
                confidence=0.6
            )

        return None

    def _analyze_vram_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze VRAM usage deviation"""

        if current > baseline and sigma > 2.5:
            if current > 85:
                # Critical VRAM
                return Suggestion(
                    id=f"vram_critical_{int(time.time())}",
                    deviation_metric="vram_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"VRAM critically high: {current:.1f}% (expected {baseline:.1f}%)",
                    root_cause="Large model or accumulated tensors",
                    suggested_action="Clear GPU cache and optimize model size",
                    action_type=ActionType.CACHE_CLEAR,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Free VRAM",
                    goal_description=f"Reduce VRAM from {current:.1f}% to ~{baseline:.1f}%",
                    goal_priority=GoalPriority.HIGH,
                    created_at=time.time(),
                    confidence=0.9
                )
            elif sigma > 4.0:
                # Moderate VRAM deviation (statistical anomaly)
                return Suggestion(
                    id=f"vram_anomaly_{int(time.time())}",
                    deviation_metric="vram_percent",
                    current_value=current,
                    baseline_value=baseline,
                    deviation_sigma=sigma,
                    problem=f"VRAM usage anomaly: {current:.1f}% (expected {baseline:.1f}%, Δ={sigma:.1f}σ)",
                    root_cause="Unusual memory allocation pattern detected",
                    suggested_action="Monitor VRAM trend and consider cache optimization",
                    action_type=ActionType.PERFORMANCE_SNAPSHOT,
                    optimization_type=OptimizationType.SAFE,
                    goal_name="Monitor VRAM anomaly",
                    goal_description=f"Track VRAM deviation from baseline (current Δ={sigma:.1f}σ)",
                    goal_priority=GoalPriority.MEDIUM,
                    created_at=time.time(),
                    confidence=0.75
                )

        return None

    def _analyze_disk_deviation(self, current: float, baseline: float, sigma: float) -> Optional[Suggestion]:
        """Analyze disk usage deviation"""

        if current > baseline and sigma > 2.5 and current > 85:
            return Suggestion(
                id=f"disk_full_{int(time.time())}",
                deviation_metric="disk_percent",
                current_value=current,
                baseline_value=baseline,
                deviation_sigma=sigma,
                problem=f"Disk space low: {current:.1f}% (expected {baseline:.1f}%)",
                root_cause="Log accumulation or temporary files",
                suggested_action="Clean old logs and temporary files",
                action_type=ActionType.LOG_ROTATION,
                optimization_type=OptimizationType.SAFE,
                goal_name="Free disk space",
                goal_description=f"Reduce disk usage from {current:.1f}% to <80%",
                goal_priority=GoalPriority.MEDIUM,
                created_at=time.time(),
                confidence=0.85
            )

        return None

    def get_suggestions_summary(self) -> Dict:
        """Get summary of recent suggestions"""
        recent = self.suggestions_history[-20:]

        return {
            "total_suggestions": len(self.suggestions_history),
            "recent_count": len(recent),
            "recent_suggestions": [
                {
                    "id": s.id,
                    "metric": s.deviation_metric,
                    "problem": s.problem,
                    "suggested_action": s.suggested_action,
                    "optimization_type": s.optimization_type.value,
                    "confidence": s.confidence,
                    "created_at": s.created_at
                }
                for s in recent
            ]
        }


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("🧠 Testing Cognitive Advisor")
    print("=" * 70)

    advisor = CognitiveAdvisor(mode="2")

    # Test CPU deviation
    cpu_deviation = {
        'metric': 'cpu_percent',
        'current': 95.0,
        'baseline_mean': 15.0,
        'deviation': 4.5,
        'severity': 'HIGH'
    }

    suggestion = advisor.analyze_deviation(cpu_deviation)

    if suggestion:
        print(f"\n✅ Suggestion Generated:")
        print(f"   Problem: {suggestion.problem}")
        print(f"   Root Cause: {suggestion.root_cause}")
        print(f"   Action: {suggestion.suggested_action}")
        print(f"   Type: {suggestion.optimization_type.value}")
        print(f"   Confidence: {suggestion.confidence:.0%}")
        print(f"   Goal: {suggestion.goal_name}")
        print(f"   Priority: {suggestion.goal_priority.value}")

    # Test VRAM deviation
    vram_deviation = {
        'metric': 'vram_percent',
        'current': 92.0,
        'baseline_mean': 25.0,
        'deviation': 5.2,
        'severity': 'HIGH'
    }

    suggestion2 = advisor.analyze_deviation(vram_deviation)

    if suggestion2:
        print(f"\n✅ Second Suggestion:")
        print(f"   Problem: {suggestion2.problem}")
        print(f"   Action: {suggestion2.suggested_action}")
        print(f"   Type: {suggestion2.optimization_type.value}")

    print("\n" + "=" * 70)
    print("✅ Cognitive Advisor Working!")
