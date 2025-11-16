"""
🧠 Brain Adjuster - Self-Adjustment Engine
Uses feedback to automatically adjust Brain Hub parameters for improved performance

Features:
- Analyze feedback patterns
- Generate adjustment recommendations
- Apply safe adjustments automatically
- Track adjustment effectiveness
- Learn from adjustment outcomes
- Rollback ineffective adjustments
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomy.feedback_collector import get_feedback_collector, MinisterStats

logger = logging.getLogger(__name__)


class AdjustmentType(Enum):
    """Types of adjustments"""
    PARAMETER_TUNING = "parameter_tuning"
    RESOURCE_ALLOCATION = "resource_allocation"
    MINISTER_CONFIGURATION = "minister_configuration"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"


class AdjustmentStatus(Enum):
    """Adjustment status"""
    PENDING = "pending"
    APPLIED = "applied"
    TESTING = "testing"
    EFFECTIVE = "effective"
    INEFFECTIVE = "ineffective"
    ROLLED_BACK = "rolled_back"


@dataclass
class Adjustment:
    """Brain adjustment record"""
    adjustment_id: str
    adjustment_type: AdjustmentType
    target: str  # Minister name or system component
    description: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    reasoning: str
    status: AdjustmentStatus
    applied_at: Optional[str] = None
    effectiveness_score: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['adjustment_type'] = self.adjustment_type.value
        data['status'] = self.status.value
        return data


@dataclass
class AdjustmentRecommendation:
    """Adjustment recommendation"""
    recommendation_id: str
    priority: str  # "high", "medium", "low"
    adjustment_type: AdjustmentType
    target: str
    description: str
    expected_impact: str
    risk_level: str  # "low", "medium", "high"
    parameters: Dict[str, Any]
    reasoning: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['adjustment_type'] = self.adjustment_type.value
        return data


class BrainAdjuster:
    """
    Brain self-adjustment engine

    Features:
    - Analyze feedback patterns
    - Generate adjustment recommendations
    - Apply adjustments automatically (if safe)
    - Track adjustment effectiveness
    - Rollback ineffective adjustments
    - Learn from past adjustments
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        risk_tolerance: str = "low"
    ):
        """
        Initialize brain adjuster

        Args:
            auto_adjust: Automatically apply low-risk adjustments
            risk_tolerance: Risk tolerance level (low/medium/high)
        """
        self.auto_adjust = auto_adjust
        self.risk_tolerance = risk_tolerance

        self.feedback_collector = get_feedback_collector()

        # Adjustment storage
        self.adjustments: deque = deque(maxlen=500)
        self.recommendations: deque = deque(maxlen=100)
        self.active_adjustments: Dict[str, Adjustment] = {}

        # Statistics
        self.start_time = datetime.now()
        self.total_recommendations = 0
        self.total_adjustments_applied = 0
        self.total_effective_adjustments = 0
        self.total_rolled_back = 0

        logger.info("🧠 Brain Adjuster initialized")

    def generate_adjustment_id(self) -> str:
        """Generate unique adjustment ID"""
        return f"adj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def generate_recommendation_id(self) -> str:
        """Generate unique recommendation ID"""
        return f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def analyze_and_recommend(self) -> List[AdjustmentRecommendation]:
        """
        Analyze feedback and generate adjustment recommendations

        Returns:
            List of recommendations
        """
        logger.info("🔍 Analyzing feedback patterns...")

        recommendations = []

        # Get all minister stats
        all_stats = self.feedback_collector.get_all_minister_stats()

        # Analyze each minister
        for stats in all_stats:
            minister_recommendations = self._analyze_minister_performance(stats)
            recommendations.extend(minister_recommendations)

        # Get overall insights
        insights = self.feedback_collector.get_insights()

        # Generate system-level recommendations
        system_recommendations = self._analyze_system_performance(insights)
        recommendations.extend(system_recommendations)

        # Store recommendations
        for rec in recommendations:
            self.recommendations.append(rec)
            self.total_recommendations += 1

        logger.info(f"✅ Generated {len(recommendations)} recommendations")

        return recommendations

    def _analyze_minister_performance(
        self,
        stats: MinisterStats
    ) -> List[AdjustmentRecommendation]:
        """Analyze minister performance and generate recommendations"""
        recommendations = []

        # Low success rate
        if stats.success_rate < 0.7:
            rec = AdjustmentRecommendation(
                recommendation_id=self.generate_recommendation_id(),
                priority="high",
                adjustment_type=AdjustmentType.MINISTER_CONFIGURATION,
                target=stats.minister_name,
                description=f"Improve {stats.minister_name} success rate (current: {stats.success_rate*100:.1f}%)",
                expected_impact=f"Increase success rate by 10-15%",
                risk_level="low",
                parameters={
                    "increase_validation": True,
                    "enable_retry_logic": True,
                    "max_retries": 3
                },
                reasoning=f"Success rate of {stats.success_rate*100:.1f}% is below target 70%"
            )
            recommendations.append(rec)

        # Low confidence
        if stats.avg_confidence < 0.7:
            rec = AdjustmentRecommendation(
                recommendation_id=self.generate_recommendation_id(),
                priority="medium",
                adjustment_type=AdjustmentType.CONFIDENCE_THRESHOLD,
                target=stats.minister_name,
                description=f"Adjust confidence threshold for {stats.minister_name}",
                expected_impact="Reduce low-confidence decisions",
                risk_level="low",
                parameters={
                    "min_confidence_threshold": 0.75,
                    "require_validation_below": 0.8
                },
                reasoning=f"Average confidence {stats.avg_confidence:.2f} is low"
            )
            recommendations.append(rec)

        # Slow execution
        if stats.avg_execution_time_ms > 1500:
            rec = AdjustmentRecommendation(
                recommendation_id=self.generate_recommendation_id(),
                priority="medium",
                adjustment_type=AdjustmentType.TIMEOUT_ADJUSTMENT,
                target=stats.minister_name,
                description=f"Optimize {stats.minister_name} execution time",
                expected_impact="Reduce average execution time by 20-30%",
                risk_level="medium",
                parameters={
                    "timeout_ms": int(stats.avg_execution_time_ms * 0.8),
                    "enable_caching": True,
                    "parallel_processing": True
                },
                reasoning=f"Average execution time {stats.avg_execution_time_ms:.0f}ms is slow"
            )
            recommendations.append(rec)

        # Declining performance
        if stats.recent_performance_trend == "declining":
            rec = AdjustmentRecommendation(
                recommendation_id=self.generate_recommendation_id(),
                priority="high",
                adjustment_type=AdjustmentType.PARAMETER_TUNING,
                target=stats.minister_name,
                description=f"Address declining performance in {stats.minister_name}",
                expected_impact="Stabilize and reverse performance decline",
                risk_level="medium",
                parameters={
                    "reset_to_baseline": True,
                    "increase_monitoring": True,
                    "enable_diagnostic_mode": True
                },
                reasoning="Performance trend is declining"
            )
            recommendations.append(rec)

        return recommendations

    def _analyze_system_performance(
        self,
        insights: Dict[str, Any]
    ) -> List[AdjustmentRecommendation]:
        """Analyze overall system performance"""
        recommendations = []

        # Low overall success rate
        if self.feedback_collector.total_successes / max(self.feedback_collector.total_feedback_received, 1) < 0.75:
            rec = AdjustmentRecommendation(
                recommendation_id=self.generate_recommendation_id(),
                priority="high",
                adjustment_type=AdjustmentType.RESOURCE_ALLOCATION,
                target="system",
                description="Improve overall system success rate",
                expected_impact="Increase system-wide success rate by 10%",
                risk_level="medium",
                parameters={
                    "increase_worker_threads": True,
                    "enable_request_queuing": True,
                    "prioritize_high_confidence": True
                },
                reasoning=f"Overall success rate is below 75%"
            )
            recommendations.append(rec)

        # Ministers needing support
        if insights.get('needs_support'):
            for minister in insights['needs_support']:
                rec = AdjustmentRecommendation(
                    recommendation_id=self.generate_recommendation_id(),
                    priority="high",
                    adjustment_type=AdjustmentType.MINISTER_CONFIGURATION,
                    target=minister,
                    description=f"Provide additional support to {minister}",
                    expected_impact="Bring minister up to acceptable performance",
                    risk_level="low",
                    parameters={
                        "enable_mentor_mode": True,
                        "increase_guidance": True,
                        "validate_all_decisions": True
                    },
                    reasoning=f"{minister} identified as needing support"
                )
                recommendations.append(rec)

        return recommendations

    def apply_adjustment(
        self,
        recommendation: AdjustmentRecommendation,
        force: bool = False
    ) -> Optional[Adjustment]:
        """
        Apply an adjustment based on recommendation

        Args:
            recommendation: Recommendation to apply
            force: Force application even if auto-adjust is disabled

        Returns:
            Adjustment object if applied, None otherwise
        """
        # Check if should auto-apply
        should_apply = force or (
            self.auto_adjust and
            recommendation.risk_level == "low"
        )

        if not should_apply:
            logger.info(f"⏸️  Adjustment requires manual approval: {recommendation.description}")
            return None

        logger.info(f"⚙️  Applying adjustment: {recommendation.description}")

        try:
            # Create adjustment record
            adjustment = Adjustment(
                adjustment_id=self.generate_adjustment_id(),
                adjustment_type=recommendation.adjustment_type,
                target=recommendation.target,
                description=recommendation.description,
                parameters_before=self._get_current_parameters(recommendation.target),
                parameters_after=recommendation.parameters,
                reasoning=recommendation.reasoning,
                status=AdjustmentStatus.APPLIED,
                applied_at=datetime.now().isoformat()
            )

            # Apply the adjustment (in real system, this would make actual changes)
            # For now, we simulate the application
            self._simulate_apply_adjustment(adjustment)

            # Store adjustment
            self.adjustments.append(adjustment)
            self.active_adjustments[adjustment.adjustment_id] = adjustment
            self.total_adjustments_applied += 1

            logger.info(f"✅ Adjustment applied: {adjustment.adjustment_id}")

            return adjustment

        except Exception as e:
            logger.error(f"Failed to apply adjustment: {e}")
            return None

    def _get_current_parameters(self, target: str) -> Dict[str, Any]:
        """Get current parameters for a target"""
        # In real system, this would fetch actual parameters from Brain Hub
        # For now, return simulated baseline
        return {
            "confidence_threshold": 0.7,
            "timeout_ms": 1000,
            "retry_enabled": False,
            "max_retries": 1
        }

    def _simulate_apply_adjustment(self, adjustment: Adjustment):
        """Simulate applying an adjustment (placeholder for real implementation)"""
        # In real system, this would:
        # 1. Update Brain Hub configuration
        # 2. Notify affected ministers
        # 3. Apply parameter changes
        # 4. Restart services if needed

        logger.debug(f"Simulated adjustment application for {adjustment.target}")

    def evaluate_adjustment_effectiveness(
        self,
        adjustment_id: str,
        test_duration_hours: float = 1.0
    ) -> Tuple[bool, float, str]:
        """
        Evaluate if an adjustment was effective

        Args:
            adjustment_id: ID of adjustment to evaluate
            test_duration_hours: How long to test (default 1 hour)

        Returns:
            Tuple of (is_effective, effectiveness_score, reason)
        """
        adjustment = self.active_adjustments.get(adjustment_id)

        if not adjustment:
            return (False, 0.0, "Adjustment not found")

        # Get feedback before and after adjustment
        applied_time = datetime.fromisoformat(adjustment.applied_at)
        cutoff_time = applied_time + timedelta(hours=test_duration_hours)

        if datetime.now() < cutoff_time:
            return (False, 0.0, "Test duration not elapsed yet")

        # Get minister stats
        minister_stats = self.feedback_collector.get_minister_stats(adjustment.target)

        if not minister_stats:
            return (False, 0.0, "No feedback data available")

        # Calculate effectiveness based on improvement
        # In real system, would compare before/after metrics
        # For now, use success rate as proxy

        if minister_stats.success_rate > 0.75:
            effectiveness_score = minister_stats.success_rate
            is_effective = True
            reason = f"Success rate improved to {minister_stats.success_rate*100:.1f}%"
        else:
            effectiveness_score = minister_stats.success_rate
            is_effective = False
            reason = f"Success rate still low at {minister_stats.success_rate*100:.1f}%"

        # Update adjustment
        adjustment.status = (
            AdjustmentStatus.EFFECTIVE if is_effective else AdjustmentStatus.INEFFECTIVE
        )
        adjustment.effectiveness_score = effectiveness_score

        if is_effective:
            self.total_effective_adjustments += 1

        logger.info(
            f"📊 Adjustment {adjustment_id} evaluation: "
            f"{'EFFECTIVE' if is_effective else 'INEFFECTIVE'} ({effectiveness_score:.2f})"
        )

        return (is_effective, effectiveness_score, reason)

    def rollback_adjustment(self, adjustment_id: str) -> bool:
        """
        Rollback an adjustment

        Args:
            adjustment_id: ID of adjustment to rollback

        Returns:
            True if successful, False otherwise
        """
        adjustment = self.active_adjustments.get(adjustment_id)

        if not adjustment:
            logger.warning(f"Adjustment not found for rollback: {adjustment_id}")
            return False

        logger.warning(f"⏪ Rolling back adjustment: {adjustment.description}")

        try:
            # Restore previous parameters (simulated)
            self._simulate_apply_adjustment(Adjustment(
                adjustment_id=self.generate_adjustment_id(),
                adjustment_type=adjustment.adjustment_type,
                target=adjustment.target,
                description=f"Rollback: {adjustment.description}",
                parameters_before=adjustment.parameters_after,
                parameters_after=adjustment.parameters_before,
                reasoning="Rolling back ineffective adjustment",
                status=AdjustmentStatus.ROLLED_BACK
            ))

            # Update status
            adjustment.status = AdjustmentStatus.ROLLED_BACK
            del self.active_adjustments[adjustment_id]

            self.total_rolled_back += 1

            logger.info(f"✅ Adjustment rolled back: {adjustment_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to rollback adjustment: {e}")
            return False

    def get_recent_adjustments(self, limit: int = 10) -> List[Adjustment]:
        """Get recent adjustments"""
        return list(self.adjustments)[-limit:]

    def get_recent_recommendations(self, limit: int = 10) -> List[AdjustmentRecommendation]:
        """Get recent recommendations"""
        return list(self.recommendations)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get brain adjuster statistics"""
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        return {
            "total_recommendations": self.total_recommendations,
            "total_adjustments_applied": self.total_adjustments_applied,
            "total_effective": self.total_effective_adjustments,
            "total_rolled_back": self.total_rolled_back,
            "effectiveness_rate": (
                self.total_effective_adjustments / max(self.total_adjustments_applied, 1) * 100
            ),
            "active_adjustments": len(self.active_adjustments),
            "auto_adjust_enabled": self.auto_adjust,
            "risk_tolerance": self.risk_tolerance,
            "uptime_hours": round(uptime_hours, 2)
        }


# Global brain adjuster instance
_brain_adjuster = None

def get_brain_adjuster() -> BrainAdjuster:
    """Get global brain adjuster instance"""
    global _brain_adjuster
    if _brain_adjuster is None:
        _brain_adjuster = BrainAdjuster()
    return _brain_adjuster
