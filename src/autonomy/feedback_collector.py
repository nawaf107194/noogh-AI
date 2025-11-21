"""
🔄 Feedback Collector - Minister Performance & Decision Tracking
Collects feedback from ministers about their decisions and outcomes

Features:
- Track minister decision outcomes (success/failure)
- Collect performance metrics from each minister
- Identify patterns in successful decisions
- Provide improvement recommendations
- Store feedback history for learning
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Outcome types for minister decisions"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class FeedbackEntry:
    """Individual feedback entry from a minister"""
    feedback_id: str
    minister_name: str
    task_type: str
    outcome: OutcomeType
    confidence_score: float  # 0.0-1.0
    execution_time_ms: float
    context: Dict[str, Any]
    learned_insights: List[str]
    error_details: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['outcome'] = self.outcome.value
        return data


@dataclass
class MinisterStats:
    """Statistics for a minister"""
    minister_name: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_confidence: float
    avg_execution_time_ms: float
    success_rate: float
    improvement_areas: List[str]
    strengths: List[str]
    recent_performance_trend: str  # "improving", "stable", "declining"


class FeedbackCollector:
    """
    Feedback collector for minister performance tracking

    Features:
    - Collect feedback from ministers
    - Analyze performance patterns
    - Identify improvement areas
    - Track performance trends
    - Generate recommendations
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize feedback collector

        Args:
            history_size: Maximum feedback entries to keep in memory
        """
        self.history_size = history_size

        # Feedback storage
        self.feedback_history: deque = deque(maxlen=history_size)
        self.minister_feedback: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

        # Statistics
        self.start_time = datetime.now()
        self.total_feedback_received = 0
        self.total_successes = 0
        self.total_failures = 0

        logger.info("📊 Feedback Collector initialized")

    def generate_feedback_id(self) -> str:
        """Generate unique feedback ID"""
        return f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def submit_feedback(
        self,
        minister_name: str,
        task_type: str,
        outcome: str,
        confidence_score: float,
        execution_time_ms: float,
        context: Optional[Dict[str, Any]] = None,
        learned_insights: Optional[List[str]] = None,
        error_details: Optional[str] = None
    ) -> FeedbackEntry:
        """
        Submit feedback from a minister

        Args:
            minister_name: Name of the minister
            task_type: Type of task performed
            outcome: Outcome type (success/partial_success/failure/error)
            confidence_score: Confidence in the decision (0.0-1.0)
            execution_time_ms: Execution time in milliseconds
            context: Task context and parameters
            learned_insights: Insights learned from the task
            error_details: Error details if failed

        Returns:
            FeedbackEntry object
        """
        try:
            # Create feedback entry
            feedback = FeedbackEntry(
                feedback_id=self.generate_feedback_id(),
                minister_name=minister_name,
                task_type=task_type,
                outcome=OutcomeType(outcome),
                confidence_score=confidence_score,
                execution_time_ms=execution_time_ms,
                context=context or {},
                learned_insights=learned_insights or [],
                error_details=error_details
            )

            # Store feedback
            self.feedback_history.append(feedback)
            self.minister_feedback[minister_name].append(feedback)

            # Update statistics
            self.total_feedback_received += 1

            if feedback.outcome == OutcomeType.SUCCESS:
                self.total_successes += 1
            elif feedback.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]:
                self.total_failures += 1

            logger.debug(
                f"📝 Feedback received from {minister_name}: {outcome} "
                f"(confidence: {confidence_score:.2f})"
            )

            return feedback

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise

    def get_minister_stats(self, minister_name: str) -> Optional[MinisterStats]:
        """
        Get statistics for a specific minister

        Args:
            minister_name: Name of the minister

        Returns:
            MinisterStats object or None if no feedback
        """
        feedback_list = list(self.minister_feedback.get(minister_name, []))

        if not feedback_list:
            return None

        # Calculate statistics
        total_tasks = len(feedback_list)
        successful_tasks = sum(
            1 for f in feedback_list if f.outcome == OutcomeType.SUCCESS
        )
        failed_tasks = sum(
            1 for f in feedback_list
            if f.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]
        )

        avg_confidence = sum(f.confidence_score for f in feedback_list) / total_tasks
        avg_execution_time = sum(f.execution_time_ms for f in feedback_list) / total_tasks
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        # Analyze improvement areas
        improvement_areas = self._identify_improvement_areas(feedback_list)
        strengths = self._identify_strengths(feedback_list)

        # Determine performance trend
        trend = self._calculate_performance_trend(feedback_list)

        return MinisterStats(
            minister_name=minister_name,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            avg_confidence=avg_confidence,
            avg_execution_time_ms=avg_execution_time,
            success_rate=success_rate,
            improvement_areas=improvement_areas,
            strengths=strengths,
            recent_performance_trend=trend
        )

    def _identify_improvement_areas(self, feedback_list: List[FeedbackEntry]) -> List[str]:
        """Identify areas where minister needs improvement"""
        areas = []

        # Check for low confidence tasks
        low_confidence_count = sum(1 for f in feedback_list if f.confidence_score < 0.7)
        if low_confidence_count > len(feedback_list) * 0.3:
            areas.append("Low confidence in decision-making")

        # Check for slow execution
        avg_time = sum(f.execution_time_ms for f in feedback_list) / len(feedback_list)
        if avg_time > 1000:  # More than 1 second
            areas.append("Slow execution time")

        # Check for errors
        error_count = sum(
            1 for f in feedback_list if f.outcome == OutcomeType.ERROR
        )
        if error_count > len(feedback_list) * 0.1:
            areas.append("High error rate")

        # Check task-specific failures
        task_failures = defaultdict(int)
        task_totals = defaultdict(int)

        for f in feedback_list:
            task_totals[f.task_type] += 1
            if f.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]:
                task_failures[f.task_type] += 1

        for task_type, failures in task_failures.items():
            total = task_totals[task_type]
            if failures / total > 0.3:  # More than 30% failure rate
                areas.append(f"Struggles with {task_type} tasks")

        return areas if areas else ["None identified"]

    def _identify_strengths(self, feedback_list: List[FeedbackEntry]) -> List[str]:
        """Identify minister's strengths"""
        strengths = []

        # Check for high success rate
        success_rate = sum(
            1 for f in feedback_list if f.outcome == OutcomeType.SUCCESS
        ) / len(feedback_list)

        if success_rate > 0.9:
            strengths.append("Very high success rate")

        # Check for high confidence
        avg_confidence = sum(f.confidence_score for f in feedback_list) / len(feedback_list)
        if avg_confidence > 0.85:
            strengths.append("High confidence in decisions")

        # Check for fast execution
        avg_time = sum(f.execution_time_ms for f in feedback_list) / len(feedback_list)
        if avg_time < 500:
            strengths.append("Fast execution")

        # Check task-specific excellence
        task_successes = defaultdict(int)
        task_totals = defaultdict(int)

        for f in feedback_list:
            task_totals[f.task_type] += 1
            if f.outcome == OutcomeType.SUCCESS:
                task_successes[f.task_type] += 1

        for task_type, successes in task_successes.items():
            total = task_totals[task_type]
            if total >= 5 and successes / total > 0.95:  # At least 5 tasks, >95% success
                strengths.append(f"Excellent at {task_type}")

        return strengths if strengths else ["Still learning"]

    def _calculate_performance_trend(self, feedback_list: List[FeedbackEntry]) -> str:
        """Calculate recent performance trend"""
        if len(feedback_list) < 10:
            return "insufficient_data"

        # Split into recent and older halves
        mid_point = len(feedback_list) // 2
        older_half = feedback_list[:mid_point]
        recent_half = feedback_list[mid_point:]

        # Calculate success rates
        older_success_rate = sum(
            1 for f in older_half if f.outcome == OutcomeType.SUCCESS
        ) / len(older_half)

        recent_success_rate = sum(
            1 for f in recent_half if f.outcome == OutcomeType.SUCCESS
        ) / len(recent_half)

        # Determine trend
        diff = recent_success_rate - older_success_rate

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"

    def get_all_minister_stats(self) -> List[MinisterStats]:
        """Get statistics for all ministers"""
        stats = []

        for minister_name in self.minister_feedback.keys():
            minister_stat = self.get_minister_stats(minister_name)
            if minister_stat:
                stats.append(minister_stat)

        # Sort by success rate
        stats.sort(key=lambda s: s.success_rate, reverse=True)

        return stats

    def get_recent_feedback(self, limit: int = 20) -> List[FeedbackEntry]:
        """Get recent feedback entries"""
        return list(self.feedback_history)[-limit:]

    def get_feedback_by_minister(
        self,
        minister_name: str,
        limit: int = 20
    ) -> List[FeedbackEntry]:
        """Get recent feedback for a specific minister"""
        return list(self.minister_feedback.get(minister_name, []))[-limit:]

    def get_insights(self) -> Dict[str, Any]:
        """
        Get overall insights from all feedback

        Returns:
            Dictionary with insights about system performance
        """
        all_stats = self.get_all_minister_stats()

        if not all_stats:
            return {
                "overall_assessment": "No feedback data available",
                "recommendations": []
            }

        # Find top performers
        top_performers = [
            s.minister_name for s in all_stats[:3] if s.success_rate > 0.8
        ]

        # Find ministers needing support
        needs_support = [
            s.minister_name for s in all_stats if s.success_rate < 0.6
        ]

        # Overall success rate
        overall_success_rate = (
            self.total_successes / max(self.total_feedback_received, 1)
        )

        # Generate recommendations
        recommendations = []

        if overall_success_rate < 0.7:
            recommendations.append("Overall system success rate is low - review minister configurations")

        if needs_support:
            recommendations.append(
                f"Ministers needing support: {', '.join(needs_support)}"
            )

        if len(all_stats) < 5:
            recommendations.append("Low minister participation - encourage more feedback")

        return {
            "overall_assessment": f"{overall_success_rate*100:.1f}% success rate",
            "total_feedback_received": self.total_feedback_received,
            "active_ministers": len(all_stats),
            "top_performers": top_performers,
            "needs_support": needs_support,
            "recommendations": recommendations if recommendations else ["System performing well"]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback collector statistics"""
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        return {
            "total_feedback_received": self.total_feedback_received,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": (
                self.total_successes / max(self.total_feedback_received, 1) * 100
            ),
            "active_ministers": len(self.minister_feedback),
            "uptime_hours": round(uptime_hours, 2)
        }


# Global feedback collector instance
_feedback_collector = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DI Container Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_feedback_collector() -> FeedbackCollector:
    """
    Get global feedback collector instance from DI container
    
    Returns:
        FeedbackCollector instance (singleton)
    """
    try:
        from src.core.di import Container
        collector = Container.resolve("feedback_collector")
        if collector is not None:
            return collector
    except ImportError:
        pass
    
    # Fallback to manual singleton for backward compatibility
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
