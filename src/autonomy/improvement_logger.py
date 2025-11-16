"""
📈 Improvement Logger - Track System Evolution Over Time
Records all improvements, adjustments, and performance metrics

Features:
- Log all adjustments and their effectiveness
- Track performance metrics over time
- Calculate improvement rates
- Generate improvement timeline
- Store historical data persistently
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import json
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class ImprovementEntry:
    """Single improvement entry"""
    entry_id: str
    timestamp: str
    improvement_type: str  # "adjustment", "training", "recovery", "optimization"
    target: str
    description: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement_percentage: float
    is_successful: bool
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DailyMetrics:
    """Daily aggregated metrics"""
    date: str
    cognition_score: float
    system_success_rate: float
    avg_response_time_ms: float
    active_ministers: int
    total_adjustments: int
    successful_adjustments: int
    total_training_jobs: int
    total_alerts: int


class ImprovementLogger:
    """
    Improvement Logger - Track system evolution

    Features:
    - Log all improvements
    - Track metrics over time
    - Calculate improvement rates
    - Generate statistics
    - Persistent storage
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize improvement logger

        Args:
            data_dir: Directory for storing improvement data
        """
        self.data_dir = data_dir or (PROJECT_ROOT / "data" / "improvements")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.recent_improvements: deque = deque(maxlen=1000)
        self.daily_metrics: Dict[str, DailyMetrics] = {}

        # Load existing data
        self._load_data()

        # Statistics
        self.start_time = datetime.now()
        self.total_improvements_logged = 0

        logger.info("📈 Improvement Logger initialized")

    def _load_data(self):
        """Load historical data from disk"""
        try:
            # Load recent improvements
            improvements_file = self.data_dir / "recent_improvements.json"
            if improvements_file.exists():
                with open(improvements_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        entry = ImprovementEntry(**item)
                        self.recent_improvements.append(entry)

            # Load daily metrics
            metrics_file = self.data_dir / "daily_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    for date, metrics in data.items():
                        self.daily_metrics[date] = DailyMetrics(**metrics)

            logger.info(
                f"📂 Loaded {len(self.recent_improvements)} improvements "
                f"and {len(self.daily_metrics)} daily metrics"
            )

        except Exception as e:
            logger.error(f"Failed to load improvement data: {e}")

    def _save_data(self):
        """Save data to disk"""
        try:
            # Save recent improvements
            improvements_file = self.data_dir / "recent_improvements.json"
            with open(improvements_file, 'w') as f:
                json.dump(
                    [entry.to_dict() for entry in self.recent_improvements],
                    f,
                    indent=2
                )

            # Save daily metrics
            metrics_file = self.data_dir / "daily_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(
                    {
                        date: asdict(metrics)
                        for date, metrics in self.daily_metrics.items()
                    },
                    f,
                    indent=2
                )

            logger.debug("💾 Saved improvement data to disk")

        except Exception as e:
            logger.error(f"Failed to save improvement data: {e}")

    def generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        return f"imp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def log_improvement(
        self,
        improvement_type: str,
        target: str,
        description: str,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        details: Optional[Dict[str, Any]] = None
    ) -> ImprovementEntry:
        """
        Log an improvement

        Args:
            improvement_type: Type of improvement
            target: Target component
            description: Description of improvement
            metrics_before: Metrics before improvement
            metrics_after: Metrics after improvement
            details: Additional details

        Returns:
            ImprovementEntry object
        """
        # Calculate improvement percentage
        # Use cognition_score or success_rate as primary metric
        primary_metric_before = metrics_before.get(
            'cognition_score',
            metrics_before.get('success_rate', 0)
        )
        primary_metric_after = metrics_after.get(
            'cognition_score',
            metrics_after.get('success_rate', 0)
        )

        if primary_metric_before > 0:
            improvement_pct = (
                (primary_metric_after - primary_metric_before) /
                primary_metric_before * 100
            )
        else:
            improvement_pct = 0.0

        # Determine success
        is_successful = improvement_pct > 0

        # Create entry
        entry = ImprovementEntry(
            entry_id=self.generate_entry_id(),
            timestamp=datetime.now().isoformat(),
            improvement_type=improvement_type,
            target=target,
            description=description,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement_percentage=improvement_pct,
            is_successful=is_successful,
            details=details
        )

        # Store entry
        self.recent_improvements.append(entry)
        self.total_improvements_logged += 1

        # Save to disk
        self._save_data()

        logger.info(
            f"📝 Logged improvement: {description} "
            f"({improvement_pct:+.1f}% change)"
        )

        return entry

    def log_daily_metrics(
        self,
        cognition_score: float,
        system_success_rate: float,
        avg_response_time_ms: float,
        active_ministers: int,
        total_adjustments: int,
        successful_adjustments: int,
        total_training_jobs: int,
        total_alerts: int
    ):
        """Log daily aggregated metrics"""
        date = datetime.now().strftime('%Y-%m-%d')

        metrics = DailyMetrics(
            date=date,
            cognition_score=cognition_score,
            system_success_rate=system_success_rate,
            avg_response_time_ms=avg_response_time_ms,
            active_ministers=active_ministers,
            total_adjustments=total_adjustments,
            successful_adjustments=successful_adjustments,
            total_training_jobs=total_training_jobs,
            total_alerts=total_alerts
        )

        self.daily_metrics[date] = metrics

        # Save to disk
        self._save_data()

        logger.info(f"📊 Logged daily metrics for {date}")

    def get_recent_improvements(
        self,
        limit: int = 20,
        improvement_type: Optional[str] = None
    ) -> List[ImprovementEntry]:
        """Get recent improvements"""
        improvements = list(self.recent_improvements)

        # Filter by type if specified
        if improvement_type:
            improvements = [
                i for i in improvements
                if i.improvement_type == improvement_type
            ]

        # Sort by timestamp (most recent first)
        improvements.sort(key=lambda i: i.timestamp, reverse=True)

        return improvements[:limit]

    def get_improvement_timeline(
        self,
        days: int = 7
    ) -> List[ImprovementEntry]:
        """Get improvement timeline for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        timeline = [
            entry for entry in self.recent_improvements
            if entry.timestamp >= cutoff_str
        ]

        # Sort by timestamp (chronological)
        timeline.sort(key=lambda e: e.timestamp)

        return timeline

    def get_daily_metrics_range(
        self,
        days: int = 7
    ) -> List[DailyMetrics]:
        """Get daily metrics for specified days"""
        result = []

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            if date in self.daily_metrics:
                result.append(self.daily_metrics[date])

        # Sort by date (most recent first)
        result.sort(key=lambda m: m.date, reverse=True)

        return result

    def calculate_improvement_rate(
        self,
        days: int = 7,
        metric: str = "cognition_score"
    ) -> float:
        """
        Calculate average daily improvement rate

        Args:
            days: Number of days to analyze
            metric: Metric to calculate rate for

        Returns:
            Average daily improvement rate (percentage)
        """
        metrics_list = self.get_daily_metrics_range(days=days)

        if len(metrics_list) < 2:
            return 0.0

        # Get oldest and newest values
        oldest = metrics_list[-1]
        newest = metrics_list[0]

        oldest_value = getattr(oldest, metric, 0)
        newest_value = getattr(newest, metric, 0)

        if oldest_value > 0:
            total_change_pct = (
                (newest_value - oldest_value) / oldest_value * 100
            )
            daily_rate = total_change_pct / len(metrics_list)
        else:
            daily_rate = 0.0

        return daily_rate

    def get_statistics(self) -> Dict[str, Any]:
        """Get improvement logger statistics"""
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        # Count successful improvements
        successful = sum(
            1 for entry in self.recent_improvements
            if entry.is_successful
        )

        total = len(self.recent_improvements)

        # Calculate average improvement
        avg_improvement = (
            sum(entry.improvement_percentage for entry in self.recent_improvements) /
            max(total, 1)
        )

        # Get improvement by type
        by_type = {}
        for entry in self.recent_improvements:
            type_name = entry.improvement_type
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1

        return {
            "total_improvements_logged": self.total_improvements_logged,
            "total_in_memory": total,
            "successful_improvements": successful,
            "failed_improvements": total - successful,
            "success_rate": successful / max(total, 1) * 100,
            "avg_improvement_percentage": avg_improvement,
            "improvements_by_type": by_type,
            "daily_metrics_count": len(self.daily_metrics),
            "uptime_hours": round(uptime_hours, 2)
        }

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive improvement summary"""
        stats = self.get_statistics()
        recent = self.get_recent_improvements(limit=10)
        timeline = self.get_improvement_timeline(days=days)
        metrics = self.get_daily_metrics_range(days=days)

        # Calculate improvement rates
        cognition_rate = self.calculate_improvement_rate(
            days=days,
            metric="cognition_score"
        )
        success_rate = self.calculate_improvement_rate(
            days=days,
            metric="system_success_rate"
        )

        return {
            "statistics": stats,
            "recent_improvements": [e.to_dict() for e in recent],
            "timeline_count": len(timeline),
            "daily_metrics_count": len(metrics),
            "improvement_rates": {
                "cognition_score_daily": round(cognition_rate, 3),
                "success_rate_daily": round(success_rate, 3),
            },
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }


# Global improvement logger instance
_improvement_logger = None

def get_improvement_logger() -> ImprovementLogger:
    """Get global improvement logger instance"""
    global _improvement_logger
    if _improvement_logger is None:
        _improvement_logger = ImprovementLogger()
    return _improvement_logger
