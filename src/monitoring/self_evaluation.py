#!/usr/bin/env python3
"""
🧠 Self-Evaluation System
نظام التقييم الذاتي

يسجل ويحلل أداء جلسات التدريب لتحسين القرارات المستقبلية.

Features:
- Session logging (VRAM, CPU, GPU, duration, success)
- Performance analysis
- Decision recommendations
- Trend detection
- Weekly summaries
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session Data Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainingSession:
    """Single training session data"""

    # Basic info
    session_id: str
    timestamp: str
    model_name: str

    # Resource usage
    vram_peak_gb: float
    vram_before_percent: float
    vram_after_percent: float
    cpu_peak_percent: float
    gpu_temp_peak_c: int

    # Training details
    device_used: str  # "cpu" or "gpu"
    duration_seconds: float
    epochs: int
    success: bool

    # Decision data
    estimated_vram_gb: float
    actual_vram_gb: float
    vram_estimation_error: float  # (actual - estimated) / estimated * 100

    # Load balancer decision
    lb_recommendation: str  # "cpu" or "gpu"
    lb_confidence: float
    lb_was_correct: bool

    # Ministers status
    ministers_paused: bool
    ministers_count: int

    # Errors (if any)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-Evaluation System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SelfEvaluationSystem:
    """
    نظام التقييم الذاتي - Self-Evaluation System

    يسجل جلسات التدريب، يحللها، ويقدم توصيات لتحسين الأداء.
    """

    def __init__(
        self,
        storage_dir: str = "/home/noogh/projects/noogh_unified_system/logs/evaluation",
        max_sessions_in_memory: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize self-evaluation system.

        Args:
            storage_dir: Directory to store session logs
            max_sessions_in_memory: Max sessions to keep in memory
            verbose: Enable verbose logging
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_sessions_in_memory = max_sessions_in_memory
        self.verbose = verbose

        # Session storage
        self.sessions_file = self.storage_dir / "sessions.jsonl"
        self.summary_file = self.storage_dir / "summary.json"

        # In-memory cache
        self.recent_sessions: List[TrainingSession] = []
        self._load_recent_sessions()

        if self.verbose:
            logger.info(f"📊 Self-Evaluation System initialized")
            logger.info(f"   Storage: {self.storage_dir}")
            logger.info(f"   Recent sessions: {len(self.recent_sessions)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Session Logging
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_session(self, session: TrainingSession) -> bool:
        """
        Log a training session.

        Args:
            session: TrainingSession object

        Returns:
            True if logged successfully
        """
        try:
            # Add to in-memory cache
            self.recent_sessions.append(session)

            # Keep only recent sessions in memory
            if len(self.recent_sessions) > self.max_sessions_in_memory:
                self.recent_sessions = self.recent_sessions[-self.max_sessions_in_memory:]

            # Append to JSONL file
            with open(self.sessions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(session.to_dict(), ensure_ascii=False) + '\n')

            if self.verbose:
                status = "✅ Success" if session.success else "❌ Failed"
                logger.info(f"📝 Session logged: {session.model_name} - {status}")
                logger.info(f"   VRAM: {session.vram_peak_gb:.2f}GB (est: {session.estimated_vram_gb:.2f}GB)")
                logger.info(f"   Duration: {session.duration_seconds:.1f}s")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to log session: {e}")
            return False

    def _load_recent_sessions(self):
        """Load recent sessions from disk"""
        if not self.sessions_file.exists():
            return

        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Load last N sessions
            recent_lines = lines[-self.max_sessions_in_memory:]

            for line in recent_lines:
                try:
                    data = json.loads(line.strip())
                    session = TrainingSession(**data)
                    self.recent_sessions.append(session)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse session: {e}")

            if self.verbose:
                logger.info(f"📂 Loaded {len(self.recent_sessions)} recent sessions")

        except Exception as e:
            logger.error(f"❌ Failed to load sessions: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance over last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Analysis results
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        # Filter sessions
        recent = [
            s for s in self.recent_sessions
            if datetime.fromisoformat(s.timestamp) > cutoff_time
        ]

        if not recent:
            return {
                "status": "no_data",
                "message": f"No sessions in last {days} days",
                "days_analyzed": days
            }

        # Calculate metrics
        total_sessions = len(recent)
        successful_sessions = [s for s in recent if s.success]
        failed_sessions = [s for s in recent if not s.success]

        success_rate = len(successful_sessions) / total_sessions * 100 if total_sessions > 0 else 0

        # VRAM analysis
        vram_peaks = [s.vram_peak_gb for s in successful_sessions]
        vram_errors = [s.vram_estimation_error for s in successful_sessions]

        # Load balancer analysis
        lb_correct = [s for s in successful_sessions if s.lb_was_correct]
        lb_accuracy = len(lb_correct) / len(successful_sessions) * 100 if successful_sessions else 0

        # Device usage
        gpu_sessions = [s for s in successful_sessions if s.device_used == "gpu"]
        cpu_sessions = [s for s in successful_sessions if s.device_used == "cpu"]

        # Duration analysis
        durations = [s.duration_seconds for s in successful_sessions]

        analysis = {
            "status": "success",
            "period": {
                "days": days,
                "from": cutoff_time.isoformat(),
                "to": datetime.now().isoformat()
            },
            "sessions": {
                "total": total_sessions,
                "successful": len(successful_sessions),
                "failed": len(failed_sessions),
                "success_rate_percent": round(success_rate, 2)
            },
            "vram": {
                "peak_avg_gb": round(mean(vram_peaks), 2) if vram_peaks else 0,
                "peak_median_gb": round(median(vram_peaks), 2) if vram_peaks else 0,
                "peak_max_gb": round(max(vram_peaks), 2) if vram_peaks else 0,
                "estimation_error_avg_percent": round(mean(vram_errors), 2) if vram_errors else 0,
                "estimation_error_stdev": round(stdev(vram_errors), 2) if len(vram_errors) > 1 else 0
            },
            "load_balancer": {
                "accuracy_percent": round(lb_accuracy, 2),
                "correct_decisions": len(lb_correct),
                "total_decisions": len(successful_sessions)
            },
            "device_usage": {
                "gpu_sessions": len(gpu_sessions),
                "cpu_sessions": len(cpu_sessions),
                "gpu_preference_percent": round(len(gpu_sessions) / len(successful_sessions) * 100, 2) if successful_sessions else 0
            },
            "duration": {
                "avg_seconds": round(mean(durations), 2) if durations else 0,
                "median_seconds": round(median(durations), 2) if durations else 0,
                "total_hours": round(sum(durations) / 3600, 2) if durations else 0
            }
        }

        return analysis

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations based on analysis.

        Returns:
            List of recommendations
        """
        analysis = self.analyze_performance(days=7)

        if analysis["status"] != "success":
            return []

        recommendations = []

        # Check success rate
        if analysis["sessions"]["success_rate_percent"] < 80:
            recommendations.append({
                "type": "warning",
                "category": "success_rate",
                "message": f"Success rate is low: {analysis['sessions']['success_rate_percent']:.1f}%",
                "suggestion": "Review failed sessions and check for common patterns"
            })

        # Check VRAM estimation
        vram_error = abs(analysis["vram"]["estimation_error_avg_percent"])
        if vram_error > 20:
            recommendations.append({
                "type": "warning",
                "category": "vram_estimation",
                "message": f"VRAM estimation error is high: {vram_error:.1f}%",
                "suggestion": "Improve VRAM estimation model or use more conservative estimates"
            })

        # Check load balancer accuracy
        if analysis["load_balancer"]["accuracy_percent"] < 70:
            recommendations.append({
                "type": "warning",
                "category": "load_balancer",
                "message": f"Load balancer accuracy is low: {analysis['load_balancer']['accuracy_percent']:.1f}%",
                "suggestion": "Review load balancer decision logic and thresholds"
            })

        # Check GPU utilization
        gpu_pref = analysis["device_usage"]["gpu_preference_percent"]
        if gpu_pref < 30:
            recommendations.append({
                "type": "info",
                "category": "gpu_usage",
                "message": f"Low GPU utilization: {gpu_pref:.1f}%",
                "suggestion": "Consider lowering GPU allocation thresholds to use GPU more often"
            })
        elif gpu_pref > 90:
            recommendations.append({
                "type": "info",
                "category": "gpu_usage",
                "message": f"Very high GPU utilization: {gpu_pref:.1f}%",
                "suggestion": "Consider if CPU could handle some workloads to free GPU"
            })

        # Positive feedback
        if analysis["sessions"]["success_rate_percent"] >= 95:
            recommendations.append({
                "type": "success",
                "category": "performance",
                "message": f"Excellent success rate: {analysis['sessions']['success_rate_percent']:.1f}%",
                "suggestion": "System is performing well, maintain current configuration"
            })

        return recommendations

    def get_insights(self) -> Dict[str, Any]:
        """
        Get high-level insights.

        Returns:
            Insights dictionary
        """
        analysis = self.analyze_performance(days=7)
        recommendations = self.get_recommendations()

        if analysis["status"] != "success":
            return {
                "status": "no_data",
                "message": "Not enough data for insights"
            }

        # Determine overall health
        success_rate = analysis["sessions"]["success_rate_percent"]
        lb_accuracy = analysis["load_balancer"]["accuracy_percent"]

        if success_rate >= 95 and lb_accuracy >= 80:
            health = "excellent"
            health_emoji = "🟢"
        elif success_rate >= 85 and lb_accuracy >= 70:
            health = "good"
            health_emoji = "🟡"
        elif success_rate >= 70:
            health = "fair"
            health_emoji = "🟠"
        else:
            health = "poor"
            health_emoji = "🔴"

        insights = {
            "status": "success",
            "overall_health": health,
            "health_emoji": health_emoji,
            "summary": {
                "total_sessions": analysis["sessions"]["total"],
                "success_rate": f"{success_rate:.1f}%",
                "lb_accuracy": f"{lb_accuracy:.1f}%",
                "avg_duration": f"{analysis['duration']['avg_seconds']:.1f}s",
                "total_training_hours": f"{analysis['duration']['total_hours']:.2f}h"
            },
            "recommendations_count": len(recommendations),
            "recommendations": recommendations,
            "analysis": analysis
        }

        return insights

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Weekly Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate_weekly_summary(self) -> Dict[str, Any]:
        """
        Generate weekly summary report.

        Returns:
            Weekly summary
        """
        analysis = self.analyze_performance(days=7)
        insights = self.get_insights()

        if analysis["status"] != "success":
            return {
                "status": "no_data",
                "message": "Not enough data for weekly summary"
            }

        # Get top models
        recent_7_days = [
            s for s in self.recent_sessions
            if datetime.fromisoformat(s.timestamp) > datetime.now() - timedelta(days=7)
        ]

        model_counts = {}
        for session in recent_7_days:
            if session.success:
                model_counts[session.model_name] = model_counts.get(session.model_name, 0) + 1

        top_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        summary = {
            "status": "success",
            "week": {
                "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d")
            },
            "overall_health": insights["overall_health"],
            "health_emoji": insights["health_emoji"],
            "metrics": analysis,
            "top_models": [
                {"name": name, "sessions": count}
                for name, count in top_models
            ],
            "recommendations": insights["recommendations"],
            "generated_at": datetime.now().isoformat()
        }

        # Save to file
        try:
            summary_file = self.storage_dir / f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            if self.verbose:
                logger.info(f"📊 Weekly summary saved: {summary_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save weekly summary: {e}")

        return summary

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Statistics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            Statistics dictionary
        """
        total_sessions = len(self.recent_sessions)

        if total_sessions == 0:
            return {
                "status": "no_data",
                "total_sessions": 0
            }

        successful = [s for s in self.recent_sessions if s.success]

        return {
            "status": "success",
            "total_sessions": total_sessions,
            "successful_sessions": len(successful),
            "failed_sessions": total_sessions - len(successful),
            "success_rate_percent": round(len(successful) / total_sessions * 100, 2),
            "oldest_session": self.recent_sessions[0].timestamp if self.recent_sessions else None,
            "newest_session": self.recent_sessions[-1].timestamp if self.recent_sessions else None,
            "storage_location": str(self.storage_dir)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Instance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create global instance (singleton pattern)
_evaluation_system = None

def get_evaluation_system(verbose: bool = False) -> SelfEvaluationSystem:
    """
    Get global evaluation system instance.

    Args:
        verbose: Enable verbose logging

    Returns:
        SelfEvaluationSystem instance
    """
    global _evaluation_system

    if _evaluation_system is None:
        _evaluation_system = SelfEvaluationSystem(verbose=verbose)

    return _evaluation_system


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_session_from_training(
    model_name: str,
    device_used: str,
    duration_seconds: float,
    epochs: int,
    success: bool,
    resources_before: Dict[str, Any],
    resources_after: Dict[str, Any],
    estimated_vram_gb: float,
    lb_recommendation: str,
    lb_confidence: float,
    ministers_paused: bool = True,
    ministers_count: int = 0,
    error_message: Optional[str] = None
) -> TrainingSession:
    """
    Create TrainingSession from training data.

    Helper function to make session creation easier.
    """
    import uuid

    # Calculate actual VRAM used
    vram_before = resources_before.get("gpu_memory_percent", 0)
    vram_after = resources_after.get("gpu_memory_percent", 0)
    vram_total = resources_before.get("gpu_memory_total", 24.0)

    vram_peak_gb = vram_after / 100 * vram_total if vram_after > vram_before else vram_before / 100 * vram_total
    actual_vram_gb = vram_peak_gb

    # Calculate estimation error
    if estimated_vram_gb > 0:
        vram_error = (actual_vram_gb - estimated_vram_gb) / estimated_vram_gb * 100
    else:
        vram_error = 0.0

    # Check if LB was correct
    lb_was_correct = (lb_recommendation == device_used)

    session = TrainingSession(
        session_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        vram_peak_gb=round(vram_peak_gb, 2),
        vram_before_percent=round(vram_before, 2),
        vram_after_percent=round(vram_after, 2),
        cpu_peak_percent=round(resources_after.get("cpu_percent", 0), 2),
        gpu_temp_peak_c=resources_after.get("gpu_temperature", 0),
        device_used=device_used,
        duration_seconds=round(duration_seconds, 2),
        epochs=epochs,
        success=success,
        estimated_vram_gb=round(estimated_vram_gb, 2),
        actual_vram_gb=round(actual_vram_gb, 2),
        vram_estimation_error=round(vram_error, 2),
        lb_recommendation=lb_recommendation,
        lb_confidence=round(lb_confidence, 2),
        lb_was_correct=lb_was_correct,
        ministers_paused=ministers_paused,
        ministers_count=ministers_count,
        error_message=error_message
    )

    return session
