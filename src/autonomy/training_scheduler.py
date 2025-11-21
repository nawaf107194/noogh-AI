"""
⚙️ Training Scheduler - Automatic Training & Model Rotation
Schedules training runs, manages training jobs, and orchestrates model updates

Features:
- Scheduled training runs (daily/weekly/manual)
- Training job queue management
- Performance testing and comparison
- Automatic model deployment or rollback
- Integration with monitoring system
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time
import logging
# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
from src.autonomy.model_manager import (
    get_model_manager, PerformanceMetrics, ModelVersion
)
from src.autonomy.monitor_service import get_monitor_service

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    TESTING = "testing"
    COMPARING = "comparing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TrainingJob:
    """Training job"""
    job_id: str
    status: TrainingStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    training_params: Optional[Dict[str, Any]] = None
    new_model_version: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    comparison_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    deployed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data


class TrainingScheduler:
    """
    Training scheduler with auto-rotation

    Features:
    - Schedule training runs
    - Execute training jobs
    - Test and compare models
    - Deploy or rollback automatically
    - Send alerts via monitoring system
    """

    def __init__(
        self,
        schedule_interval_hours: int = 24,  # Daily by default
        auto_deploy: bool = True,
        auto_rollback: bool = True
    ):
        """
        Initialize training scheduler

        Args:
            schedule_interval_hours: Hours between scheduled training runs
            auto_deploy: Automatically deploy better models
            auto_rollback: Automatically rollback on degradation
        """
        self.schedule_interval = schedule_interval_hours * 3600  # Convert to seconds
        self.auto_deploy = auto_deploy
        self.auto_rollback = auto_rollback

        self.model_manager = get_model_manager()
        self.monitor_service = get_monitor_service()

        self.running = False
        self.training_jobs: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None

        # Callbacks
        self.training_callbacks: List[Callable] = []

        self.start_time = datetime.now()
        self.last_training_time: Optional[datetime] = None
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0

    def create_job_id(self) -> str:
        """Generate unique job ID"""
        return f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def schedule_training(
        self,
        training_params: Optional[Dict[str, Any]] = None
    ) -> TrainingJob:
        """
        Schedule a new training job

        Args:
            training_params: Training parameters

        Returns:
            TrainingJob object
        """
        job = TrainingJob(
            job_id=self.create_job_id(),
            status=TrainingStatus.PENDING,
            created_at=datetime.now().isoformat(),
            training_params=training_params or {}
        )

        self.training_jobs.append(job)
        logger.info(f"📋 Scheduled training job: {job.job_id}")

        return job

    async def execute_training_job(self, job: TrainingJob) -> bool:
        """
        Execute a training job

        Args:
            job: Training job to execute

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🚀 Starting training job: {job.job_id}")

        try:
            # Update status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            self.current_job = job

            # Simulate training (in real system, call actual training function)
            logger.info("🔧 Training model...")
            await asyncio.sleep(2)  # Simulate training time

            # In real system, this would be actual training:
            # trained_model_path = await train_model(job.training_params)

            # For now, simulate a trained model
            trained_model_path = "models/simulated_model"

            # Test the model
            logger.info("🧪 Testing model...")
            job.status = TrainingStatus.TESTING
            await asyncio.sleep(1)  # Simulate testing

            # Get performance metrics (simulated)
            new_metrics = await self._test_model(trained_model_path)
            job.performance_metrics = new_metrics.to_dict()

            # Compare with active model
            logger.info("📊 Comparing with active model...")
            job.status = TrainingStatus.COMPARING

            should_deploy, reason, comparison = self.model_manager.compare_with_active(new_metrics)
            job.comparison_result = comparison

            logger.info(f"📈 Comparison result: {reason}")

            # Register the new model version
            new_version = self.model_manager.register_model(
                model_path=trained_model_path,
                performance_metrics=new_metrics,
                training_params=job.training_params,
                set_active=False  # Don't activate yet
            )

            job.new_model_version = new_version.version_id

            # Decide on deployment
            if should_deploy and self.auto_deploy:
                logger.info("✅ New model is better, deploying...")
                self.model_manager._set_active_version(new_version.version_id)
                self.model_manager._save_metadata()
                job.deployed = True

                # Send success alert
                self.monitor_service.generate_alert(
                    "recovery",
                    f"New model deployed: {new_version.version_id} (cognition: {new_metrics.cognition_score:.3f})"
                )

            else:
                logger.info("⚠️ New model is not better, keeping current version")

                # Send warning alert
                self.monitor_service.generate_alert(
                    "warning",
                    f"Training completed but new model not deployed (cognition: {new_metrics.cognition_score:.3f})"
                )

            # Complete job
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            self.current_job = None

            self.total_jobs_completed += 1
            self.last_training_time = datetime.now()

            logger.info(f"✅ Training job completed: {job.job_id}")

            # Call callbacks
            for callback in self.training_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Training callback failed: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Training job failed: {e}")

            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            job.error = str(e)
            self.current_job = None

            self.total_jobs_failed += 1

            # Send failure alert
            self.monitor_service.generate_alert(
                "critical",
                f"Training job failed: {job.job_id} - {str(e)}"
            )

            return False

    async def _test_model(self, model_path: str) -> PerformanceMetrics:
        """
        Test a model and return performance metrics

        Args:
            model_path: Path to model to test

        Returns:
            PerformanceMetrics object
        """
        # In real system, this would run actual model testing
        # For now, simulate metrics with small variations

        import random

        # Get current active model metrics as baseline
        if self.model_manager.active_version:
            base_metrics = self.model_manager.active_version.performance_metrics
            base_cognition = base_metrics.get('cognition_score', 0.975)
            base_accuracy = base_metrics.get('accuracy', 0.95)
        else:
            base_cognition = 0.975
            base_accuracy = 0.95

        # Simulate slight variation (±5%)
        variation = random.uniform(-0.05, 0.05)

        metrics = PerformanceMetrics(
            accuracy=min(1.0, max(0.0, base_accuracy + variation)),
            avg_response_time_ms=random.uniform(10, 20),
            cognition_score=min(1.0, max(0.0, base_cognition + variation)),
            success_rate=random.uniform(0.95, 1.0),
            memory_usage_mb=random.uniform(500, 800),
            test_samples=1000,
            timestamp=datetime.now().isoformat()
        )

        return metrics

    async def scheduler_loop(self):
        """Main scheduler loop"""
        logger.info(f"🔄 Training scheduler started (interval: {self.schedule_interval/3600:.1f}h)")

        while self.running:
            try:
                # Check if it's time for scheduled training
                should_train = False

                if self.last_training_time is None:
                    # First run
                    should_train = True
                else:
                    # Check if interval has passed
                    time_since_last = (datetime.now() - self.last_training_time).total_seconds()
                    if time_since_last >= self.schedule_interval:
                        should_train = True

                if should_train:
                    logger.info("⏰ Scheduled training time reached")

                    # Create and execute training job
                    job = self.schedule_training()
                    await self.execute_training_job(job)

                # Sleep for 1 hour before checking again
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    def start(self):
        """Start the training scheduler"""
        if self.running:
            logger.warning("Training scheduler already running")
            return

        self.running = True
        logger.info("▶️  Starting training scheduler...")

        # Start scheduler loop in background
        asyncio.create_task(self.scheduler_loop())

    def stop(self):
        """Stop the training scheduler"""
        if not self.running:
            logger.warning("Training scheduler not running")
            return

        self.running = False
        logger.info("⏸️  Training scheduler stopped")

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get currently running job"""
        return self.current_job

    def get_recent_jobs(self, limit: int = 10) -> List[TrainingJob]:
        """Get recent training jobs"""
        return sorted(
            self.training_jobs,
            key=lambda j: j.created_at,
            reverse=True
        )[:limit]

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a specific job"""
        return next((j for j in self.training_jobs if j.job_id == job_id), None)

    def register_training_callback(self, callback: Callable):
        """
        Register a callback to be called when training completes

        Args:
            callback: Function that takes TrainingJob as argument
        """
        self.training_callbacks.append(callback)
        logger.debug(f"Registered training callback: {callback.__name__}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training scheduler statistics"""
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        next_training_in = None
        if self.last_training_time:
            next_training_time = self.last_training_time + timedelta(seconds=self.schedule_interval)
            next_training_in = (next_training_time - datetime.now()).total_seconds() / 3600

        return {
            'running': self.running,
            'schedule_interval_hours': self.schedule_interval / 3600,
            'auto_deploy': self.auto_deploy,
            'auto_rollback': self.auto_rollback,
            'total_jobs': len(self.training_jobs),
            'total_completed': self.total_jobs_completed,
            'total_failed': self.total_jobs_failed,
            'success_rate': (
                self.total_jobs_completed / max(len(self.training_jobs), 1) * 100
            ),
            'last_training_time': (
                self.last_training_time.isoformat() if self.last_training_time else None
            ),
            'next_training_in_hours': round(next_training_in, 2) if next_training_in else None,
            'current_job': self.current_job.job_id if self.current_job else None,
            'uptime_hours': round(uptime_hours, 2)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        stats = self.get_statistics()
        model_stats = self.model_manager.get_statistics()

        return {
            'scheduler': stats,
            'models': model_stats,
            'current_job': self.current_job.to_dict() if self.current_job else None,
            'recent_jobs': [j.to_dict() for j in self.get_recent_jobs(5)]
        }


# Global training scheduler instance
_training_scheduler = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DI Container Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_training_scheduler() -> TrainingScheduler:
    """
    Get global training scheduler instance from DI container
    
    Returns:
        TrainingScheduler instance (singleton)
    """
    try:
        from src.core.di import Container
        scheduler = Container.resolve("training_scheduler")
        if scheduler is not None:
            return scheduler
    except ImportError:
        pass
    
    # Fallback to manual singleton for backward compatibility
    global _training_scheduler
    if _training_scheduler is None:
        _training_scheduler = TrainingScheduler()
    return _training_scheduler
