"""
🤖 Autonomous Runner - 24/7 Background Service Manager
Manages all autonomous services and ensures continuous operation

Services Managed:
- Self-Monitoring (every 10 minutes)
- Feedback Collection (continuous)
- Training Scheduler (daily)
- Improvement Logging (continuous)
- Daily Reporter (daily at 08:00)

Features:
- Auto-restart on failure
- Health monitoring
- Service coordination
- Crash recovery
- Performance tracking
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import logging
import signal
import traceback
from enum import Enum

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomy.monitor_service import get_monitor_service
from src.autonomy.training_scheduler import get_training_scheduler
from src.autonomy.feedback_collector import get_feedback_collector
from src.autonomy.brain_adjuster import get_brain_adjuster

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class ServiceHealth:
    """Health status of a service"""
    name: str
    status: ServiceStatus
    uptime_seconds: float
    restarts: int
    last_error: Optional[str] = None
    last_restart: Optional[str] = None
    is_healthy: bool = True


class AutonomousRunner:
    """
    Autonomous Runner - Manages all autonomous services 24/7

    Features:
    - Start/stop all services
    - Auto-restart on failure
    - Health monitoring
    - Coordinated shutdown
    - Performance tracking
    """

    def __init__(self):
        """Initialize autonomous runner"""
        self.running = False
        self.services: Dict[str, ServiceHealth] = {}

        # Service instances
        self.monitor_service = None
        self.training_scheduler = None
        self.feedback_collector = None
        self.brain_adjuster = None

        # Statistics
        self.start_time = datetime.now()
        self.total_service_restarts = 0
        self.total_errors = 0

        # Configuration
        self.restart_delay_seconds = 5
        self.max_restart_attempts = 5

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🤖 Autonomous Runner initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"⚠️ Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    async def start_all_services(self):
        """Start all autonomous services"""
        logger.info("🚀 Starting all autonomous services...")

        self.running = True

        # Initialize services
        self.monitor_service = get_monitor_service()
        self.training_scheduler = get_training_scheduler()
        self.feedback_collector = get_feedback_collector()
        self.brain_adjuster = get_brain_adjuster()

        # Initialize service health tracking
        self.services = {
            "monitor_service": ServiceHealth(
                name="Self-Monitoring",
                status=ServiceStatus.STOPPED,
                uptime_seconds=0,
                restarts=0
            ),
            "training_scheduler": ServiceHealth(
                name="Training Scheduler",
                status=ServiceStatus.STOPPED,
                uptime_seconds=0,
                restarts=0
            ),
            "feedback_analyzer": ServiceHealth(
                name="Feedback Analyzer",
                status=ServiceStatus.STOPPED,
                uptime_seconds=0,
                restarts=0
            ),
        }

        # Start services
        tasks = [
            self._run_service("monitor_service", self._monitor_loop),
            self._run_service("training_scheduler", self._training_loop),
            self._run_service("feedback_analyzer", self._feedback_loop),
            self._health_check_loop(),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_service(self, service_name: str, service_func):
        """Run a service with auto-restart"""
        restart_count = 0

        while self.running:
            try:
                # Update service status
                self.services[service_name].status = ServiceStatus.STARTING
                logger.info(f"▶️  Starting {self.services[service_name].name}...")

                # Start service
                service_start = datetime.now()
                self.services[service_name].status = ServiceStatus.RUNNING

                await service_func()

                # If service exits normally, stop
                break

            except Exception as e:
                restart_count += 1
                self.total_service_restarts += 1
                self.total_errors += 1

                error_msg = f"{str(e)}\n{traceback.format_exc()}"

                logger.error(
                    f"❌ {self.services[service_name].name} failed: {e} "
                    f"(restart {restart_count}/{self.max_restart_attempts})"
                )

                # Update service health
                self.services[service_name].status = ServiceStatus.FAILED
                self.services[service_name].last_error = error_msg
                self.services[service_name].restarts = restart_count
                self.services[service_name].is_healthy = False

                # Check restart limit
                if restart_count >= self.max_restart_attempts:
                    logger.error(
                        f"🛑 {self.services[service_name].name} exceeded max restart attempts, giving up"
                    )
                    break

                # Wait before restart
                self.services[service_name].status = ServiceStatus.RESTARTING
                self.services[service_name].last_restart = datetime.now().isoformat()

                logger.info(
                    f"⏳ Restarting {self.services[service_name].name} in {self.restart_delay_seconds}s..."
                )
                await asyncio.sleep(self.restart_delay_seconds)

    async def _monitor_loop(self):
        """Run monitoring service"""
        logger.info("📊 Starting Self-Monitoring service...")

        # Start monitoring service
        self.monitor_service.start()

        # Keep running
        while self.running:
            await asyncio.sleep(60)

    async def _training_loop(self):
        """Run training scheduler"""
        logger.info("⚙️ Starting Training Scheduler...")

        # Start training scheduler
        self.training_scheduler.start()

        # Keep running
        while self.running:
            await asyncio.sleep(60)

    async def _feedback_loop(self):
        """Run feedback analysis loop"""
        logger.info("🔄 Starting Feedback Analyzer...")

        while self.running:
            try:
                # Every hour, analyze feedback and generate recommendations
                await asyncio.sleep(3600)  # 1 hour

                logger.info("🔍 Analyzing feedback patterns...")

                # Generate recommendations
                recommendations = self.brain_adjuster.analyze_and_recommend()

                if recommendations:
                    logger.info(f"💡 Generated {len(recommendations)} recommendations")

                    # Auto-apply low-risk recommendations
                    for rec in recommendations:
                        if rec.risk_level == "low":
                            adjustment = self.brain_adjuster.apply_adjustment(
                                recommendation=rec,
                                force=False
                            )

                            if adjustment:
                                logger.info(
                                    f"✅ Applied adjustment: {adjustment.description}"
                                )

            except Exception as e:
                logger.error(f"Feedback analysis error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _health_check_loop(self):
        """Monitor service health"""
        logger.info("💓 Starting Health Check loop...")

        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Update service uptime
                for service_name, health in self.services.items():
                    if health.status == ServiceStatus.RUNNING:
                        health.uptime_seconds = (
                            datetime.now() - self.start_time
                        ).total_seconds()

                # Log health status
                healthy_count = sum(
                    1 for h in self.services.values() if h.is_healthy
                )

                if healthy_count < len(self.services):
                    logger.warning(
                        f"⚠️ Health check: {healthy_count}/{len(self.services)} services healthy"
                    )

            except Exception as e:
                logger.error(f"Health check error: {e}")

    def stop(self):
        """Stop all services gracefully"""
        logger.info("🛑 Stopping all autonomous services...")

        self.running = False

        # Stop services
        if self.monitor_service:
            self.monitor_service.stop()

        if self.training_scheduler:
            self.training_scheduler.stop()

        # Update service status
        for health in self.services.values():
            health.status = ServiceStatus.STOPPED

        logger.info("✅ All services stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive runner status"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "uptime_hours": round(uptime / 3600, 2),
            "services": {
                name: {
                    "name": health.name,
                    "status": health.status.value,
                    "uptime_seconds": health.uptime_seconds,
                    "uptime_hours": round(health.uptime_seconds / 3600, 2),
                    "restarts": health.restarts,
                    "is_healthy": health.is_healthy,
                    "last_error": health.last_error,
                    "last_restart": health.last_restart,
                }
                for name, health in self.services.items()
            },
            "statistics": {
                "total_service_restarts": self.total_service_restarts,
                "total_errors": self.total_errors,
                "healthy_services": sum(
                    1 for h in self.services.values() if h.is_healthy
                ),
                "total_services": len(self.services),
            },
            "start_time": self.start_time.isoformat(),
        }

    def get_summary(self) -> str:
        """Get human-readable status summary"""
        status = self.get_status()

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🤖 Autonomous Runner Status",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"Running: {'✅ YES' if status['running'] else '❌ NO'}",
            f"Uptime: {status['uptime_hours']} hours",
            f"Healthy Services: {status['statistics']['healthy_services']}/{status['statistics']['total_services']}",
            f"Total Restarts: {status['statistics']['total_service_restarts']}",
            f"Total Errors: {status['statistics']['total_errors']}",
            f"",
            "Services:",
        ]

        for name, service in status['services'].items():
            status_icon = {
                "running": "✅",
                "stopped": "⏹️",
                "failed": "❌",
                "restarting": "🔄",
                "starting": "▶️",
            }.get(service['status'], "❓")

            lines.append(
                f"  {status_icon} {service['name']}: {service['status']} "
                f"(uptime: {service['uptime_hours']}h, restarts: {service['restarts']})"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(lines)


# Global runner instance
_autonomous_runner = None

def get_autonomous_runner() -> AutonomousRunner:
    """Get global autonomous runner instance"""
    global _autonomous_runner
    if _autonomous_runner is None:
        _autonomous_runner = AutonomousRunner()
    return _autonomous_runner


async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('autonomous_runner.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🤖 Noogh Autonomous Runner Starting...")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Get runner
    runner = get_autonomous_runner()

    # Start all services
    await runner.start_all_services()


if __name__ == "__main__":
    asyncio.run(main())
