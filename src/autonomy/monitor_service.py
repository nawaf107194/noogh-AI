"""
⏰ Monitor Service - Scheduled Health Checks & Alerts
Runs periodic health checks and sends alerts when issues detected

Features:
- Scheduled health checks (every 10 minutes)
- Alert generation and notification
- Auto-recovery attempts
- Performance tracking
- Alert history
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import time
import asyncio
import logging
import json
from collections import deque
# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
from src.autonomy.self_monitor import get_monitor, HealthStatus

logger = logging.getLogger(__name__)


class Alert:
    """System alert"""

    def __init__(
        self,
        alert_type: str,  # "warning", "critical", "emergency", "recovery"
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ):
        self.alert_type = alert_type
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.acknowledged = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_type": self.alert_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged
        }

    def __repr__(self):
        return f"Alert({self.alert_type}: {self.message})"


class MonitorService:
    """
    Monitoring service with scheduled checks and alerting

    Features:
    - Periodic health checks (default: every 10 minutes)
    - Alert generation when issues detected
    - Alert history tracking
    - Optional alert callbacks (email, webhook, etc.)
    - Auto-recovery attempts
    """

    def __init__(self, check_interval_minutes: int = 10):
        self.monitor = get_monitor()
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.running = False
        self.task = None

        # Alert tracking
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.unacknowledged_alerts = []

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            "total_checks": 0,
            "total_alerts": 0,
            "last_check_time": None,
            "service_start_time": datetime.now().isoformat(),
        }

    def add_alert_callback(self, callback: Callable):
        """Add callback function to be called when alert is generated"""
        self.alert_callbacks.append(callback)

    def generate_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Generate and store alert"""
        alert = Alert(alert_type, message, details)

        self.alerts.append(alert)
        self.unacknowledged_alerts.append(alert)
        self.stats["total_alerts"] += 1

        # Log alert
        emoji = "⚠️" if alert_type == "warning" else "🚨" if alert_type in ["critical", "emergency"] else "✅"
        logger.warning(f"{emoji} ALERT: [{alert_type.upper()}] {message}")

        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return alert

    def acknowledge_alert(self, alert: Alert):
        """Acknowledge an alert"""
        alert.acknowledged = True
        if alert in self.unacknowledged_alerts:
            self.unacknowledged_alerts.remove(alert)

    def acknowledge_all_alerts(self):
        """Acknowledge all pending alerts"""
        for alert in self.unacknowledged_alerts:
            alert.acknowledged = True
        self.unacknowledged_alerts.clear()

    def attempt_recovery(self, health_status: HealthStatus):
        """
        Attempt automatic recovery for known issues

        Returns:
            True if recovery attempted, False otherwise
        """
        recovery_attempted = False

        # Check for specific recoverable issues
        for issue in health_status.issues:
            if "API not responsive" in issue:
                logger.info("🔧 Attempting API recovery...")
                # Could attempt to restart API here
                # For now, just log
                recovery_attempted = True

            elif "Brain Hub not ready" in issue:
                logger.info("🔧 Attempting Brain Hub recovery...")
                # Could attempt to reinitialize Brain Hub
                recovery_attempted = True

        if recovery_attempted:
            self.generate_alert(
                "recovery",
                "Auto-recovery attempted",
                {"issues": health_status.issues}
            )

        return recovery_attempted

    async def perform_scheduled_check(self):
        """Perform a scheduled health check"""
        logger.info("🔍 Starting scheduled health check...")

        # Perform check
        health_status = self.monitor.perform_health_check()

        self.stats["total_checks"] += 1
        self.stats["last_check_time"] = health_status.timestamp

        # Generate alerts based on status
        if health_status.overall_status == "emergency":
            self.generate_alert(
                "emergency",
                f"EMERGENCY: System in critical state!",
                {
                    "cognition_score": health_status.cognition_score,
                    "issues": health_status.issues,
                    "active_ministers": health_status.active_ministers
                }
            )
            # Attempt recovery
            self.attempt_recovery(health_status)

        elif health_status.overall_status == "critical":
            self.generate_alert(
                "critical",
                f"CRITICAL: System has serious issues",
                {
                    "issues": health_status.issues,
                    "warnings": health_status.warnings
                }
            )
            # Attempt recovery
            self.attempt_recovery(health_status)

        elif health_status.overall_status == "warning":
            self.generate_alert(
                "warning",
                f"WARNING: System performance degraded",
                {
                    "warnings": health_status.warnings,
                    "cognition_score": health_status.cognition_score
                }
            )

        elif health_status.overall_status == "healthy":
            # Only log recovery if there were recent alerts
            if len(self.unacknowledged_alerts) > 0:
                self.generate_alert(
                    "recovery",
                    "System back to healthy state",
                    {"cognition_score": health_status.cognition_score}
                )
                # Auto-acknowledge previous alerts
                self.acknowledge_all_alerts()

        logger.info(f"✅ Scheduled check complete: {health_status.overall_status}")

        return health_status

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info(f"🤖 Monitor Service started (check interval: {self.check_interval/60} minutes)")

        while self.running:
            try:
                await self.perform_scheduled_check()

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)

        logger.info("🛑 Monitor Service stopped")

    def start(self):
        """Start the monitoring service"""
        if self.running:
            logger.warning("Monitor service already running")
            return

        self.running = True

        # Create and start async task
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.task = loop.create_task(self.monitor_loop())

        logger.info("✅ Monitor service started")

    def stop(self):
        """Stop the monitoring service"""
        if not self.running:
            return

        self.running = False

        if self.task:
            self.task.cancel()

        logger.info("🛑 Monitor service stopped")

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        alerts_list = list(self.alerts)
        return alerts_list[-limit:]

    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts"""
        return list(self.unacknowledged_alerts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "unacknowledged_alerts_count": len(self.unacknowledged_alerts),
            "total_alerts_in_history": len(self.alerts),
            "service_uptime_hours": (
                datetime.now() - datetime.fromisoformat(self.stats["service_start_time"])
            ).total_seconds() / 3600
        }

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "running": self.running,
            "check_interval_minutes": self.check_interval / 60,
            "statistics": self.get_statistics(),
            "health_summary": self.monitor.get_health_summary()
        }


# Global service instance
_service = None

def get_monitor_service(check_interval_minutes: int = 10) -> MonitorService:
    """Get global monitor service instance"""
    global _service
    if _service is None:
        _service = MonitorService(check_interval_minutes)
    return _service


# Example alert callback functions
def log_alert_callback(alert: Alert):
    """Simple callback that logs alerts"""
    logger.info(f"📧 Alert callback: {alert}")


def webhook_alert_callback(alert: Alert):
    """Example webhook callback (not implemented)"""
    # Could send to Slack, Discord, email, etc.
    pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    service = get_monitor_service(check_interval_minutes=1)  # Check every 1 minute for testing
    service.add_alert_callback(log_alert_callback)

    try:
        service.start()
        # Keep running
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        service.stop()
