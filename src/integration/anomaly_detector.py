"""
Anomaly Detector - Intelligent System Health Monitoring and Alerting
نظام إنذارات ذكي لكشف الشذوذ
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """مستويات الإنذار"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """إنذار نظام"""

    def __init__(self, level: AlertLevel, message: str, source: str,
                 metric_name: Optional[str] = None, metric_value: Optional[float] = None,
                 threshold: Optional[float] = None):
        self.level = level
        self.message = message
        self.source = source
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold
        }

    def __repr__(self):
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(self.level, "")

        return f"{emoji} [{self.level.value.upper()}] {self.message}"


class AnomalyDetector:
    """كاشف الشذوذ والمراقبة الذكية"""

    def __init__(self):
        self.thresholds = {
            # System resources
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 85, "critical": 95},
            "disk_percent": {"warning": 85, "critical": 95},
            "gpu_memory_percent": {"warning": 90, "critical": 98},
            "gpu_temp_celsius": {"warning": 80, "critical": 90},

            # API performance
            "api_error_rate_pct": {"warning": 5, "critical": 15},
            "api_response_time_ms": {"warning": 1000, "critical": 3000},
            "api_p95_response_time_ms": {"warning": 2000, "critical": 5000},

            # Training metrics
            "training_loss_stagnation_epochs": {"warning": 5, "critical": 10},
            "validation_loss_increase_pct": {"warning": 10, "critical": 25},

            # Brain health
            "brain_health_score": {"warning": 60, "critical": 40, "inverse": True}  # inverse: lower is worse
        }

        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.recent_alerts: List[Alert] = []
        self.max_recent_alerts = 100

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """إضافة معالج للإنذارات (مثل logging, email, slack, etc.)"""
        self.alert_handlers.append(handler)

    def check_system_resources(self, system_stats: Dict[str, Any]) -> List[Alert]:
        """فحص موارد النظام"""
        alerts = []

        # CPU check
        cpu = system_stats.get("cpu_percent", 0)
        if cpu >= self.thresholds["cpu_percent"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"استخدام CPU حرج: {cpu}%",
                "system_resources",
                "cpu_percent",
                cpu,
                self.thresholds["cpu_percent"]["critical"]
            ))
        elif cpu >= self.thresholds["cpu_percent"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"استخدام CPU مرتفع: {cpu}%",
                "system_resources",
                "cpu_percent",
                cpu,
                self.thresholds["cpu_percent"]["warning"]
            ))

        # Memory check
        mem = system_stats.get("memory_percent", 0)
        if mem >= self.thresholds["memory_percent"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"استخدام الذاكرة حرج: {mem}%",
                "system_resources",
                "memory_percent",
                mem,
                self.thresholds["memory_percent"]["critical"]
            ))
        elif mem >= self.thresholds["memory_percent"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"استخدام الذاكرة مرتفع: {mem}%",
                "system_resources",
                "memory_percent",
                mem,
                self.thresholds["memory_percent"]["warning"]
            ))

        # Disk check
        disk = system_stats.get("disk_percent", 0)
        if disk >= self.thresholds["disk_percent"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"مساحة القرص منخفضة جداً: {disk}% مستخدم",
                "system_resources",
                "disk_percent",
                disk,
                self.thresholds["disk_percent"]["critical"]
            ))
        elif disk >= self.thresholds["disk_percent"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"مساحة القرص منخفضة: {disk}% مستخدم",
                "system_resources",
                "disk_percent",
                disk,
                self.thresholds["disk_percent"]["warning"]
            ))

        # GPU checks
        gpu_info = system_stats.get("gpu", {})
        if gpu_info:
            # GPU memory
            if "memory_allocated_mb" in gpu_info and "memory_total_mb" in gpu_info:
                gpu_mem_pct = (gpu_info["memory_allocated_mb"] / gpu_info["memory_total_mb"]) * 100
                if gpu_mem_pct >= self.thresholds["gpu_memory_percent"]["critical"]:
                    alerts.append(Alert(
                        AlertLevel.CRITICAL,
                        f"ذاكرة GPU ممتلئة تقريباً: {gpu_mem_pct:.1f}%",
                        "system_resources",
                        "gpu_memory_percent",
                        gpu_mem_pct,
                        self.thresholds["gpu_memory_percent"]["critical"]
                    ))
                elif gpu_mem_pct >= self.thresholds["gpu_memory_percent"]["warning"]:
                    alerts.append(Alert(
                        AlertLevel.WARNING,
                        f"ذاكرة GPU مرتفعة: {gpu_mem_pct:.1f}%",
                        "system_resources",
                        "gpu_memory_percent",
                        gpu_mem_pct,
                        self.thresholds["gpu_memory_percent"]["warning"]
                    ))

            # GPU temperature (if available)
            if "temperature_celsius" in gpu_info:
                temp = gpu_info["temperature_celsius"]
                if temp >= self.thresholds["gpu_temp_celsius"]["critical"]:
                    alerts.append(Alert(
                        AlertLevel.CRITICAL,
                        f"حرارة GPU حرجة: {temp}°C",
                        "system_resources",
                        "gpu_temp_celsius",
                        temp,
                        self.thresholds["gpu_temp_celsius"]["critical"]
                    ))
                elif temp >= self.thresholds["gpu_temp_celsius"]["warning"]:
                    alerts.append(Alert(
                        AlertLevel.WARNING,
                        f"حرارة GPU مرتفعة: {temp}°C",
                        "system_resources",
                        "gpu_temp_celsius",
                        temp,
                        self.thresholds["gpu_temp_celsius"]["warning"]
                    ))

        return alerts

    def check_api_performance(self, api_stats: Dict[str, Any]) -> List[Alert]:
        """فحص أداء API"""
        alerts = []

        # Error rate check
        error_rate = api_stats.get("error_rate_pct", 0)
        if error_rate >= self.thresholds["api_error_rate_pct"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"معدل أخطاء API حرج: {error_rate}%",
                "api_performance",
                "api_error_rate_pct",
                error_rate,
                self.thresholds["api_error_rate_pct"]["critical"]
            ))
        elif error_rate >= self.thresholds["api_error_rate_pct"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"معدل أخطاء API مرتفع: {error_rate}%",
                "api_performance",
                "api_error_rate_pct",
                error_rate,
                self.thresholds["api_error_rate_pct"]["warning"]
            ))

        # Response time check
        avg_response = api_stats.get("avg_response_time_ms", 0)
        if avg_response >= self.thresholds["api_response_time_ms"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"زمن استجابة API بطيء جداً: {avg_response}ms",
                "api_performance",
                "api_response_time_ms",
                avg_response,
                self.thresholds["api_response_time_ms"]["critical"]
            ))
        elif avg_response >= self.thresholds["api_response_time_ms"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"زمن استجابة API بطيء: {avg_response}ms",
                "api_performance",
                "api_response_time_ms",
                avg_response,
                self.thresholds["api_response_time_ms"]["warning"]
            ))

        # P95 response time check
        p95_response = api_stats.get("p95_response_time_ms", 0)
        if p95_response >= self.thresholds["api_p95_response_time_ms"]["critical"]:
            alerts.append(Alert(
                AlertLevel.ERROR,
                f"P95 زمن استجابة API بطيء جداً: {p95_response}ms",
                "api_performance",
                "api_p95_response_time_ms",
                p95_response,
                self.thresholds["api_p95_response_time_ms"]["critical"]
            ))
        elif p95_response >= self.thresholds["api_p95_response_time_ms"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"P95 زمن استجابة API بطيء: {p95_response}ms",
                "api_performance",
                "api_p95_response_time_ms",
                p95_response,
                self.thresholds["api_p95_response_time_ms"]["warning"]
            ))

        return alerts

    def check_brain_health(self, health_score: Dict[str, Any]) -> List[Alert]:
        """فحص صحة الدماغ"""
        alerts = []

        overall = health_score.get("overall_health", 0)

        # Brain health is inverse: lower score is worse
        if overall <= self.thresholds["brain_health_score"]["critical"]:
            alerts.append(Alert(
                AlertLevel.CRITICAL,
                f"صحة الدماغ منخفضة جداً: {overall}/100",
                "brain_health",
                "brain_health_score",
                overall,
                self.thresholds["brain_health_score"]["critical"]
            ))
        elif overall <= self.thresholds["brain_health_score"]["warning"]:
            alerts.append(Alert(
                AlertLevel.WARNING,
                f"صحة الدماغ منخفضة: {overall}/100",
                "brain_health",
                "brain_health_score",
                overall,
                self.thresholds["brain_health_score"]["warning"]
            ))

        # Check individual components
        components = health_score.get("components", {})
        for component, score in components.items():
            if score < 50:
                alerts.append(Alert(
                    AlertLevel.WARNING,
                    f"مكون '{component}' في الدماغ يحتاج تحسين: {score}/100",
                    "brain_health",
                    f"brain_{component}_score",
                    score,
                    50
                ))

        return alerts

    def check_all(self, system_stats: Optional[Dict] = None,
                  api_stats: Optional[Dict] = None,
                  brain_health: Optional[Dict] = None) -> List[Alert]:
        """فحص شامل لكل المقاييس"""
        all_alerts = []

        if system_stats:
            all_alerts.extend(self.check_system_resources(system_stats))

        if api_stats:
            all_alerts.extend(self.check_api_performance(api_stats))

        if brain_health:
            all_alerts.extend(self.check_brain_health(brain_health))

        # تخزين الإنذارات الحديثة
        self.recent_alerts.extend(all_alerts)
        self.recent_alerts = self.recent_alerts[-self.max_recent_alerts:]

        # تفعيل معالجات الإنذارات
        for alert in all_alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")

        return all_alerts

    def get_health_status(self) -> Dict[str, Any]:
        """الحصول على ملخص الحالة الصحية"""
        critical_count = sum(1 for a in self.recent_alerts if a.level == AlertLevel.CRITICAL)
        error_count = sum(1 for a in self.recent_alerts if a.level == AlertLevel.ERROR)
        warning_count = sum(1 for a in self.recent_alerts if a.level == AlertLevel.WARNING)

        if critical_count > 0:
            status = "حرج 🔴"
            status_code = "critical"
        elif error_count > 0:
            status = "خطأ 🟠"
            status_code = "error"
        elif warning_count > 0:
            status = "تحذير 🟡"
            status_code = "warning"
        else:
            status = "صحي 🟢"
            status_code = "healthy"

        return {
            "status": status,
            "status_code": status_code,
            "critical_alerts": critical_count,
            "error_alerts": error_count,
            "warning_alerts": warning_count,
            "total_alerts": len(self.recent_alerts),
            "recent_alerts": [a.to_dict() for a in self.recent_alerts[-10:]]  # آخر 10 إنذارات
        }


# معالجات الإنذارات الافتراضية
def log_alert_handler(alert: Alert):
    """معالج يسجل الإنذار في logs"""
    log_method = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.ERROR: logger.error,
        AlertLevel.CRITICAL: logger.critical
    }.get(alert.level, logger.info)

    log_method(str(alert))


# مثال على الاستخدام
if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.add_alert_handler(log_alert_handler)

    # محاكاة بعض المقاييس
    system_stats = {
        "cpu_percent": 92,
        "memory_percent": 78,
        "disk_percent": 65,
        "gpu": {
            "memory_allocated_mb": 10000,
            "memory_total_mb": 11764,
            "temperature_celsius": 75
        }
    }

    api_stats = {
        "error_rate_pct": 8,
        "avg_response_time_ms": 850,
        "p95_response_time_ms": 1800
    }

    brain_health = {
        "overall_health": 85,
        "components": {
            "architecture": 90,
            "training": 85,
            "generalization": 80
        }
    }

    alerts = detector.check_all(system_stats, api_stats, brain_health)

    print(f"عدد الإنذارات: {len(alerts)}\n")
    for alert in alerts:
        print(alert)

    print("\n" + "="*50)
    print("حالة النظام:")
    import json
    print(json.dumps(detector.get_health_status(), indent=2, ensure_ascii=False))
