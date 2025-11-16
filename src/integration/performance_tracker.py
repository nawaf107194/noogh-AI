"""
Performance Tracker - Real-time API and System Performance Monitoring
تتبع أداء API في الزمن الحقيقي
"""

import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """متتبع الأداء في الزمن الحقيقي"""

    def __init__(self, history_size: int = 1000, persist_path: Optional[str] = None):
        """
        Args:
            history_size: عدد القياسات التي يتم الاحتفاظ بها في الذاكرة
            persist_path: مسار حفظ البيانات (اختياري)
        """
        self.history_size = history_size
        self.persist_path = Path(persist_path) if persist_path else None

        # تخزين القياسات
        self.api_calls = deque(maxlen=history_size)
        self.system_snapshots = deque(maxlen=history_size)
        self.errors = deque(maxlen=history_size)

        # إحصائيات مباشرة
        self.total_api_calls = 0
        self.total_errors = 0
        self.start_time = datetime.now()

        # قفل للتعامل مع التزامن
        self.lock = threading.Lock()

        # تحميل البيانات المحفوظة إن وجدت
        self._load_persisted_data()

    def record_api_call(self, endpoint: str, method: str, status_code: int,
                       response_time: float, user_id: Optional[str] = None,
                       error: Optional[str] = None):
        """تسجيل استدعاء API"""
        with self.lock:
            call_data = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": round(response_time * 1000, 2),
                "user_id": user_id,
                "error": error
            }

            self.api_calls.append(call_data)
            self.total_api_calls += 1

            if error or status_code >= 400:
                self.errors.append(call_data)
                self.total_errors += 1

    def record_system_snapshot(self):
        """تسجيل لقطة لحالة النظام"""
        with self.lock:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                # GPU info if available
                gpu_info = {}
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_info = {
                            "name": torch.cuda.get_device_name(0),
                            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
                            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
                            "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2),
                            "utilization_pct": torch.cuda.utilization()
                        }
                except Exception as e:
                    # Error caught: {e}
                    pass

                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": mem.percent,
                    "memory_used_gb": round(mem.used / (1024**3), 2),
                    "memory_total_gb": round(mem.total / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "gpu": gpu_info
                }

                self.system_snapshots.append(snapshot)

            except Exception as e:
                logger.error(f"Failed to record system snapshot: {e}")

    def get_api_stats(self, last_minutes: Optional[int] = None) -> Dict[str, Any]:
        """الحصول على إحصائيات API"""
        with self.lock:
            if last_minutes:
                cutoff = datetime.now() - timedelta(minutes=last_minutes)
                relevant_calls = [
                    c for c in self.api_calls
                    if datetime.fromisoformat(c["timestamp"]) >= cutoff
                ]
            else:
                relevant_calls = list(self.api_calls)

            if not relevant_calls:
                return {
                    "total_calls": 0,
                    "error_count": 0,
                    "error_rate_pct": 0,
                    "avg_response_time_ms": 0,
                    "min_response_time_ms": 0,
                    "max_response_time_ms": 0,
                    "p50_response_time_ms": 0,
                    "p95_response_time_ms": 0,
                    "p99_response_time_ms": 0,
                    "endpoints": {},
                    "status_codes": {}
                }

            response_times = [c["response_time_ms"] for c in relevant_calls]
            response_times_sorted = sorted(response_times)
            n = len(response_times_sorted)

            # حساب percentiles
            p50_idx = int(n * 0.50)
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)

            # تجميع حسب endpoint
            endpoints = {}
            for call in relevant_calls:
                ep = call["endpoint"]
                if ep not in endpoints:
                    endpoints[ep] = {"count": 0, "errors": 0, "total_time": 0}
                endpoints[ep]["count"] += 1
                endpoints[ep]["total_time"] += call["response_time_ms"]
                if call.get("error") or call["status_code"] >= 400:
                    endpoints[ep]["errors"] += 1

            # حساب متوسط لكل endpoint
            for ep in endpoints:
                endpoints[ep]["avg_time_ms"] = round(endpoints[ep]["total_time"] / endpoints[ep]["count"], 2)
                endpoints[ep]["error_rate_pct"] = round((endpoints[ep]["errors"] / endpoints[ep]["count"]) * 100, 2)

            # تجميع حسب status code
            status_codes = {}
            for call in relevant_calls:
                code = str(call["status_code"])
                status_codes[code] = status_codes.get(code, 0) + 1

            error_count = sum(1 for c in relevant_calls if c.get("error") or c["status_code"] >= 400)

            return {
                "total_calls": len(relevant_calls),
                "error_count": error_count,
                "error_rate_pct": round((error_count / len(relevant_calls)) * 100, 2) if relevant_calls else 0,
                "avg_response_time_ms": round(sum(response_times) / len(response_times), 2),
                "min_response_time_ms": round(min(response_times), 2),
                "max_response_time_ms": round(max(response_times), 2),
                "p50_response_time_ms": round(response_times_sorted[p50_idx], 2) if n > 0 else 0,
                "p95_response_time_ms": round(response_times_sorted[p95_idx], 2) if n > 0 else 0,
                "p99_response_time_ms": round(response_times_sorted[p99_idx], 2) if n > 0 else 0,
                "endpoints": endpoints,
                "status_codes": status_codes
            }

    def get_system_stats(self, last_minutes: Optional[int] = None) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام"""
        with self.lock:
            if last_minutes:
                cutoff = datetime.now() - timedelta(minutes=last_minutes)
                relevant_snapshots = [
                    s for s in self.system_snapshots
                    if datetime.fromisoformat(s["timestamp"]) >= cutoff
                ]
            else:
                relevant_snapshots = list(self.system_snapshots)

            if not relevant_snapshots:
                return {
                    "snapshot_count": 0,
                    "avg_cpu_percent": 0,
                    "avg_memory_percent": 0,
                    "avg_disk_percent": 0,
                    "peak_cpu_percent": 0,
                    "peak_memory_percent": 0
                }

            cpu_values = [s["cpu_percent"] for s in relevant_snapshots]
            mem_values = [s["memory_percent"] for s in relevant_snapshots]
            disk_values = [s["disk_percent"] for s in relevant_snapshots]

            return {
                "snapshot_count": len(relevant_snapshots),
                "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2),
                "avg_memory_percent": round(sum(mem_values) / len(mem_values), 2),
                "avg_disk_percent": round(sum(disk_values) / len(disk_values), 2),
                "peak_cpu_percent": round(max(cpu_values), 2),
                "peak_memory_percent": round(max(mem_values), 2),
                "latest_snapshot": relevant_snapshots[-1] if relevant_snapshots else None
            }

    def get_comprehensive_report(self, last_minutes: Optional[int] = None) -> Dict[str, Any]:
        """تقرير شامل عن الأداء"""
        uptime = datetime.now() - self.start_time

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_human": str(uptime).split('.')[0],
            "api_stats": self.get_api_stats(last_minutes),
            "system_stats": self.get_system_stats(last_minutes),
            "overall": {
                "total_api_calls_lifetime": self.total_api_calls,
                "total_errors_lifetime": self.total_errors,
                "lifetime_error_rate_pct": round((self.total_errors / self.total_api_calls * 100), 2) if self.total_api_calls > 0 else 0
            }
        }

    def _load_persisted_data(self):
        """تحميل البيانات المحفوظة"""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    self.total_api_calls = data.get("total_api_calls", 0)
                    self.total_errors = data.get("total_errors", 0)
                    # يمكن إضافة المزيد من البيانات حسب الحاجة
                logger.info(f"✅ Loaded persisted performance data from {self.persist_path}")
            except Exception as e:
                logger.warning(f"Could not load persisted data: {e}")

    def persist_data(self):
        """حفظ البيانات للتخزين الدائم"""
        if self.persist_path:
            try:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_api_calls": self.total_api_calls,
                    "total_errors": self.total_errors,
                    "uptime_seconds": int((datetime.now() - self.start_time).total_seconds())
                }

                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"✅ Persisted performance data to {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to persist data: {e}")


# نموذج استخدام مع FastAPI middleware
class PerformanceMiddleware:
    """Middleware لتتبع الأداء تلقائياً"""

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    async def __call__(self, request, call_next):
        start_time = time.time()
        error = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error = str(e)
            status_code = 500
            raise
        finally:
            response_time = time.time() - start_time
            self.tracker.record_api_call(
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                response_time=response_time,
                error=error
            )

        return response


# مثال على الاستخدام
if __name__ == "__main__":
    tracker = PerformanceTracker(persist_path="/tmp/performance_data.json")

    # محاكاة بعض الاستدعاءات
    tracker.record_api_call("/api/chat", "POST", 200, 0.15)
    tracker.record_api_call("/api/status", "GET", 200, 0.05)
    tracker.record_api_call("/api/chat", "POST", 200, 0.12)
    tracker.record_api_call("/api/chat", "POST", 500, 0.25, error="Internal error")

    tracker.record_system_snapshot()

    report = tracker.get_comprehensive_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
