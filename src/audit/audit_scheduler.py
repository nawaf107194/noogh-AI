"""
Audit Scheduler - مُجدول التدقيق الدوري
========================================

نظام لتشغيل التدقيق الذاتي بشكل دوري تلقائي
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import threading
import time
import schedule
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audit.self_audit_engine import SelfAuditEngine, AuditReport
from audit.automated_tests_extended import ExtendedAutomatedTests


@dataclass
class ScheduleConfig:
    """إعدادات الجدولة"""
    enabled: bool = True
    interval_hours: int = 168  # أسبوعياً (7 * 24)
    run_on_startup: bool = False
    max_history_days: int = 90  # الاحتفاظ بالسجل لمدة 90 يوم


class AuditScheduler:
    """
    مُجدول التدقيق الدوري

    يقوم بـ:
    1. تشغيل تدقيق دوري (أسبوعي افتراضياً)
    2. إرسال إشعارات عند انخفاض مستوى الوعي
    3. تنظيف السجلات القديمة
    4. توليد تقارير الاتجاهات
    """

    def __init__(self,
                 config: Optional[ScheduleConfig] = None,
                 on_audit_complete: Optional[Callable] = None,
                 on_consciousness_drop: Optional[Callable] = None):
        """
        Args:
            config: إعدادات الجدولة
            on_audit_complete: callback عند اكتمال التدقيق
            on_consciousness_drop: callback عند انخفاض مستوى الوعي
        """
        self.config = config or ScheduleConfig()
        self.on_audit_complete = on_audit_complete
        self.on_consciousness_drop = on_consciousness_drop

        # تهيئة المحرك
        self.engine = SelfAuditEngine(db_path="data/self_audit.db")
        self.test_suite = ExtendedAutomatedTests()
        self.engine.automated_tests = self.test_suite.get_all_extended_tests()

        # حالة الجدولة
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # آخر تدقيق
        self.last_audit_time: Optional[datetime] = None
        self.last_audit_score: Optional[float] = None

    def start(self):
        """بدء الجدولة"""
        if self.is_running:
            print("⚠️  Scheduler already running")
            return

        print("🚀 Starting Audit Scheduler...")
        print(f"   Interval: Every {self.config.interval_hours} hours")
        print(f"   Run on startup: {self.config.run_on_startup}")

        self.is_running = True
        self._stop_event.clear()

        # جدولة التدقيق الدوري
        schedule.every(self.config.interval_hours).hours.do(self._run_scheduled_audit)

        # تشغيل فوري إذا مطلوب
        if self.config.run_on_startup:
            self._run_scheduled_audit()

        # بدء thread الجدولة
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

        print("✅ Scheduler started successfully")

    def stop(self):
        """إيقاف الجدولة"""
        if not self.is_running:
            return

        print("🛑 Stopping Audit Scheduler...")
        self.is_running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5)

        schedule.clear()
        print("✅ Scheduler stopped")

    def _scheduler_loop(self):
        """حلقة الجدولة الرئيسية"""
        while self.is_running and not self._stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # فحص كل دقيقة
            except Exception as e:
                print(f"❌ Scheduler error: {e}")
                time.sleep(60)

    def _run_scheduled_audit(self):
        """تشغيل تدقيق مجدول"""
        try:
            print(f"\n{'=' * 70}")
            print(f"🔄 Running Scheduled Audit - {datetime.now(timezone.utc).isoformat()}")
            print(f"{'=' * 70}\n")

            # تشغيل التدقيق
            report = self.engine.run_full_audit()

            # حفظ الحالة
            self.last_audit_time = report.timestamp
            prev_score = self.last_audit_score
            self.last_audit_score = report.overall_score

            # طباعة النتائج
            self._print_audit_summary(report)

            # فحص انخفاض مستوى الوعي
            if prev_score is not None and report.overall_score < prev_score - 0.1:
                print(f"\n⚠️  WARNING: Consciousness level dropped!")
                print(f"   Previous: {prev_score*100:.1f}%")
                print(f"   Current:  {report.overall_score*100:.1f}%")
                print(f"   Drop:     {(prev_score - report.overall_score)*100:.1f}%\n")

                if self.on_consciousness_drop:
                    self.on_consciousness_drop(report, prev_score)

            # callback اكتمال التدقيق
            if self.on_audit_complete:
                self.on_audit_complete(report)

            # تنظيف السجلات القديمة
            self._cleanup_old_audits()

            print(f"\n{'=' * 70}")
            print(f"✅ Scheduled Audit Complete")
            print(f"{'=' * 70}\n")

        except Exception as e:
            print(f"❌ Audit failed: {e}")
            import traceback
            traceback.print_exc()

    def _print_audit_summary(self, report: AuditReport):
        """طباعة ملخص التدقيق"""
        print(f"📊 Audit Results:")
        print(f"   Overall Score:     {report.overall_score*100:.1f}%")
        print(f"   Consciousness:     {report.consciousness_level.name}")
        print(f"   Passed:            {report.passed_questions}/{report.total_questions}")
        print(f"   Execution Time:    {report.execution_time_seconds:.2f}s")

        if report.critical_issues:
            print(f"\n⚠️  Critical Issues ({len(report.critical_issues)}):")
            for issue in report.critical_issues[:3]:
                print(f"   • {issue}")
            if len(report.critical_issues) > 3:
                print(f"   ... and {len(report.critical_issues) - 3} more")

    def _cleanup_old_audits(self):
        """تنظيف التدقيقات القديمة"""
        if self.config.max_history_days <= 0:
            return

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.max_history_days)

        # يمكن إضافة منطق الحذف من قاعدة البيانات هنا
        print(f"🗑️  Cleanup: Keeping audits newer than {cutoff_date.date()}")

    def run_audit_now(self) -> AuditReport:
        """تشغيل تدقيق فوري"""
        print("⚡ Running immediate audit...")
        return self.engine.run_full_audit()

    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الجدولة"""
        next_run = None
        if self.is_running and schedule.jobs:
            next_run = schedule.next_run()

        return {
            "is_running": self.is_running,
            "interval_hours": self.config.interval_hours,
            "last_audit_time": self.last_audit_time.isoformat() if self.last_audit_time else None,
            "last_audit_score": self.last_audit_score,
            "next_run": next_run.isoformat() if next_run else None,
            "scheduled_jobs": len(schedule.jobs)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example & CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def on_audit_complete_callback(report: AuditReport):
    """Callback عند اكتمال التدقيق"""
    print(f"\n📧 Sending notification: Audit completed with score {report.overall_score*100:.1f}%")


def on_consciousness_drop_callback(report: AuditReport, previous_score: float):
    """Callback عند انخفاض مستوى الوعي"""
    print(f"\n🚨 ALERT: Consciousness dropped from {previous_score*100:.1f}% to {report.overall_score*100:.1f}%")
    print(f"   Action required: Review critical issues")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit Scheduler")
    parser.add_argument("--mode", choices=["start", "once", "status"], default="once",
                       help="Scheduler mode")
    parser.add_argument("--interval", type=int, default=168,
                       help="Audit interval in hours (default: 168 = weekly)")
    parser.add_argument("--startup", action="store_true",
                       help="Run audit on startup")

    args = parser.parse_args()

    config = ScheduleConfig(
        enabled=True,
        interval_hours=args.interval,
        run_on_startup=args.startup
    )

    scheduler = AuditScheduler(
        config=config,
        on_audit_complete=on_audit_complete_callback,
        on_consciousness_drop=on_consciousness_drop_callback
    )

    if args.mode == "start":
        print("🚀 Starting scheduler in daemon mode...")
        scheduler.start()

        try:
            # Keep running
            while True:
                time.sleep(60)
                status = scheduler.get_status()
                if status["next_run"]:
                    print(f"⏰ Next audit: {status['next_run']}")
        except KeyboardInterrupt:
            print("\n\n⚠️  Received interrupt signal")
            scheduler.stop()

    elif args.mode == "once":
        print("⚡ Running single audit...")
        report = scheduler.run_audit_now()
        print(f"\n✅ Audit complete: {report.overall_score*100:.1f}% ({report.consciousness_level.name})")

    elif args.mode == "status":
        scheduler.start()
        time.sleep(1)
        status = scheduler.get_status()
        print("\n📊 Scheduler Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        scheduler.stop()
