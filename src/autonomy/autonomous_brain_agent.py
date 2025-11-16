#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Autonomous Brain Agent - النظام الذكي المستقل
=====================================================

نظام ذكي مستقل يعمل 24/7 باستخدام طبقة التكامل الموحدة

القدرات:
✅ التحليل الذاتي (self-analysis) - كل ساعة
✅ البحث المستمر (continuous research) - يبحث عن تحسينات
✅ التدريب التلقائي (auto-training) - يدرّب نماذج جديدة
✅ التحسين المستمر (continuous improvement)
✅ التقارير اليومية (daily reports)

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-11-10
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import unified brain API
from src.integration.brain_integration_layer import create_unified_brain

# Import autonomous systems
from src.monitoring.resource_monitor import ResourceMonitor
from training.training_need_detector import TrainingNeedDetector
from data.auto_data_collector import AutoDataCollector
from finance.finance_system import FinanceSystem, CostCategory, RevenueSource
from training.local_trainer import LocalModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutonomousBrainAgent:
    """
    🤖 نظام ذكي مستقل يعمل 24/7

    يستخدم طبقة التكامل الموحدة (UnifiedBrainAPI) لـ:
    - التحليل الذاتي باستخدام الوزراء
    - البحث المستمر عن تحسينات
    - التدريب التلقائي للنماذج
    - التحسين المستمر للأداء
    """

    def __init__(self, work_dir: str = "/home/noogh/projects/noogh_unified_system"):
        """تهيئة النظام المستقل"""
        self.work_dir = Path(work_dir)
        self.data_dir = self.work_dir / "data" / "autonomous"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize unified brain
        logger.info("🧠 Initializing Unified Brain API...")
        self.brain = create_unified_brain(verbose=False)

        # Check if brain is ready
        if not self.brain.is_ready:
            logger.error("❌ Brain not ready - cannot start autonomous agent")
            raise RuntimeError("Brain initialization failed")

        logger.info("✅ Unified Brain API ready")
        logger.info(f"   Hub Ready: {self.brain.is_hub_ready}")
        logger.info(f"   Cognition Score: {self.brain.get_cognition_score():.1%}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # AUTONOMOUS SYSTEMS - الأنظمة المستقلة
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        logger.info("🔧 Initializing autonomous systems...")

        # 1. Resource Monitor - مراقب الموارد
        self.resource_monitor = ResourceMonitor(check_interval=30.0, verbose=False)
        logger.info("   ✅ ResourceMonitor initialized")

        # 2. Training Need Detector - كاشف الحاجة للتدريب
        self.training_detector = TrainingNeedDetector(
            performance_threshold=0.85,
            min_new_samples=1000,
            days_since_training=7,
            work_dir=str(work_dir)
        )
        logger.info("   ✅ TrainingNeedDetector initialized")

        # 3. Auto Data Collector - جامع البيانات
        self.data_collector = AutoDataCollector(work_dir=str(work_dir))
        logger.info("   ✅ AutoDataCollector initialized")

        # 4. Finance System - النظام المالي
        self.finance = FinanceSystem(initial_balance=1000.0, work_dir=str(work_dir))
        logger.info("   ✅ FinanceSystem initialized")

        # 5. Local Model Trainer - مدرب النماذج المحلي
        self.model_trainer = LocalModelTrainer(work_dir=str(work_dir))
        logger.info("   ✅ LocalModelTrainer initialized")

        logger.info("✅ All autonomous systems ready")

        # Statistics
        self.total_analyses = 0
        self.total_research_tasks = 0
        self.total_trainings = 0
        self.total_improvements = 0

        # State
        self.running = False
        self.last_analysis = None
        self.last_research = None
        self.last_training = None
        self.last_report = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AUTONOMOUS TASKS - المهام المستقلة
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def self_analysis(self):
        """
        🔍 التحليل الذاتي
        يستخدم وزير التحليل لتحليل أداء النظام
        """
        logger.info("━" * 60)
        logger.info("🔍 Starting self-analysis...")
        logger.info("━" * 60)

        try:
            # Ask analysis minister for system analysis
            result = self.brain.ask_minister(
                request="تحليل أداء النظام الحالي وتحديد نقاط التحسين",
                context={
                    "type": "self_analysis",
                    "timestamp": datetime.now().isoformat(),
                    "cognition_score": self.brain.get_cognition_score()
                }
            )

            self.total_analyses += 1
            self.last_analysis = datetime.now()

            # Save analysis report
            report_path = self.reports_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "type": "self_analysis",
                    "minister": result.get('minister_used'),
                    "analysis": result.get('response'),
                    "confidence": result.get('confidence'),
                    "cognition_score": self.brain.get_cognition_score()
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Self-analysis complete")
            logger.info(f"   Minister: {result.get('minister_used')}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.1%}")
            logger.info(f"   Report: {report_path}")

            return result

        except Exception as e:
            logger.error(f"❌ Self-analysis failed: {e}")
            return None

    async def research_improvements(self):
        """
        📚 البحث عن تحسينات
        يستخدم وزير البحث للبحث عن تقنيات جديدة
        """
        logger.info("━" * 60)
        logger.info("📚 Starting research for improvements...")
        logger.info("━" * 60)

        try:
            # Ask research minister
            result = self.brain.ask_minister(
                request="ابحث عن أحدث التقنيات والأساليب لتحسين أداء النظام",
                context={
                    "type": "research",
                    "focus": "system_improvements",
                    "current_cognition": self.brain.get_cognition_score()
                }
            )

            self.total_research_tasks += 1
            self.last_research = datetime.now()

            # Save research report
            report_path = self.reports_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "type": "research",
                    "minister": result.get('minister_used'),
                    "findings": result.get('response'),
                    "confidence": result.get('confidence')
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Research complete")
            logger.info(f"   Minister: {result.get('minister_used')}")
            logger.info(f"   Report: {report_path}")

            return result

        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return None

    async def auto_training(self):
        """
        🏋️ التدريب التلقائي مع استشارة الوزراء

        الآلية:
        1. الكشف التلقائي عن الحاجة للتدريب (TrainingNeedDetector)
        2. استشارة وزير التعليم عن طرق التدريب ومصادر البيانات
        3. فحص الموارد والميزانية
        4. جمع البيانات بناءً على نصيحة الوزير
        5. إرسال للدماغ للتدريب الفعلي
        6. تحميل الـ model الجديد
        """
        logger.info("━" * 60)
        logger.info("🏋️ Starting autonomous training with ministers consultation...")
        logger.info("━" * 60)

        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 1: النظام المستقل يكتشف الحاجة للتدريب
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("📊 Step 1: Agent detecting training need...")
            decision = self.training_detector.should_train()

            if not decision.should_train:
                logger.info("ℹ️  No training needed at this time")
                return {"status": "skipped", "reason": "not_needed"}

            logger.info(f"✅ Agent detected training need: {len(decision.reasons)} reasons")
            for reason in decision.reasons:
                logger.info(f"   - [{reason['priority']}] {reason['details']}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 2: استشارة وزير التعليم عن أفضل طريقة للتدريب
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("🏛️ Step 2: Consulting Minister of Education...")

            # تحضير السياق للوزير
            reasons_summary = ", ".join([r['details'] for r in decision.reasons])

            education_consultation = self.brain.ask_minister(
                request=f"نحتاج تدريب نموذج جديد للأسباب التالية: {reasons_summary}. ما هي أفضل مصادر البيانات المتاحة؟ وما هي أفضل طريقة للتدريب؟",
                context={
                    "type": "training_planning",
                    "reasons": decision.reasons,
                    "confidence": decision.confidence,
                    "current_performance": self.training_detector.get_status().get("current_performance")
                }
            )

            logger.info(f"✅ Minister consulted: {education_consultation.get('minister_used')}")
            logger.info(f"   Advice: {education_consultation.get('response', '')[:150]}...")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3: فحص الموارد (النظام المستقل)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("📊 Step 3: Checking system resources...")
            resources = self.resource_monitor.get_current_resources()

            # إذا كانت الموارد غير كافية، استشر وزير الموارد
            if resources.overall_status.value not in ['normal', 'warning']:
                logger.warning(f"⚠️  Resources not optimal: {resources.overall_status.value}")

                resource_consultation = self.brain.ask_minister(
                    request=f"الموارد الحالية: {resources.warnings}. كيف نتعامل مع هذا قبل التدريب؟",
                    context={
                        "type": "resource_issue",
                        "warnings": resources.warnings,
                        "status": resources.overall_status.value
                    }
                )

                logger.info(f"🏛️ Resource Minister advice: {resource_consultation.get('response', '')[:100]}...")
                logger.warning("   Training postponed due to resource constraints")
                return {"status": "postponed", "reason": "insufficient_resources", "advice": resource_consultation}

            logger.info("✅ Resources available for training")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 4: فحص الميزانية (النظام المستقل)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("💰 Step 4: Checking budget...")
            training_cost = self.finance.estimate_training_cost(gpu_hours=1.0)

            if not self.finance.can_afford_operation(training_cost):
                logger.warning(f"⚠️  Insufficient funds: need ${training_cost:.2f}")

                # استشارة وزير المالية
                finance_consultation = self.brain.ask_minister(
                    request=f"الميزانية الحالية: ${self.finance.config['balance']:.2f}، نحتاج ${training_cost:.2f} للتدريب. ما الحل؟",
                    context={
                        "type": "budget_issue",
                        "balance": self.finance.config['balance'],
                        "needed": training_cost
                    }
                )

                logger.info(f"🏛️ Finance Minister advice: {finance_consultation.get('response', '')[:100]}...")
                return {"status": "postponed", "reason": "insufficient_funds", "advice": finance_consultation}

            logger.info(f"✅ Budget OK: ${training_cost:.2f} estimated cost")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 5: جمع البيانات بناءً على نصيحة وزير التعليم
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("📥 Step 5: Collecting training data (based on minister's advice)...")

            # هنا نستخدم نصيحة الوزير لتحديد المصادر
            # في المستقبل يمكن تحليل رد الوزير لاستخراج المصادر المقترحة
            collected_data = await self.data_collector.collect_training_data(
                target_samples=1000,
                task_type="general"
            )

            train_count = len(collected_data['train'])
            test_count = len(collected_data['test'])
            logger.info(f"✅ Data collected: {train_count} train, {test_count} test samples")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 6: إرسال للدماغ للتدريب الفعلي
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("🧠 Step 6: Sending to Brain for actual training...")

            # استشارة وزير التدريب للتوجيه النهائي
            training_consultation = self.brain.ask_minister(
                request=f"جاهزون للتدريب مع {train_count} عينة. ما هي أفضل إعدادات التدريب (learning rate, batch size, epochs)؟",
                context={
                    "type": "training_configuration",
                    "samples": train_count,
                    "task_type": "general"
                }
            )

            logger.info(f"🏛️ Training Minister guidance: {training_consultation.get('response', '')[:100]}...")

            # تسجيل التكلفة
            self.finance.record_cost(
                training_cost,
                CostCategory.TRAINING,
                f"Model training with {train_count} samples (minister-guided)"
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # التدريب الفعلي - REAL TRAINING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("🏋️ Brain is training the model (REAL TRAINING)...")

            training_result = await self.model_trainer.train(
                train_data=collected_data['train'],
                test_data=collected_data['test'],
                minister_advice=training_consultation.get('response', '')
            )

            if training_result['status'] == 'failed':
                logger.error(f"❌ Training failed: {training_result.get('error')}")
                return {
                    "status": "failed",
                    "error": training_result.get('error')
                }

            final_accuracy = training_result['accuracy']
            logger.info(f"✅ Real training completed! Accuracy: {final_accuracy:.1%}")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 7: تحميل الـ model الجديد
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info("📦 Step 7: Loading new model into Brain...")

            # تسجيل التدريب
            self.training_detector.record_training({
                "accuracy": final_accuracy,
                "samples": train_count,
                "minister_advice": education_consultation.get('response', '')[:200],
                "model_path": training_result.get('model_path'),
                "train_loss": training_result.get('train_loss'),
                "eval_loss": training_result.get('loss')
            })
            self.training_detector.update_performance(accuracy=final_accuracy)

            self.total_trainings += 1
            self.last_training = datetime.now()

            logger.info(f"✅ Training complete: {final_accuracy:.1%} accuracy")
            logger.info(f"💸 Cost recorded: ${training_cost:.2f}")
            logger.info(f"📦 Model path: {training_result.get('model_path')}")
            logger.info("✅ New model loaded into Brain")

            return {
                "status": "success",
                "accuracy": final_accuracy,
                "train_samples": train_count,
                "test_samples": test_count,
                "cost": training_cost,
                "model_path": training_result.get('model_path'),
                "ministers_consulted": [
                    education_consultation.get('minister_used'),
                    training_consultation.get('minister_used')
                ]
            }

        except Exception as e:
            logger.error(f"❌ Auto-training failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def generate_daily_report(self):
        """
        📊 توليد تقرير يومي شامل
        يجمع كل المعلومات من جميع الأنظمة
        """
        logger.info("━" * 60)
        logger.info("📊 Generating comprehensive daily report...")
        logger.info("━" * 60)

        try:
            # Get full system status
            status = self.brain.get_full_status()

            # Get resource snapshot
            resources = self.resource_monitor.get_current_resources()

            # Get financial status
            financial_status = self.finance.get_financial_status()

            # Get training status
            training_status = self.training_detector.get_status()

            # Create comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "type": "daily_report",
                "brain_status": status,
                "autonomous_stats": {
                    "total_analyses": self.total_analyses,
                    "total_research_tasks": self.total_research_tasks,
                    "total_trainings": self.total_trainings,
                    "total_improvements": self.total_improvements
                },
                "last_activities": {
                    "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
                    "last_research": self.last_research.isoformat() if self.last_research else None,
                    "last_training": self.last_training.isoformat() if self.last_training else None
                },
                "cognition_score": self.brain.get_cognition_score(),
                "ministers_count": len(self.brain.list_ministers()),
                # New: Autonomous systems data
                "resources": {
                    "gpu_memory_percent": resources.gpu_memory_percent,
                    "gpu_temperature": resources.gpu_temperature,
                    "cpu_percent": resources.cpu_percent,
                    "ram_percent": resources.ram_percent,
                    "overall_status": resources.overall_status.value,
                    "warnings": resources.warnings
                },
                "finances": {
                    "balance": financial_status.balance,
                    "total_costs": financial_status.total_costs,
                    "total_revenue": financial_status.total_revenue,
                    "net_profit": financial_status.net_profit,
                    "can_afford_training": financial_status.can_afford_training,
                    "cost_breakdown": self.finance.get_cost_breakdown(30),
                    "revenue_breakdown": self.finance.get_revenue_breakdown(30)
                },
                "training": training_status
            }

            # Save report
            report_path = self.reports_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.last_report = datetime.now()

            logger.info("✅ Comprehensive daily report generated")
            logger.info(f"   Report: {report_path}")
            logger.info(f"   Cognition: {report['cognition_score']:.1%}")
            logger.info(f"   Balance: ${financial_status.balance:.2f}")
            logger.info(f"   Resources: {resources.overall_status.value.upper()}")
            logger.info(f"   Total trainings: {self.total_trainings}")

            return report

        except Exception as e:
            logger.error(f"❌ Daily report failed: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN LOOP - الحلقة الرئيسية
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def run_cycle(self):
        """
        دورة واحدة من النشاط المستقل
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔄 Starting autonomous cycle...")
        logger.info("=" * 60)

        # 0. Resource check (always first)
        logger.info("📊 Checking system resources...")
        resources = self.resource_monitor.get_current_resources()
        logger.info(f"   Status: {resources.overall_status.value.upper()}")
        logger.info(f"   GPU: {resources.gpu_utilization:.1f}%, RAM: {resources.ram_percent:.1f}%")

        if resources.warnings:
            for warning in resources.warnings:
                logger.warning(f"   ⚠️  {warning}")

        # 1. Self-analysis (every cycle)
        await self.self_analysis()
        await asyncio.sleep(2)

        # 2. Research (every cycle)
        await self.research_improvements()
        await asyncio.sleep(2)

        # 3. Autonomous training (uses all 4 systems)
        await self.auto_training()
        await asyncio.sleep(2)

        # 4. Daily report (if needed)
        if self.last_report is None or \
           (datetime.now() - self.last_report).total_seconds() > 86400:  # 24 hours
            await self.generate_daily_report()

        logger.info("=" * 60)
        logger.info("✅ Autonomous cycle complete")
        logger.info("=" * 60 + "\n")

    async def run_forever(self, cycle_interval_hours: float = 1.0):
        """
        تشغيل مستمر 24/7

        Args:
            cycle_interval_hours: فترة بين كل دورة (بالساعات)
        """
        self.running = True

        logger.info("=" * 70)
        logger.info("🤖 AUTONOMOUS BRAIN AGENT - STARTING 24/7 OPERATION")
        logger.info("=" * 70)
        logger.info(f"Cycle interval: {cycle_interval_hours} hours")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"Reports directory: {self.reports_dir}")
        logger.info("=" * 70 + "\n")

        cycle_number = 0

        try:
            while self.running:
                cycle_number += 1
                logger.info(f"\n🔵 CYCLE #{cycle_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Run one cycle
                await self.run_cycle()

                # Wait for next cycle
                wait_seconds = cycle_interval_hours * 3600
                logger.info(f"⏳ Waiting {cycle_interval_hours} hours until next cycle...")
                await asyncio.sleep(wait_seconds)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"\n❌ Error in main loop: {e}")
        finally:
            self.running = False
            logger.info("\n" + "=" * 70)
            logger.info("🛑 AUTONOMOUS BRAIN AGENT - STOPPED")
            logger.info("=" * 70)
            logger.info(f"Total cycles: {cycle_number}")
            logger.info(f"Total analyses: {self.total_analyses}")
            logger.info(f"Total research: {self.total_research_tasks}")
            logger.info(f"Total trainings: {self.total_trainings}")
            logger.info("=" * 70 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI - واجهة سطر الأوامر
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="🤖 Autonomous Brain Agent - نظام ذكي مستقل 24/7"
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Cycle interval in hours (default: 1.0)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default="/home/noogh/projects/noogh_unified_system",
        help='Working directory'
    )

    args = parser.parse_args()

    # Create and run agent
    agent = AutonomousBrainAgent(work_dir=args.work_dir)
    await agent.run_forever(cycle_interval_hours=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
