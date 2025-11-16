#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - Base Minister
نظام الحكومة الداخلية لنوغ - الوزير الأساسي

Version: 2.0.0
Features:
- ✅ Base class for all ministers (الوزراء)
- ✅ Standardized reporting system
- ✅ Authority and permission management
- ✅ Communication with President
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


from .minister_types_universal import MinisterType


class Priority(Enum):
    """أولوية المهام"""
    URGENT = "urgent"       # عاجل
    HIGH = "high"          # عالي
    MEDIUM = "medium"      # متوسط
    LOW = "low"           # منخفض


class TaskStatus(Enum):
    """حالة المهام"""
    PENDING = "pending"           # في الانتظار
    IN_PROGRESS = "in_progress"   # قيد التنفيذ
    COMPLETED = "completed"       # مكتمل
    FAILED = "failed"            # فشل
    DELEGATED = "delegated"      # مفوض لوزير آخر


class MinisterReport:
    """تقرير الوزير للرئيس"""

    def __init__(
        self,
        minister_type: "MinisterType",
        task_id: str,
        status: TaskStatus,
        result: Dict[str, Any],
        recommendations: List[str] = None,
        next_actions: List[str] = None,
        resources_needed: List[str] = None,
        confidence: float = 1.0
    ):
        self.minister_type = minister_type
        self.task_id = task_id
        self.status = status
        self.result = result
        self.recommendations = recommendations or []
        self.next_actions = next_actions or []
        self.resources_needed = resources_needed or []
        self.confidence = confidence
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """تحويل التقرير إلى dict"""
        return {
            'minister': self.minister_type.value,
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'recommendations': self.recommendations,
            'next_actions': self.next_actions,
            'resources_needed': self.resources_needed,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

    def to_arabic_summary(self) -> str:
        """ملخص بالعربية للتقرير"""
        minister_names = {
            MinisterType.FINANCE: "وزير المالية",
            MinisterType.EDUCATION: "وزير التعليم",
            MinisterType.COMMUNICATION: "وزير الخارجية",
            MinisterType.SECURITY: "وزير الداخلية",
            MinisterType.DEVELOPMENT: "وزير الصناعة",
            MinisterType.STRATEGY: "وزير التخطيط"
        }

        status_names = {
            TaskStatus.PENDING: "في الانتظار",
            TaskStatus.IN_PROGRESS: "قيد التنفيذ",
            TaskStatus.COMPLETED: "مكتمل",
            TaskStatus.FAILED: "فشل",
            TaskStatus.DELEGATED: "مفوض"
        }

        minister_name = minister_names.get(self.minister_type, "وزير غير معروف")
        status_name = status_names.get(self.status, "حالة غير معروفة")

        summary = f"📋 تقرير {minister_name}\n"
        summary += f"   الحالة: {status_name}\n"
        summary += f"   الثقة: {self.confidence:.1%}\n"

        if self.recommendations:
            summary += f"   التوصيات: {len(self.recommendations)} توصية\n"

        return summary


class MinisterResponse:
    """
    استجابة الوزير لطلب معين
    Minister's response to a specific request
    """

    def __init__(
        self,
        minister_type: MinisterType,
        action: str,
        description: str,
        confidence: float = 1.0,
        data: Dict[str, Any] = None,
        recommendations: List[str] = None
    ):
        """
        Args:
            minister_type: نوع الوزير
            action: الإجراء المقترح
            description: وصف الاستجابة
            confidence: مستوى الثقة (0-1)
            data: بيانات إضافية
            recommendations: توصيات
        """
        self.minister_type = minister_type
        self.action = action
        self.description = description
        self.confidence = confidence
        self.data = data or {}
        self.recommendations = recommendations or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """تحويل الاستجابة إلى dict"""
        return {
            'minister': self.minister_type.value,
            'action': self.action,
            'description': self.description,
            'confidence': self.confidence,
            'data': self.data,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return f"MinisterResponse(minister={self.minister_type.value}, action={self.action}, confidence={self.confidence:.2f})"


class BaseMinister(ABC):
    """
    الفئة الأساسية لجميع الوزراء

    كل وزير يجب أن:
    - يحدد صلاحياته ومجال عمله
    - ينفذ المهام المطلوبة منه
    - يرفع تقارير للرئيس
    - يتعاون مع الوزراء الآخرين
    """

    def __init__(
        self,
        minister_type: MinisterType,
        name: str,
        authorities: List[str],
        resources: Dict[str, Any] = None,
        verbose: bool = True,
        specialty: str = None,
        description: str = None,
        expertise_level: float = 0.85
    ):
        """
        Args:
            minister_type: نوع الوزير
            name: اسم الوزير
            authorities: الصلاحيات (list of strings)
            resources: الموارد المتاحة
            verbose: عرض التفاصيل
            specialty: التخصص
            description: الوصف
            expertise_level: مستوى الخبرة (0-1)
        """
        self.minister_type = minister_type
        self.name = name
        self.authorities = authorities
        self.resources = resources or {}
        self.specialty = specialty or name
        self.description = description or f"Minister of {name}"
        self.expertise_level = expertise_level
        self.verbose = verbose

        # إدارة المهام
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

        # إحصائيات
        self.tasks_count = 0
        self.success_rate = 0.0
        self.last_activity = datetime.now()

        if self.verbose:
            logger.info(f"🏛️ {self.get_arabic_title()} initialized")
            logger.info(f"   Authorities: {len(self.authorities)}")
            logger.info(f"   Resources: {len(self.resources)}")

    def get_arabic_title(self) -> str:
        """الحصول على اللقب بالعربية"""
        titles = {
            MinisterType.FINANCE: "💰 وزير المالية",
            MinisterType.EDUCATION: "📚 وزير التعليم",
            MinisterType.COMMUNICATION: "🌍 وزير الخارجية",
            MinisterType.SECURITY: "🔐 وزير الداخلية",
            MinisterType.DEVELOPMENT: "⚙️ وزير الصناعة",
            MinisterType.STRATEGY: "📊 وزير التخطيط"
        }
        return titles.get(self.minister_type, f"🏛️ {self.name}")

    def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """
        فحص قدرة الوزير على تنفيذ المهمة

        Args:
            task_type: نوع المهمة
            task_data: بيانات المهمة

        Returns:
            True إذا كان يستطيع تنفيذ المهمة
        """
        # فحص الصلاحيات
        if task_type not in self.authorities:
            return False

        # فحص إضافي خاص بكل وزير
        return self._can_handle_specific_task(task_type, task_data)

    @abstractmethod
    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بكل وزير (يجب تنفيذه في كل وزير)"""
        pass

    async def execute_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> MinisterReport:
        """
        تنفيذ المهمة الأساسية

        Args:
            task_id: معرف المهمة
            task_type: نوع المهمة
            task_data: بيانات المهمة
            priority: أولوية المهمة

        Returns:
            MinisterReport تقرير الإنجاز
        """
        start_time = datetime.now()

        # تسجيل المهمة
        self.active_tasks[task_id] = {
            'type': task_type,
            'data': task_data,
            'priority': priority,
            'start_time': start_time,
            'status': TaskStatus.IN_PROGRESS
        }

        self.tasks_count += 1
        self.last_activity = start_time

        if self.verbose:
            logger.info(f"\n🎯 {self.get_arabic_title()} - New Task")
            logger.info(f"   Task ID: {task_id}")
            logger.info(f"   Type: {task_type}")
            logger.info(f"   Priority: {priority.value}")

        try:
            # تنفيذ المهمة (يجب تنفيذه في كل وزير)
            result = await self._execute_specific_task(task_id, task_type, task_data)

            # حساب الوقت المستغرق
            duration = (datetime.now() - start_time).total_seconds()

            # تحديث حالة المهمة
            self.active_tasks[task_id]['status'] = TaskStatus.COMPLETED
            self.active_tasks[task_id]['duration'] = duration
            self.active_tasks[task_id]['result'] = result

            # نقل إلى المهام المكتملة
            self.completed_tasks.append(task_id)
            del self.active_tasks[task_id]

            # حساب معدل النجاح
            total_completed = len(self.completed_tasks)
            total_failed = len(self.failed_tasks)
            if total_completed + total_failed > 0:
                self.success_rate = total_completed / (total_completed + total_failed)

            # إنشاء التقرير
            report = MinisterReport(
                minister_type=self.minister_type,
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                confidence=self._calculate_confidence(result)
            )

            if self.verbose:
                logger.info(f"✅ Task {task_id} completed ({duration:.2f}s)")
                logger.info(f"   Success Rate: {self.success_rate:.1%}")

            return report

        except Exception as e:
            # التعامل مع الأخطاء
            error_msg = str(e)

            # تحديث حالة المهمة
            self.active_tasks[task_id]['status'] = TaskStatus.FAILED
            self.active_tasks[task_id]['error'] = error_msg

            # نقل إلى المهام الفاشلة
            self.failed_tasks.append(task_id)
            del self.active_tasks[task_id]

            # حساب معدل النجاح
            total_completed = len(self.completed_tasks)
            total_failed = len(self.failed_tasks)
            if total_completed + total_failed > 0:
                self.success_rate = total_completed / (total_completed + total_failed)

            # إنشاء تقرير الخطأ
            report = MinisterReport(
                minister_type=self.minister_type,
                task_id=task_id,
                status=TaskStatus.FAILED,
                result={'error': error_msg},
                confidence=0.0
            )

            if self.verbose:
                logger.error(f"❌ Task {task_id} failed: {error_msg}")
                logger.info(f"   Success Rate: {self.success_rate:.1%}")

            return report

    @abstractmethod
    async def _execute_specific_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """تنفيذ المهمة الخاصة بكل وزير (يجب تنفيذه في كل وزير)"""
        pass

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """حساب مستوى الثقة في النتيجة"""
        # حساب أساسي - يمكن تخصيصه في كل وزير
        if 'error' in result:
            return 0.0

        if 'confidence' in result:
            return float(result['confidence'])

        # ثقة افتراضية بناءً على معدل النجاح
        return self.success_rate

    def get_status_report(self) -> Dict[str, Any]:
        """تقرير حالة الوزير"""
        return {
            'minister': {
                'type': self.minister_type.value,
                'name': self.name,
                'title': self.get_arabic_title()
            },
            'statistics': {
                'total_tasks': self.tasks_count,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'success_rate': self.success_rate,
                'last_activity': self.last_activity.isoformat()
            },
            'authorities': self.authorities,
            'resources': list(self.resources.keys())
        }

    def print_status(self):
        """عرض حالة الوزير"""
        status = self.get_status_report()

        logger.info(f"\n📊 {self.get_arabic_title()} - Status Report")
        logger.info("=" * 60)

        stats = status['statistics']
        logger.info(f"Total Tasks: {stats['total_tasks']}")
        logger.info(f"Active: {stats['active_tasks']}")
        logger.info(f"Completed: {stats['completed_tasks']}")
        logger.info(f"Failed: {stats['failed_tasks']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1%}")

        logger.info(f"\nAuthorities ({len(self.authorities)}):")
        for auth in self.authorities[:3]:  # Show first 3
            logger.info(f"  • {auth}")
        if len(self.authorities) > 3:
            logger.info(f"  ... and {len(self.authorities) - 3} more")

        logger.info("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_task_id() -> str:
    """توليد معرف فريد للمهمة"""
    from uuid import uuid4
    return f"task_{uuid4().hex[:8]}"


if __name__ == "__main__":
    # اختبار سريع للفئة الأساسية
    logger.info("🧪 Testing Base Minister...\n")

    # لا يمكن إنشاء BaseMinister مباشرة لأنه abstract
    # سنختبر في الوزراء المحددين

    logger.info("✅ Base Minister framework ready!")
    logger.info("   Ready for specific ministers implementation...")
