#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - President
نظام الحكومة الداخلية لنوغ - الرئيس

Version: 3.0.0 - Simplified Implementation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_minister import BaseMinister, MinisterReport, TaskStatus, Priority

logger = logging.getLogger(__name__)


class President:
    """
    رئيس الحكومة الداخلية لنوغ

    المسؤوليات:
    - إدارة وتنسيق جميع الوزراء
    - اتخاذ القرارات الاستراتيجية
    - توزيع المهام على الوزراء المناسبين
    - جمع التقارير وتنسيق الردود
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: عرض التفاصيل
        """
        self.verbose = verbose

        # Cabinet - مجلس الوزراء
        self.cabinet: Dict[str, BaseMinister] = {}
        self.initialize_cabinet()

        # إحصائيات رئاسية
        self.total_requests = 0
        self.successful_requests = 0

        if self.verbose:
            logger.info("🎩 President initialized")
            logger.info("   Cabinet ready with 4 ministers")

    def initialize_cabinet(self):
        """Initialize and register all available ministers."""
        from .education_minister import EducationMinister
        from .security_minister import SecurityMinister
        from .development_minister import DevelopmentMinister
        from .communication_minister import CommunicationMinister

        self.cabinet = {
            "education": EducationMinister(),
            "security": SecurityMinister(),
            "development": DevelopmentMinister(),
            "communication": CommunicationMinister()
        }

    async def process_request(self, user_input: str, context: Optional[dict] = None, priority: str = "medium"):
        """
        Process a user request through the government system.

        Args:
            user_input: طلب المستخدم
            context: سياق إضافي
            priority: أولوية المهمة

        Returns:
            نتيجة المعالجة
        """
        self.total_requests += 1

        # تحديد الوزير المناسب بناءً على نوع الطلب
        minister_key = self._determine_minister(user_input)
        
        if minister_key in self.cabinet:
            try:
                from .base_minister import generate_task_id
                task_id = generate_task_id()
                
                result = await self.cabinet[minister_key].execute_task(
                    task_id=task_id,
                    task_type="general",
                    task_data={"input": user_input, "context": context or {}},
                    priority=Priority(priority.upper())
                )
                self.successful_requests += 1
                return result.to_dict()
            except Exception as e:
                logger.error(f"Error processing request with {minister_key}: {e}")
                return {
                    "success": False,
                    "error": f"Error processing request: {str(e)}",
                    "minister": minister_key
                }
        else:
            return {
                "success": False,
                "error": f"No suitable minister found for request",
                "input": user_input
            }

    def _determine_minister(self, user_input: str) -> str:
        """
        Determine the appropriate minister based on user input keywords.
        """
        user_input_lower = user_input.lower()
        
        # التعليم والبحث
        if any(keyword in user_input_lower for keyword in ["علمني", "تعلم", "دورة", "شرح", "مفهوم", "درس"]):
            return "education"
        
        # الأمن والحماية
        elif any(keyword in user_input_lower for keyword in ["أمن", "حماية", "تهديد", "اختراق", "مراقبة"]):
            return "security"
        
        # التطوير والبرمجة
        elif any(keyword in user_input_lower for keyword in ["طور", "حسّن", "اصلح", "كود", "برمجة", "bug"]):
            return "development"
        
        # التواصل والترجمة
        elif any(keyword in user_input_lower for keyword in ["اكتب", "تقرير", "ترجم", "لخص", "رد"]):
            return "communication"
        
        # افتراضي: التعليم
        else:
            return "education"

    def get_cabinet_status(self) -> Dict[str, Any]:
        """
        Get the status of the entire cabinet.

        Returns:
            حالة مجلس الوزراء
        """
        # جميع الوزراء يعتبرون نشطين في هذا التنفيذ المبسط
        active_ministers = len(self.cabinet)
        
        return {
            "total_ministers": len(self.cabinet),
            "active_ministers": active_ministers,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0,
            "ministers": list(self.cabinet.keys())
        }

    def print_status(self):
        """
        Print the status of the government system.
        """
        status = self.get_cabinet_status()
        logger.info("\n" + "="*50)
        logger.info("🏛️ Noogh Government Status")
        logger.info("="*50)
        logger.info(f"📊 Total Ministers: {status['total_ministers']}")
        logger.info(f"✅ Active Ministers: {status['active_ministers']}")
        logger.info(f"📨 Total Requests: {status['total_requests']}")
        logger.info(f"✅ Successful: {status['successful_requests']}")
        logger.info(f"📈 Success Rate: {status['success_rate']:.1%}")
        logger.info("="*50)


# Helper function for creating president instance
def create_president(verbose: bool = True) -> President:
    """
    إنشاء رئيس نوغ

    Usage:
        president = create_president()
        result = await president.process_request("ما هو سعر BTC؟")
    """
    return President(verbose=verbose)


if __name__ == "__main__":
    # اختبار سريع
    async def test_president():
        logger.info("🧪 Testing Noogh President...\n")

        president = create_president(verbose=True)

        # عرض حالة مجلس الوزراء
        president.print_status()

        # اختبار معالجة طلب
        result = await president.process_request("علمني عن الذكاء الاصطناعي")
        logger.info(f"Test result: {result}")

        logger.info(f"\n✅ President test complete!")

    asyncio.run(test_president())
