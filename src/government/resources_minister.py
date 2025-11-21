#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resources Minister - وزير الموارد
Responsible for resource management, allocation, and optimization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_minister import BaseMinister, MinisterReport, TaskStatus, Priority, generate_task_id
from .minister_types_universal import MinisterType

logger = logging.getLogger(__name__)


class ResourcesMinister(BaseMinister):
    """
    وزير الموارد - مسؤول عن إدارة وتوزيع وتحسين الموارد
    """

    def __init__(self, verbose: bool = True, brain_hub: Any = None):
        super().__init__(
            minister_type=MinisterType.RESOURCES,
            name="Resources Minister",
            authorities=["resource_management", "allocation", "optimization", "capacity_planning"],
            verbose=verbose,
            specialty="Resource Management and Optimization",
            description="إدارة الموارد، التوزيع، التحسين، تخطيط السعة"
        ,
            brain_hub=brain_hub
        )

    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بوزير الموارد"""
        input_text = task_data.get("input", "").lower()
        return any(keyword in input_text for keyword in ["موارد", "توزيع", "تحسين", "سعة", "إدارة"])

    async def _execute_specific_task(self, task_id: str, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ المهمة الخاصة بوزير الموارد"""
        input_text = task_data.get("input", "")
        context = task_data.get("context", {})
        
        # Perform resource tasks based on input
        if "توزيع" in input_text or "allocation" in input_text.lower():
            allocation_result = self._allocate_resources(input_text, context)
            return {
                "allocation_result": allocation_result,
                "resource_type": "allocation",
                "confidence": 0.86
            }
        elif "تحسين" in input_text or "optimization" in input_text.lower():
            optimization_result = self._optimize_resources(input_text, context)
            return {
                "optimization_result": optimization_result,
                "resource_type": "optimization",
                "confidence": 0.84
            }
        elif "سعة" in input_text or "capacity" in input_text.lower():
            capacity_result = self._plan_capacity(input_text, context)
            return {
                "capacity_result": capacity_result,
                "resource_type": "capacity_planning",
                "confidence": 0.82
            }
        else:
            return {
                "resource_result": "الطلب لا يتطلب إدارة موارد متخصصة",
                "resource_type": "basic",
                "confidence": 0.3
            }

    def _allocate_resources(self, input_text: str, context: Dict[str, Any]) -> str:
        """توزيع الموارد"""
        return "توزيع الموارد:\n- الموارد المتاحة: 100 وحدة\n- التوزيع الحالي:\n  - التطوير: 40 وحدة\n  - البحث: 25 وحدة\n  - العمليات: 20 وحدة\n  - الاحتياطي: 15 وحدة\n\nالتوصيات:\n- زيادة تخصيص البحث إلى 30 وحدة\n- تقليل الاحتياطي إلى 10 وحدة\n- إعادة توزيع 5 وحدات للتطوير"

    def _optimize_resources(self, input_text: str, context: Dict[str, Any]) -> str:
        """تحسين الموارد"""
        return "تحسين الموارد:\n- الكفاءة الحالية: 75%\n- إمكانية التحسين: 15 نقطة مئوية\n- المجالات:\n  - الحوسبة: تحسين 8% through better scheduling\n  - التخزين: تحسين 5% through compression\n  - الشبكة: تحسين 2% through optimization\n- التوفير المتوقع: 20% من التكاليف"

    def _plan_capacity(self, input_text: str, context: Dict[str, Any]) -> str:
        """تخطيط السعة"""
        return "تخطيط السعة:\n- الاستخدام الحالي: 60%\n- النمو المتوقع: 25% سنوياً\n- السعة المطلوبة:\n  - 3 أشهر: 70%\n  - 6 أشهر: 80%\n  - 12 شهر: 95%\n- التوصيات:\n  - توسيع السعة بنسبة 30%\n  - استثمار في بنية تحتية قابلة للتطوير\n  - تنفيذ خطة توسع تدريجية"


# Helper function for creating resources minister
def create_resources_minister(verbose: bool = True) -> ResourcesMinister:
    """إنشاء وزير الموارد"""
    return ResourcesMinister(verbose=verbose)


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test_resources():
        minister = create_resources_minister(verbose=True)
        
        # Test resources task
        result = await minister.execute_task(
            task_id="test_001",
            task_type="resources",
            task_data={"input": "حسن لي توزيع الموارد", "context": {}},
            priority=Priority.HIGH
        )
        
        print(f"Resources Result: {result.result}")
        print(f"Confidence: {result.confidence}")
    
    asyncio.run(test_resources())
