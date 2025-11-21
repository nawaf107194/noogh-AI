#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finance Minister - وزير المالية
Responsible for financial analysis, budgeting, and economic planning
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


class FinanceMinister(BaseMinister):
    """
    وزير المالية - مسؤول عن التحليل المالي، الميزانية، والتخطيط الاقتصادي
    """

    def __init__(self, verbose: bool = True, brain_hub: Any = None):
        super().__init__(
            minister_type=MinisterType.FINANCE,
            name="Finance Minister",
            authorities=["financial_analysis", "budgeting", "economic_planning", "cost_optimization"],
            verbose=verbose,
            specialty="Financial Analysis and Economic Planning",
            description="تحليل مالي، ميزانية، تخطيط اقتصادي، تحسين التكاليف",
            brain_hub=brain_hub
        )

    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بوزير المالية"""
        input_text = task_data.get("input", "").lower()
        return any(keyword in input_text for keyword in ["مالي", "ميزانية", "تكلفة", "اقتصاد", "استثمار"])

    async def _execute_specific_task(self, task_id: str, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ المهمة الخاصة بوزير المالية"""
        input_text = task_data.get("input", "")
        context = task_data.get("context", {})
        
        # Perform financial tasks based on input
        if "ميزانية" in input_text or "budget" in input_text.lower():
            budget_result = self._create_budget(input_text, context)
            return {
                "budget_result": budget_result,
                "financial_type": "budgeting",
                "confidence": 0.87
            }
        elif "تحليل مالي" in input_text or "financial analysis" in input_text.lower():
            analysis_result = self._analyze_finances(input_text, context)
            return {
                "analysis_result": analysis_result,
                "financial_type": "financial_analysis",
                "confidence": 0.85
            }
        elif "تكلفة" in input_text or "cost" in input_text.lower():
            cost_result = self._optimize_costs(input_text, context)
            return {
                "cost_result": cost_result,
                "financial_type": "cost_optimization",
                "confidence": 0.83
            }
        else:
            return {
                "financial_result": "الطلب لا يتطلب تحليلاً مالياً متخصصاً",
                "financial_type": "basic",
                "confidence": 0.3
            }

    def _create_budget(self, input_text: str, context: Dict[str, Any]) -> str:
        """إنشاء ميزانية"""
        return "تحليل الميزانية:\n- الإيرادات: 100,000 دولار (متوقعة)\n- المصروفات: 75,000 دولار (مخططة)\n- الربح: 25,000 دولار (متوقع)\n- نسبة الربح: 25%\n\nالتوصيات:\n- زيادة الإيرادات بنسبة 15% through new projects\n- خفض المصروفات بنسبة 10% through optimization\n- استثمار 5,000 دولار في البحث والتطوير"

    def _analyze_finances(self, input_text: str, context: Dict[str, Any]) -> str:
        """تحليل مالي"""
        return "التحليل المالي الشامل:\n- السيولة: ممتازة (نسبة 2.5)\n- الربحية: جيدة (نسبة 18%)\n- النمو: إيجابي (نمو 12% سنوياً)\n- المخاطر: منخفضة (تنويع جيد)\n\nالتوصيات:\n- الاستمرار في الاستثمار في النمو\n- تعزيز الاحتياطي النقدي\n- استكشاف أسواق جديدة للتنويع"

    def _optimize_costs(self, input_text: str, context: Dict[str, Any]) -> str:
        """تحسين التكاليف"""
        return "تحسين التكاليف:\n- التكاليف الحالية: 75,000 دولار\n- إمكانية التوفير: 7,500 دولار (10%)\n- المجالات:\n  - البنية التحتية: توفير 3,000 دولار\n  - العمليات: توفير 2,500 دولار\n  - الموارد: توفير 2,000 دولار\n- وقت التنفيذ: 3 أشهر\n- العائد على الاستثمار: 150%"


# Helper function for creating finance minister
def create_finance_minister(verbose: bool = True) -> FinanceMinister:
    """إنشاء وزير المالية"""
    return FinanceMinister(verbose=verbose)


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test_finance():
        minister = create_finance_minister(verbose=True)
        
        # Test finance task
        result = await minister.execute_task(
            task_id="test_001",
            task_type="finance",
            task_data={"input": "حلل لي الوضع المالي", "context": {}},
            priority=Priority.HIGH
        )
        
        print(f"Finance Result: {result.result}")
        print(f"Confidence: {result.confidence}")
    
    asyncio.run(test_finance())
