#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Minister - وزير الاستراتيجية
Responsible for strategic planning, SWOT analysis, and priority setting
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_minister import BaseMinister, MinisterReport, TaskStatus, Priority, generate_task_id
from .minister_types_universal import MinisterType

logger = logging.getLogger(__name__)


class StrategyMinister(BaseMinister):
    """
    وزير الاستراتيجية - مسؤول عن التخطيط الاستراتيجي وتحليل SWOT وتحديد الأولويات
    """

    def __init__(self, verbose: bool = True, brain_hub: Any = None):
        super().__init__(
            minister_type=MinisterType.STRATEGY,
            name="Strategy Minister",
            authorities=["strategy", "planning", "swot_analysis", "priority_setting"],
            verbose=verbose,
            specialty="Strategic Planning and Analysis",
            description="التخطيط الاستراتيجي، تحليل SWOT، تحديد الأولويات"
        ,
            brain_hub=brain_hub
        )

    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بوزير الاستراتيجية"""
        input_text = task_data.get("input", "").lower()
        return any(keyword in input_text for keyword in ["استراتيجية", "خطة", "swot", "أولويات", "تخطيط"])

    async def _execute_specific_task(self, task_id: str, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ المهمة الخاصة بوزير الاستراتيجية"""
        input_text = task_data.get("input", "")
        context = task_data.get("context", {})
        
        # Perform strategic analysis based on input
        if "استراتيجية" in input_text or "خطة" in input_text or "swot" in input_text:
            strategy_result = self._develop_strategy(input_text, context)
            return {
                "strategy_result": strategy_result,
                "analysis_type": "strategic",
                "confidence": 0.88
            }
        elif "أولويات" in input_text or "تخطيط" in input_text:
            priority_result = self._set_priorities(input_text, context)
            return {
                "priority_result": priority_result,
                "analysis_type": "priority_planning",
                "confidence": 0.82
            }
        else:
            return {
                "strategy_result": "الطلب لا يتطلب تخطيطاً استراتيجياً",
                "analysis_type": "basic",
                "confidence": 0.3
            }

    def _develop_strategy(self, input_text: str, context: Dict[str, Any]) -> str:
        """تطوير استراتيجية شاملة"""
        if "مشروع" in input_text or "نظام" in input_text:
            return "استراتيجية تطوير المشروع:\n- المرحلة 1: تحليل المتطلبات (أسبوعان)\n- المرحلة 2: التصميم والتنفيذ (4 أسابيع)\n- المرحلة 3: الاختبار والتقييم (أسبوعان)\n- المرحلة 4: النشر والتحسين المستمر\n\nنقاط القوة: فريق متميز، تقنيات حديثة\nنقاط الضعف: موارد محدودة\nالفرص: سوق متنامٍ\nالتهديدات: منافسة شديدة"
        elif "سوق" in input_text or "تداول" in input_text:
            return "استراتيجية التداول:\n- المدى القصير: شراء على الدعم، بيع على المقاومة\n- المدى المتوسط: تتبع الاتجاه العام\n- المدى الطويل: تنويع المحفظة\n\nتحليل SWOT:\nالقوة: تحليل فني قوي\nالضعف: تقلبات السوق\nالفرص: نمو القطاع\nالتهديدات: تشريعات جديدة"
        else:
            return "استراتيجية عامة:\n- تحديد الأهداف بوضوح\n- تحليل البيئة الداخلية والخارجية\n- وضع خطط بديلة\n- المتابعة والتقييم المستمر\n- التكيف مع المتغيرات"

    def _set_priorities(self, input_text: str, context: Dict[str, Any]) -> str:
        """تحديد الأولويات"""
        return "أولويات المهام:\n1. المهام العاجلة ذات الأثر الكبير\n2. المهام المهمة ذات الأثر المتوسط\n3. المهام الروتينية\n4. المهام المؤجلة\n\nنصيحة: ركز 80% من الجهد على 20% من المهام الأكثر تأثيراً"


# Helper function for creating strategy minister
def create_strategy_minister(verbose: bool = True) -> StrategyMinister:
    """إنشاء وزير الاستراتيجية"""
    return StrategyMinister(verbose=verbose)


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test_strategy():
        minister = create_strategy_minister(verbose=True)
        
        # Test strategy task
        result = await minister.execute_task(
            task_id="test_001",
            task_type="strategy",
            task_data={"input": "ضع لي خطة استراتيجية للمشروع", "context": {}},
            priority=Priority.HIGH
        )
        
        print(f"Strategy Result: {result.result}")
        print(f"Confidence: {result.confidence}")
    
    asyncio.run(test_strategy())
