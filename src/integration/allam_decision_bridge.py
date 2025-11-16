#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌉 ALLaM Decision Bridge - جسر ALLaM مع محرك القرار
==================================================

Integrates ALLaM (Arabic LLM) with Decision Engine for enhanced reasoning.

Author: Noogh AI Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from decision import (
    DecisionContext,
    DecisionOption,
    DecisionType,
    DecisionPriority
)


logger = logging.getLogger(__name__)


class ALLaMDecisionBridge:
    """
    🌉 جسر بين ALLaM ومحرك القرار

    Uses ALLaM for:
    - Arabic language understanding
    - Complex reasoning
    - Generating decision options
    - Explaining decisions in Arabic
    """

    def __init__(self, use_allam: bool = True):
        self.use_allam = use_allam
        self.allam_model = None

        if use_allam:
            try:
                from brain.allam_model import ALLaMModel
                self.allam_model = ALLaMModel(backend="production")
                self.allam_model.load_model()
                logger.info("✅ ALLaM model loaded in production mode")
            except Exception as e:
                logger.warning(f"⚠️ Could not load ALLaM: {e}")
                logger.info("   Falling back to rule-based reasoning")
                self.use_allam = False

        logger.info("🌉 ALLaM Decision Bridge initialized")

    def enhance_decision_context(
        self,
        context: DecisionContext,
        user_request: str
    ) -> DecisionContext:
        """
        تحسين سياق القرار باستخدام ALLaM

        Args:
            context: سياق القرار الأصلي
            user_request: الطلب من المستخدم

        Returns:
            سياق محسّن
        """
        if not self.use_allam or not self.allam_model:
            return context  # No enhancement

        try:
            # Use ALLaM to understand user intent better
            analysis = self._analyze_intent(user_request)

            # Enhance context with ALLaM insights
            if "intent" in analysis:
                context.metadata["allam_intent"] = analysis["intent"]

            if "entities" in analysis:
                context.metadata["allam_entities"] = analysis["entities"]

            if "sentiment" in analysis:
                context.metadata["allam_sentiment"] = analysis["sentiment"]

            logger.debug("✨ Context enhanced with ALLaM insights")

        except Exception as e:
            logger.warning(f"ALLaM enhancement failed: {e}")

        return context

    def generate_options_with_allam(
        self,
        user_request: str,
        context: Dict[str, Any],
        ministers_input: Dict[str, Any]
    ) -> List[DecisionOption]:
        """
        توليد خيارات قرار باستخدام ALLaM

        Args:
            user_request: طلب المستخدم
            context: السياق
            ministers_input: مدخلات الوزراء

        Returns:
            خيارات قرار مقترحة من ALLaM
        """
        options = []

        if not self.use_allam or not self.allam_model:
            return options  # Return empty, let ministers handle it

        try:
            # Prepare prompt for ALLaM
            prompt = self._build_decision_prompt(
                user_request,
                ministers_input
            )

            # Get ALLaM's suggestions
            response = self._query_allam(prompt)

            # Parse response into decision options
            options = self._parse_allam_response(response)

            logger.info(f"✨ ALLaM generated {len(options)} options")

        except Exception as e:
            logger.warning(f"ALLaM option generation failed: {e}")

        return options

    def explain_decision_in_arabic(
        self,
        decision_action: str,
        reasoning: List[str],
        context: Dict[str, Any]
    ) -> str:
        """
        شرح القرار بالعربية باستخدام ALLaM

        Args:
            decision_action: القرار المُتخذ
            reasoning: الأسباب
            context: السياق

        Returns:
            شرح بالعربية
        """
        if not self.use_allam or not self.allam_model:
            # Fallback: simple Arabic template
            return self._simple_arabic_explanation(decision_action, reasoning)

        try:
            prompt = f"""
اشرح القرار التالي بالعربية بشكل واضح ومفصل:

القرار: {decision_action}
الأسباب:
{chr(10).join('- ' + r for r in reasoning)}

اكتب شرحاً مختصراً وواضحاً للمستخدم:
"""

            explanation = self._query_allam(prompt)
            return explanation

        except Exception as e:
            logger.warning(f"ALLaM explanation failed: {e}")
            return self._simple_arabic_explanation(decision_action, reasoning)

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════

    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """تحليل النية من النص"""

        if not self.use_allam:
            return self._rule_based_intent(text)

        # Use ALLaM for intent analysis
        prompt = f"""
حلل النية من النص التالي:
"{text}"

أعط النتيجة بصيغة JSON:
{{
    "intent": "trading" أو "analysis" أو "learning",
    "entities": ["كيان1", "كيان2"],
    "sentiment": "positive" أو "negative" أو "neutral"
}}
"""

        try:
            response = self._query_allam(prompt)
            # Parse JSON response (simplified for demo)
            return self._rule_based_intent(text)  # Fallback for now
        except:
            return self._rule_based_intent(text)

    def _rule_based_intent(self, text: str) -> Dict[str, Any]:
        """تحليل بسيط بدون ALLaM"""

        text_lower = text.lower()

        # Intent
        if any(word in text_lower for word in ["تداول", "شراء", "بيع", "استثمار", "trade", "buy", "sell", "invest"]):
            intent = "trading"
        elif any(word in text_lower for word in ["تحليل", "فحص", "analyze", "check"]):
            intent = "analysis"
        elif any(word in text_lower for word in ["تعلم", "تدريب", "learn", "train"]):
            intent = "learning"
        else:
            intent = "general"

        # Entities (simple extraction)
        entities = []
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "بتكوين", "إيثيريوم"]
        for keyword in crypto_keywords:
            if keyword in text_lower:
                entities.append(keyword)

        # Sentiment (very simple)
        positive_words = ["جيد", "ممتاز", "رائع", "good", "great", "excellent"]
        negative_words = ["سيء", "خطير", "bad", "risky"]

        if any(word in text_lower for word in positive_words):
            sentiment = "positive"
        elif any(word in text_lower for word in negative_words):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment
        }

    def _build_decision_prompt(
        self,
        user_request: str,
        ministers_input: Dict[str, Any]
    ) -> str:
        """بناء prompt لـ ALLaM"""

        prompt = f"""
أنت مستشار ذكي متخصص في اتخاذ القرارات.

طلب المستخدم:
{user_request}

تحليل الخبراء:
"""

        for minister, data in ministers_input.items():
            if isinstance(data, dict):
                recs = data.get("recommendations", [])
                if recs:
                    prompt += f"\n{minister}:\n"
                    for rec in recs[:2]:  # Top 2
                        if isinstance(rec, dict):
                            prompt += f"  - {rec.get('description', rec.get('action', ''))}\n"

        prompt += """
بناءً على التحليل أعلاه، اقترح 2-3 خيارات للقرار مع شرح مختصر لكل خيار.
"""

        return prompt

    def _query_allam(self, prompt: str) -> str:
        """استعلام ALLaM"""

        if not self.allam_model:
            raise Exception("ALLaM model not loaded")

        # For demo mode, return simulated response
        if self.allam_model.backend == "demo":
            return self._simulate_allam_response(prompt)

        # Real ALLaM inference
        try:
            response = self.allam_model.generate(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"ALLaM query failed: {e}")
            return ""

    def _simulate_allam_response(self, prompt: str) -> str:
        """محاكاة استجابة ALLaM في وضع Demo"""

        # Simple simulated responses based on keywords
        if "قرار" in prompt or "خيار" in prompt:
            return """
الخيارات المقترحة:
1. تنفيذ القرار بناءً على توصية الخبراء (ثقة عالية)
2. الانتظار وجمع المزيد من المعلومات (حذر)
3. طلب رأي إضافي من مستشارين آخرين (متوازن)
"""
        elif "شرح" in prompt:
            return "تم اتخاذ هذا القرار بناءً على تحليل شامل للبيانات المتاحة وتوصيات الخبراء، مع الأخذ بعين الاعتبار المخاطر والفوائد المحتملة."
        else:
            return "فهمت طلبك. سأعمل على تحليله بعناية."

    def _parse_allam_response(self, response: str) -> List[DecisionOption]:
        """تحليل استجابة ALLaM وتحويلها لخيارات قرار"""

        options = []

        # Simple parsing (for demo)
        # In production, would use more sophisticated parsing

        lines = response.strip().split('\n')
        option_id = 0

        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                # Extract option text
                option_text = line[2:].strip()

                # Create decision option
                option = DecisionOption(
                    option_id=f"ALLAM-{option_id}",
                    action=f"allam_suggestion_{option_id}",
                    description=option_text,
                    confidence=0.75,  # Default confidence from ALLaM
                    reasoning=[f"ALLaM suggestion based on Arabic understanding"]
                )

                options.append(option)
                option_id += 1

        return options

    def _simple_arabic_explanation(
        self,
        action: str,
        reasoning: List[str]
    ) -> str:
        """شرح بسيط بالعربية بدون ALLaM"""

        explanation = f"تم اتخاذ القرار: {action}\n\nالأسباب:\n"

        for i, reason in enumerate(reasoning[:3], 1):
            explanation += f"{i}. {reason}\n"

        return explanation


# Singleton
_allam_bridge = None

def get_allam_bridge(use_allam: bool = True) -> ALLaMDecisionBridge:
    """الحصول على جسر ALLaM"""
    global _allam_bridge
    if _allam_bridge is None:
        _allam_bridge = ALLaMDecisionBridge(use_allam=use_allam)
    return _allam_bridge
