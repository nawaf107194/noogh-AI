"""
Confidence Measurement System - نظام قياس الثقة
================================================

يقيس ثقة القرارات والاستنتاجات رقمياً بناءً على عدة عوامل
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import statistics


class ConfidenceLevel(Enum):
    """مستويات الثقة"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                 # 20-40%
    MODERATE = "moderate"       # 40-60%
    HIGH = "high"               # 60-80%
    VERY_HIGH = "very_high"     # 80-100%


@dataclass
class ConfidenceFactor:
    """عامل واحد يؤثر على الثقة"""
    name: str
    score: float  # 0.0 - 1.0
    weight: float  # أهمية هذا العامل
    evidence: str


@dataclass
class ConfidenceScore:
    """نتيجة قياس الثقة"""
    overall_score: float  # 0.0 - 1.0
    level: ConfidenceLevel
    factors: List[ConfidenceFactor]
    timestamp: datetime
    reasoning: str


class ConfidenceScorer:
    """
    نظام قياس الثقة الشامل

    يقيس الثقة بناءً على:
    1. جودة البيانات المصدرية
    2. تناسق الاستنتاجات
    3. عدد المصادر المؤيدة
    4. التأكيدات الداخلية
    5. سجل الأداء السابق
    6. وضوح النتائج
    """

    def __init__(self):
        # أوزان العوامل
        self.factor_weights = {
            "data_quality": 0.25,
            "consistency": 0.20,
            "source_count": 0.15,
            "internal_validation": 0.15,
            "past_performance": 0.15,
            "result_clarity": 0.10
        }

    def calculate_confidence(self,
                           decision: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> ConfidenceScore:
        """
        حساب ثقة قرار معين

        Args:
            decision: القرار أو الاستنتاج
            context: السياق الإضافي

        Returns:
            ConfidenceScore: نتيجة قياس الثقة
        """
        context = context or {}
        factors = []

        # 1. جودة البيانات
        data_quality = self._assess_data_quality(decision, context)
        factors.append(ConfidenceFactor(
            name="data_quality",
            score=data_quality,
            weight=self.factor_weights["data_quality"],
            evidence=f"Data quality assessment: {data_quality:.2f}"
        ))

        # 2. تناسق الاستنتاجات
        consistency = self._assess_consistency(decision, context)
        factors.append(ConfidenceFactor(
            name="consistency",
            score=consistency,
            weight=self.factor_weights["consistency"],
            evidence=f"Internal consistency: {consistency:.2f}"
        ))

        # 3. عدد المصادر
        source_score = self._assess_sources(decision, context)
        factors.append(ConfidenceFactor(
            name="source_count",
            score=source_score,
            weight=self.factor_weights["source_count"],
            evidence=f"Source reliability: {source_score:.2f}"
        ))

        # 4. التحقق الداخلي
        validation_score = self._assess_internal_validation(decision, context)
        factors.append(ConfidenceFactor(
            name="internal_validation",
            score=validation_score,
            weight=self.factor_weights["internal_validation"],
            evidence=f"Internal validation: {validation_score:.2f}"
        ))

        # 5. سجل الأداء السابق
        performance_score = self._assess_past_performance(decision, context)
        factors.append(ConfidenceFactor(
            name="past_performance",
            score=performance_score,
            weight=self.factor_weights["past_performance"],
            evidence=f"Historical performance: {performance_score:.2f}"
        ))

        # 6. وضوح النتائج
        clarity_score = self._assess_result_clarity(decision, context)
        factors.append(ConfidenceFactor(
            name="result_clarity",
            score=clarity_score,
            weight=self.factor_weights["result_clarity"],
            evidence=f"Result clarity: {clarity_score:.2f}"
        ))

        # حساب النقاط الإجمالية
        overall_score = sum(f.score * f.weight for f in factors)

        # تحديد المستوى
        level = self._determine_level(overall_score)

        # توليد التفسير
        reasoning = self._generate_reasoning(factors, overall_score)

        return ConfidenceScore(
            overall_score=overall_score,
            level=level,
            factors=factors,
            timestamp=datetime.now(timezone.utc),
            reasoning=reasoning
        )

    def _assess_data_quality(self, decision: Dict, context: Dict) -> float:
        """تقييم جودة البيانات المصدرية"""
        score = 0.5  # قيمة افتراضية

        # فحص وجود بيانات كافية
        if "sources" in decision:
            sources = decision["sources"]
            if isinstance(sources, list):
                # كلما زادت المصادر، زادت الثقة
                score += min(0.3, len(sources) * 0.1)

        # فحص جودة المدخلات
        if "input_quality" in context:
            score += context["input_quality"] * 0.2

        # فحص اكتمال البيانات
        if "data_completeness" in context:
            score += context["data_completeness"] * 0.2

        return min(1.0, score)

    def _assess_consistency(self, decision: Dict, context: Dict) -> float:
        """تقييم تناسق الاستنتاجات"""
        score = 0.6  # قيمة افتراضية

        # فحص التناسق الداخلي
        if "internal_consistency" in decision:
            score = decision["internal_consistency"]

        # فحص التناسق مع السياق
        if "context_alignment" in context:
            score = (score + context["context_alignment"]) / 2

        # فحص عدم وجود تناقضات
        if "contradictions" in decision:
            contradictions = len(decision.get("contradictions", []))
            score -= min(0.4, contradictions * 0.1)

        return max(0.0, min(1.0, score))

    def _assess_sources(self, decision: Dict, context: Dict) -> float:
        """تقييم المصادر"""
        if "sources" not in decision:
            return 0.3  # لا توجد مصادر

        sources = decision["sources"]

        if not sources:
            return 0.3

        # عدد المصادر
        source_count = len(sources) if isinstance(sources, list) else 1

        # نقاط بناءً على العدد
        if source_count >= 5:
            score = 1.0
        elif source_count >= 3:
            score = 0.8
        elif source_count >= 2:
            score = 0.6
        else:
            score = 0.4

        # تعديل بناءً على موثوقية المصادر
        if "source_reliability" in context:
            reliability = context["source_reliability"]
            score = (score + reliability) / 2

        return score

    def _assess_internal_validation(self, decision: Dict, context: Dict) -> float:
        """تقييم التحقق الداخلي"""
        score = 0.5

        # فحص وجود تحقق
        if "validated" in decision and decision["validated"]:
            score += 0.3

        # فحص نتائج التحقق
        if "validation_results" in decision:
            results = decision["validation_results"]
            if isinstance(results, dict):
                passed = results.get("passed", 0)
                total = results.get("total", 1)
                score += (passed / total) * 0.4

        return min(1.0, score)

    def _assess_past_performance(self, decision: Dict, context: Dict) -> float:
        """تقييم الأداء السابق"""
        # النظر في سجل الأداء السابق
        if "performance_history" in context:
            history = context["performance_history"]

            if isinstance(history, list) and history:
                # حساب متوسط الأداء السابق
                avg_performance = statistics.mean(history)
                return avg_performance

        # فحص دقة القرارات المشابهة
        if "similar_decisions_accuracy" in context:
            return context["similar_decisions_accuracy"]

        return 0.5  # قيمة افتراضية (غير معروف)

    def _assess_result_clarity(self, decision: Dict, context: Dict) -> float:
        """تقييم وضوح النتائج"""
        score = 0.5

        # فحص وضوح الإجابة
        if "result" in decision:
            result = decision["result"]

            # طول الإجابة المناسب
            if isinstance(result, str):
                length = len(result)
                if 10 <= length <= 500:
                    score += 0.2

        # فحص الغموض
        if "ambiguity_score" in decision:
            ambiguity = decision["ambiguity_score"]
            score -= ambiguity * 0.3

        # فحص وجود تفسير
        if "explanation" in decision and decision["explanation"]:
            score += 0.2

        return max(0.0, min(1.0, score))

    def _determine_level(self, score: float) -> ConfidenceLevel:
        """تحديد مستوى الثقة بناءً على النقاط"""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_reasoning(self, factors: List[ConfidenceFactor], overall_score: float) -> str:
        """توليد تفسير لمستوى الثقة"""
        # العوامل الأقوى
        sorted_factors = sorted(factors, key=lambda f: f.score * f.weight, reverse=True)

        reasoning_parts = [
            f"Overall confidence: {overall_score*100:.1f}%"
        ]

        # أفضل 3 عوامل
        reasoning_parts.append("Strongest factors:")
        for factor in sorted_factors[:3]:
            reasoning_parts.append(f"  • {factor.name}: {factor.score*100:.1f}%")

        # أضعف عامل
        weakest = sorted_factors[-1]
        if weakest.score < 0.4:
            reasoning_parts.append(f"Weakest factor: {weakest.name} ({weakest.score*100:.1f}%)")

        return "\n".join(reasoning_parts)

    def should_proceed_with_decision(self, confidence: ConfidenceScore,
                                    min_threshold: float = 0.5) -> Tuple[bool, str]:
        """
        هل يجب المضي قدماً بالقرار؟

        Returns:
            (should_proceed, reason)
        """
        if confidence.overall_score >= min_threshold:
            return True, f"Confidence {confidence.overall_score*100:.1f}% meets threshold"

        # فحص العوامل الحرجة
        critical_factors = ["data_quality", "consistency", "internal_validation"]

        low_critical_factors = [
            f.name for f in confidence.factors
            if f.name in critical_factors and f.score < 0.4
        ]

        if low_critical_factors:
            return False, f"Critical factors too low: {', '.join(low_critical_factors)}"

        return False, f"Overall confidence {confidence.overall_score*100:.1f}% below threshold"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    print("📊 Confidence Measurement System - Test")
    print("=" * 70)

    scorer = ConfidenceScorer()

    # مثال 1: قرار بثقة عالية
    print("\n1️⃣ High confidence decision:")
    decision1 = {
        "result": "Paris is the capital of France",
        "sources": ["encyclopedia", "wikipedia", "official_source"],
        "validated": True,
        "validation_results": {"passed": 5, "total": 5}
    }

    context1 = {
        "input_quality": 0.9,
        "data_completeness": 0.95,
        "source_reliability": 0.9,
        "performance_history": [0.92, 0.89, 0.91]
    }

    confidence1 = scorer.calculate_confidence(decision1, context1)
    print(f"   Score: {confidence1.overall_score*100:.1f}%")
    print(f"   Level: {confidence1.level.value}")
    print(f"   {confidence1.reasoning}")

    should_proceed, reason = scorer.should_proceed_with_decision(confidence1)
    print(f"   Proceed: {should_proceed} - {reason}")

    # مثال 2: قرار بثقة منخفضة
    print("\n2️⃣ Low confidence decision:")
    decision2 = {
        "result": "The answer is unclear",
        "sources": [],
        "validated": False,
        "contradictions": ["conflict1", "conflict2"]
    }

    context2 = {
        "input_quality": 0.3,
        "data_completeness": 0.4
    }

    confidence2 = scorer.calculate_confidence(decision2, context2)
    print(f"   Score: {confidence2.overall_score*100:.1f}%")
    print(f"   Level: {confidence2.level.value}")
    print(f"   {confidence2.reasoning}")

    should_proceed, reason = scorer.should_proceed_with_decision(confidence2)
    print(f"   Proceed: {should_proceed} - {reason}")

    print("\n" + "=" * 70)
