"""
Meta-Confidence Calibration System - نظام معايرة الثقة الذاتية
==================================================================

يقيس ثقة النظام في قراراته بناءً على عوامل متعددة ويُنتج
Certainty Index لكل مهمة

Addresses Deep Cognition Q16 (Meta-confidence calibration)

Author: Noogh AI Team
Date: 2025-11-10
Priority: CRITICAL
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math


class CertaintyLevel(str, Enum):
    """مستوى اليقين"""
    ABSOLUTE = "absolute"  # 95%+ - يقين مطلق
    HIGH = "high"  # 85-94% - ثقة عالية
    MODERATE = "moderate"  # 70-84% - ثقة متوسطة
    LOW = "low"  # 50-69% - ثقة منخفضة
    UNCERTAIN = "uncertain"  # 30-49% - غير متأكد
    VERY_UNCERTAIN = "very_uncertain"  # 0-29% - شك كبير


class ConfidenceSource(str, Enum):
    """مصدر الثقة"""
    DATA_QUALITY = "data_quality"  # جودة البيانات
    MODEL_AGREEMENT = "model_agreement"  # توافق النماذج
    HISTORICAL_ACCURACY = "historical_accuracy"  # الدقة التاريخية
    CONTEXT_CLARITY = "context_clarity"  # وضوح السياق
    REASONING_DEPTH = "reasoning_depth"  # عمق التفكير
    CROSS_VALIDATION = "cross_validation"  # التحقق المتقاطع


@dataclass
class ConfidenceFactor:
    """عامل ثقة واحد"""
    source: ConfidenceSource
    score: float  # 0.0-1.0
    weight: float  # 0.0-1.0
    evidence: str


@dataclass
class CertaintyIndex:
    """مؤشر اليقين الكامل"""
    overall_confidence: float  # 0.0-1.0
    certainty_level: CertaintyLevel
    factors: List[ConfidenceFactor]
    calibration_quality: float  # مدى دقة المعايرة
    recommendation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": self.overall_confidence,
            "certainty_level": self.certainty_level.value,
            "factors": [
                {
                    "source": f.source.value,
                    "score": f.score,
                    "weight": f.weight,
                    "evidence": f.evidence
                }
                for f in self.factors
            ],
            "calibration_quality": self.calibration_quality,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat()
        }


class MetaConfidenceCalibrator:
    """
    معايِر الثقة الذاتية

    يحسب مؤشر اليقين (Certainty Index) لكل قرار بناءً على:
    1. جودة البيانات المدخلة
    2. توافق النماذج المختلفة
    3. الدقة التاريخية
    4. وضوح السياق
    5. عمق التفكير
    6. التحقق المتقاطع
    """

    def __init__(self):
        # أوزان العوامل الافتراضية
        self.default_weights = {
            ConfidenceSource.DATA_QUALITY: 0.25,
            ConfidenceSource.MODEL_AGREEMENT: 0.20,
            ConfidenceSource.HISTORICAL_ACCURACY: 0.15,
            ConfidenceSource.CONTEXT_CLARITY: 0.15,
            ConfidenceSource.REASONING_DEPTH: 0.15,
            ConfidenceSource.CROSS_VALIDATION: 0.10
        }

        # سجل القرارات السابقة (للتعلم من الأخطاء)
        self.decision_history: List[Dict[str, Any]] = []

        # إحصائيات المعايرة
        self.total_decisions = 0
        self.calibration_errors: List[float] = []  # فرق بين الثقة المتوقعة والفعلية

    def calculate_certainty(self,
                           data_quality: Optional[float] = None,
                           model_agreement: Optional[float] = None,
                           historical_accuracy: Optional[float] = None,
                           context_clarity: Optional[float] = None,
                           reasoning_depth: Optional[float] = None,
                           cross_validation: Optional[float] = None,
                           custom_weights: Optional[Dict[ConfidenceSource, float]] = None) -> CertaintyIndex:
        """
        حساب مؤشر اليقين

        Args:
            data_quality: جودة البيانات (0.0-1.0)
            model_agreement: توافق النماذج (0.0-1.0)
            historical_accuracy: الدقة التاريخية (0.0-1.0)
            context_clarity: وضوح السياق (0.0-1.0)
            reasoning_depth: عمق التفكير (0.0-1.0)
            cross_validation: التحقق المتقاطع (0.0-1.0)
            custom_weights: أوزان مخصصة (اختياري)

        Returns:
            CertaintyIndex
        """
        weights = custom_weights or self.default_weights

        # جمع العوامل المتوفرة
        factors: List[ConfidenceFactor] = []

        if data_quality is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.DATA_QUALITY,
                score=data_quality,
                weight=weights[ConfidenceSource.DATA_QUALITY],
                evidence=self._get_data_quality_evidence(data_quality)
            ))

        if model_agreement is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.MODEL_AGREEMENT,
                score=model_agreement,
                weight=weights[ConfidenceSource.MODEL_AGREEMENT],
                evidence=self._get_model_agreement_evidence(model_agreement)
            ))

        if historical_accuracy is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.HISTORICAL_ACCURACY,
                score=historical_accuracy,
                weight=weights[ConfidenceSource.HISTORICAL_ACCURACY],
                evidence=self._get_historical_accuracy_evidence(historical_accuracy)
            ))

        if context_clarity is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.CONTEXT_CLARITY,
                score=context_clarity,
                weight=weights[ConfidenceSource.CONTEXT_CLARITY],
                evidence=self._get_context_clarity_evidence(context_clarity)
            ))

        if reasoning_depth is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.REASONING_DEPTH,
                score=reasoning_depth,
                weight=weights[ConfidenceSource.REASONING_DEPTH],
                evidence=self._get_reasoning_depth_evidence(reasoning_depth)
            ))

        if cross_validation is not None:
            factors.append(ConfidenceFactor(
                source=ConfidenceSource.CROSS_VALIDATION,
                score=cross_validation,
                weight=weights[ConfidenceSource.CROSS_VALIDATION],
                evidence=self._get_cross_validation_evidence(cross_validation)
            ))

        # حساب الثقة الإجمالية (weighted average)
        if factors:
            total_weight = sum(f.weight for f in factors)
            overall_confidence = sum(f.score * f.weight for f in factors) / total_weight if total_weight > 0 else 0.0
        else:
            overall_confidence = 0.5  # قيمة افتراضية عند عدم توفر بيانات

        # تطبيق معايرة ديناميكية (تعلم من الأخطاء السابقة)
        calibrated_confidence = self._apply_calibration(overall_confidence, factors)

        # تحديد مستوى اليقين
        certainty_level = self._determine_certainty_level(calibrated_confidence)

        # حساب جودة المعايرة
        calibration_quality = self._assess_calibration_quality(factors)

        # توليد توصية
        recommendation = self._generate_recommendation(calibrated_confidence, certainty_level, factors)

        index = CertaintyIndex(
            overall_confidence=calibrated_confidence,
            certainty_level=certainty_level,
            factors=factors,
            calibration_quality=calibration_quality,
            recommendation=recommendation
        )

        self.total_decisions += 1
        return index

    def _apply_calibration(self, raw_confidence: float, factors: List[ConfidenceFactor]) -> float:
        """
        تطبيق معايرة ديناميكية بناءً على الأخطاء السابقة

        يستخدم calibration_errors لتعديل الثقة
        """
        if not self.calibration_errors:
            return raw_confidence

        # حساب متوسط خطأ المعايرة
        avg_error = sum(self.calibration_errors) / len(self.calibration_errors)

        # تطبيق تصحيح بسيط
        # إذا كنا دائماً نبالغ في الثقة (avg_error > 0)، نخفض الثقة
        # إذا كنا دائماً نقلل من الثقة (avg_error < 0)، نرفع الثقة
        correction = -avg_error * 0.1  # تصحيح بنسبة 10%

        calibrated = raw_confidence + correction

        # Clamp to [0, 1]
        return max(0.0, min(1.0, calibrated))

    def _determine_certainty_level(self, confidence: float) -> CertaintyLevel:
        """تحديد مستوى اليقين"""
        if confidence >= 0.95:
            return CertaintyLevel.ABSOLUTE
        elif confidence >= 0.85:
            return CertaintyLevel.HIGH
        elif confidence >= 0.70:
            return CertaintyLevel.MODERATE
        elif confidence >= 0.50:
            return CertaintyLevel.LOW
        elif confidence >= 0.30:
            return CertaintyLevel.UNCERTAIN
        else:
            return CertaintyLevel.VERY_UNCERTAIN

    def _assess_calibration_quality(self, factors: List[ConfidenceFactor]) -> float:
        """
        تقييم جودة المعايرة

        كلما زاد عدد العوامل المتوفرة، زادت جودة المعايرة
        """
        available_factors = len(factors)
        total_factors = len(ConfidenceSource)

        coverage = available_factors / total_factors

        # حساب تباين النقاط (كلما قل التباين، زادت الجودة)
        if len(factors) >= 2:
            scores = [f.score for f in factors]
            variance = sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            consistency = 1.0 - min(variance, 1.0)
        else:
            consistency = 0.5

        # جودة المعايرة = (coverage * 0.6) + (consistency * 0.4)
        quality = (coverage * 0.6) + (consistency * 0.4)

        return quality

    def _generate_recommendation(self,
                                 confidence: float,
                                 level: CertaintyLevel,
                                 factors: List[ConfidenceFactor]) -> str:
        """توليد توصية بناءً على مستوى اليقين"""
        if level == CertaintyLevel.ABSOLUTE:
            return "Proceed with high confidence - decision is well-calibrated"
        elif level == CertaintyLevel.HIGH:
            return "Proceed - confidence is strong"
        elif level == CertaintyLevel.MODERATE:
            return "Proceed with caution - consider validation"
        elif level == CertaintyLevel.LOW:
            return "Review decision - seek additional evidence"
        elif level == CertaintyLevel.UNCERTAIN:
            return "High uncertainty - request clarification or more data"
        else:
            return "Do not proceed - confidence too low, re-evaluate approach"

    def record_outcome(self, predicted_confidence: float, actual_success: bool):
        """
        تسجيل نتيجة قرار فعلي لتحسين المعايرة

        Args:
            predicted_confidence: الثقة المتوقعة (0.0-1.0)
            actual_success: هل نجح القرار فعلياً؟
        """
        actual_confidence = 1.0 if actual_success else 0.0
        error = predicted_confidence - actual_confidence

        self.calibration_errors.append(error)

        # الاحتفاظ بآخر 100 خطأ فقط
        if len(self.calibration_errors) > 100:
            self.calibration_errors.pop(0)

        self.decision_history.append({
            "predicted": predicted_confidence,
            "actual": actual_confidence,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def get_calibration_stats(self) -> Dict[str, Any]:
        """إحصائيات المعايرة"""
        if not self.calibration_errors:
            return {
                "total_decisions": self.total_decisions,
                "calibration_quality": "no_data",
                "mean_error": 0.0,
                "rmse": 0.0
            }

        mean_error = sum(self.calibration_errors) / len(self.calibration_errors)
        rmse = math.sqrt(sum(e ** 2 for e in self.calibration_errors) / len(self.calibration_errors))

        # تحديد جودة المعايرة بناءً على RMSE
        if rmse < 0.1:
            quality = "excellent"
        elif rmse < 0.2:
            quality = "good"
        elif rmse < 0.3:
            quality = "acceptable"
        else:
            quality = "needs_improvement"

        return {
            "total_decisions": self.total_decisions,
            "calibration_quality": quality,
            "mean_error": mean_error,
            "rmse": rmse,
            "recent_errors": self.calibration_errors[-10:]
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Evidence Generation (helper methods)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_data_quality_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Data quality: excellent - complete and validated"
        elif score >= 0.7:
            return "Data quality: good - mostly complete"
        elif score >= 0.5:
            return "Data quality: acceptable - some gaps exist"
        else:
            return "Data quality: poor - significant gaps"

    def _get_model_agreement_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Model agreement: strong consensus across all models"
        elif score >= 0.7:
            return "Model agreement: majority consensus"
        elif score >= 0.5:
            return "Model agreement: partial consensus"
        else:
            return "Model agreement: significant disagreement"

    def _get_historical_accuracy_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Historical accuracy: excellent track record"
        elif score >= 0.7:
            return "Historical accuracy: good performance"
        elif score >= 0.5:
            return "Historical accuracy: moderate performance"
        else:
            return "Historical accuracy: poor track record"

    def _get_context_clarity_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Context clarity: fully clear and unambiguous"
        elif score >= 0.7:
            return "Context clarity: mostly clear"
        elif score >= 0.5:
            return "Context clarity: somewhat ambiguous"
        else:
            return "Context clarity: highly ambiguous"

    def _get_reasoning_depth_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Reasoning depth: comprehensive multi-step analysis"
        elif score >= 0.7:
            return "Reasoning depth: solid analysis"
        elif score >= 0.5:
            return "Reasoning depth: basic analysis"
        else:
            return "Reasoning depth: shallow analysis"

    def _get_cross_validation_evidence(self, score: float) -> str:
        if score >= 0.9:
            return "Cross-validation: verified across multiple sources"
        elif score >= 0.7:
            return "Cross-validation: verified with some sources"
        elif score >= 0.5:
            return "Cross-validation: limited verification"
        else:
            return "Cross-validation: no verification"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🎯 Meta-Confidence Calibration System - Test")
    print("=" * 70)
    print()

    calibrator = MetaConfidenceCalibrator()

    # مثال 1: ثقة عالية (جميع العوامل قوية)
    print("1️⃣ High Confidence Decision:")
    print("-" * 70)

    index = calibrator.calculate_certainty(
        data_quality=0.95,
        model_agreement=0.90,
        historical_accuracy=0.88,
        context_clarity=0.92,
        reasoning_depth=0.85,
        cross_validation=0.90
    )

    print(f"Overall Confidence: {index.overall_confidence * 100:.1f}%")
    print(f"Certainty Level: {index.certainty_level.value}")
    print(f"Calibration Quality: {index.calibration_quality:.2f}")
    print(f"Recommendation: {index.recommendation}")
    print()

    # مثال 2: ثقة منخفضة (عوامل ضعيفة)
    print("2️⃣ Low Confidence Decision:")
    print("-" * 70)

    index2 = calibrator.calculate_certainty(
        data_quality=0.45,
        model_agreement=0.40,
        context_clarity=0.35
    )

    print(f"Overall Confidence: {index2.overall_confidence * 100:.1f}%")
    print(f"Certainty Level: {index2.certainty_level.value}")
    print(f"Calibration Quality: {index2.calibration_quality:.2f}")
    print(f"Recommendation: {index2.recommendation}")
    print()

    # تسجيل نتائج (للتعلم)
    print("3️⃣ Recording Outcomes:")
    print("-" * 70)

    calibrator.record_outcome(predicted_confidence=0.90, actual_success=True)
    calibrator.record_outcome(predicted_confidence=0.85, actual_success=True)
    calibrator.record_outcome(predicted_confidence=0.70, actual_success=False)

    stats = calibrator.get_calibration_stats()
    print(f"Total Decisions: {stats['total_decisions']}")
    print(f"Calibration Quality: {stats['calibration_quality']}")
    print(f"Mean Error: {stats['mean_error']:.3f}")
    print(f"RMSE: {stats['rmse']:.3f}")
    print()

    print("=" * 70)
    print("✅ Meta-Confidence Calibration System Test Complete!")
    print()
