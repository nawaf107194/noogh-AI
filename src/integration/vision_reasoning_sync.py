"""
Vision-Reasoning Synchronizer - موحّد الرؤية والتفكير
========================================================

يربط بين نظام الرؤية الحاسوبية ونظام التفكير المنطقي
لتحويل المعلومات البصرية إلى رموز معرفية قابلة للتحليل

Addresses Deep Cognition Q8 (Cross-modal validation)

Author: Noogh AI Team
Date: 2025-11-10
Priority: CRITICAL
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import os
import sys

# إضافة مسار المشروع
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModalityType(str, Enum):
    """نوع الوسيط"""
    VISION = "vision"
    TEXT = "text"
    REASONING = "reasoning"
    COMBINED = "combined"


class SyncStatus(str, Enum):
    """حالة التزامن"""
    ALIGNED = "aligned"  # متوافق
    PARTIAL_CONFLICT = "partial_conflict"  # تعارض جزئي
    MAJOR_CONFLICT = "major_conflict"  # تعارض كبير
    UNSYNCHRONIZED = "unsynchronized"  # غير متزامن


@dataclass
class VisualConcept:
    """مفهوم بصري"""
    concept_type: str  # "object", "scene", "text", "pattern"
    label: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningConcept:
    """مفهوم منطقي"""
    concept_type: str  # "entity", "relation", "action", "state"
    label: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class SyncResult:
    """نتيجة التزامن"""
    vision_concepts: List[VisualConcept]
    reasoning_concepts: List[ReasoningConcept]
    text_context: Optional[str]

    # التحليل
    sync_status: SyncStatus
    alignment_score: float  # 0.0-1.0
    conflicts: List[str]
    resolved_concepts: List[str]

    # التوصيات
    recommendation: str
    confidence_adjustment: float  # تعديل الثقة النهائي

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vision_concepts": [
                {
                    "type": v.concept_type,
                    "label": v.label,
                    "confidence": v.confidence,
                    "attributes": v.attributes
                }
                for v in self.vision_concepts
            ],
            "reasoning_concepts": [
                {
                    "type": r.concept_type,
                    "label": r.label,
                    "confidence": r.confidence,
                    "evidence": r.evidence
                }
                for r in self.reasoning_concepts
            ],
            "text_context": self.text_context,
            "sync_status": self.sync_status.value,
            "alignment_score": self.alignment_score,
            "conflicts": self.conflicts,
            "resolved_concepts": self.resolved_concepts,
            "recommendation": self.recommendation,
            "confidence_adjustment": self.confidence_adjustment
        }


class VisionReasoningSynchronizer:
    """
    موحّد الرؤية والتفكير

    يقوم بـ:
    1. استخراج مفاهيم بصرية من الصور (via image_analyzer + OCR)
    2. استخراج مفاهيم منطقية من النصوص (via NLP)
    3. مقارنة وموازنة المفاهيم
    4. كشف التعارضات
    5. حل التعارضات أو الإبلاغ عنها
    """

    def __init__(self):
        self.project_root = "/home/noogh/projects/noogh_unified_system"

        # محاولة تحميل الأنظمة
        self.vision_available = self._check_vision_systems()
        self.reasoning_available = self._check_reasoning_systems()
        self.nlp_available = self._check_nlp_systems()

        # إحصائيات
        self.total_syncs = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0

    def _check_vision_systems(self) -> bool:
        """فحص توفر أنظمة الرؤية"""
        return (
            os.path.exists(f"{self.project_root}/core/vision/image_analyzer.py") and
            os.path.exists(f"{self.project_root}/core/vision/ocr_engine.py")
        )

    def _check_reasoning_systems(self) -> bool:
        """فحص توفر أنظمة التفكير"""
        return os.path.exists(f"{self.project_root}/core/reasoning")

    def _check_nlp_systems(self) -> bool:
        """فحص توفر أنظمة NLP"""
        return os.path.exists(f"{self.project_root}/core/nlp")

    def synchronize(self,
                   image_path: Optional[str] = None,
                   text_description: Optional[str] = None,
                   expected_labels: Optional[List[str]] = None) -> SyncResult:
        """
        تزامن بين الرؤية والتفكير

        Args:
            image_path: مسار الصورة (اختياري)
            text_description: وصف نصي (اختياري)
            expected_labels: تسميات متوقعة للتحقق (اختياري)

        Returns:
            SyncResult
        """
        self.total_syncs += 1

        # 1. استخراج مفاهيم بصرية
        vision_concepts = self._extract_visual_concepts(image_path) if image_path else []

        # 2. استخراج مفاهيم منطقية من النص
        reasoning_concepts = self._extract_reasoning_concepts(text_description) if text_description else []

        # 3. التحقق من expected_labels
        if expected_labels:
            label_concepts = [
                ReasoningConcept(
                    concept_type="entity",
                    label=label,
                    confidence=0.9,
                    evidence=["user_provided_label"]
                )
                for label in expected_labels
            ]
            reasoning_concepts.extend(label_concepts)

        # 4. مقارنة المفاهيم
        alignment_score, conflicts = self._compare_concepts(vision_concepts, reasoning_concepts)

        # 5. تحديد حالة التزامن
        sync_status = self._determine_sync_status(alignment_score, len(conflicts))

        # 6. حل التعارضات (إن أمكن)
        resolved_concepts, confidence_adjustment = self._resolve_conflicts(
            vision_concepts, reasoning_concepts, conflicts
        )

        if resolved_concepts:
            self.conflicts_resolved += len(resolved_concepts)

        if conflicts:
            self.conflicts_detected += len(conflicts)

        # 7. توليد توصية
        recommendation = self._generate_recommendation(sync_status, conflicts, resolved_concepts)

        return SyncResult(
            vision_concepts=vision_concepts,
            reasoning_concepts=reasoning_concepts,
            text_context=text_description,
            sync_status=sync_status,
            alignment_score=alignment_score,
            conflicts=conflicts,
            resolved_concepts=resolved_concepts,
            recommendation=recommendation,
            confidence_adjustment=confidence_adjustment
        )

    def _extract_visual_concepts(self, image_path: str) -> List[VisualConcept]:
        """استخراج مفاهيم من صورة"""
        if not self.vision_available:
            return []

        concepts = []

        try:
            # استخدام Image Analyzer
            from vision.image_analyzer import ImageAnalyzer

            analyzer = ImageAnalyzer()
            result = analyzer.analyze_image(image_path)

            # تحويل لمفاهيم
            # 1. نوع الصورة
            concepts.append(VisualConcept(
                concept_type="scene",
                label=result.image_type.value,
                confidence=0.8,
                attributes={
                    "dimensions": f"{result.dimensions.width}x{result.dimensions.height}",
                    "orientation": result.dimensions.orientation
                }
            ))

            # 2. نظام الألوان
            concepts.append(VisualConcept(
                concept_type="pattern",
                label=f"color_scheme_{result.colors.color_scheme.value}",
                confidence=0.7,
                attributes={
                    "brightness": result.colors.brightness_avg,
                    "diversity": result.colors.color_diversity
                }
            ))

            # 3. المحتوى
            if result.content.has_text:
                concepts.append(VisualConcept(
                    concept_type="text",
                    label="text_detected",
                    confidence=0.8 if result.content.text_density > 0.3 else 0.5
                ))

        except Exception as e:
            # في حالة فشل التحليل
            concepts.append(VisualConcept(
                concept_type="error",
                label=f"analysis_failed: {str(e)}",
                confidence=0.0
            ))

        # محاولة OCR
        try:
            from vision.ocr_engine import OCREngine

            ocr = OCREngine()
            ocr_result = ocr.extract_text(image_path)

            if ocr_result.full_text.strip():
                concepts.append(VisualConcept(
                    concept_type="text",
                    label="ocr_extracted",
                    confidence=ocr_result.average_confidence,
                    attributes={
                        "text": ocr_result.full_text[:100],  # أول 100 حرف
                        "languages": ocr_result.detected_languages
                    }
                ))

        except Exception:
            pass

        return concepts

    def _extract_reasoning_concepts(self, text: str) -> List[ReasoningConcept]:
        """استخراج مفاهيم منطقية من نص"""
        if not self.nlp_available:
            return []

        concepts = []

        # تحليل بسيط (في نظام كامل، نستخدم NER + Dependency Parsing)
        text_lower = text.lower()

        # كشف كيانات بسيط
        entities = [
            ("dog", "animal"),
            ("cat", "animal"),
            ("car", "vehicle"),
            ("tree", "plant"),
            ("house", "building"),
            ("كلب", "animal"),
            ("قطة", "animal"),
            ("سيارة", "vehicle")
        ]

        for entity, entity_type in entities:
            if entity in text_lower:
                concepts.append(ReasoningConcept(
                    concept_type="entity",
                    label=entity,
                    confidence=0.8,
                    evidence=[f"mentioned in text: '{entity}'"]
                ))

        # كشف أفعال (actions)
        actions = ["running", "walking", "sitting", "يركض", "يمشي", "يجلس"]
        for action in actions:
            if action in text_lower:
                concepts.append(ReasoningConcept(
                    concept_type="action",
                    label=action,
                    confidence=0.7,
                    evidence=[f"action detected: '{action}'"]
                ))

        return concepts

    def _compare_concepts(self,
                         vision: List[VisualConcept],
                         reasoning: List[ReasoningConcept]) -> Tuple[float, List[str]]:
        """
        مقارنة المفاهيم البصرية والمنطقية

        Returns:
            (alignment_score, conflicts)
        """
        if not vision and not reasoning:
            return 1.0, []

        if not vision or not reasoning:
            return 0.5, ["Partial data: only one modality available"]

        conflicts = []

        # البحث عن تطابق
        vision_labels = set(v.label.lower() for v in vision)
        reasoning_labels = set(r.label.lower() for r in reasoning)

        # حساب Jaccard similarity
        intersection = vision_labels & reasoning_labels
        union = vision_labels | reasoning_labels

        if union:
            alignment = len(intersection) / len(union)
        else:
            alignment = 0.0

        # كشف تعارضات واضحة
        # مثال: النص يقول "dog" لكن الصورة تُظهر "cat"
        contradictory_pairs = [
            ("dog", "cat"),
            ("كلب", "قطة"),
            ("car", "bicycle"),
            ("day", "night")
        ]

        for v_concept in vision:
            for r_concept in reasoning:
                for pair in contradictory_pairs:
                    if (pair[0] in v_concept.label.lower() and pair[1] in r_concept.label.lower()) or \
                       (pair[1] in v_concept.label.lower() and pair[0] in r_concept.label.lower()):
                        conflicts.append(
                            f"Contradiction: Vision shows '{v_concept.label}' but text mentions '{r_concept.label}'"
                        )

        return alignment, conflicts

    def _determine_sync_status(self, alignment: float, conflict_count: int) -> SyncStatus:
        """تحديد حالة التزامن"""
        if conflict_count > 0:
            if alignment < 0.3:
                return SyncStatus.MAJOR_CONFLICT
            else:
                return SyncStatus.PARTIAL_CONFLICT

        if alignment >= 0.7:
            return SyncStatus.ALIGNED
        else:
            return SyncStatus.UNSYNCHRONIZED

    def _resolve_conflicts(self,
                          vision: List[VisualConcept],
                          reasoning: List[ReasoningConcept],
                          conflicts: List[str]) -> Tuple[List[str], float]:
        """
        محاولة حل التعارضات

        Returns:
            (resolved_concepts, confidence_adjustment)
        """
        resolved = []
        confidence_adjustment = 1.0

        if not conflicts:
            return resolved, confidence_adjustment

        # استراتيجية الحل:
        # 1. إذا كانت ثقة الرؤية أعلى، نفضل الرؤية
        # 2. إذا كانت ثقة التفكير أعلى، نفضل التفكير
        # 3. إذا متساوية، نُخفّض الثقة ونُبلغ عن التعارض

        for conflict in conflicts:
            # استخراج المفاهيم المتعارضة
            # (في نظام حقيقي، parsing أدق)
            if "Vision shows" in conflict and "text mentions" in conflict:
                # تخفيض الثقة
                confidence_adjustment *= 0.7
                resolved.append(f"Unresolved: {conflict} - confidence reduced")

        return resolved, confidence_adjustment

    def _generate_recommendation(self,
                                status: SyncStatus,
                                conflicts: List[str],
                                resolved: List[str]) -> str:
        """توليد توصية"""
        if status == SyncStatus.ALIGNED:
            return "Proceed - vision and reasoning are aligned"

        if status == SyncStatus.MAJOR_CONFLICT:
            return f"Do not proceed - major conflicts detected: {len(conflicts)} issue(s)"

        if status == SyncStatus.PARTIAL_CONFLICT:
            if resolved:
                return f"Proceed with caution - {len(resolved)} conflict(s) resolved"
            else:
                return f"Review required - {len(conflicts)} unresolved conflict(s)"

        return "Insufficient data - unable to synchronize modalities"

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات النظام"""
        return {
            "total_syncs": self.total_syncs,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "resolution_rate": (self.conflicts_resolved / self.conflicts_detected * 100) if self.conflicts_detected > 0 else 0.0,
            "vision_available": self.vision_available,
            "reasoning_available": self.reasoning_available,
            "nlp_available": self.nlp_available
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🔗 Vision-Reasoning Synchronizer - Test")
    print("=" * 70)
    print()

    sync = VisionReasoningSynchronizer()

    print(f"Vision Systems: {'✅ Available' if sync.vision_available else '❌ Not available'}")
    print(f"Reasoning Systems: {'✅ Available' if sync.reasoning_available else '❌ Not available'}")
    print(f"NLP Systems: {'✅ Available' if sync.nlp_available else '❌ Not available'}")
    print()

    # Test Case: تزامن نص فقط (بدون صورة)
    print("1️⃣ Test: Text-only synchronization")
    print("-" * 70)

    result = sync.synchronize(
        text_description="A dog is running in the park",
        expected_labels=["dog", "park", "running"]
    )

    print(f"Sync Status: {result.sync_status.value}")
    print(f"Alignment Score: {result.alignment_score:.2f}")
    print(f"Reasoning Concepts: {len(result.reasoning_concepts)}")
    print(f"Conflicts: {len(result.conflicts)}")
    print(f"Recommendation: {result.recommendation}")
    print()

    # Test Case: تعارض
    print("2️⃣ Test: Conflict detection")
    print("-" * 70)

    result2 = sync.synchronize(
        text_description="There is a cat sitting on the couch",
        expected_labels=["dog"]  # تعارض: النص يقول cat، التسمية dog
    )

    print(f"Sync Status: {result2.sync_status.value}")
    print(f"Alignment Score: {result2.alignment_score:.2f}")
    print(f"Conflicts Detected: {len(result2.conflicts)}")
    if result2.conflicts:
        for conflict in result2.conflicts:
            print(f"  ⚠️ {conflict}")
    print(f"Confidence Adjustment: {result2.confidence_adjustment:.2f}")
    print(f"Recommendation: {result2.recommendation}")
    print()

    print("=" * 70)
    stats = sync.get_statistics()
    print("📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    print("✅ Vision-Reasoning Synchronizer Test Complete!")
    print()
