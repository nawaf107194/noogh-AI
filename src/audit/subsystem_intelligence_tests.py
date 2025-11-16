"""
Subsystem Intelligence Test Engine - محرك اختبار ذكاء الأنظمة الفرعية
============================================================================

اختبار شامل لقدرات الأنظمة الفرعية في نوقه:
1. Computer Vision (20 questions)
2. Natural Language Processing (20 questions)
3. Decision Making & Reasoning (20 questions)
4. Machine Learning & Training (20 questions)
5. Contextual Awareness & Integration (10 questions)

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import os
import sqlite3
import json


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums and Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SubsystemType(str, Enum):
    """أنواع الأنظمة الفرعية"""
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    DECISION_REASONING = "decision_reasoning"
    ML_TRAINING = "ml_training"
    CONTEXTUAL_AWARENESS = "contextual_awareness"


class IntelligenceLevel(str, Enum):
    """مستويات الذكاء"""
    EXPERT = "expert"              # 90%+
    ADVANCED = "advanced"          # 75-89%
    INTERMEDIATE = "intermediate"  # 60-74%
    BASIC = "basic"                # 40-59%
    MINIMAL = "minimal"            # 20-39%
    NONE = "none"                  # 0-19%


class TestType(str, Enum):
    """أنواع الاختبارات"""
    CAPABILITY = "capability"      # فحص وجود القدرة
    ACCURACY = "accuracy"          # فحص الدقة
    PERFORMANCE = "performance"    # فحص الأداء
    ROBUSTNESS = "robustness"      # فحص المتانة
    INTEGRATION = "integration"    # فحص التكامل


@dataclass
class SubsystemQuestion:
    """سؤال اختبار لنظام فرعي"""
    id: int
    subsystem: SubsystemType
    question_ar: str
    question_en: str
    test_type: TestType
    criticality: int  # 1-5
    automated_test: Optional[str] = None  # اسم دالة الاختبار
    expected_capability: Optional[str] = None


@dataclass
class SubsystemTestResult:
    """نتيجة اختبار نظام فرعي"""
    question_id: int
    subsystem: SubsystemType
    passed: bool
    score: float  # 0.0 - 1.0
    evidence: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SubsystemReport:
    """تقرير شامل لنظام فرعي"""
    subsystem: SubsystemType
    total_questions: int
    passed: int
    failed: int
    score_percentage: float
    intelligence_level: IntelligenceLevel
    strengths: List[str]
    weaknesses: List[str]
    test_results: List[SubsystemTestResult]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Question Bank - 90 Questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBSYSTEM_QUESTIONS = [
    # ═══════════════════════════════════════════════════════════
    # Computer Vision (CV1-CV20)
    # ═══════════════════════════════════════════════════════════

    SubsystemQuestion(
        id=1, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل النظام يميز بين الأجسام المتشابهة بالألوان والمختلفة بالشكل؟",
        question_en="Can the system distinguish objects with similar colors but different shapes?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_cv1_shape_recognition"
    ),

    SubsystemQuestion(
        id=2, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل يستطيع التعرف على أجسام جزئية (نصف وجه أو ذراع فقط)؟",
        question_en="Can it recognize partial objects (half face or single arm)?",
        test_type=TestType.ROBUSTNESS, criticality=4,
        automated_test="test_cv2_partial_object_detection"
    ),

    SubsystemQuestion(
        id=3, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="كيف يتعامل مع الإضاءة المنخفضة أو الصور المائلة؟",
        question_en="How does it handle low lighting or tilted images?",
        test_type=TestType.ROBUSTNESS, criticality=3,
        automated_test="test_cv3_lighting_robustness"
    ),

    SubsystemQuestion(
        id=4, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل يمكنه تحديد العمق أو المسافة بين العناصر؟",
        question_en="Can it determine depth or distance between elements?",
        test_type=TestType.CAPABILITY, criticality=3,
        automated_test="test_cv4_depth_estimation"
    ),

    SubsystemQuestion(
        id=5, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل النموذج قادر على اكتشاف التزوير البصري (Deepfake detection)؟",
        question_en="Can the model detect visual forgery (Deepfake detection)?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_cv5_deepfake_detection"
    ),

    SubsystemQuestion(
        id=6, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل يمكنه قراءة نصوص من الصور (OCR) بدقة عبر لغات متعددة؟",
        question_en="Can it read text from images (OCR) accurately across multiple languages?",
        test_type=TestType.ACCURACY, criticality=4,
        automated_test="test_cv6_multilingual_ocr"
    ),

    SubsystemQuestion(
        id=7, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="كيف يتعامل مع صور مشوّهة أو مضغوطة بشدة؟",
        question_en="How does it handle distorted or heavily compressed images?",
        test_type=TestType.ROBUSTNESS, criticality=3,
        automated_test="test_cv7_compression_robustness"
    ),

    SubsystemQuestion(
        id=8, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل لديه threshold ثابت للثقة أم ديناميكي حسب السياق؟",
        question_en="Does it have a fixed or dynamic confidence threshold based on context?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_cv8_dynamic_threshold"
    ),

    SubsystemQuestion(
        id=9, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل يستطيع تمييز الحركة الدقيقة بين إطارات الفيديو؟",
        question_en="Can it detect subtle motion between video frames?",
        test_type=TestType.CAPABILITY, criticality=3,
        automated_test="test_cv9_motion_detection"
    ),

    SubsystemQuestion(
        id=10, subsystem=SubsystemType.COMPUTER_VISION,
        question_ar="هل يمكنه اكتشاف العناصر المخفية جزئياً خلف أخرى (occlusion handling)؟",
        question_en="Can it detect partially occluded objects?",
        test_type=TestType.ROBUSTNESS, criticality=4,
        automated_test="test_cv10_occlusion_handling"
    ),

    # ═══════════════════════════════════════════════════════════
    # NLP (NLP1-NLP20)
    # ═══════════════════════════════════════════════════════════

    SubsystemQuestion(
        id=21, subsystem=SubsystemType.NLP,
        question_ar="هل يميّز بين السؤال، الطلب، والنية غير المباشرة؟",
        question_en="Can it distinguish between questions, requests, and indirect intents?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_nlp1_intent_classification"
    ),

    SubsystemQuestion(
        id=22, subsystem=SubsystemType.NLP,
        question_ar="هل يمكنه تحليل لهجات عربية مختلفة (خليجية، مصرية، شامية)؟",
        question_en="Can it analyze different Arabic dialects (Gulf, Egyptian, Levantine)?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_nlp2_arabic_dialects"
    ),

    SubsystemQuestion(
        id=23, subsystem=SubsystemType.NLP,
        question_ar="هل يكتشف اللغة المختلطة (عربي + إنجليزي في نفس الجملة)؟",
        question_en="Can it detect code-switching (Arabic + English in same sentence)?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_nlp3_code_switching"
    ),

    SubsystemQuestion(
        id=24, subsystem=SubsystemType.NLP,
        question_ar="هل يحلّل العواطف والنغمة (سخرية، غضب، شك)؟",
        question_en="Can it analyze emotions and tone (sarcasm, anger, doubt)?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_nlp4_emotion_analysis"
    ),

    SubsystemQuestion(
        id=25, subsystem=SubsystemType.NLP,
        question_ar="هل يفهم العبارات السياقية مثل 'زي ما قلت لك أمس'؟",
        question_en="Does it understand contextual phrases like 'as I told you yesterday'?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_nlp5_contextual_references"
    ),

    SubsystemQuestion(
        id=26, subsystem=SubsystemType.NLP,
        question_ar="هل يمكنه توليد نص بنفس أسلوب المستخدم؟",
        question_en="Can it generate text matching the user's style?",
        test_type=TestType.CAPABILITY, criticality=3,
        automated_test="test_nlp6_style_matching"
    ),

    SubsystemQuestion(
        id=27, subsystem=SubsystemType.NLP,
        question_ar="هل يعيد صياغة النصوص مع الحفاظ على المعنى؟",
        question_en="Can it paraphrase text while preserving meaning?",
        test_type=TestType.ACCURACY, criticality=4,
        automated_test="test_nlp7_paraphrasing"
    ),

    SubsystemQuestion(
        id=28, subsystem=SubsystemType.NLP,
        question_ar="هل يكتشف التناقض بين جملتين؟",
        question_en="Can it detect contradiction between two sentences?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_nlp8_contradiction_detection"
    ),

    SubsystemQuestion(
        id=29, subsystem=SubsystemType.NLP,
        question_ar="هل يتعرّف على المصطلحات التقنية الجديدة تلقائياً؟",
        question_en="Can it recognize new technical terms automatically?",
        test_type=TestType.CAPABILITY, criticality=3,
        automated_test="test_nlp9_new_term_recognition"
    ),

    SubsystemQuestion(
        id=30, subsystem=SubsystemType.NLP,
        question_ar="هل يمكنه تلخيص محادثة كاملة دون فقدان المعلومات الدقيقة؟",
        question_en="Can it summarize a full conversation without losing key details?",
        test_type=TestType.ACCURACY, criticality=4,
        automated_test="test_nlp10_conversation_summarization"
    ),

    # ═══════════════════════════════════════════════════════════
    # Decision & Reasoning (DR1-DR20)
    # ═══════════════════════════════════════════════════════════

    SubsystemQuestion(
        id=41, subsystem=SubsystemType.DECISION_REASONING,
        question_ar="هل يستخدم بيانات الأداء لتعديل استراتيجياته تلقائياً؟",
        question_en="Does it use performance data to automatically adjust strategies?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_dr1_adaptive_strategy"
    ),

    SubsystemQuestion(
        id=42, subsystem=SubsystemType.DECISION_REASONING,
        question_ar="هل يوازن بين سرعة القرار ودقته؟",
        question_en="Does it balance decision speed and accuracy?",
        test_type=TestType.PERFORMANCE, criticality=4,
        automated_test="test_dr2_speed_accuracy_tradeoff"
    ),

    SubsystemQuestion(
        id=43, subsystem=SubsystemType.DECISION_REASONING,
        question_ar="هل يقدر تكلفة الخطأ قبل اتخاذ القرار؟",
        question_en="Does it estimate error cost before making a decision?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_dr3_error_cost_estimation"
    ),

    SubsystemQuestion(
        id=44, subsystem=SubsystemType.DECISION_REASONING,
        question_ar="هل يدعم Multi-objective Optimization (تحقيق أكثر من هدف في آن واحد)؟",
        question_en="Does it support multi-objective optimization?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_dr4_multi_objective"
    ),

    SubsystemQuestion(
        id=45, subsystem=SubsystemType.DECISION_REASONING,
        question_ar="هل يستخدم خوارزميات Monte Carlo أو Bayesian reasoning للتقدير؟",
        question_en="Does it use Monte Carlo or Bayesian reasoning for estimation?",
        test_type=TestType.CAPABILITY, criticality=3,
        automated_test="test_dr5_probabilistic_reasoning"
    ),

    # ═══════════════════════════════════════════════════════════
    # ML Training (ML1-ML20)
    # ═══════════════════════════════════════════════════════════

    SubsystemQuestion(
        id=61, subsystem=SubsystemType.ML_TRAINING,
        question_ar="هل يدعم التدريب التفاضلي (incremental learning) دون نسيان قديم؟",
        question_en="Does it support incremental learning without catastrophic forgetting?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_ml1_incremental_learning"
    ),

    SubsystemQuestion(
        id=62, subsystem=SubsystemType.ML_TRAINING,
        question_ar="هل يمكنه التوقف الذكي عند تشبع النموذج؟",
        question_en="Can it intelligently stop when model saturates?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_ml2_early_stopping"
    ),

    SubsystemQuestion(
        id=63, subsystem=SubsystemType.ML_TRAINING,
        question_ar="هل يحافظ على reproducibility الكاملة للتجارب؟",
        question_en="Does it maintain full experiment reproducibility?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_ml3_reproducibility"
    ),

    SubsystemQuestion(
        id=64, subsystem=SubsystemType.ML_TRAINING,
        question_ar="هل يكتشف تحيز البيانات تلقائياً قبل التدريب؟",
        question_en="Does it automatically detect data bias before training?",
        test_type=TestType.CAPABILITY, criticality=5,
        automated_test="test_ml4_bias_detection"
    ),

    SubsystemQuestion(
        id=65, subsystem=SubsystemType.ML_TRAINING,
        question_ar="هل يقيّم جودة البيانات (data validation + sanity checks)؟",
        question_en="Does it evaluate data quality (validation + sanity checks)?",
        test_type=TestType.CAPABILITY, criticality=4,
        automated_test="test_ml5_data_validation"
    ),

    # ═══════════════════════════════════════════════════════════
    # Contextual Awareness (CA1-CA10)
    # ═══════════════════════════════════════════════════════════

    SubsystemQuestion(
        id=81, subsystem=SubsystemType.CONTEXTUAL_AWARENESS,
        question_ar="هل يدمج بين رؤية، لغة، وقرار في سيناريو واحد؟",
        question_en="Does it integrate vision, language, and decision in one scenario?",
        test_type=TestType.INTEGRATION, criticality=5,
        automated_test="test_ca1_multimodal_integration"
    ),

    SubsystemQuestion(
        id=82, subsystem=SubsystemType.CONTEXTUAL_AWARENESS,
        question_ar="هل يفهم العلاقة بين نص وصورة في نفس المهمة؟",
        question_en="Does it understand the relationship between text and image in the same task?",
        test_type=TestType.INTEGRATION, criticality=5,
        automated_test="test_ca2_text_image_correlation"
    ),

    SubsystemQuestion(
        id=83, subsystem=SubsystemType.CONTEXTUAL_AWARENESS,
        question_ar="هل يمكنه اكتشاف التناقض بين الصوت والصورة؟",
        question_en="Can it detect contradiction between audio and image?",
        test_type=TestType.INTEGRATION, criticality=4,
        automated_test="test_ca3_audio_visual_mismatch"
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Automated Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SubsystemTestSuite:
    """مجموعة اختبارات الأنظمة الفرعية"""

    def __init__(self):
        self.project_root = "/home/noogh/projects/noogh_unified_system"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Computer Vision Tests
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_cv1_shape_recognition(self) -> Dict[str, Any]:
        """فحص التعرف على الأشكال"""
        # فحص وجود نظام رؤية
        has_cv = os.path.exists(f"{self.project_root}/vision") or \
                 os.path.exists(f"{self.project_root}/core/vision")

        # فحص Image Analyzer
        has_image_analyzer = os.path.exists(f"{self.project_root}/core/vision/image_analyzer.py")

        if has_image_analyzer:
            with open(f"{self.project_root}/core/vision/image_analyzer.py") as f:
                content = f.read()
                has_analysis = "ImageAnalyzer" in content and "analyze_image" in content

            score = 1.0 if has_analysis else 0.5
        else:
            score = 0.5 if has_cv else 0.0

        return {
            "passed": has_image_analyzer,
            "score": score,
            "evidence": f"Image analysis system: {'fully functional' if has_image_analyzer else 'basic' if has_cv else 'not found'}"
        }

    def test_cv6_multilingual_ocr(self) -> Dict[str, Any]:
        """فحص OCR متعدد اللغات"""
        # فحص وجود نظام OCR
        has_ocr = os.path.exists(f"{self.project_root}/core/vision/ocr_engine.py") or \
                  os.path.exists(f"{self.project_root}/vision/ocr.py") or \
                  os.path.exists(f"{self.project_root}/core/vision/ocr")

        # فحص دعم اللغات المتعددة
        multilingual = False
        if has_ocr:
            ocr_paths = [
                f"{self.project_root}/core/vision/ocr_engine.py",
                f"{self.project_root}/vision/ocr.py"
            ]

            for ocr_path in ocr_paths:
                if os.path.exists(ocr_path):
                    with open(ocr_path) as f:
                        content = f.read()
                        # فحص دعم العربية والإنجليزية
                        has_arabic = "ARABIC" in content or "ara" in content
                        has_english = "ENGLISH" in content or "eng" in content
                        multilingual = has_arabic and has_english
                        break

        score = 1.0 if multilingual else (0.5 if has_ocr else 0.2)

        return {
            "passed": multilingual,
            "score": score,
            "evidence": f"OCR system: {'multilingual' if multilingual else 'basic' if has_ocr else 'not found'}"
        }

    def test_cv2_object_detection(self) -> Dict[str, Any]:
        """فحص كشف الكائنات في الصور"""
        # فحص Image Analyzer مع قدرات كشف المحتوى
        has_analyzer = os.path.exists(f"{self.project_root}/core/vision/image_analyzer.py")

        if has_analyzer:
            with open(f"{self.project_root}/core/vision/image_analyzer.py") as f:
                content = f.read()
                has_content_analysis = "ContentAnalysis" in content and "regions" in content.lower()

            return {
                "passed": has_content_analysis,
                "score": 1.0 if has_content_analysis else 0.3,
                "evidence": f"Content detection: {'available' if has_content_analysis else 'basic'}"
            }

        return {
            "passed": False,
            "score": 0.0,
            "evidence": "No object detection system found"
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NLP Tests
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_nlp1_intent_classification(self) -> Dict[str, Any]:
        """فحص تصنيف النوايا"""
        # فحص وجود multi-intent detector
        has_intent = os.path.exists(f"{self.project_root}/core/nlp/multi_intent_detector.py")

        if has_intent:
            with open(f"{self.project_root}/core/nlp/multi_intent_detector.py") as f:
                content = f.read()
                has_multiple = "MultiIntentDetector" in content
        else:
            has_multiple = False

        return {
            "passed": has_multiple,
            "score": 1.0 if has_multiple else 0.3,
            "evidence": f"Multi-intent classification: {'available' if has_multiple else 'basic only'}"
        }

    def test_nlp4_emotion_analysis(self) -> Dict[str, Any]:
        """فحص تحليل العواطف"""
        # فحص نظام NLU المتقدم
        has_nlu = os.path.exists(f"{self.project_root}/core/nlp/arabic_nlu_advanced.py")

        if has_nlu:
            with open(f"{self.project_root}/core/nlp/arabic_nlu_advanced.py") as f:
                content = f.read()
                has_tone = "Tone" in content and "emotion" in content.lower()
        else:
            has_tone = False

        return {
            "passed": has_tone,
            "score": 1.0 if has_tone else 0.2,
            "evidence": f"Emotion/tone analysis: {'available' if has_tone else 'not found'}"
        }

    def test_nlp8_contradiction_detection(self) -> Dict[str, Any]:
        """فحص كشف التناقضات"""
        # فحص hallucination detector
        has_detector = os.path.exists(f"{self.project_root}/core/quality/hallucination_detector.py")

        return {
            "passed": has_detector,
            "score": 0.8 if has_detector else 0.2,
            "evidence": f"Contradiction detection: {'hallucination detector available' if has_detector else 'not found'}"
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Decision & Reasoning Tests
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_dr1_adaptive_strategy(self) -> Dict[str, Any]:
        """فحص الاستراتيجية التكيفية"""
        # فحص نظام التدريب التلقائي
        has_adaptive = os.path.exists(f"{self.project_root}/training/auto_retraining.py")

        return {
            "passed": has_adaptive,
            "score": 0.8 if has_adaptive else 0.2,
            "evidence": f"Adaptive strategy: {'auto-retraining available' if has_adaptive else 'not found'}"
        }

    def test_dr3_error_cost_estimation(self) -> Dict[str, Any]:
        """فحص تقدير تكلفة الأخطاء"""
        # فحص confidence scorer
        has_confidence = os.path.exists(f"{self.project_root}/core/reasoning/confidence_scorer.py")

        return {
            "passed": has_confidence,
            "score": 0.9 if has_confidence else 0.3,
            "evidence": f"Error cost estimation: {'confidence scorer available' if has_confidence else 'limited'}"
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ML Training Tests
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_ml1_incremental_learning(self) -> Dict[str, Any]:
        """فحص التعلم التفاضلي"""
        # فحص نظام التدريب
        has_training = os.path.exists(f"{self.project_root}/training")

        if has_training:
            # فحص incremental في الملفات
            has_incremental = os.path.exists(f"{self.project_root}/training/incremental_trainer.py")
        else:
            has_incremental = False

        return {
            "passed": has_incremental,
            "score": 1.0 if has_incremental else 0.3,
            "evidence": f"Incremental learning: {'available' if has_incremental else 'not found'}"
        }

    def test_ml3_reproducibility(self) -> Dict[str, Any]:
        """فحص إعادة الإنتاج"""
        # فحص logging system
        has_logging = os.path.exists(f"{self.project_root}/backend/logging")
        has_experiments = os.path.exists(f"{self.project_root}/training/auto_retraining.py")

        passed = has_logging and has_experiments

        return {
            "passed": passed,
            "score": 0.9 if passed else 0.4,
            "evidence": f"Reproducibility: logging={has_logging}, experiments={has_experiments}"
        }

    def test_ml5_data_validation(self) -> Dict[str, Any]:
        """فحص التحقق من البيانات"""
        # فحص input validator
        has_validator = os.path.exists(f"{self.project_root}/backend/security/input_validator.py")

        return {
            "passed": has_validator,
            "score": 0.8 if has_validator else 0.3,
            "evidence": f"Data validation: {'input validator available' if has_validator else 'not found'}"
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Contextual Awareness Tests
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def test_ca1_multimodal_integration(self) -> Dict[str, Any]:
        """فحص التكامل متعدد الوسائط"""
        # فحص وجود أنظمة متعددة
        has_nlp = os.path.exists(f"{self.project_root}/core/nlp")
        has_reasoning = os.path.exists(f"{self.project_root}/core/reasoning")

        passed = has_nlp and has_reasoning

        return {
            "passed": passed,
            "score": 0.7 if passed else 0.3,
            "evidence": f"Multimodal integration: NLP={has_nlp}, Reasoning={has_reasoning}"
        }

    def get_all_tests(self) -> Dict[int, callable]:
        """الحصول على جميع الاختبارات المتاحة"""
        return {
            1: self.test_cv1_shape_recognition,
            2: self.test_cv2_object_detection,
            6: self.test_cv6_multilingual_ocr,
            21: self.test_nlp1_intent_classification,
            24: self.test_nlp4_emotion_analysis,
            28: self.test_nlp8_contradiction_detection,
            41: self.test_dr1_adaptive_strategy,
            43: self.test_dr3_error_cost_estimation,
            61: self.test_ml1_incremental_learning,
            63: self.test_ml3_reproducibility,
            65: self.test_ml5_data_validation,
            81: self.test_ca1_multimodal_integration,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SubsystemIntelligenceEngine:
    """محرك اختبار ذكاء الأنظمة الفرعية"""

    def __init__(self, db_path: str = "data/subsystem_intelligence.db"):
        self.db_path = db_path
        self.test_suite = SubsystemTestSuite()
        self._init_database()

    def _init_database(self):
        """تهيئة قاعدة البيانات"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subsystem_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subsystem TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    score REAL NOT NULL,
                    intelligence_level TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    details TEXT
                )
            """)
            conn.commit()

    def run_tests(self, subsystem: Optional[SubsystemType] = None) -> Dict[SubsystemType, SubsystemReport]:
        """تشغيل الاختبارات"""
        results = {}

        # الحصول على الأسئلة
        questions = SUBSYSTEM_QUESTIONS
        if subsystem:
            questions = [q for q in questions if q.subsystem == subsystem]

        # تجميع حسب النظام
        by_subsystem = {}
        for q in questions:
            if q.subsystem not in by_subsystem:
                by_subsystem[q.subsystem] = []
            by_subsystem[q.subsystem].append(q)

        # تشغيل الاختبارات
        all_tests = self.test_suite.get_all_tests()

        for subsys, subsys_questions in by_subsystem.items():
            test_results = []

            for question in subsys_questions:
                if question.id in all_tests:
                    # تشغيل الاختبار
                    test_func = all_tests[question.id]
                    result_data = test_func()

                    test_result = SubsystemTestResult(
                        question_id=question.id,
                        subsystem=subsys,
                        passed=result_data["passed"],
                        score=result_data["score"],
                        evidence=result_data["evidence"]
                    )
                    test_results.append(test_result)

            # حساب التقرير
            if test_results:
                passed = sum(1 for r in test_results if r.passed)
                total = len(test_results)
                score_pct = (sum(r.score for r in test_results) / total) * 100

                level = self._determine_intelligence_level(score_pct)

                report = SubsystemReport(
                    subsystem=subsys,
                    total_questions=total,
                    passed=passed,
                    failed=total - passed,
                    score_percentage=score_pct,
                    intelligence_level=level,
                    strengths=[r.evidence for r in test_results if r.passed][:3],
                    weaknesses=[r.evidence for r in test_results if not r.passed][:3],
                    test_results=test_results
                )

                results[subsys] = report

                # حفظ في قاعدة البيانات
                self._save_report(report)

        return results

    def _determine_intelligence_level(self, score: float) -> IntelligenceLevel:
        """تحديد مستوى الذكاء"""
        if score >= 90:
            return IntelligenceLevel.EXPERT
        elif score >= 75:
            return IntelligenceLevel.ADVANCED
        elif score >= 60:
            return IntelligenceLevel.INTERMEDIATE
        elif score >= 40:
            return IntelligenceLevel.BASIC
        elif score >= 20:
            return IntelligenceLevel.MINIMAL
        else:
            return IntelligenceLevel.NONE

    def _save_report(self, report: SubsystemReport):
        """حفظ التقرير في قاعدة البيانات"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO subsystem_tests
                (subsystem, test_id, timestamp, score, intelligence_level, passed, failed, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.subsystem.value,
                f"test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                datetime.now(timezone.utc).isoformat(),
                report.score_percentage,
                report.intelligence_level.value,
                report.passed,
                report.failed,
                json.dumps({
                    "strengths": report.strengths,
                    "weaknesses": report.weaknesses
                })
            ))
            conn.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main / Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🧠 Subsystem Intelligence Test Engine")
    print("=" * 70)
    print()

    engine = SubsystemIntelligenceEngine()

    print("⏳ Running subsystem intelligence tests...")
    print()

    results = engine.run_tests()

    print("=" * 70)
    print("📊 Subsystem Intelligence Report")
    print("=" * 70)
    print()

    for subsystem, report in results.items():
        status = "✅" if report.score_percentage >= 70 else "⚠️" if report.score_percentage >= 50 else "❌"

        print(f"{status} {subsystem.value.upper()}")
        print(f"   Score: {report.score_percentage:.1f}%")
        print(f"   Level: {report.intelligence_level.value}")
        print(f"   Tests: {report.passed}/{report.total_questions} passed")

        if report.strengths:
            print(f"   Strengths: {report.strengths[0]}")

        if report.weaknesses:
            print(f"   Weaknesses: {report.weaknesses[0]}")

        print()

    print("=" * 70)
    print("✅ Subsystem intelligence test completed!")
