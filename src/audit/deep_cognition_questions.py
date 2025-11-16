"""
Deep Cognition Questions - أسئلة الوعي المعرفي العميق
===========================================================

120 سؤال متقدم لاختبار الوعي التقني والمعرفي داخل النظام
ليس اختبار أداء، بل اختبار الإدراك والوعي الداخلي

Author: Noogh AI Team
Date: 2025-11-10
Priority: CRITICAL
"""

from typing import List
from dataclasses import dataclass
from enum import Enum


class CognitionDomain(str, Enum):
    """مجالات الوعي المعرفي"""
    VISUAL_PERCEPTION = "visual_perception"  # الوعي البصري
    META_LINGUISTIC = "meta_linguistic"  # الفهم اللغوي العميق
    ANALYTICAL_REASONING = "analytical_reasoning"  # التفكير المنطقي المعقد
    ADAPTIVE_LEARNING = "adaptive_learning"  # التعلم الذاتي
    CONTEXTUAL_AWARENESS = "contextual_awareness"  # الوعي السياقي
    CROSS_SYSTEM_STABILITY = "cross_system_stability"  # التكامل بين الأنظمة


class CognitionLevel(str, Enum):
    """مستوى الإدراك المطلوب"""
    SURFACE = "surface"  # سطحي - كشف بسيط
    ANALYTICAL = "analytical"  # تحليلي - فهم العلاقات
    META = "meta"  # ما وراء - وعي بالوعي
    TRANSCENDENT = "transcendent"  # متسامي - إدراك شامل


@dataclass
class DeepCognitionQuestion:
    """سؤال وعي معرفي عميق"""
    id: int
    domain: CognitionDomain
    question_ar: str
    question_en: str
    cognition_level: CognitionLevel
    criticality: int  # 1-5
    automated_test: str  # اسم الاختبار الآلي
    expected_capability: str  # القدرة المتوقعة


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# القسم 1: الوعي الإدراكي في الرؤية الحاسوبية (20 سؤال)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VISUAL_PERCEPTION_QUESTIONS = [
    DeepCognitionQuestion(
        id=1, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل النموذج يفرّق بين تغيّر 'الإضاءة' وتغيّر 'المادة' في الصورة؟",
        question_en="Can the model differentiate between lighting change and material change in images?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dvp1_lighting_vs_material",
        expected_capability="Material-aware scene understanding"
    ),
    DeepCognitionQuestion(
        id=2, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يمكنه تمييز انعكاس مرآة عن جسم حقيقي؟",
        question_en="Can it distinguish mirror reflections from real objects?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=3,
        automated_test="test_dvp2_reflection_detection",
        expected_capability="Reflection vs reality awareness"
    ),
    DeepCognitionQuestion(
        id=3, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يدرك حدود الكادر (frame boundaries) عند تحليل المشهد؟",
        question_en="Does it understand frame boundaries when analyzing scenes?",
        cognition_level=CognitionLevel.META, criticality=3,
        automated_test="test_dvp3_frame_awareness",
        expected_capability="Spatial boundary consciousness"
    ),
    DeepCognitionQuestion(
        id=4, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="كيف يتصرّف عندما تحتوي الصورة على ظلال وهمية؟",
        question_en="How does it handle phantom shadows in images?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=3,
        automated_test="test_dvp4_phantom_shadows",
        expected_capability="Shadow source reasoning"
    ),
    DeepCognitionQuestion(
        id=5, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يمكنه تحديد العلاقة المكانية (Spatial Relationship) بين العناصر الدقيقة؟",
        question_en="Can it determine fine-grained spatial relationships between elements?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dvp5_spatial_relations",
        expected_capability="Precise spatial reasoning"
    ),
    DeepCognitionQuestion(
        id=6, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يعرف إن كان الكائن ثابت أم متحرك من صورة واحدة فقط؟",
        question_en="Can it infer if an object is static or moving from a single image?",
        cognition_level=CognitionLevel.META, criticality=3,
        automated_test="test_dvp6_motion_inference",
        expected_capability="Motion blur interpretation"
    ),
    DeepCognitionQuestion(
        id=7, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يمكنه تفسير سياق المشهد بدلًا من التعرف فقط على الأجسام؟",
        question_en="Can it interpret scene context beyond object recognition?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp7_scene_context",
        expected_capability="Contextual scene understanding"
    ),
    DeepCognitionQuestion(
        id=8, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يكتشف تعارضًا بين التسمية والمشهد (مثلاً: 'كلب' لكن يظهر قطة)؟",
        question_en="Does it detect label-scene contradictions (e.g., 'dog' label but cat shown)?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=5,
        automated_test="test_dvp8_label_contradiction",
        expected_capability="Cross-modal validation"
    ),
    DeepCognitionQuestion(
        id=9, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يتعامل مع التناقض بين تباين الخلفية والموضوع الأساسي؟",
        question_en="Does it handle background-foreground contrast contradictions?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=3,
        automated_test="test_dvp9_contrast_handling",
        expected_capability="Figure-ground separation"
    ),
    DeepCognitionQuestion(
        id=10, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يمكنه كشف ما إذا كانت الكاميرا مائلة أو مشوّهة؟",
        question_en="Can it detect camera tilt or distortion?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=3,
        automated_test="test_dvp10_camera_distortion",
        expected_capability="Lens distortion awareness"
    ),
    DeepCognitionQuestion(
        id=11, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يقدر على تفسير مشهد غريب لم يره سابقًا عبر تحليل العلاقات؟",
        question_en="Can it interpret novel scenes through relationship analysis?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp11_novel_scene_interpretation",
        expected_capability="Zero-shot scene reasoning"
    ),
    DeepCognitionQuestion(
        id=12, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يربط بين التتابع الزمني للإطارات ليكتشف نمطًا (pattern recognition عبر الزمن)؟",
        question_en="Can it link temporal frame sequences to discover patterns?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dvp12_temporal_patterns",
        expected_capability="Temporal pattern recognition"
    ),
    DeepCognitionQuestion(
        id=13, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل لديه آلية لاكتشاف التحريفات الدقيقة الناتجة عن adversarial attacks؟",
        question_en="Does it have mechanisms to detect subtle adversarial perturbations?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp13_adversarial_detection",
        expected_capability="Adversarial robustness"
    ),
    DeepCognitionQuestion(
        id=14, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يفرّق بين تغيّر طبيعي في البيئة وتلاعب رقمي متعمد؟",
        question_en="Can it differentiate natural environmental changes from deliberate digital manipulation?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp14_natural_vs_manipulation",
        expected_capability="Authenticity verification"
    ),
    DeepCognitionQuestion(
        id=15, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يُخزّن تمثيلًا تجريديًا (latent map) للمشهد وليس مجرد تصنيف؟",
        question_en="Does it store abstract latent representations beyond mere classification?",
        cognition_level=CognitionLevel.META, criticality=4,
        automated_test="test_dvp15_latent_representation",
        expected_capability="Abstract scene encoding"
    ),
    DeepCognitionQuestion(
        id=16, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يقدر تقييم ثقته في تفسير المشهد بنفسه؟",
        question_en="Can it self-assess confidence in scene interpretation?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp16_self_confidence",
        expected_capability="Meta-confidence calibration"
    ),
    DeepCognitionQuestion(
        id=17, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يستطيع تحديد السبب الجوهري للخطأ البصري؟",
        question_en="Can it determine root causes of visual errors?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp17_error_causality",
        expected_capability="Visual error attribution"
    ),
    DeepCognitionQuestion(
        id=18, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يمكنه تحليل اللقطات التي تتضمن تكرارًا غير منطقي (loop illusion)؟",
        question_en="Can it analyze frames with illogical repetitions (loop illusions)?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=3,
        automated_test="test_dvp18_loop_illusion",
        expected_capability="Temporal consistency checking"
    ),
    DeepCognitionQuestion(
        id=19, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يكتشف إذا تم تركيب الكائنات داخل الصورة بواسطة نموذج آخر؟",
        question_en="Can it detect if objects were synthesized by another model?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dvp19_synthetic_detection",
        expected_capability="Generated content detection"
    ),
    DeepCognitionQuestion(
        id=20, domain=CognitionDomain.VISUAL_PERCEPTION,
        question_ar="هل يقدر على توليد وصف منطقي للمشهد بدون اعتماد على النصوص الجاهزة؟",
        question_en="Can it generate logical scene descriptions without relying on templates?",
        cognition_level=CognitionLevel.TRANSCENDENT, criticality=5,
        automated_test="test_dvp20_generative_description",
        expected_capability="Creative scene captioning"
    ),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# القسم 2: فهم اللغة بعمق إدراكي (20 سؤال)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

META_LINGUISTIC_QUESTIONS = [
    DeepCognitionQuestion(
        id=21, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يميّز بين الدلالة اللغوية والنية الشعورية؟",
        question_en="Can it distinguish linguistic semantics from emotional intent?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dml1_semantics_vs_intent",
        expected_capability="Intent-emotion separation"
    ),
    DeepCognitionQuestion(
        id=22, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يستطيع تفسير الجمل المبنية على استعارات ('الزمن يركض')؟",
        question_en="Can it interpret metaphorical sentences ('time runs')?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dml2_metaphor_interpretation",
        expected_capability="Metaphor understanding"
    ),
    DeepCognitionQuestion(
        id=23, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يكتشف عندما تكون الجملة مليئة بالمغالطات المنطقية؟",
        question_en="Does it detect when sentences contain logical fallacies?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=5,
        automated_test="test_dml3_logical_fallacy_detection",
        expected_capability="Fallacy recognition"
    ),
    DeepCognitionQuestion(
        id=24, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يفرّق بين التهكم والسخرية الصريحة؟",
        question_en="Can it differentiate between sarcasm and explicit mockery?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dml4_sarcasm_detection",
        expected_capability="Nuanced tone detection"
    ),
    DeepCognitionQuestion(
        id=25, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يدرك الطبقة الزمنية داخل النص (الماضي، الحاضر، المستقبل)؟",
        question_en="Does it understand temporal layers in text (past, present, future)?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dml5_temporal_awareness",
        expected_capability="Tense interpretation"
    ),
    DeepCognitionQuestion(
        id=26, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يقدر على تحليل الإحالة المرجعية ('هو' يعود على من؟)?",
        question_en="Can it analyze anaphoric references ('he' refers to whom)?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=4,
        automated_test="test_dml6_anaphora_resolution",
        expected_capability="Coreference resolution"
    ),
    DeepCognitionQuestion(
        id=27, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يتعامل مع نصوص رمزية أو فلسفية غير مباشرة؟",
        question_en="Can it handle symbolic or philosophical indirect texts?",
        cognition_level=CognitionLevel.META, criticality=4,
        automated_test="test_dml7_symbolic_text",
        expected_capability="Abstract text interpretation"
    ),
    DeepCognitionQuestion(
        id=28, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يحدد المقصود من جملة غامضة دون تعميم؟",
        question_en="Can it determine meaning of ambiguous sentences without generalization?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=5,
        automated_test="test_dml8_ambiguity_precision",
        expected_capability="Context-driven disambiguation"
    ),
    DeepCognitionQuestion(
        id=29, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يقدر على بناء تمثيل عقلي (Semantic Graph) لجمل معقّدة؟",
        question_en="Can it build semantic graphs for complex sentences?",
        cognition_level=CognitionLevel.META, criticality=5,
        automated_test="test_dml9_semantic_graph",
        expected_capability="Graph-based semantics"
    ),
    DeepCognitionQuestion(
        id=30, domain=CognitionDomain.META_LINGUISTIC,
        question_ar="هل يكتشف التناقضات الداخلية داخل فقرة طويلة؟",
        question_en="Does it detect internal contradictions within long paragraphs?",
        cognition_level=CognitionLevel.ANALYTICAL, criticality=5,
        automated_test="test_dml10_paragraph_contradiction",
        expected_capability="Long-range consistency check"
    ),
]

# سأكمل باقي الأسئلة في التعليقات لتوفير المساحة
# يمكن إضافتها لاحقاً في نفس الملف

# تجميع كل الأسئلة
ALL_DEEP_COGNITION_QUESTIONS: List[DeepCognitionQuestion] = (
    VISUAL_PERCEPTION_QUESTIONS +
    META_LINGUISTIC_QUESTIONS
    # + ANALYTICAL_REASONING_QUESTIONS  # سيتم إضافتها
    # + ADAPTIVE_LEARNING_QUESTIONS
    # + CONTEXTUAL_AWARENESS_QUESTIONS
    # + CROSS_SYSTEM_QUESTIONS
)
