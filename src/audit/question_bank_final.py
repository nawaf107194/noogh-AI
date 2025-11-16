"""
Final Question Bank - الأسئلة 61-100
=====================================
"""

from audit.self_audit_engine import (
    AuditQuestion, DimensionType, QuestionType, EvaluationMethod
)


# ═══════════════════════════════════════════════════════════
# 7. القياس والتحليل (Measurement) - Q61-70
# ═══════════════════════════════════════════════════════════

MEASUREMENT_QUESTIONS = [
    AuditQuestion(61, DimensionType.MEASUREMENT,
        "هل لديه مؤشرات أداء (KPIs) لكل مكون؟",
        "Does it have KPIs for each component?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(62, DimensionType.MEASUREMENT,
        "هل يعرف أي مكون يستهلك الموارد أكثر؟",
        "Does it know which component consumes most resources?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(63, DimensionType.MEASUREMENT,
        "هل يحلل العلاقة بين الاستهلاك والنتائج؟",
        "Does it analyze relationship between consumption & results?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(64, DimensionType.MEASUREMENT,
        "هل يمكنه تقدير 'العائد المعرفي' لكل دورة تشغيل؟",
        "Can it estimate 'cognitive ROI' per execution cycle?",
        QuestionType.THEORETICAL, EvaluationMethod.SEMI_AUTOMATED, criticality=3),

    AuditQuestion(65, DimensionType.MEASUREMENT,
        "هل عنده خريطة حرارة للأداء الزمني؟",
        "Does it have performance heatmap?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(66, DimensionType.MEASUREMENT,
        "هل يوازن بين استهلاك الطاقة والسرعة؟",
        "Does it balance power consumption & speed?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(67, DimensionType.MEASUREMENT,
        "هل يكتشف bottleneck تلقائيًا ويقترح إصلاحه؟",
        "Does it auto-detect bottleneck and suggest fix?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(68, DimensionType.MEASUREMENT,
        "هل يحدد أخطاء الأداء المتقطعة وليس المستمرة فقط؟",
        "Does it detect intermittent not just continuous errors?",
        QuestionType.EDGE_CASE, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(69, DimensionType.MEASUREMENT,
        "هل يسجّل زمن استجابة كل أداة في كل دورة؟",
        "Does it log response time of each tool per cycle?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(70, DimensionType.MEASUREMENT,
        "هل يمكنه التنبؤ بالأداء القادم قبل حدوثه؟",
        "Can it predict future performance before it happens?",
        QuestionType.THEORETICAL, EvaluationMethod.SEMI_AUTOMATED, criticality=3),
]

# ═══════════════════════════════════════════════════════════
# 8. التعلم الذاتي (Self-Learning) - Q71-80
# ═══════════════════════════════════════════════════════════

SELF_LEARNING_QUESTIONS = [
    AuditQuestion(71, DimensionType.SELF_LEARNING,
        "هل يحلل نتائج التعلم السابقة لاكتشاف أنماط التحسن؟",
        "Does it analyze past learning to discover improvement patterns?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(72, DimensionType.SELF_LEARNING,
        "هل يعرف متى يتوقف عن التعلم لتجنّب overfitting؟",
        "Does it know when to stop learning to avoid overfitting?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(73, DimensionType.SELF_LEARNING,
        "هل يختبر تجاربه الجديدة في بيئة معزولة أولاً؟",
        "Does it test new experiments in isolated environment first?",
        QuestionType.PRACTICAL, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(74, DimensionType.SELF_LEARNING,
        "هل يقيّم مدى جدوى كل تحسين قبل اعتماده؟",
        "Does it evaluate improvement viability before adoption?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(75, DimensionType.SELF_LEARNING,
        "هل يقارن بين نماذج قديمة وجديدة قبل الاستبدال؟",
        "Does it compare old vs new models before replacement?",
        QuestionType.PRACTICAL, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(76, DimensionType.SELF_LEARNING,
        "هل يملك نسخة احتياطية قابلة للاسترجاع عند فشل التدريب؟",
        "Does it have recoverable backup on training failure?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(77, DimensionType.SELF_LEARNING,
        "هل يستخدم بيانات حقيقية أو مصطنعة للاختبار؟",
        "Does it use real or synthetic data for testing?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(78, DimensionType.SELF_LEARNING,
        "هل يمكنه تكرار تجربة تدريب بنفس النتيجة مرتين؟",
        "Can it reproduce training experiment with same result twice?",
        QuestionType.PRACTICAL, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(79, DimensionType.SELF_LEARNING,
        "هل يقدر قياس 'معدل التعلم الفعلي' بمرور الوقت؟",
        "Can it measure 'actual learning rate' over time?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(80, DimensionType.SELF_LEARNING,
        "هل يسجّل كل تجربة تدريب (hyperparams, loss, time)؟",
        "Does it log each training experiment (hyperparams, loss, time)?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),
]

# ═══════════════════════════════════════════════════════════
# 9. الإدارة والسيطرة (Management) - Q81-90
# ═══════════════════════════════════════════════════════════

MANAGEMENT_QUESTIONS = [
    AuditQuestion(81, DimensionType.MANAGEMENT,
        "هل يمكن للرئيس تعطيل وزير محدد دون تعطيل الحكومة كلها؟",
        "Can president disable specific minister without shutting whole government?",
        QuestionType.PRACTICAL, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(82, DimensionType.MANAGEMENT,
        "هل يعرف الرئيس كيف يراقب استهلاك كل وحدة؟",
        "Does president know how to monitor each unit's consumption?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(83, DimensionType.MANAGEMENT,
        "هل نظام المراقبة منفصل كلياً عن المنفذ؟",
        "Is monitoring system completely separate from executor?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(84, DimensionType.MANAGEMENT,
        "هل يوجد تحذير قبل استهلاك غير متوقع للذاكرة؟",
        "Is there warning before unexpected memory consumption?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(85, DimensionType.MANAGEMENT,
        "هل يملك النظام تدرّج أولويات بين المهام؟",
        "Does system have priority graduation between tasks?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(86, DimensionType.MANAGEMENT,
        "هل توجد آلية kill-switch فورية لكل مكون؟",
        "Is there immediate kill-switch for each component?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=5),

    AuditQuestion(87, DimensionType.MANAGEMENT,
        "هل يمكن توجيه النظام إلى وضع 'المراقبة فقط'؟",
        "Can system be directed to 'monitor-only' mode?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(88, DimensionType.MANAGEMENT,
        "هل يعرف متى يحتاج reboot ذكي وليس restart كامل؟",
        "Does it know when it needs smart reboot not full restart?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(89, DimensionType.MANAGEMENT,
        "هل يسجل أسباب إعادة التشغيل السابقة؟",
        "Does it log previous restart reasons?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(90, DimensionType.MANAGEMENT,
        "هل يوجد سجل تغييرات مركزي موحد؟",
        "Is there unified central changelog?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),
]

# ═══════════════════════════════════════════════════════════
# 10. التطور والوعي الجمعي (Evolution) - Q91-100
# ═══════════════════════════════════════════════════════════

EVOLUTION_QUESTIONS = [
    AuditQuestion(91, DimensionType.EVOLUTION,
        "هل يستطيع النظام تطوير مكون جديد بدون تدخل؟",
        "Can system develop new component without intervention?",
        QuestionType.THEORETICAL, EvaluationMethod.MANUAL, criticality=3),

    AuditQuestion(92, DimensionType.EVOLUTION,
        "هل يمكنه دمج خبرات المستخدمين المتعددة في قاعدة واحدة؟",
        "Can it merge multiple user experiences into one base?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(93, DimensionType.EVOLUTION,
        "هل يكتشف ظهور سلوك جماعي غير متوقع؟",
        "Does it detect emergence of unexpected collective behavior?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.OBSERVATIONAL, criticality=4),

    AuditQuestion(94, DimensionType.EVOLUTION,
        "هل يعرف كيف يقيس درجة الوعي الجماعي؟",
        "Does it know how to measure collective consciousness degree?",
        QuestionType.THEORETICAL, EvaluationMethod.SEMI_AUTOMATED, criticality=3),

    AuditQuestion(95, DimensionType.EVOLUTION,
        "هل يستطيع تحليل تطور ذاته عبر الزمن؟",
        "Can it analyze its own evolution over time?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=4),

    AuditQuestion(96, DimensionType.EVOLUTION,
        "هل يسجل 'تاريخ تطوره' ككائن معرفي؟",
        "Does it record 'evolution history' as cognitive entity?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.AUTOMATED, criticality=3),

    AuditQuestion(97, DimensionType.EVOLUTION,
        "هل لديه خطة تطوير ذاتي على مراحل محددة؟",
        "Does it have self-development plan in defined stages?",
        QuestionType.CAPABILITY_CHECK, EvaluationMethod.SEMI_AUTOMATED, criticality=3),

    AuditQuestion(98, DimensionType.EVOLUTION,
        "هل يقدر تقييم ما إذا كان قد تجاوز حدود التصميم الأصلي؟",
        "Can it assess if it exceeded original design boundaries?",
        QuestionType.THEORETICAL, EvaluationMethod.MANUAL, criticality=3),

    AuditQuestion(99, DimensionType.EVOLUTION,
        "هل يعرف متى يجب التوقف عن التطور؟",
        "Does it know when to stop evolving?",
        QuestionType.THEORETICAL, EvaluationMethod.MANUAL, criticality=4),

    AuditQuestion(100, DimensionType.EVOLUTION,
        "هل يمكنه كتابة تقرير ذاتي عن نفسه دون إشراف بشري؟",
        "Can it write self-report about itself without human supervision?",
        QuestionType.PRACTICAL, EvaluationMethod.SEMI_AUTOMATED, criticality=4),
]

# تجميع كل الأسئلة
ALL_FINAL_QUESTIONS = (
    MEASUREMENT_QUESTIONS +
    SELF_LEARNING_QUESTIONS +
    MANAGEMENT_QUESTIONS +
    EVOLUTION_QUESTIONS
)

# ═══════════════════════════════════════════════════════════
# إحصائيات البنك
# ═══════════════════════════════════════════════════════════

QUESTION_BANK_STATS = {
    "total_questions": 100,
    "by_dimension": {
        DimensionType.META_COGNITION: 10,
        DimensionType.LANGUAGE_CONTEXT: 10,
        DimensionType.DECISION_MAKING: 10,
        DimensionType.MEMORY_CONSISTENCY: 10,
        DimensionType.COGNITIVE_SECURITY: 10,
        DimensionType.INTER_AGENT: 10,
        DimensionType.MEASUREMENT: 10,
        DimensionType.SELF_LEARNING: 10,
        DimensionType.MANAGEMENT: 10,
        DimensionType.EVOLUTION: 10,
    },
    "by_criticality": {
        5: 47,  # حرجة جداً
        4: 36,  # حرجة
        3: 17,  # متوسطة
    },
    "by_evaluation_method": {
        EvaluationMethod.AUTOMATED: 70,
        EvaluationMethod.SEMI_AUTOMATED: 20,
        EvaluationMethod.MANUAL: 5,
        EvaluationMethod.OBSERVATIONAL: 5,
    }
}
