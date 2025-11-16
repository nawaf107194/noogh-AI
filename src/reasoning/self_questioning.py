"""
Self-Questioning System - نظام إعادة صياغة الأسئلة
===================================================

يكتشف الأسئلة/الأهداف الداخلية الغامضة ويعيد صياغتها بوضوح.
يحسّن جودة التفكير الداخلي عبر توضيح المقاصد.

Addresses Q9 from Self-Audit (Meta-Cognition)

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re


class AmbiguityType(str, Enum):
    """نوع الغموض"""
    VAGUE_GOAL = "vague_goal"  # هدف غامض
    UNCLEAR_QUESTION = "unclear_question"  # سؤال غير واضح
    MISSING_CONTEXT = "missing_context"  # ينقصه سياق
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"  # يحتمل عدة تفسيرات
    CONTRADICTORY = "contradictory"  # متناقض
    NONE = "none"  # واضح


@dataclass
class AmbiguityAnalysis:
    """تحليل غموض السؤال/الهدف"""
    is_ambiguous: bool
    ambiguity_type: AmbiguityType
    ambiguity_score: float  # 0.0-1.0 (أعلى = أكثر غموضاً)
    issues: List[str]  # المشاكل المحددة
    clarification_needed: bool


@dataclass
class RephrasedQuestion:
    """سؤال/هدف معاد صياغته"""
    original: str
    rephrased: str
    improvements: List[str]  # التحسينات المطبقة
    clarity_score: float  # 0.0-1.0 (أعلى = أوضح)
    timestamp: datetime


class SelfQuestioningSystem:
    """
    نظام إعادة صياغة الأسئلة الداخلية

    يكتشف:
    1. الأسئلة/الأهداف الغامضة
    2. الأسئلة التي تحتمل عدة تفسيرات
    3. الأسئلة الناقصة للسياق

    ويعيد صياغتها لتصبح أوضح
    """

    def __init__(self):
        # Ambiguity patterns (أنماط الغموض)
        self.vague_indicators = [
            r'\b(شيء|something|thing)\b',
            r'\b(كذا|stuff|etc)\b',
            r'\b(ممكن|maybe|perhaps)\b',
            r'\b(نوعاً ما|kind of|sort of)\b',
            r'\b(تقريباً|approximately|roughly)\b'
        ]

        self.question_indicators = [
            r'\b(ماذا|what)\b',
            r'\b(كيف|how)\b',
            r'\b(لماذا|why)\b',
            r'\b(متى|when)\b',
            r'\b(أين|where)\b',
            r'\b(من|who)\b'
        ]

        # إحصائيات
        self.total_analyzed = 0
        self.total_rephrased = 0
        self.rephrasing_history: List[RephrasedQuestion] = []

    def analyze_ambiguity(self, question: str) -> AmbiguityAnalysis:
        """
        تحليل غموض السؤال/الهدف

        Returns:
            AmbiguityAnalysis مع تفاصيل الغموض
        """
        self.total_analyzed += 1

        issues = []
        ambiguity_score = 0.0

        # 1. فحص الطول (أسئلة قصيرة جداً عادةً غامضة)
        if len(question.split()) < 3:
            issues.append("Question too short - lacks detail")
            ambiguity_score += 0.3

        # 2. فحص المؤشرات الغامضة
        vague_count = 0
        for pattern in self.vague_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                vague_count += 1

        if vague_count > 0:
            issues.append(f"Contains {vague_count} vague indicator(s)")
            ambiguity_score += min(vague_count * 0.2, 0.5)

        # 3. فحص وجود كلمات استفهام (بدون context)
        has_question_word = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.question_indicators
        )

        if has_question_word and len(question.split()) < 5:
            issues.append("Question word present but lacks context")
            ambiguity_score += 0.2

        # 4. فحص الضمائر الغامضة (هو، هي، ذلك بدون إشارة واضحة)
        vague_pronouns = [r'\bهو\b', r'\bهي\b', r'\bذلك\b', r'\bهذا\b',
                          r'\bit\b', r'\bthis\b', r'\bthat\b', r'\bthey\b']

        pronoun_count = sum(
            1 for pattern in vague_pronouns
            if re.search(pattern, question, re.IGNORECASE)
        )

        if pronoun_count >= 2:
            issues.append("Multiple vague pronouns without clear referents")
            ambiguity_score += 0.2

        # 5. فحص الأسئلة المفتوحة جداً
        very_open_patterns = [
            r'^(ماذا|what)\s+(عن|about)',
            r'^(كيف|how)\s*$',
            r'^(لماذا|why)\s*$'
        ]

        if any(re.search(pattern, question, re.IGNORECASE) for pattern in very_open_patterns):
            issues.append("Question is too open-ended")
            ambiguity_score += 0.3

        # 6. فحص الأسئلة المركبة (أكثر من سؤال في واحد)
        question_marks = question.count('?')
        connectors = len(re.findall(r'\b(و|and|or|أو)\b', question, re.IGNORECASE))

        if question_marks > 1 or (has_question_word and connectors >= 2):
            issues.append("Multiple questions combined - needs splitting")
            ambiguity_score += 0.25

        # تحديد نوع الغموض
        ambiguity_type = self._determine_ambiguity_type(question, issues)

        # Capping
        ambiguity_score = min(ambiguity_score, 1.0)

        is_ambiguous = ambiguity_score >= 0.4

        return AmbiguityAnalysis(
            is_ambiguous=is_ambiguous,
            ambiguity_type=ambiguity_type,
            ambiguity_score=ambiguity_score,
            issues=issues,
            clarification_needed=is_ambiguous
        )

    def rephrase_question(self, question: str, analysis: Optional[AmbiguityAnalysis] = None) -> RephrasedQuestion:
        """
        إعادة صياغة السؤال/الهدف ليكون أوضح

        Args:
            question: السؤال الأصلي
            analysis: (اختياري) تحليل موجود مسبقاً

        Returns:
            RephrasedQuestion مع الصياغة المحسّنة
        """
        if analysis is None:
            analysis = self.analyze_ambiguity(question)

        if not analysis.is_ambiguous:
            # السؤال واضح بالفعل
            return RephrasedQuestion(
                original=question,
                rephrased=question,
                improvements=[],
                clarity_score=1.0 - analysis.ambiguity_score,
                timestamp=datetime.now(timezone.utc)
            )

        rephrased = question
        improvements = []

        # 1. إضافة تفاصيل للأسئلة القصيرة
        if "Question too short" in str(analysis.issues):
            if re.search(r'\b(ماذا|what)\b', question, re.IGNORECASE):
                rephrased = self._expand_what_question(rephrased)
                improvements.append("Expanded 'what' question with specific details")

            elif re.search(r'\b(كيف|how)\b', question, re.IGNORECASE):
                rephrased = self._expand_how_question(rephrased)
                improvements.append("Expanded 'how' question with specific goal")

        # 2. استبدال الكلمات الغامضة
        if "vague indicator" in str(analysis.issues):
            rephrased = re.sub(r'\b(شيء|something|thing)\b', 'specific item', rephrased, flags=re.IGNORECASE)
            rephrased = re.sub(r'\b(كذا|stuff)\b', 'details', rephrased, flags=re.IGNORECASE)
            rephrased = re.sub(r'\b(ممكن|maybe)\b', '', rephrased, flags=re.IGNORECASE)
            rephrased = re.sub(r'\b(نوعاً ما|kind of|sort of)\b', '', rephrased, flags=re.IGNORECASE)
            improvements.append("Replaced vague indicators with specific terms")

        # 3. توضيح الضمائر
        if "vague pronouns" in str(analysis.issues):
            rephrased = self._clarify_pronouns(rephrased)
            improvements.append("Clarified vague pronouns")

        # 4. تقسيم الأسئلة المركبة
        if "Multiple questions" in str(analysis.issues):
            rephrased = self._split_compound_question(rephrased)
            improvements.append("Split compound question into focused sub-question")

        # 5. إضافة سياق للأسئلة المفتوحة جداً
        if "too open-ended" in str(analysis.issues):
            rephrased = self._add_context_to_open_question(rephrased)
            improvements.append("Added context to open-ended question")

        # تنظيف
        rephrased = re.sub(r'\s+', ' ', rephrased).strip()

        # حساب وضوح الصياغة الجديدة
        new_analysis = self.analyze_ambiguity(rephrased)
        clarity_score = 1.0 - new_analysis.ambiguity_score

        result = RephrasedQuestion(
            original=question,
            rephrased=rephrased,
            improvements=improvements,
            clarity_score=clarity_score,
            timestamp=datetime.now(timezone.utc)
        )

        self.rephrasing_history.append(result)
        self.total_rephrased += 1

        return result

    def auto_improve_question(self, question: str) -> str:
        """
        تحسين السؤال تلقائياً إذا كان غامضاً

        Returns:
            rephrased question (أو الأصلي إذا كان واضحاً)
        """
        analysis = self.analyze_ambiguity(question)

        if analysis.is_ambiguous:
            result = self.rephrase_question(question, analysis)
            return result.rephrased
        else:
            return question

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات النظام"""
        if self.total_analyzed == 0:
            return {"status": "no_data"}

        avg_clarity = 0.0
        if self.rephrasing_history:
            avg_clarity = sum(r.clarity_score for r in self.rephrasing_history) / len(self.rephrasing_history)

        return {
            "total_analyzed": self.total_analyzed,
            "total_rephrased": self.total_rephrased,
            "rephrase_rate": (self.total_rephrased / self.total_analyzed) * 100,
            "average_clarity_after_rephrase": avg_clarity,
            "recent_improvements": [r.improvements for r in self.rephrasing_history[-5:]]
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helper Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _determine_ambiguity_type(self, question: str, issues: List[str]) -> AmbiguityType:
        """تحديد نوع الغموض الرئيسي"""
        if not issues:
            return AmbiguityType.NONE

        if "vague indicator" in str(issues):
            return AmbiguityType.VAGUE_GOAL

        if "lacks context" in str(issues):
            return AmbiguityType.MISSING_CONTEXT

        if "Multiple questions" in str(issues):
            return AmbiguityType.MULTIPLE_INTERPRETATIONS

        if "too short" in str(issues):
            return AmbiguityType.UNCLEAR_QUESTION

        return AmbiguityType.VAGUE_GOAL

    def _expand_what_question(self, question: str) -> str:
        """توسعة سؤال 'ماذا'"""
        # "ماذا؟" -> "ماذا تحديداً يجب فعله لتحقيق الهدف؟"
        if re.match(r'^(ماذا|what)\s*\?*$', question, re.IGNORECASE):
            return "What specifically should be done to achieve the goal?"

        # "What about X?" -> "What specific information about X is needed?"
        question = re.sub(
            r'^(ماذا|what)\s+(عن|about)\s+(\w+)',
            r'What specific information about \3 is needed',
            question,
            flags=re.IGNORECASE
        )

        return question

    def _expand_how_question(self, question: str) -> str:
        """توسعة سؤال 'كيف'"""
        # "كيف؟" -> "كيف يمكن تنفيذ هذه المهمة بفعالية؟"
        if re.match(r'^(كيف|how)\s*\?*$', question, re.IGNORECASE):
            return "How can this task be executed effectively?"

        return question

    def _clarify_pronouns(self, question: str) -> str:
        """توضيح الضمائر الغامضة"""
        # "Is it ready?" -> "Is the item ready?"
        question = re.sub(r'\bit\b', 'the item', question, flags=re.IGNORECASE)

        # "هل هو جاهز؟" -> "هل العنصر جاهز؟"
        question = re.sub(r'\bهو\b', 'العنصر', question)
        question = re.sub(r'\bهي\b', 'العنصر', question)

        return question

    def _split_compound_question(self, question: str) -> str:
        """تقسيم سؤال مركب للجزء الأول (الأهم)"""
        # "What is X and how does Y work?" -> "What is X?"
        parts = re.split(r'\s+(و|and|or|أو)\s+', question, maxsplit=1)

        if len(parts) >= 2:
            return parts[0] + " [Note: Focus on this first]"

        return question

    def _add_context_to_open_question(self, question: str) -> str:
        """إضافة سياق للأسئلة المفتوحة جداً"""
        # "What about that?" -> "What specific aspect of the current task should be addressed?"
        if re.search(r'\b(that|this|ذلك|هذا)\b', question, re.IGNORECASE):
            return "What specific aspect of the current task should be addressed?"

        return question


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("❓ Self-Questioning System - Test")
    print("=" * 70)

    system = SelfQuestioningSystem()

    test_questions = [
        "ماذا؟",  # غامض جداً
        "How?",  # غامض جداً
        "What about that thing?",  # غامض + ضمائر غامضة
        "Analyze the data and generate report and send email",  # مركب
        "What is the sentiment analysis accuracy on our test dataset?",  # واضح
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}️⃣ Original: \"{question}\"")
        print("-" * 70)

        # تحليل
        analysis = system.analyze_ambiguity(question)
        print(f"   Ambiguous: {analysis.is_ambiguous}")
        print(f"   Ambiguity Score: {analysis.ambiguity_score:.2f}")
        print(f"   Type: {analysis.ambiguity_type.value}")

        if analysis.issues:
            print(f"   Issues:")
            for issue in analysis.issues:
                print(f"     • {issue}")

        # إعادة صياغة
        if analysis.is_ambiguous:
            rephrased = system.rephrase_question(question, analysis)
            print(f"\n   ✅ Rephrased: \"{rephrased.rephrased}\"")
            print(f"   Clarity Score: {rephrased.clarity_score:.2f}")
            print(f"   Improvements:")
            for improvement in rephrased.improvements:
                print(f"     • {improvement}")
        else:
            print(f"\n   ✅ Already clear - no rephrasing needed")

    print("\n" + "=" * 70)
    print("📊 Statistics:")
    stats = system.get_statistics()
    for key, value in stats.items():
        if key != "recent_improvements":
            print(f"  {key}: {value}")
    print()
