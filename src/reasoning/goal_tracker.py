"""
Goal Tracker - متتبع الهدف
===============================

يراقب الهدف الأصلي ويكتشف متى يفقد النظام تركيزه.
يوفر إشارة واضحة لإيقاف التفكير عند فقدان الهدف بالكامل.

Addresses Q5 from Self-Audit (Meta-Cognition)

Author: Noogh AI Team
Date: 2025-11-10
Priority: HIGH
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import re


@dataclass
class GoalStatus:
    """حالة الهدف الحالية"""
    aligned: bool
    alignment_score: float  # 0.0-1.0
    deviation_reason: str
    recommendation: str
    critical: bool  # هل الانحراف خطير؟


@dataclass
class GoalMilestone:
    """نقطة تقدم نحو الهدف"""
    milestone_id: str
    timestamp: datetime
    description: str
    progress_percentage: float
    sub_goals_completed: List[str]


class GoalTracker:
    """
    متتبع الهدف

    يراقب:
    1. التوافق مع الهدف الأصلي
    2. التقدم نحو الهدف
    3. الأهداف الفرعية المكتملة
    4. الانحرافات الخطيرة
    """

    def __init__(self,
                 critical_deviation_threshold: float = 0.3,
                 max_history: int = 50):
        """
        Args:
            critical_deviation_threshold: حد الانحراف الخطير (0.0-1.0)
            max_history: أقصى عدد من التحديثات المخزنة
        """
        self.critical_deviation_threshold = critical_deviation_threshold
        self.max_history = max_history

        self.original_goal: Optional[str] = None
        self.current_focus: Optional[str] = None
        self.sub_goals: List[str] = []
        self.completed_sub_goals: List[str] = []

        self.milestones: deque = deque(maxlen=max_history)
        self.alignment_history: deque = deque(maxlen=max_history)

        # إحصائيات
        self.total_checks = 0
        self.critical_deviations = 0
        self.times_refocused = 0

    def set_goal(self, goal: str, sub_goals: Optional[List[str]] = None):
        """تحديد الهدف الأصلي"""
        self.original_goal = goal
        self.current_focus = goal
        self.sub_goals = sub_goals or []
        self.completed_sub_goals = []
        self.milestones.clear()
        self.alignment_history.clear()

        # تسجيل نقطة البداية
        self._add_milestone(
            description="Goal started",
            progress_percentage=0.0,
            sub_goals_completed=[]
        )

    def update_focus(self, current_focus: str) -> GoalStatus:
        """
        تحديث التركيز الحالي وفحص التوافق

        Returns:
            GoalStatus مع توصيات
        """
        if not self.original_goal:
            return GoalStatus(
                aligned=True,
                alignment_score=1.0,
                deviation_reason="No goal set",
                recommendation="Continue",
                critical=False
            )

        self.current_focus = current_focus
        self.total_checks += 1

        # حساب التوافق
        alignment_score = self._calculate_alignment(self.original_goal, current_focus)

        # تخزين في التاريخ
        self.alignment_history.append({
            "timestamp": datetime.now(timezone.utc),
            "focus": current_focus,
            "score": alignment_score
        })

        # تحديد الحالة
        if alignment_score >= 0.7:
            # توافق جيد
            status = GoalStatus(
                aligned=True,
                alignment_score=alignment_score,
                deviation_reason="",
                recommendation="Continue - well aligned with goal",
                critical=False
            )
        elif alignment_score >= self.critical_deviation_threshold:
            # انحراف متوسط
            status = GoalStatus(
                aligned=False,
                alignment_score=alignment_score,
                deviation_reason="Moderate deviation from original goal",
                recommendation="Warning - consider refocusing on original goal",
                critical=False
            )
        else:
            # انحراف خطير!
            self.critical_deviations += 1

            status = GoalStatus(
                aligned=False,
                alignment_score=alignment_score,
                deviation_reason="Critical deviation - focus completely lost",
                recommendation="STOP reasoning - refocus or terminate",
                critical=True
            )

        return status

    def complete_sub_goal(self, sub_goal: str) -> float:
        """
        تسجيل إنجاز هدف فرعي

        Returns:
            progress_percentage (0-100)
        """
        if sub_goal not in self.completed_sub_goals:
            self.completed_sub_goals.append(sub_goal)

        # حساب التقدم
        if self.sub_goals:
            progress = (len(self.completed_sub_goals) / len(self.sub_goals)) * 100
        else:
            # لا توجد أهداف فرعية محددة
            progress = 50.0  # تقدير عام

        # تسجيل milestone
        self._add_milestone(
            description=f"Completed: {sub_goal}",
            progress_percentage=progress,
            sub_goals_completed=self.completed_sub_goals.copy()
        )

        return progress

    def refocus_on_goal(self):
        """إعادة التركيز على الهدف الأصلي"""
        self.current_focus = self.original_goal
        self.times_refocused += 1

    def should_stop_reasoning(self) -> Tuple[bool, str]:
        """
        هل يجب إيقاف التفكير؟

        Returns:
            (should_stop, reason)
        """
        if not self.original_goal:
            return False, "No goal set"

        # 1. فحص الانحراف الحالي
        if self.alignment_history:
            latest = self.alignment_history[-1]

            if latest["score"] < self.critical_deviation_threshold:
                return True, "Critical goal deviation detected"

        # 2. فحص الانحراف المستمر
        if len(self.alignment_history) >= 5:
            recent_scores = [h["score"] for h in list(self.alignment_history)[-5:]]
            avg_recent = sum(recent_scores) / len(recent_scores)

            if avg_recent < 0.4:
                return True, "Sustained goal deviation over multiple steps"

        # 3. فحص الانحرافات الخطيرة المتكررة
        if self.critical_deviations >= 3:
            return True, "Too many critical deviations"

        # 4. فحص إذا كان الهدف مكتمل
        if self.sub_goals and len(self.completed_sub_goals) >= len(self.sub_goals):
            return True, "Goal completed - all sub-goals achieved"

        return False, "Continue"

    def get_progress_report(self) -> Dict[str, Any]:
        """تقرير التقدم"""
        if not self.original_goal:
            return {"status": "no_goal"}

        # حساب التقدم
        if self.sub_goals:
            progress = (len(self.completed_sub_goals) / len(self.sub_goals)) * 100
        else:
            progress = 0.0

        # حساب متوسط التوافق
        if self.alignment_history:
            avg_alignment = sum(h["score"] for h in self.alignment_history) / len(self.alignment_history)
        else:
            avg_alignment = 1.0

        return {
            "original_goal": self.original_goal,
            "current_focus": self.current_focus,
            "progress_percentage": progress,
            "sub_goals_total": len(self.sub_goals),
            "sub_goals_completed": len(self.completed_sub_goals),
            "average_alignment": avg_alignment,
            "total_checks": self.total_checks,
            "critical_deviations": self.critical_deviations,
            "times_refocused": self.times_refocused,
            "milestones_count": len(self.milestones)
        }

    def _calculate_alignment(self, goal: str, focus: str) -> float:
        """
        حساب التوافق بين الهدف والتركيز الحالي

        Returns:
            alignment_score (0.0-1.0)
        """
        # تحويل لأحرف صغيرة
        goal_lower = goal.lower()
        focus_lower = focus.lower()

        # 1. تشابه النص المباشر
        if goal_lower == focus_lower:
            return 1.0

        if goal_lower in focus_lower or focus_lower in goal_lower:
            return 0.9

        # 2. Jaccard Similarity على مستوى الكلمات
        goal_words = set(self._tokenize(goal_lower))
        focus_words = set(self._tokenize(focus_lower))

        if not goal_words or not focus_words:
            return 0.0

        intersection = len(goal_words & focus_words)
        union = len(goal_words | focus_words)

        jaccard = intersection / union if union > 0 else 0.0

        # 3. فحص الكلمات المفتاحية المهمة
        # (الأفعال والأسماء الرئيسية تحمل وزن أكبر)
        important_words = self._extract_important_words(goal_lower)

        if important_words:
            important_in_focus = sum(1 for w in important_words if w in focus_lower)
            keyword_match = important_in_focus / len(important_words)

            # دمج jaccard مع keyword match (وزن 60% للكلمات المفتاحية)
            final_score = (0.4 * jaccard) + (0.6 * keyword_match)
        else:
            final_score = jaccard

        return final_score

    def _tokenize(self, text: str) -> List[str]:
        """تقسيم النص لكلمات"""
        # إزالة علامات الترقيم
        text = re.sub(r'[^\w\s]', ' ', text)

        # تقسيم وإزالة الكلمات القصيرة جداً
        words = [w for w in text.split() if len(w) > 2]

        return words

    def _extract_important_words(self, text: str) -> List[str]:
        """استخراج الكلمات المهمة (أفعال، أسماء رئيسية)"""
        # كلمات شائعة للتجاهل
        stop_words = {
            'في', 'من', 'إلى', 'على', 'عن', 'هذا', 'ذلك', 'التي', 'الذي',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }

        words = self._tokenize(text)

        # إزالة stop words
        important = [w for w in words if w not in stop_words]

        return important

    def _add_milestone(self, description: str, progress_percentage: float,
                      sub_goals_completed: List[str]):
        """إضافة milestone للتقدم"""
        milestone = GoalMilestone(
            milestone_id=f"milestone_{len(self.milestones)}",
            timestamp=datetime.now(timezone.utc),
            description=description,
            progress_percentage=progress_percentage,
            sub_goals_completed=sub_goals_completed.copy()
        )

        self.milestones.append(milestone)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🎯 Goal Tracker - Test")
    print("=" * 70)

    tracker = GoalTracker(critical_deviation_threshold=0.3)

    print("\n1️⃣ Setting goal...")
    tracker.set_goal(
        goal="Analyze user sentiment from customer reviews",
        sub_goals=[
            "Load review data",
            "Preprocess text",
            "Run sentiment analysis",
            "Generate report"
        ]
    )

    print(f"Goal set: {tracker.original_goal}")
    print(f"Sub-goals: {len(tracker.sub_goals)}")
    print()

    # تحديث 1: توافق جيد
    print("2️⃣ Update 1: Good alignment")
    status = tracker.update_focus("Loading customer review data from database")
    print(f"  Aligned: {status.aligned}")
    print(f"  Alignment Score: {status.alignment_score:.2f}")
    print(f"  Recommendation: {status.recommendation}")
    print()

    # إكمال هدف فرعي
    print("3️⃣ Completing sub-goal...")
    progress = tracker.complete_sub_goal("Load review data")
    print(f"  Progress: {progress:.1f}%")
    print()

    # تحديث 2: انحراف متوسط
    print("4️⃣ Update 2: Moderate deviation")
    status = tracker.update_focus("Researching latest NLP models on arXiv")
    print(f"  Aligned: {status.aligned}")
    print(f"  Alignment Score: {status.alignment_score:.2f}")
    print(f"  Recommendation: {status.recommendation}")
    print(f"  Critical: {status.critical}")
    print()

    # تحديث 3: انحراف خطير!
    print("5️⃣ Update 3: Critical deviation")
    status = tracker.update_focus("Planning weekend vacation to the beach")
    print(f"  Aligned: {status.aligned}")
    print(f"  Alignment Score: {status.alignment_score:.2f}")
    print(f"  Deviation Reason: {status.deviation_reason}")
    print(f"  Recommendation: {status.recommendation}")
    print(f"  Critical: {status.critical} ⚠️")
    print()

    # فحص إذا يجب الإيقاف
    should_stop, reason = tracker.should_stop_reasoning()
    print(f"6️⃣ Should stop reasoning: {should_stop}")
    if should_stop:
        print(f"   Reason: {reason}")
    print()

    # تقرير التقدم
    print("=" * 70)
    print("📊 Progress Report:")
    report = tracker.get_progress_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    print()
