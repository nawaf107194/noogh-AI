"""
Loop Detection System - نظام كشف حلقات التفكير
===================================================

يكتشف عندما يدخل النظام في حلقة تفكير دائرية ويوقفها تلقائياً
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import hashlib
import json


@dataclass
class ThinkingState:
    """حالة تفكير واحدة"""
    state_id: str
    timestamp: datetime
    context: Dict[str, Any]
    reasoning_step: str
    outputs: List[str]

    def get_signature(self) -> str:
        """الحصول على توقيع فريد للحالة"""
        # نستخدم hash من المحتوى الرئيسي
        content = json.dumps({
            "reasoning": self.reasoning_step,
            "outputs": sorted(self.outputs),
            "key_context": {k: v for k, v in self.context.items() if k in ['intent', 'query', 'action']}
        }, sort_keys=True)

        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class LoopDetection:
    """كشف حلقة تفكير"""
    loop_detected: bool
    loop_length: int
    repeated_states: List[str]
    confidence: float  # 0.0 - 1.0
    recommendation: str


class LoopDetector:
    """
    كاشف حلقات التفكير الدائرية

    يكتشف:
    1. التكرار المباشر (نفس الحالة تظهر مرتين)
    2. الحلقات الطويلة (دورة من عدة حالات تتكرر)
    3. الحالات المتشابهة جداً (quasi-loops)
    """

    def __init__(self,
                 max_history: int = 100,
                 loop_threshold: int = 3,
                 similarity_threshold: float = 0.85):
        """
        Args:
            max_history: أقصى عدد من الحالات للاحتفاظ بها
            loop_threshold: كم مرة يجب تكرار النمط ليعتبر حلقة
            similarity_threshold: نسبة التشابه المطلوبة لاعتبار حالتين متماثلتين
        """
        self.max_history = max_history
        self.loop_threshold = loop_threshold
        self.similarity_threshold = similarity_threshold

        # سجل الحالات
        self.state_history: deque = deque(maxlen=max_history)
        self.signature_counts: Dict[str, int] = {}
        self.signature_positions: Dict[str, List[int]] = {}

        # إحصائيات
        self.total_states = 0
        self.loops_detected = 0
        self.loops_prevented = 0

    def add_state(self,
                  reasoning_step: str,
                  context: Dict[str, Any],
                  outputs: List[str]) -> LoopDetection:
        """
        إضافة حالة تفكير جديدة والتحقق من وجود حلقات

        Returns:
            LoopDetection: نتيجة الكشف
        """
        # إنشاء حالة جديدة
        state = ThinkingState(
            state_id=f"state_{self.total_states}",
            timestamp=datetime.now(timezone.utc),
            context=context,
            reasoning_step=reasoning_step,
            outputs=outputs
        )

        # الحصول على التوقيع
        signature = state.get_signature()

        # تحديث الإحصائيات
        self.total_states += 1

        # تسجيل الموقع
        if signature not in self.signature_positions:
            self.signature_positions[signature] = []
        self.signature_positions[signature].append(len(self.state_history))

        # تحديث عداد التكرار
        self.signature_counts[signature] = self.signature_counts.get(signature, 0) + 1

        # إضافة للسجل
        self.state_history.append(state)

        # الكشف عن الحلقات
        return self._detect_loop(signature)

    def _detect_loop(self, current_signature: str) -> LoopDetection:
        """كشف الحلقات بناءً على التوقيع الحالي"""

        # 1. الكشف البسيط: تكرار مباشر
        repeat_count = self.signature_counts[current_signature]

        if repeat_count >= self.loop_threshold:
            self.loops_detected += 1

            return LoopDetection(
                loop_detected=True,
                loop_length=1,
                repeated_states=[current_signature],
                confidence=1.0,
                recommendation="إيقاف فوري - تكرار مباشر لنفس الحالة"
            )

        # 2. كشف الحلقات الطويلة (patterns)
        if len(self.state_history) >= 6:
            pattern_detection = self._detect_pattern_loop()
            if pattern_detection.loop_detected:
                self.loops_detected += 1
                return pattern_detection

        # 3. كشف الحالات المتشابهة جداً (quasi-loops)
        if len(self.state_history) >= 4:
            similarity_detection = self._detect_similarity_loop(current_signature)
            if similarity_detection.loop_detected:
                return similarity_detection

        # لا توجد حلقة
        return LoopDetection(
            loop_detected=False,
            loop_length=0,
            repeated_states=[],
            confidence=0.0,
            recommendation="متابعة التفكير"
        )

    def _detect_pattern_loop(self) -> LoopDetection:
        """
        كشف حلقات الأنماط (A->B->C->A->B->C)
        """
        if len(self.state_history) < 6:
            return LoopDetection(False, 0, [], 0.0, "")

        # فحص آخر 12 حالة
        recent_states = list(self.state_history)[-12:]
        recent_sigs = [s.get_signature() for s in recent_states]

        # فحص أنماط بطول 2-6
        for pattern_length in range(2, 7):
            if len(recent_sigs) < pattern_length * 2:
                continue

            # استخراج آخر نمط
            last_pattern = recent_sigs[-pattern_length:]

            # البحث عن تكرار
            occurrences = 0
            for i in range(len(recent_sigs) - pattern_length, -1, -pattern_length):
                current_pattern = recent_sigs[i:i+pattern_length]
                if current_pattern == last_pattern:
                    occurrences += 1
                else:
                    break

            if occurrences >= 2:
                return LoopDetection(
                    loop_detected=True,
                    loop_length=pattern_length,
                    repeated_states=last_pattern,
                    confidence=min(1.0, occurrences / 3),
                    recommendation=f"حلقة نمط بطول {pattern_length} - تغيير الاستراتيجية"
                )

        return LoopDetection(False, 0, [], 0.0, "")

    def _detect_similarity_loop(self, current_signature: str) -> LoopDetection:
        """
        كشف الحالات المتشابهة جداً (ليست نفس الحالة لكن قريبة جداً)
        """
        if len(self.state_history) < 4:
            return LoopDetection(False, 0, [], 0.0, "")

        # مقارنة مع آخر 10 حالات
        recent_states = list(self.state_history)[-10:]
        current_state = recent_states[-1]

        similar_count = 0
        similar_sigs = []

        for state in recent_states[:-1]:
            similarity = self._calculate_similarity(current_state, state)

            if similarity >= self.similarity_threshold:
                similar_count += 1
                similar_sigs.append(state.get_signature())

        if similar_count >= 2:
            return LoopDetection(
                loop_detected=True,
                loop_length=similar_count,
                repeated_states=similar_sigs + [current_signature],
                confidence=similar_count / 5,
                recommendation="حالات متشابهة جداً - تنويع الاستراتيجية"
            )

        return LoopDetection(False, 0, [], 0.0, "")

    def _calculate_similarity(self, state1: ThinkingState, state2: ThinkingState) -> float:
        """
        حساب التشابه بين حالتين (0.0 - 1.0)
        """
        # مقارنة خطوات التفكير
        reasoning_similarity = self._text_similarity(
            state1.reasoning_step,
            state2.reasoning_step
        )

        # مقارنة المخرجات
        outputs_similarity = self._list_similarity(
            state1.outputs,
            state2.outputs
        )

        # مقارنة السياق الرئيسي
        context_similarity = self._dict_similarity(
            state1.context,
            state2.context
        )

        # متوسط مرجح
        return (
            reasoning_similarity * 0.5 +
            outputs_similarity * 0.3 +
            context_similarity * 0.2
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """حساب تشابه نصين"""
        if not text1 or not text2:
            return 0.0

        # تحويل لكلمات
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """حساب تشابه قائمتين"""
        if not list1 or not list2:
            return 0.0

        set1 = set(list1)
        set2 = set(list2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _dict_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """حساب تشابه قاموسين"""
        if not dict1 or not dict2:
            return 0.0

        # فحص المفاتيح المشتركة
        common_keys = set(dict1.keys()) & set(dict2.keys())

        if not common_keys:
            return 0.0

        matches = sum(1 for k in common_keys if dict1[k] == dict2[k])

        return matches / len(common_keys)

    def should_stop_reasoning(self, detection: LoopDetection) -> bool:
        """
        هل يجب إيقاف التفكير؟
        """
        if not detection.loop_detected:
            return False

        # إيقاف فوري للحلقات المباشرة
        if detection.loop_length == 1 and detection.confidence >= 0.8:
            self.loops_prevented += 1
            return True

        # إيقاف للحلقات الطويلة الواضحة
        if detection.confidence >= 0.7:
            self.loops_prevented += 1
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الكاشف"""
        return {
            "total_states": self.total_states,
            "unique_states": len(self.signature_counts),
            "loops_detected": self.loops_detected,
            "loops_prevented": self.loops_prevented,
            "current_history_size": len(self.state_history),
            "loop_detection_rate": self.loops_detected / max(1, self.total_states) * 100
        }

    def reset(self):
        """إعادة تعيين الكاشف"""
        self.state_history.clear()
        self.signature_counts.clear()
        self.signature_positions.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


if __name__ == "__main__":
    print("🔄 Loop Detection System - Test")
    print("=" * 60)

    detector = LoopDetector(loop_threshold=2)

    # محاكاة حلقة تفكير
    print("\n1️⃣ Testing direct repetition...")
    for i in range(3):
        detection = detector.add_state(
            reasoning_step="Analyzing user query",
            context={"intent": "search", "query": "test"},
            outputs=["No results found"]
        )

        print(f"   Step {i+1}: Loop={detection.loop_detected}, Confidence={detection.confidence:.2f}")

        if detector.should_stop_reasoning(detection):
            print(f"   ⚠️  {detection.recommendation}")
            break

    detector.reset()

    # محاكاة حلقة نمط
    print("\n2️⃣ Testing pattern loop (A->B->A->B)...")
    pattern = [
        ("Step A", {"intent": "analyze"}, ["Result A"]),
        ("Step B", {"intent": "process"}, ["Result B"]),
    ]

    for i in range(6):
        step = pattern[i % 2]
        detection = detector.add_state(step[0], step[1], step[2])

        print(f"   Step {i+1} ({step[0]}): Loop={detection.loop_detected}")

        if detection.loop_detected:
            print(f"   ⚠️  Pattern detected! Length={detection.loop_length}")
            break

    print("\n" + "=" * 60)
    print("📊 Statistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
