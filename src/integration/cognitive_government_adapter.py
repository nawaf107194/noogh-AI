#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️🧠 Cognitive Government Adapter - محول حكومي معرفي
=======================================================

Enhanced Government Adapter that uses Cognitive Decision Bridge
to provide self-improving government decision-making.

الحكومة + الذكاء المعرفي = حكومة ذاتية التحسين

Author: Noogh AI Team
Version: 3.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .cognitive_decision_bridge import get_cognitive_decision_bridge


logger = logging.getLogger(__name__)


class CognitiveGovernmentAdapter:
    """
    🏛️🧠 محول حكومي معرفي

    يربط 14 وزيراً مع:
    - Neural Brain v3.0 (4096 neurons)
    - Cognitive Core (self-improvement)
    - Decision Engine (intelligent decisions)

    كل قرار حكومي يمر عبر:
    1. تحليل الوزراء (14 ministers)
    2. معالجة عصبية (4096 neurons)
    3. صنع قرار ذكي
    4. حفظ في الذاكرة
    5. تحسين ذاتي تلقائي كل 24 ساعة

    النتيجة: حكومة تتعلم من أخطائها وتتحسن تلقائياً!
    """

    def __init__(self, enable_autonomous_improvement: bool = True):
        """
        تهيئة المحول الحكومي المعرفي

        Args:
            enable_autonomous_improvement: تفعيل التحسين الذاتي (افتراضياً True)
        """

        # Get cognitive decision bridge
        self.bridge = get_cognitive_decision_bridge(
            enable_autonomous_improvement=enable_autonomous_improvement,
            improvement_interval_hours=24
        )

        # Statistics
        self.total_government_decisions = 0
        self.decisions_by_minister = {}

        logger.info("🏛️🧠 Cognitive Government Adapter initialized")
        logger.info(f"   Autonomous improvement: {enable_autonomous_improvement}")

    def process_government_request(
        self,
        user_request: str,
        ministers_analysis: Dict[str, Any],
        priority: str = "MEDIUM",
        decision_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        معالجة طلب حكومي

        Args:
            user_request: طلب المستخدم
            ministers_analysis: تحليلات الوزراء
                مثال:
                {
                    "finance": {
                        "recommendations": ["buy_btc", "hold_cash"],
                        "confidence": 0.85,
                        "reasoning": ["market is bullish", "low risk"]
                    },
                    "technical_analysis": {
                        "recommendations": ["buy_signal"],
                        "confidence": 0.90
                    },
                    ...
                }
            priority: CRITICAL, HIGH, MEDIUM, LOW, INFO
            decision_type: نوع القرار (اختياري)

        Returns:
            القرار النهائي مع تحليل معرفي
        """

        logger.info("=" * 70)
        logger.info("🏛️ GOVERNMENT DECISION REQUEST")
        logger.info("=" * 70)
        logger.info(f"Request: {user_request[:100]}...")
        logger.info(f"Ministers involved: {list(ministers_analysis.keys())}")
        logger.info(f"Priority: {priority}")

        # Update statistics
        self.total_government_decisions += 1
        for minister in ministers_analysis.keys():
            self.decisions_by_minister[minister] = \
                self.decisions_by_minister.get(minister, 0) + 1

        # Process through cognitive bridge
        response = self.bridge.process_government_decision(
            user_request=user_request,
            ministers_analysis=ministers_analysis,
            priority=priority,
            decision_type=decision_type
        )

        # Enhance response with government-specific info
        response["government_info"] = {
            "total_government_decisions": self.total_government_decisions,
            "ministers_involved": list(ministers_analysis.keys()),
            "minister_participation": self.decisions_by_minister
        }

        logger.info("=" * 70)
        logger.info("✅ GOVERNMENT DECISION COMPLETE")
        logger.info(f"   Action: {response['action']}")
        logger.info(f"   Confidence: {response['confidence']:.2%}")
        logger.info("=" * 70)

        return response

    def record_government_outcome(
        self,
        decision_id: str,
        cognitive_record_id: str,
        success: bool,
        execution_results: Optional[Dict] = None,
        user_feedback: Optional[str] = None,
        user_rating: Optional[float] = None
    ):
        """
        تسجيل نتيجة القرار الحكومي

        Args:
            decision_id: معرّف القرار
            cognitive_record_id: معرّف السجل المعرفي
            success: نجح أم فشل
            execution_results: نتائج التنفيذ
            user_feedback: ملاحظات المستخدم
            user_rating: تقييم (0-1)
        """

        logger.info("📝 Recording government outcome...")

        metrics = {}
        if execution_results:
            metrics["execution_results"] = execution_results

        # Record through bridge
        self.bridge.record_outcome(
            decision_id=decision_id,
            cognitive_record_id=cognitive_record_id,
            success=success,
            user_feedback=user_feedback,
            user_rating=user_rating,
            metrics=metrics
        )

        logger.info("✅ Government outcome recorded")

    def get_government_daily_report(self) -> Dict[str, Any]:
        """
        تقرير يومي للحكومة

        Returns:
            تقرير شامل عن أداء الحكومة والتوصيات
        """

        logger.info("📊 Generating government daily report...")

        # Get reflection from bridge
        reflection = self.bridge.get_daily_reflection()

        # Add government-specific analysis
        report = {
            "timestamp": datetime.now().isoformat(),
            "government_statistics": {
                "total_decisions": self.total_government_decisions,
                "decisions_by_minister": self.decisions_by_minister,
                "most_active_minister": max(
                    self.decisions_by_minister,
                    key=self.decisions_by_minister.get
                ) if self.decisions_by_minister else "none"
            },
            "cognitive_reflection": reflection,
            "recommendations": self._generate_government_recommendations(reflection)
        }

        logger.info("✅ Government daily report generated")

        return report

    def get_minister_performance(self, minister_name: str) -> Dict[str, Any]:
        """
        تحليل أداء وزير معين

        Args:
            minister_name: اسم الوزير

        Returns:
            إحصائيات أداء الوزير
        """

        # Get cognitive stats
        cognitive_stats = self.bridge.cognitive_core.get_statistics()
        memory_vault = self.bridge.cognitive_core.memory_vault

        # Get decisions involving this minister
        all_decisions = memory_vault.get_all_decisions(limit=1000)
        minister_decisions = [
            d for d in all_decisions
            if minister_name in d.ministers_involved
        ]

        if not minister_decisions:
            return {
                "minister": minister_name,
                "total_decisions": 0,
                "success_rate": 0.0,
                "message": "No decisions found for this minister"
            }

        # Calculate statistics
        total = len(minister_decisions)
        successful = len([d for d in minister_decisions if d.outcome == "success"])
        failed = len([d for d in minister_decisions if d.outcome == "failure"])

        success_rate = successful / total if total > 0 else 0.0
        avg_confidence = sum(d.confidence for d in minister_decisions) / total

        return {
            "minister": minister_name,
            "total_decisions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "recent_decisions": [
                {
                    "timestamp": d.timestamp.isoformat(),
                    "decision": d.decision,
                    "confidence": d.confidence,
                    "outcome": d.outcome
                }
                for d in minister_decisions[-5:]  # Last 5
            ]
        }

    def trigger_government_improvement(self) -> Dict[str, Any]:
        """
        تفعيل تحسين حكومي فوري

        Returns:
            نتائج التحسين
        """

        logger.info("🔧 Triggering government-wide improvement...")

        result = self.bridge.trigger_manual_improvement()

        logger.info("✅ Government improvement triggered")

        return result

    def get_minister_recommendations(self) -> Dict[str, List[str]]:
        """
        الحصول على توصيات لكل وزير

        Returns:
            توصيات لتحسين أداء كل وزير
        """

        recommendations = {}

        # Get reflection
        reflection = self.bridge.get_daily_reflection()
        failure_patterns = reflection.get("reflection", {}).get("failure_patterns", {})

        # Analyze by minister
        by_minister = failure_patterns.get("by_minister", {})

        for minister, failure_count in by_minister.items():
            recs = []

            if failure_count > 5:
                recs.append(f"⚠️ High failure count ({failure_count}). Review decision criteria.")

            # Get minister performance
            perf = self.get_minister_performance(minister)

            if perf["success_rate"] < 0.5:
                recs.append(f"📉 Success rate low ({perf['success_rate']:.1%}). Needs improvement.")

            if perf["avg_confidence"] < 0.6:
                recs.append(f"🤔 Low confidence average ({perf['avg_confidence']:.1%}). Review analysis methods.")

            if not recs:
                recs.append("✅ Performance good. Continue current approach.")

            recommendations[minister] = recs

        return recommendations

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════

    def _generate_government_recommendations(
        self,
        reflection: Dict[str, Any]
    ) -> List[str]:
        """توليد توصيات للحكومة"""

        recommendations = []

        # Get statistics
        cognitive_stats = reflection.get("cognitive_statistics", {})
        memory_stats = cognitive_stats.get("memory_vault", {})

        success_rate = memory_stats.get("success_rate", 0.0)

        # Recommendation based on success rate
        if success_rate < 0.5:
            recommendations.append(
                "🔴 Success rate is low (<50%). Consider reviewing decision criteria."
            )
        elif success_rate < 0.7:
            recommendations.append(
                "🟡 Success rate is moderate (50-70%). Room for improvement."
            )
        else:
            recommendations.append(
                "🟢 Success rate is good (>70%). Keep up the good work!"
            )

        # Check autonomous improvement
        bridge_stats = reflection.get("bridge_statistics", {})
        improvement_triggers = bridge_stats.get("improvement_triggers", 0)

        if improvement_triggers > 10:
            recommendations.append(
                f"⚠️ High number of critical failures ({improvement_triggers}). "
                "Autonomous improvement is actively learning from these."
            )

        # Minister-specific
        minister_participation = self.decisions_by_minister
        if minister_participation:
            most_active = max(minister_participation, key=minister_participation.get)
            least_active = min(minister_participation, key=minister_participation.get)

            recommendations.append(
                f"📊 Most active minister: {most_active} "
                f"({minister_participation[most_active]} decisions)"
            )

            if minister_participation[least_active] < 5:
                recommendations.append(
                    f"💤 {least_active} has low participation "
                    f"({minister_participation[least_active]} decisions). "
                    "Consider involving more."
                )

        return recommendations


# Singleton
_adapter = None

def get_cognitive_government_adapter(
    enable_autonomous_improvement: bool = True
) -> CognitiveGovernmentAdapter:
    """
    الحصول على المحول الحكومي المعرفي (Singleton)

    Args:
        enable_autonomous_improvement: تفعيل التحسين الذاتي

    Returns:
        المحول الحكومي المعرفي
    """
    global _adapter
    if _adapter is None:
        _adapter = CognitiveGovernmentAdapter(
            enable_autonomous_improvement=enable_autonomous_improvement
        )
    return _adapter
