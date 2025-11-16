#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠🌉 Cognitive Decision Bridge - جسر الإدراك مع محرك القرار
==========================================================

Integrates Cognitive Core (4096-neuron brain + self-improvement)
with Decision Engine (government ministers + decision making).

This creates a complete cognitive loop:
Decision → Neural Processing → Storage → Analysis → Improvement → Better Decision

Author: Noogh AI Team
Version: 3.0.0
"""

import logging
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..decision import (
    DecisionEngine,
    DecisionContext,
    DecisionOption,
    DecisionType,
    DecisionPriority,
    get_decision_engine
)

from ..brain.unified_brain import create_brain, UnifiedNooghBrain


logger = logging.getLogger(__name__)


class CognitiveDecisionBridge:
    """
    🧠🌉 جسر متكامل بين الإدراك والقرار (v4.0 - Unified Brain)

    يربط بين:
    1. UnifiedNooghBrain (MegaBrain V5) - الفهم العميق
    2. Decision Engine - صنع القرار
    3. Government System - الوزراء

    النتيجة:
    - نظام قرار ذكي يتحسن تلقائياً باستخدام النظام الذاتي المدمج
    """

    def __init__(
        self,
        unified_brain: Optional[UnifiedNooghBrain] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        """
        تهيئة الجسر المعرفي

        Args:
            unified_brain: العقل الموحد
            decision_engine: محرك القرار
        """

        # Initialize unified brain
        if unified_brain is None:
            logger.info("🧠 Creating Unified Noogh Brain...")
            self.unified_brain = create_brain(use_autonomous=True)
            # Create a default model for inference if none exists
            if not self.unified_brain.is_ready:
                self.unified_brain.create_mega_brain(config="micro")
        else:
            self.unified_brain = unified_brain

        # Initialize decision engine
        if decision_engine is None:
            self.decision_engine = get_decision_engine()
        else:
            self.decision_engine = decision_engine

        # Statistics
        self.total_decisions = 0
        self.cognitive_enhanced_decisions = 0

        logger.info("✅ Cognitive Decision Bridge initialized (Unified Brain)")
        brain_info = self.unified_brain.get_system_info()
        model_stats = brain_info.get("model", {}).get("stats", {})
        logger.info(f"   Model: {model_stats.get('model_name', 'N/A')}")
        logger.info(f"   Parameters: {model_stats.get('parameters_millions', 0):.2f}M")
        logger.info(f"   Autonomous System: {'✅' if self.unified_brain.use_autonomous else '❌'}")

    def process_government_decision(
        self,
        user_request: str,
        ministers_analysis: Dict[str, Any],
        priority: str = "MEDIUM",
        decision_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        معالجة قرار حكومي بالكامل مع التحسين المعرفي

        هذه الدالة هي النقطة المركزية التي تجمع كل شيء:

        Flow:
        1. تحليل الطلب عبر Neural Brain (4096 neurons)
        2. استخراج options من الوزراء
        3. صنع القرار عبر Decision Engine
        4. حفظ القرار في Memory Vault
        5. تحليل النتائج (عند تسجيل الـ outcome)
        6. تحسين ذاتي تلقائي كل 24 ساعة

        Args:
            user_request: طلب المستخدم
            ministers_analysis: تحليلات الوزراء
            priority: CRITICAL, HIGH, MEDIUM, LOW, INFO
            decision_type: نوع القرار (اختياري)

        Returns:
            القرار النهائي + معلومات معرفية
        """

        logger.info("=" * 70)
        logger.info("🧠 COGNITIVE DECISION PROCESSING")
        logger.info("=" * 70)

        self.total_decisions += 1

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Neural Processing (Unified Brain)
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 1: Neural processing (Unified Brain)...")

        # The new brain takes a dictionary as input
        # In a real scenario, this would be a proper feature vector
        dummy_input = {"user_request": user_request}
        neural_output = self.unified_brain.inference(dummy_input)

        # Extract neural insights
        neural_insights = {
            "confidence": torch.sigmoid(neural_output).mean().item() if neural_output is not None else 0.5,
        }

        logger.info(f"   Neural confidence: {neural_insights['confidence']:.2%}")

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Decision Context Creation
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 2: Creating decision context...")

        # Determine decision type
        inferred_decision_type = self._infer_decision_type(ministers_analysis)
        final_decision_type = decision_type or inferred_decision_type.value

        # Create context
        context = DecisionContext(
            request_id=f"COG-{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            decision_type=final_decision_type,
            priority=DecisionPriority[priority.upper()],
            user_request=user_request,
            ministers_input=ministers_analysis,
            current_state={},
            metadata={
                "neural_confidence": neural_insights["confidence"],
                "cognitive_enhanced": True,
                "brain_version": "v3.0_4096neurons"
            }
        )

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Extract Options from Ministers
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 3: Extracting minister options...")

        options = self._extract_minister_options(ministers_analysis, neural_insights)

        logger.info(f"   Extracted {len(options)} options from ministers")

        if not options:
            logger.warning("   No options from ministers, creating defaults")
            options = self._create_default_options(user_request)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Make Decision (Decision Engine)
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 4: Making decision...")

        decision = self.decision_engine.make_decision(context, options)

        logger.info(f"   Decision: {decision.selected_option.action}")
        logger.info(f"   Confidence: {decision.confidence:.2%}")

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Cognitive Processing (Knowledge Graph)
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 5: Cognitive processing (knowledge graph)...")

        # Extract ministers involved
        ministers_involved = list(ministers_analysis.keys())
        
        # Add decision to knowledge graph
        record_id = f"decision_{decision.decision_id}"
        self.unified_brain.add_knowledge(record_id, data={
            "type": "decision",
            "request": user_request,
            "decision": decision.selected_option.action,
            "confidence": decision.confidence,
            "ministers": ministers_involved,
            "timestamp": decision.timestamp.isoformat()
        })

        self.cognitive_enhanced_decisions += 1

        logger.info(f"   Stored in knowledge graph: {record_id}")

        # ═══════════════════════════════════════════════════════════
        # STEP 6: Prepare Response
        # ═══════════════════════════════════════════════════════════

        logger.info("Step 6: Preparing response...")

        brain_info = self.unified_brain.get_system_info()
        model_stats = brain_info.get("model", {}).get("stats", {})

        response = {
            # Decision info
            "decision_id": decision.decision_id,
            "cognitive_record_id": record_id,
            "action": decision.selected_option.action,
            "description": decision.selected_option.description,
            "confidence": decision.confidence,
            "should_execute": decision.should_execute,

            # Reasoning
            "reasoning": decision.reasoning,
            "alternatives": [
                {
                    "action": alt.action,
                    "confidence": alt.confidence,
                    "description": alt.description
                }
                for alt in decision.alternatives
            ],

            # Neural insights
            "neural_insights": {
                "brain_confidence": neural_insights["confidence"],
                "model_name": model_stats.get("model_name"),
                "parameters_millions": model_stats.get("parameters_millions", 0)
            },

            # Cognitive insights
            "cognitive_insights": {
                "record_id": record_id,
                "knowledge_nodes": brain_info.get("knowledge", {}).get("nodes", 0),
                "autonomous_system_active": self.unified_brain.use_autonomous,
            },

            # Ministers involved
            "ministers_involved": ministers_involved,

            # Timestamps
            "timestamp": decision.timestamp.isoformat(),
            "decision_time_ms": decision.decision_time_ms
        }

        logger.info("=" * 70)
        logger.info("✅ COGNITIVE DECISION COMPLETE")
        logger.info("=" * 70)

        return response

    def record_outcome(
        self,
        decision_id: str,
        cognitive_record_id: str,
        success: bool,
        user_feedback: Optional[str] = None,
        user_rating: Optional[float] = None,
        metrics: Optional[Dict] = None
    ):
        """
        تسجيل نتيجة القرار

        هذا يُفعّل:
        1. تحديث Knowledge Graph
        2. قد يُفعّل تدريب ذاتي إذا كان فشل حرج
        """

        logger.info("=" * 70)
        logger.info("📝 RECORDING OUTCOME")
        logger.info("=" * 70)

        if metrics is None:
            metrics = {}

        # Record in Decision Engine
        self.decision_engine.record_outcome(decision_id, success, metrics)

        # Update knowledge graph with outcome
        outcome = "success" if success else "failure"
        outcome_score = user_rating if user_rating is not None else (1.0 if success else 0.0)
        
        self.unified_brain.add_knowledge(cognitive_record_id, data={
            "outcome": outcome,
            "outcome_score": outcome_score,
            "user_feedback": user_feedback,
            "metrics": metrics
        })

        logger.info(f"   Outcome: {outcome}")
        logger.info(f"   Score: {outcome_score:.2f}")
        if user_feedback:
            logger.info(f"   Feedback: {user_feedback[:100]}")

        # The autonomous system in UnifiedNooghBrain will handle improvements
        if not success and self.unified_brain.use_autonomous:
            logger.warning("⚠️ Critical failure detected! Autonomous system will handle it.")

        logger.info("=" * 70)
        logger.info("✅ OUTCOME RECORDED")
        logger.info("=" * 70)

    def get_daily_reflection(self) -> Dict[str, Any]:
        """
        الحصول على تأمل يومي
        NOTE: This is now handled by the autonomous system within UnifiedNooghBrain.
        This method provides a summary from the brain's perspective.
        """
        logger.info("🔍 Generating daily reflection from Unified Brain...")
        # This is a simplified representation. A real implementation would query
        # the brain's knowledge graph for performance metrics.
        return self.unified_brain.get_system_info()

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════

    def _infer_decision_type(self, ministers_data: Dict) -> DecisionType:
        """استنتاج نوع القرار من الوزراء"""
        ministers = set(ministers_data.keys())
        if any(m in ministers for m in ["finance", "quantitative", "technical_analysis", "portfolio"]):
            return DecisionType.TRADING
        if any(m in ministers for m in ["training", "research"]):
            return DecisionType.LEARNING
        if any(m in ministers for m in ["resource", "performance"]):
            return DecisionType.RESOURCE
        if "security" in ministers or "privacy" in ministers:
            return DecisionType.SECURITY
        return DecisionType.ANALYSIS

    def _extract_minister_options(
        self,
        ministers_data: Dict[str, Any],
        neural_insights: Dict[str, Any]
    ) -> List[DecisionOption]:
        """استخراج الخيارات من تحليلات الوزراء"""
        options = []
        option_id = 0
        for minister, data in ministers_data.items():
            if not isinstance(data, dict):
                continue
            recommendations = data.get("recommendations", []) or data.get("actions", []) or data.get("suggestions", [])
            for rec in recommendations:
                if isinstance(rec, str):
                    option = DecisionOption(
                        option_id=f"OPT-{option_id}",
                        action=rec,
                        description=f"{minister} suggests: {rec}",
                        confidence=data.get("confidence", 0.5)
                    )
                elif isinstance(rec, dict):
                    base_confidence = rec.get("confidence", 0.5)
                    neural_boost = neural_insights["confidence"] * 0.2
                    adjusted_confidence = min(1.0, base_confidence + neural_boost)
                    option = DecisionOption(
                        option_id=f"OPT-{option_id}",
                        action=rec.get("action", "unknown"),
                        description=rec.get("description", ""),
                        confidence=adjusted_confidence,
                        expected_value=rec.get("expected_value", 0.0),
                        risk_score=rec.get("risk", 0.0),
                        cost=rec.get("cost", 0.0)
                    )
                    option.reasoning = rec.get("reasoning", [])
                option.ministers_votes[minister] = data.get("confidence", 0.5)
                option.supporting_evidence["neural_enhanced"] = True
                option.supporting_evidence["neural_confidence"] = neural_insights["confidence"]
                options.append(option)
                option_id += 1
        return options

    def _create_default_options(self, user_request: str) -> List[DecisionOption]:
        """خيارات افتراضية"""
        return [
            DecisionOption(
                option_id="DEFAULT-1",
                action="analyze_further",
                description="Analyze the request in more detail using neural brain",
                confidence=0.6,
                risk_score=0.1
            ),
            DecisionOption(
                option_id="DEFAULT-2",
                action="request_clarification",
                description="Ask user for more information",
                confidence=0.5,
                risk_score=0.0
            )
        ]


# Singleton
_bridge = None

def get_cognitive_decision_bridge() -> CognitiveDecisionBridge:
    """
    الحصول على جسر القرار المعرفي (Singleton)

    Returns:
        الجسر المعرفي
    """
    global _bridge
    if _bridge is None:
        _bridge = CognitiveDecisionBridge()
    return _bridge
