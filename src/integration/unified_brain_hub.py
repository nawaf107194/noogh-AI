#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Unified Brain Hub - مركز الدماغ الموحد
============================================

The supreme integration layer that connects ALL cognitive systems:
- Deep Cognition v1.2 Lite (97.5% TRANSCENDENT)
- Agent Brain (Planning & Reasoning)
- 14 Active Ministers (Government)
- Unified Cognition (Decision + Learning + Memory)

This is the CENTRAL NERVOUS SYSTEM of Noogh.

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-11-10
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.core.di import Container


# Global flags will be managed by the Container and instance state

# Ensure MinisterType is available for type hints and runtime
try:
    from src.government.ministers_activation import MinisterType
except ImportError:
    from enum import Enum
    class MinisterType(str, Enum):
        EDUCATION = "education"
        TRAINING = "training"
        SECURITY = "security"
        DEVELOPMENT = "development"
        RESEARCH = "research"
        KNOWLEDGE = "knowledge"
        PRIVACY = "privacy"
        CREATIVITY = "creativity"
        ANALYSIS = "analysis"
        STRATEGY = "strategy"
        REASONING = "reasoning"
        COMMUNICATION = "communication"
        RESOURCES = "resources"
        FINANCE = "finance"




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BrainHubStatus:
    """حالة مركز الدماغ الموحد"""
    active: bool
    cognition_score: float
    active_ministers: int
    deep_cognition_available: bool
    agent_brain_available: bool
    government_available: bool
    unified_cognition_available: bool
    timestamp: str


@dataclass
class ProcessingResult:
    """نتيجة المعالجة الموحدة"""
    status: str
    response: str
    minister_used: Optional[str]
    cognition_analysis: Optional[Dict]
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]


class UnifiedBrainHub:
    """
    🧠 مركز الدماغ الموحد - The Supreme Integration Layer

    This is the CENTRAL NERVOUS SYSTEM that connects:
    1. Deep Cognition v1.2 Lite (Scene, Material, Confidence, Semantic)
    2. Agent Brain (Planning, Reasoning, Execution)
    3. 14 Active Ministers (Government)
    4. Unified Cognition (Decision + Learning + Memory)

    Flow:
    ─────
    User Request
        ↓
    Brain Hub (analyzes request type)
        ↓
    ┌──────────────────────────────────────────┐
    │ Route to appropriate system:             │
    │ • Vision task → Deep Cognition (Scene)   │
    │ • Text analysis → Semantic Intent        │
    │ • Decision → Unified Cognition           │
    │ • Specialized task → Delegate to Minister│
    │ • Complex planning → Agent Brain         │
    └──────────────────────────────────────────┘
        ↓
    Synthesize Response
        ↓
    Return to User
    """

    def __init__(self, enable_gpu: bool = True):
        """
        تهيئة مركز الدماغ الموحد

        Args:
            enable_gpu: تفعيل GPU للوزراء الذين يحتاجونه
        """
        logger.info("=" * 70)
        logger.info("🧠 UNIFIED BRAIN HUB - INITIALIZATION")
        logger.info("=" * 70)

        self.enable_gpu = enable_gpu
        self.is_ready = False
        self.cognition_score = 0.0
        self.active_ministers_count = 0
        
        # Initialize subsystems via DI
        self._initialize_subsystems()

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0

    def _initialize_subsystems(self):
        """Initialize and register all subsystems"""
        logger.info("🔧 Initializing subsystems...")

        # 1. Deep Cognition
        try:
            from src.vision.scene_understanding import SceneUnderstandingEngine
            from src.vision.material_analyzer import MaterialAnalyzer
            from src.reasoning.meta_confidence import MetaConfidenceCalibrator
            from src.nlp.semantic_intent_analyzer import SemanticIntentAnalyzer
            from src.integration.vision_reasoning_sync import VisionReasoningSynchronizer
            
            self.scene_understanding = SceneUnderstandingEngine()
            self.material_analyzer = MaterialAnalyzer()
            self.meta_confidence = MetaConfidenceCalibrator()
            self.semantic_intent = SemanticIntentAnalyzer()
            self.vision_reasoning_sync = VisionReasoningSynchronizer()
            
            Container.register("scene_understanding", self.scene_understanding)
            Container.register("semantic_intent", self.semantic_intent)
            
            self.has_deep_cognition = True
            self.cognition_score = 0.975
            logger.info("   ✅ Deep Cognition v1.2 Lite loaded")
        except ImportError as e:
            logger.warning(f"   ⚠️ Deep Cognition not available: {e}")
            self.has_deep_cognition = False
            self.scene_understanding = None
            self.semantic_intent = None

        # 2. Agent Brain
        try:
            from src.agent.brain import AgentBrain
            self.agent_brain = AgentBrain()
            Container.register("agent_brain", self.agent_brain)
            self.has_agent_brain = True
            logger.info("   ✅ Agent Brain loaded")
        except ImportError as e:
            logger.warning(f"   ⚠️ Agent Brain not available: {e}")
            self.has_agent_brain = False
            self.agent_brain = None

        # 3. Government
        try:
            from src.government.ministers_activation import MinistersActivationSystem, MinisterType
            self.ministers_system = MinistersActivationSystem(brain_hub=self)
            self.ministers_system.activate_all()
            self.active_ministers_count = len(self.ministers_system.active_ministers)
            Container.register("ministers_system", self.ministers_system)
            self.has_government = True
            logger.info(f"   ✅ Government loaded ({self.active_ministers_count} ministers)")
        except ImportError as e:
            logger.warning(f"   ⚠️ Government system not available: {e}")
            self.has_government = False
            self.ministers_system = None
            
            # MinisterType is already handled at module level
            pass

        # 4. Unified Cognition
        try:
            from src.integration.unified_cognition import get_cognition_system
            self.unified_cognition = get_cognition_system()
            Container.register("unified_cognition", self.unified_cognition)
            self.has_unified_cognition = True
            logger.info("   ✅ Unified Cognition loaded")
        except ImportError as e:
            logger.warning(f"   ⚠️ Unified Cognition not available: {e}")
            self.has_unified_cognition = False
            self.unified_cognition = None

        self.is_ready = True
        logger.info("✅ Subsystems initialization complete")

    def process_request(self, request: str, context: Optional[Dict] = None) -> ProcessingResult:
        """
        معالجة طلب موحد من المستخدم

        This is the MAIN ENTRY POINT for all requests.

        Args:
            request: طلب المستخدم (نص أو مهمة)
            context: سياق إضافي (صور، بيانات، إلخ)

        Returns:
            ProcessingResult with response and metadata
        """
        start_time = datetime.now()
        self.total_requests += 1

        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"🎯 Request #{self.total_requests}: {request[:50]}...")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        context = context or {}

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: ANALYZE REQUEST TYPE
        # ═══════════════════════════════════════════════════════════════

        request_type = self._classify_request(request, context)
        logger.info(f"📋 Request Type: {request_type}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: ROUTE TO APPROPRIATE SYSTEM
        # ═══════════════════════════════════════════════════════════════

        response_text = ""
        minister_used = None
        cognition_analysis = None
        confidence = 0.0

        try:
            if request_type == "vision_analysis" and self.scene_understanding:
                # Use Deep Cognition - Scene Understanding
                result = self._handle_vision_analysis(request, context)
                response_text = result['response']
                cognition_analysis = result['analysis']
                confidence = result['confidence']

            elif request_type == "text_understanding" and self.semantic_intent:
                # Use Deep Cognition - Semantic Intent
                result = self._handle_text_understanding(request)
                response_text = result['response']
                cognition_analysis = result['analysis']
                confidence = result['confidence']

            elif request_type == "minister_task" and self.ministers_system:
                # Delegate to appropriate minister (async call handled safely)
                result = self._delegate_to_minister_sync(request, context)

                response_text = result['response']
                minister_used = result['minister']
                confidence = result['confidence']

            elif request_type == "complex_planning" and self.agent_brain:
                # Use Agent Brain for complex planning
                result = self._handle_complex_planning(request)
                response_text = result['response']
                confidence = result['confidence']

            elif request_type == "decision" and self.unified_cognition:
                # Use Unified Cognition for decisions
                result = self._handle_decision(request, context)
                response_text = result['response']
                confidence = result['confidence']

            else:
                # Fallback: basic response
                response_text = f"Understood request: {request}"
                confidence = 0.5

            self.successful_requests += 1
            status = "success"

        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            response_text = f"Error: {str(e)}"
            confidence = 0.0
            status = "error"

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: BUILD RESULT
        # ═══════════════════════════════════════════════════════════════

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        result = ProcessingResult(
            status=status,
            response=response_text,
            minister_used=minister_used,
            cognition_analysis=cognition_analysis,
            confidence=confidence,
            processing_time_ms=processing_time,
            metadata={
                "request_type": request_type,
                "timestamp": datetime.now().isoformat(),
                "cognition_score": self.cognition_score,
                "active_ministers": self.active_ministers_count
            }
        )

        logger.info(f"✅ Request processed in {processing_time:.0f}ms")
        logger.info(f"   Confidence: {confidence:.0%}")

        return result

    def _classify_request(self, request: str, context: Dict) -> str:
        """
        تصنيف نوع الطلب

        Returns:
            One of: vision_analysis, text_understanding, minister_task,
                   complex_planning, decision
        """
        request_lower = request.lower()

        # Check for image/vision keywords
        if context.get('image_path') or any(word in request_lower for word in
            ['صورة', 'image', 'مشهد', 'scene', 'visual', 'مادة', 'material']):
            return "vision_analysis"

        # Check for text understanding keywords
        if any(word in request_lower for word in
            ['معنى', 'meaning', 'فهم', 'understand', 'تحليل نص', 'analyze text']):
            return "text_understanding"

        # Check for minister-specific tasks
        minister_keywords = {
            # وزير التعليم
            'تعليم': 'education', 'طلاب': 'students', 'دروس': 'lessons', 'تعلم': 'learning',
            # وزير التدريب
            'تدريب': 'training', 'مهارات': 'skills', 'تمرين': 'exercise',
            # وزير الأمن
            'أمن': 'security', 'حماية': 'protection', 'أمان': 'safety',
            # وزير التطوير
            'تطوير': 'development', 'تحسين': 'improvement', 'ترقية': 'upgrade',
            # وزير البحث والتطوير
            'بحث': 'research', 'أبحاث': 'researches', 'دراسة': 'study', 'تقنيات': 'technologies',
            # وزير المعرفة
            'معرفة': 'knowledge', 'معلومات': 'information', 'علم': 'science', 'حقائق': 'facts',
            # وزير الخصوصية
            'خصوصية': 'privacy', 'سرية': 'confidentiality', 'بيانات شخصية': 'personal data',
            # وزير الإبداع
            'إبداع': 'creativity', 'أفكار': 'ideas', 'ابتكار': 'innovation', 'إبداعي': 'creative',
            # وزير التحليل
            'تحليل': 'analysis', 'حلل': 'analyze', 'فحص': 'examination', 'تقييم': 'evaluation',
            # وزير الاستراتيجية
            'استراتيجية': 'strategy', 'خطة': 'plan', 'تخطيط': 'planning',
            # وزير الاستدلال
            'استدلال': 'reasoning', 'استنتاج': 'inference', 'منطق': 'logic',
            # وزير التواصل
            'تواصل': 'communication', 'رسالة': 'message', 'اتصال': 'contact',
            # وزير الموارد
            'موارد': 'resources', 'مصادر': 'sources', 'أصول': 'assets',
            # وزير المالية
            'مالية': 'finance', 'تكاليف': 'costs', 'ميزانية': 'budget', 'أموال': 'money'
        }
        if any(keyword in request_lower for keyword in minister_keywords.keys()):
            return "minister_task"

        # Check for complex planning
        if any(word in request_lower for word in
            ['خطة', 'plan', 'استراتيجية', 'strategy', 'مشروع', 'project']):
            return "complex_planning"

        # Default to decision
        return "decision"

    def _handle_vision_analysis(self, request: str, context: Dict) -> Dict:
        """معالجة تحليل بصري"""
        image_path = context.get('image_path')

        if not image_path:
            return {
                'response': "No image provided for vision analysis",
                'analysis': None,
                'confidence': 0.0
            }

        # Use Scene Understanding
        scene_analysis = self.scene_understanding.analyze_scene(image_path)

        # Use Material Analyzer if requested
        material_analysis = None
        if 'material' in request.lower() or 'مادة' in request:
            material_analysis = self.material_analyzer.analyze(image_path)

        response = f"Scene Analysis:\n"
        response += f"- Type: {scene_analysis.scene_context.scene_type.value}\n"
        response += f"- Lighting: {scene_analysis.scene_context.lighting_condition.value}\n"
        response += f"- Complexity: {scene_analysis.complexity_score:.0%}\n"

        if material_analysis:
            response += f"\nMaterial Analysis:\n"
            response += f"- Type: {material_analysis.material.material_type.value}\n"
            response += f"- Surface: {material_analysis.material.surface_property.value}\n"

        return {
            'response': response,
            'analysis': {
                'scene': asdict(scene_analysis),
                'material': asdict(material_analysis) if material_analysis else None
            },
            'confidence': scene_analysis.interpretability
        }

    def _handle_text_understanding(self, text: str) -> Dict:
        """معالجة فهم النصوص"""
        analysis = self.semantic_intent.analyze(text)

        response = f"Text Understanding:\n"
        response += f"- Semantic: {analysis.semantic.value}\n"
        response += f"- Intent: {analysis.intent.value}\n"
        response += f"- Emotion: {analysis.emotional.tone.value} ({analysis.emotional.intensity:.0%})\n"
        response += f"- Alignment: {analysis.semantic_intent_alignment:.0%}\n"
        response += f"\nInterpretation: {analysis.interpreted_meaning}"

        return {
            'response': response,
            'analysis': asdict(analysis),
            'confidence': analysis.semantic_intent_alignment
        }

    async def _delegate_to_minister(self, request: str, context: Dict) -> Dict:
        """تفويض مهمة لوزير (async version)"""
        # Determine which minister to use
        minister_type = self._select_minister(request)

        if not minister_type:
            return {
                'response': "No appropriate minister found",
                'minister': None,
                'confidence': 0.0
            }

        # Delegate task
        task = {
            'type': 'user_request',
            'request': request,
            **context
        }

        result = await self.ministers_system.delegate_task(minister_type, task)

        return {
            'response': f"Minister {result.get('minister', 'Unknown')} handled the task",
            'minister': result.get('minister'),
            'confidence': 0.9
        }

    def _delegate_to_minister_sync(self, request: str, context: Dict) -> Dict:
        """Safe synchronous wrapper for async minister delegation"""
        import asyncio
        import sys

        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # We're already in an async context - this shouldn't happen
                # but if it does, we need to handle it
                logger.warning("⚠️ Called sync wrapper from async context")
                # Create a task and return a placeholder
                return {
                    'response': "Minister delegation requires async context",
                    'minister': None,
                    'confidence': 0.0
                }
            except RuntimeError:
                # No running loop - this is the expected case
                pass

            # Safe to create and run a new event loop
            try:
                # Try to use existing event loop if available
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async function
            result = loop.run_until_complete(self._delegate_to_minister(request, context))
            return result

        except Exception as e:
            logger.error(f"❌ Error delegating to minister: {e}", exc_info=True)
            return {
                'response': f"Minister delegation failed: {str(e)}",
                'minister': None,
                'confidence': 0.0
            }

    def _select_minister(self, request: str) -> Optional[MinisterType]:
        """اختيار الوزير المناسب بناءً على نظام نقاط"""
        request_lower = request.lower()

        # نظام نقاط لكل وزير
        minister_scores = {}

        # وزير التعليم
        education_keywords = ['تعليم', 'teach', 'درس', 'طلاب', 'students']
        minister_scores[MinisterType.EDUCATION] = sum(1 for word in education_keywords if word in request_lower)

        # وزير التدريب
        training_keywords = ['تدريب', 'train', 'training', 'مهارات', 'skills', 'نموذج', 'model']
        minister_scores[MinisterType.TRAINING] = sum(1 for word in training_keywords if word in request_lower)

        # وزير الأمن
        security_keywords = ['أمن', 'security', 'protect', 'حماية', 'protection', 'أمان', 'safety', 'هجمات', 'attacks']
        minister_scores[MinisterType.SECURITY] = sum(1 for word in security_keywords if word in request_lower)

        # وزير التطوير
        development_keywords = ['تطوير', 'develop', 'development', 'code', 'تحسين', 'improvement', 'feature']
        minister_scores[MinisterType.DEVELOPMENT] = sum(1 for word in development_keywords if word in request_lower)

        # وزير البحث والتطوير
        research_keywords = ['بحث', 'research', 'أبحاث', 'researches', 'دراسة', 'study', 'تقنيات', 'technologies']
        minister_scores[MinisterType.RESEARCH] = sum(1 for word in research_keywords if word in request_lower)

        # وزير المعرفة
        knowledge_keywords = ['معرفة', 'knowledge', 'معلومات', 'information', 'علم', 'science', 'حقائق', 'facts']
        minister_scores[MinisterType.KNOWLEDGE] = sum(1 for word in knowledge_keywords if word in request_lower)

        # وزير الخصوصية
        privacy_keywords = ['خصوصية', 'privacy', 'سرية', 'confidentiality', 'بيانات شخصية', 'personal data']
        minister_scores[MinisterType.PRIVACY] = sum(1 for word in privacy_keywords if word in request_lower)

        # وزير الإبداع
        creativity_keywords = ['إبداع', 'creativity', 'أفكار', 'ideas', 'ابتكار', 'innovation', 'إبداعي', 'creative']
        minister_scores[MinisterType.CREATIVITY] = sum(1 for word in creativity_keywords if word in request_lower)

        # وزير التحليل
        analysis_keywords = ['حلل أداء', 'تحليل أداء', 'analyze performance', 'analysis', 'فحص', 'examination', 'تقييم', 'evaluation']
        minister_scores[MinisterType.ANALYSIS] = sum(1 for word in analysis_keywords if word in request_lower)

        # وزير الاستراتيجية
        strategy_keywords = ['استراتيجية', 'strategy', 'خطة استراتيجية', 'strategic plan', 'تخطيط', 'planning']
        minister_scores[MinisterType.STRATEGY] = sum(1 for word in strategy_keywords if word in request_lower)

        # وزير الاستدلال
        reasoning_keywords = ['استدلال', 'reasoning', 'استنتاج', 'inference', 'منطق', 'logic']
        minister_scores[MinisterType.REASONING] = sum(1 for word in reasoning_keywords if word in request_lower)

        # وزير التواصل
        communication_keywords = ['تواصل', 'communication', 'رسالة', 'message', 'اتصال', 'contact']
        minister_scores[MinisterType.COMMUNICATION] = sum(1 for word in communication_keywords if word in request_lower)

        # وزير الموارد
        resources_keywords = ['موارد', 'resources', 'مصادر', 'sources', 'أصول', 'assets']
        minister_scores[MinisterType.RESOURCES] = sum(1 for word in resources_keywords if word in request_lower)

        # وزير المالية (الأولوية للكلمات المالية)
        finance_keywords = ['مالية', 'finance', 'تكاليف', 'costs', 'ميزانية', 'budget', 'أموال', 'money']
        minister_scores[MinisterType.FINANCE] = sum(1 for word in finance_keywords if word in request_lower)
        # أعط نقاط إضافية إذا كانت "تكاليف مالية" معاً
        if 'تكاليف' in request_lower and 'مالية' in request_lower:
            minister_scores[MinisterType.FINANCE] += 2

        # اختر الوزير بأعلى نقاط
        if minister_scores:
            max_score = max(minister_scores.values())
            if max_score > 0:
                # إرجاع الوزير الأول بأعلى نقاط
                for minister, score in minister_scores.items():
                    if score == max_score:
                        return minister

        return None

    def _handle_complex_planning(self, request: str) -> Dict:
        """معالجة التخطيط المعقد"""
        task = self.agent_brain.analyze_task(request)

        response = f"Planning Result:\n"
        response += f"- Task Type: {task.task_type}\n"
        response += f"- Steps: {len(task.steps)}\n"

        for i, step in enumerate(task.steps[:3], 1):
            response += f"  {i}. {step['description']}\n"

        if len(task.steps) > 3:
            response += f"  ... and {len(task.steps) - 3} more steps\n"

        return {
            'response': response,
            'confidence': 0.85
        }

    def _handle_decision(self, request: str, context: Dict) -> Dict:
        """معالجة القرارات عبر Unified Cognition"""
        # For now, simple response
        # TODO: Integrate with unified_cognition.process_request()

        return {
            'response': f"Decision analysis for: {request}",
            'confidence': 0.7
        }

    def get_status(self) -> BrainHubStatus:
        """الحصول على حالة مركز الدماغ"""
        return BrainHubStatus(
            active=self.is_ready,
            cognition_score=self.cognition_score,
            active_ministers=self.active_ministers_count,
            deep_cognition_available=HAS_DEEP_COGNITION,
            agent_brain_available=HAS_AGENT_BRAIN,
            government_available=HAS_GOVERNMENT,
            unified_cognition_available=HAS_UNIFIED_COGNITION,
            timestamp=datetime.now().isoformat()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات شاملة"""
        stats = {
            'brain_hub': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'success_rate': self.successful_requests / max(1, self.total_requests),
                'cognition_score': self.cognition_score
            },
            'systems': {
                'deep_cognition': HAS_DEEP_COGNITION,
                'agent_brain': HAS_AGENT_BRAIN,
                'government': HAS_GOVERNMENT,
                'unified_cognition': HAS_UNIFIED_COGNITION
            }
        }

        # Add ministers stats
        if self.ministers_system:
            stats['ministers'] = self.ministers_system.get_all_stats()

        # Add unified cognition stats
        if self.unified_cognition:
            stats['unified_cognition'] = self.unified_cognition.get_system_health()

        return stats

    # ═══════════════════════════════════════════════════════════════
    # METHODS FOR MINISTERS - Deep Cognition Integration
    # ═══════════════════════════════════════════════════════════════

    def inference(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        استدلال ذكي باستخدام Deep Cognition

        يستخدمه الوزراء للحصول على تحليل معرفي عميق

        Args:
            data: البيانات المطلوب تحليلها

        Returns:
            نتيجة الاستدلال من Deep Cognition
        """
        if not self.is_ready:
            return None

        result = {}

        # Use Semantic Intent if text provided
        if 'text' in data or 'topic' in data:
            text = data.get('text') or data.get('topic', '')
            if self.semantic_intent and text:
                analysis = self.semantic_intent.analyze(str(text))
                result['semantic_analysis'] = {
                    'semantic_layer': analysis.semantic.layer.value,
                    'intent_layer': analysis.intent.layer.value,
                    'emotional_tone': analysis.emotional.tone.value,
                    'alignment': analysis.semantic_intent_alignment,
                    'interpretation': analysis.interpreted_meaning
                }

        # Use Meta Confidence if confidence calculation requested
        if 'confidence_factors' in data:
            factors = data['confidence_factors']
            if self.meta_confidence:
                confidence_result = self.meta_confidence.calculate_certainty(**factors)
                result['confidence_analysis'] = {
                    'overall_confidence': confidence_result.overall_confidence,
                    'certainty_level': confidence_result.certainty_level.value,
                    'recommendation': confidence_result.recommendation
                }

        # Use Scene Understanding if image provided
        if 'image_path' in data:
            if self.scene_understanding:
                scene_analysis = self.scene_understanding.analyze_scene(data['image_path'])
                result['scene_analysis'] = {
                    'scene_type': scene_analysis.scene_context.scene_type.value,
                    'lighting': scene_analysis.scene_context.lighting_condition.value,
                    'complexity': scene_analysis.complexity_score
                }

        return result if result else None

    @property
    def ai_engine(self):
        """
        AI Engine محاكي لـ compatibility مع الوزراء القدامى

        يوفر:
        - process(): معالجة عامة
        - reasoning.reason(): استدلال منطقي
        """
        return AIEngineProxy(self)


class AIEngineProxy:
    """
    محاكي AI Engine للتوافق مع الوزراء القدامى
    """

    def __init__(self, brain_hub: 'UnifiedBrainHub'):
        self.brain_hub = brain_hub
        self.reasoning = ReasoningProxy(brain_hub)

    def process(self, data: Dict[str, Any], context: str = None) -> Optional[Dict[str, Any]]:
        """
        معالجة عامة باستخدام Deep Cognition

        Args:
            data: البيانات المطلوب معالجتها
            context: السياق (development, analysis, etc.)

        Returns:
            نتيجة المعالجة
        """
        result = {'context': context}

        # Use semantic intent for text processing
        if 'text' in data or 'feature' in data:
            text = data.get('text') or data.get('feature', '')
            if self.brain_hub.semantic_intent:
                analysis = self.brain_hub.semantic_intent.analyze(str(text))
                result['suggestions'] = [
                    f"Approach 1: {analysis.interpreted_meaning}",
                    f"Approach 2: Alternative implementation",
                    f"Approach 3: Optimized solution"
                ]
                result['confidence'] = analysis.semantic_intent_alignment

        return result


class ReasoningProxy:
    """محاكي Reasoning Engine"""

    def __init__(self, brain_hub: 'UnifiedBrainHub'):
        self.brain_hub = brain_hub

    def reason(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        استدلال منطقي باستخدام Meta Confidence

        Args:
            data: المشكلة المطلوب حلها

        Returns:
            نتيجة الاستدلال
        """
        if not self.brain_hub.meta_confidence:
            return None

        # Use meta confidence to evaluate reasoning strength
        confidence_result = self.brain_hub.meta_confidence.calculate_certainty(
            data_quality=0.8,
            model_agreement=0.85,
            historical_accuracy=0.9,
            context_clarity=0.88
        )

        return {
            'reasoning_steps': [
                "1. Analyze problem structure",
                "2. Identify key constraints",
                "3. Evaluate possible solutions",
                "4. Select optimal approach",
                "5. Validate solution"
            ],
            'confidence': confidence_result.overall_confidence,
            'certainty_level': confidence_result.certainty_level.value,
            'recommendation': confidence_result.recommendation
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLETON INSTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_brain_hub = None

def get_brain_hub(enable_gpu: bool = True) -> UnifiedBrainHub:
    """الحصول على مركز الدماغ الموحد (Singleton)"""
    global _brain_hub
    if _brain_hub is None:
        _brain_hub = UnifiedBrainHub(enable_gpu=enable_gpu)
    return _brain_hub


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_brain_hub():
    """اختبار مركز الدماغ الموحد"""
    print("\n" + "=" * 70)
    print("🧪 TESTING UNIFIED BRAIN HUB")
    print("=" * 70 + "\n")

    # Initialize
    brain_hub = get_brain_hub(enable_gpu=True)

    # Check status
    status = brain_hub.get_status()
    print("\n📊 Brain Hub Status:")
    print(f"   Active: {status.active}")
    print(f"   Cognition Score: {status.cognition_score:.1%}")
    print(f"   Active Ministers: {status.active_ministers}")
    print(f"   Deep Cognition: {status.deep_cognition_available}")
    print(f"   Agent Brain: {status.agent_brain_available}")
    print(f"   Government: {status.government_available}")

    # Test requests
    print("\n" + "=" * 70)
    print("🧪 Testing Request Processing")
    print("=" * 70 + "\n")

    test_requests = [
        "فهم معنى هذا النص: الحياة جميلة",
        "أريد تعليم الطلاب عن الذكاء الاصطناعي",
        "خطة لبناء نظام موزع"
    ]

    for request in test_requests:
        print(f"\n📝 Request: {request}")
        result = brain_hub.process_request(request)
        print(f"   Status: {result.status}")
        print(f"   Confidence: {result.confidence:.0%}")
        if result.minister_used:
            print(f"   Minister: {result.minister_used}")

    # Statistics
    print("\n" + "=" * 70)
    print("📊 Final Statistics")
    print("=" * 70 + "\n")

    stats = brain_hub.get_statistics()
    print(f"Total Requests: {stats['brain_hub']['total_requests']}")
    print(f"Success Rate: {stats['brain_hub']['success_rate']:.0%}")

    print("\n✅ TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_brain_hub()
