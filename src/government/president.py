#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - President
نظام الحكومة الداخلية لنوغ - الرئيس

Version: 3.0.0 - Simplified Implementation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.nlp.intent import IntentRouter, Intent
from src.knowledge.kernel import KnowledgeKernelV41
from src.government.base_minister import BaseMinister, Priority

logger = logging.getLogger(__name__)



class President:
    """
    رئيس الحكومة الداخلية لنوغ

    المسؤوليات:
    - إدارة وتنسيق جميع الوزراء
    - اتخاذ القرارات الاستراتيجية
    - توزيع المهام على الوزراء المناسبين
    - جمع التقارير وتنسيق الردود
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: عرض التفاصيل
        """
        self.verbose = verbose

        # Neural Core - Local Brain (Meta-Llama-3-8B on RTX 5070)
        # ⚠️ CRITICAL: Initialize brain BEFORE cabinet so ministers can receive it
        try:
            from src.services.local_brain_service import LocalBrainService
            self.brain = LocalBrainService()
            if self.verbose:
                logger.info("🧠 Neural Core initialized (Meta-Llama-3-8B)")
        except Exception as e:
            logger.warning(f"⚠️ Neural Core not available: {e}")
            self.brain = None

        # Cabinet - مجلس الوزراء
        # Ministers will receive brain instance during initialization
        self.cabinet: Dict[str, BaseMinister] = {}
        self.initialize_cabinet()

        # Intent Router for dispatching
        # In a real app, the kernel would be injected, not created here.
        self.kernel = KnowledgeKernelV41()
        self.intent_router = IntentRouter(self.kernel)
        self.intent_to_minister_map = self._create_intent_map()

        # إحصائيات رئاسية
        self.total_requests = 0
        self.successful_requests = 0

        if self.verbose:
            logger.info("🎩 President initialized")
            logger.info(f"   Cabinet ready with {len(self.cabinet)} ministers")
            logger.info("   IntentRouter is now responsible for dispatch.")

    def initialize_cabinet(self):
        """
        Initialize the cabinet with AI-powered ministers.
        
        Each minister gets access to the LocalBrainService for intelligence.
        """
        logger.info("Initializing Smart Cabinet...")
        
        # Get brain instance for ministers
        brain = self.brain if hasattr(self, 'brain') and self.brain else None
        
        if brain is None:
            logger.warning("⚠️ LocalBrainService not available, ministers will have limited capability")
        
        # Import all ministers
        from src.government.ministers.education_minister import EducationMinister
        from src.government.ministers.security_minister import SecurityMinister
        from src.government.ministers.development_minister import DevelopmentMinister
        from src.government.ministers.finance_minister import FinanceMinister
        from src.government.ministers.health_minister import HealthMinister
        from src.government.ministers.foreign_minister import ForeignMinister
        from src.government.ministers.communication_minister import CommunicationMinister
        
        # Initialize complete cabinet (7 ministers)
        self.cabinet = {
            "education": EducationMinister(brain=brain),
            "security": SecurityMinister(brain=brain),
            "development": DevelopmentMinister(brain=brain),
            "finance": FinanceMinister(brain=brain),
            "health": HealthMinister(brain=brain),
            "foreign": ForeignMinister(brain=brain),
            "communication": CommunicationMinister(brain=brain),
        }
        
        logger.info(f"✅ Complete Cabinet initialized with {len(self.cabinet)} AI-powered ministers")

    def _create_intent_map(self) -> Dict[Intent, str]:
        """Creates a mapping from Intent to minister key."""
        return {
            Intent.QUESTION_KB: "education",
            Intent.QUESTION_WEB: "communication",
            Intent.CHITCHAT: "communication",
            Intent.COMMAND_NOT_IMPLEMENTED: "development",
            Intent.UNKNOWN: "education", # Default fallback
        }

    def process_request(self, user_input: str, context: Optional[dict] = None, priority: str = "medium") -> dict:
        """
        Process a user request through the government system.
        
        Args:
            user_input: طلب المستخدم
            context: سياق إضافي
            priority: أولوية المهمة
            
        Returns:
            dict: نتيجة المعالجة
        """
        self.total_requests += 1
        context = context or {}
        
        # 0. Recall relevant memories (Context Augmentation)
        # We use the kernel to find similar past interactions or facts
        memories = self.kernel.recall(user_input, top_k=2)
        if memories:
            context['memories'] = memories
            if self.verbose:
                logger.info(f"🧠 Recalled {len(memories)} relevant memories")

        # 1. Determine intent using the router (Contextual)
        # We pass the conversation history if available in context
        history = context.get('history', [])
        
        # Use the analyzer directly for contextual analysis if possible, 
        # otherwise fallback to router which might not expose it directly yet.
        # For now, we assume intent_router has access to the analyzer or we use it directly if we had it.
        # Since IntentRouter wraps the logic, we'll stick to its route method but ideally it should support context.
        # TODO: Update IntentRouter to support context fully. For now, we rely on the kernel's enhanced capabilities.
        
        intent_response = self.intent_router.route(user_input, context)
        intent = intent_response.intent
        
        if self.verbose:
            logger.info(f"Intent determined: {intent.value}")

        # 2. Determine the minister from the intent
        minister_key = self.intent_to_minister_map.get(intent, "education") # Default to education
        
        if self.verbose:
            logger.info(f"Dispatching to: {minister_key.upper()} Minister")

        result_data = {}
        if minister_key in self.cabinet:
            try:
                from .base_minister import generate_task_id
                task_id = generate_task_id()

                # Safe Priority enum access with fallback
                try:
                    priority_enum = Priority[priority.upper()]
                except (KeyError, AttributeError):
                    logger.warning(f"Invalid priority '{priority}', using MEDIUM")
                    priority_enum = Priority.MEDIUM

                # Use the specific intent value as the task_type
                # Call minister synchronously (ministers have sync execute_task_sync fallback)
                import asyncio
                try:
                    # Try to run async method in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    minister_result = loop.run_until_complete(
                        self.cabinet[minister_key].execute_task(
                            task_id=task_id,
                            task_type=intent.value,
                            task_data={"user_input": user_input, "context": context},
                            priority=priority_enum
                        )
                    )
                    loop.close()
                except Exception as async_err:
                    logger.error(f"Async execution failed: {async_err}")
                    raise
                self.successful_requests += 1
                result_data = minister_result.to_dict()
                
                # 3. Learn from the interaction (Reinforcement / Memory)
                # If the request was successful, store it in memory
                if result_data.get('status') == 'completed':
                    learning_text = f"User asked: '{user_input}'. System answered via {minister_key}: '{result_data.get('result', {}).get('message', 'Done')}'"
                    self.kernel.learn(learning_text, metadata={"intent": intent.value, "minister": minister_key})
                
                return result_data
            except Exception as e:
                logger.error(f"Error processing request with {minister_key}: {e}", exc_info=True)
                # Fall through to Neural Core if minister fails
                logger.info("⚡ Minister failed, trying Neural Core...")
        
        # If no minister handles it OR minister failed, use Neural Core (Brain)
        if self.brain is not None:
            try:
                logger.info("🧠 Activating Neural Core (Meta-Llama-3-8B)...")
                # Call brain synchronously
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                brain_response = loop.run_until_complete(self.brain.think(user_input, max_tokens=512))
                loop.close()
                
                self.successful_requests += 1
                
                # Structure the brain response like a minister response
                result_data = {
                    "status": "completed",
                    "task_id": f"neural_{self.total_requests}",
                    "task_type": "neural_inference",
                    "minister": "neural_core",
                    "result": {
                        "message": brain_response,
                        "source": "Meta-Llama-3-8B-Instruct"
                    }
                }
                
                # Learn from neural interaction
                learning_text = f"User asked: '{user_input}'. Neural Core answered: '{brain_response[:200]}...'"
                self.kernel.learn(learning_text, metadata={"intent": "neural_inference", "minister": "neural_core"})
                
                return result_data
                
            except Exception as e:
                logger.error(f"Error with Neural Core: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Neural Core error: {str(e)}",
                    "input": user_input
                }
        else:
            # No minister and no brain available
            return {
                "success": False,
                "error": f"No suitable handler found for intent '{intent.value}' and Neural Core unavailable",
                "input": user_input
            }


    def get_cabinet_status(self) -> Dict[str, Any]:
        """
        Get the status of the entire cabinet.

        Returns:
            حالة مجلس الوزراء
        """
        # جميع الوزراء يعتبرون نشطين في هذا التنفيذ المبسط
        active_ministers = len(self.cabinet)
        
        return {
            "total_ministers": len(self.cabinet),
            "active_ministers": active_ministers,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0,
            "ministers": list(self.cabinet.keys())
        }

    def print_status(self):
        """
        Print the status of the government system.
        """
        status = self.get_cabinet_status()
        logger.info("\n" + "="*50)
        logger.info("🏛️ Noogh Government Status")
        logger.info("="*50)
        logger.info(f"📊 Total Ministers: {status['total_ministers']}")
        logger.info(f"✅ Active Ministers: {status['active_ministers']}")
        logger.info(f"📨 Total Requests: {status['total_requests']}")
        logger.info(f"✅ Successful: {status['successful_requests']}")
        logger.info(f"📈 Success Rate: {status['success_rate']:.1%}")
        logger.info("="*50)


# Helper function for creating president instance
def create_president(verbose: bool = True) -> President:
    """
    إنشاء رئيس نوغ

    Usage:
        president = create_president()
        result = await president.process_request("ما هو سعر BTC؟")
    """
    return President(verbose=verbose)


if __name__ == "__main__":
    # اختبار سريع
    async def test_president():
        logger.info("🧪 Testing Noogh President...\n")

        president = create_president(verbose=True)

        # عرض حالة مجلس الوزراء
        president.print_status()

        # اختبار معالجة طلب
        result = await president.process_request("علمني عن الذكاء الاصطناعي")
        logger.info(f"Test result: {result}")

        logger.info(f"\n✅ President test complete!")

    asyncio.run(test_president())
