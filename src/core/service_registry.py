#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service Registry - Central service registration for DI container
نظام تسجيل الخدمات - التسجيل المركزي للخدمات في حاوية DI
"""
import logging
from typing import Optional
from src.core.di import Container
from src.core.database import SessionLocal

logger = logging.getLogger(__name__)


def register_core_services():
    """
    Register all core services with the DI container
    تسجيل جميع الخدمات الأساسية في حاوية DI
    """
    logger.info("🚀 Registering core services...")
    
    # Database session factory
    Container.register_factory("db_session", lambda: SessionLocal())
    
    # Cache Manager
    try:
        from src.api.utils.cache_manager import CacheManager
        Container.register_factory("cache_manager", lambda: CacheManager())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register cache_manager: {e}")
    
    # Device Manager
    try:
        from src.api.utils.device_manager import DeviceManager
        Container.register_factory("device_manager", lambda: DeviceManager())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register device_manager: {e}")
    
    logger.info("✅ Core services registered")


def register_autonomy_services():
    """
    Register autonomy-related services
    تسجيل خدمات الاستقلالية
    """
    logger.info("🤖 Registering autonomy services...")
    
    # Brain Adjuster
    try:
        from src.autonomy.brain_adjuster import BrainAdjuster
        Container.register_factory("brain_adjuster", lambda: BrainAdjuster())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register brain_adjuster: {e}")
    
    # Feedback Collector
    try:
        from src.autonomy.feedback_collector import FeedbackCollector
        Container.register_factory("feedback_collector", lambda: FeedbackCollector())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register feedback_collector: {e}")
    
    # Improvement Logger
    try:
        from src.autonomy.improvement_logger import ImprovementLogger
        Container.register_factory("improvement_logger", lambda: ImprovementLogger())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register improvement_logger: {e}")
    
    # Training Scheduler
    try:
        from src.autonomy.training_scheduler import TrainingScheduler
        Container.register_factory("training_scheduler", lambda: TrainingScheduler())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register training_scheduler: {e}")
    
    # Model Manager
    try:
        from src.autonomy.model_manager import ModelManager
        Container.register_factory("model_manager", lambda: ModelManager())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register model_manager: {e}")
    
    logger.info("✅ Autonomy services registered")


def register_integration_services():
    """
    Register integration services
    تسجيل خدمات التكامل
    """
    logger.info("🔗 Registering integration services...")
    
    # Cognitive Decision Bridge
    try:
        from src.integration.cognitive_decision_bridge import CognitiveDecisionBridge
        Container.register_factory("cognitive_decision_bridge", lambda: CognitiveDecisionBridge())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register cognitive_decision_bridge: {e}")
    
    # Cognitive Government Adapter
    try:
        from src.integration.cognitive_government_adapter import CognitiveGovernmentAdapter
        Container.register_factory(
            "cognitive_government_adapter",
            lambda: CognitiveGovernmentAdapter(enable_autonomous_improvement=True)
        )
    except ImportError as e:
        logger.warning(f"⚠️ Could not register cognitive_government_adapter: {e}")
    
    logger.info("✅ Integration services registered")


def register_monitoring_services():
    """
    Register monitoring services
    تسجيل خدمات المراقبة
    """
    logger.info("📊 Registering monitoring services...")
    
    # Self Evaluation System
    try:
        from src.monitoring.self_evaluation import SelfEvaluationSystem
        Container.register_factory("evaluation_system", lambda: SelfEvaluationSystem(verbose=False))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register evaluation_system: {e}")
    
    # VRAM Predictor
    try:
        from src.monitoring.ml_predictor import VRAMPredictor
        Container.register_factory("vram_predictor", lambda: VRAMPredictor(verbose=False))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register vram_predictor: {e}")
    
    logger.info("✅ Monitoring services registered")


def register_api_services():
    """
    Register API-specific services
    تسجيل خدمات API
    """
    logger.info("🌐 Registering API services...")
    
    # WebSocket Connection Manager
    try:
        from src.api.routes.websocket import ConnectionManager
        Container.register_factory("connection_manager", lambda: ConnectionManager())
    except ImportError as e:
        logger.warning(f"⚠️ Could not register connection_manager: {e}")
    
    logger.info("✅ API services registered")


def register_all_services():
    """
    Register all services in the correct order
    تسجيل جميع الخدمات بالترتيب الصحيح
    """
    logger.info("=" * 70)
    logger.info("🎯 INITIALIZING DEPENDENCY INJECTION CONTAINER")
    logger.info("=" * 70)
    
    # Register in dependency order
    register_core_services()
    register_autonomy_services()
    register_integration_services()
    register_monitoring_services()
    register_api_services()
    
    # Log statistics
    stats = Container.stats()
    logger.info("=" * 70)
    logger.info("📈 DI CONTAINER STATISTICS")
    logger.info(f"   Total Services: {stats['total_services']}")
    logger.info(f"   Instantiated: {stats['instantiated_services']}")
    logger.info(f"   Factories: {stats['registered_factories']}")
    logger.info("=" * 70)
    
    return stats


if __name__ == "__main__":
    # Test service registration
    logging.basicConfig(level=logging.INFO)
    register_all_services()
    
    # Test resolution
    print("\n🧪 Testing service resolution...")
    cache = Container.resolve("cache_manager")
    print(f"Cache Manager: {cache}")
    
    device = Container.resolve("device_manager")
    print(f"Device Manager: {device}")
    
    print("\n✅ Service registry test complete!")
