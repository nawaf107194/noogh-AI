#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Brain Integration Layer - طبقة التكامل الذكية
===================================================

Unified interface that combines:
- UnifiedBrainHub (التنسيق والوزراء)
- UnifiedNooghBrain (التدريب والنماذج)

Single, clean API for all brain capabilities.

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-11-10
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedBrainAPI:
    """
    🧠 واجهة برمجية موحدة للوصول لكل قدرات نظام نوغ

    Unified Brain API - Single interface for all Noogh brain capabilities

    Features:
    ✅ Minister delegation (14 ministers)
    ✅ Cognition scoring (97.5% TRANSCENDENT)
    ✅ Deep Cognition v1.2 Lite
    ✅ Neural training (MegaBrain V5)
    ✅ GPU acceleration
    ✅ Autonomous system integration
    ✅ Hugging Face integration

    Usage:
        brain = UnifiedBrainAPI()

        # Ask ministers
        result = brain.ask_minister("أريد تعليم الطلاب")

        # Train models (lazy loaded)
        brain.init_neural_brain(config='small')
        brain.train_model(train_data, train_labels)

        # Full status
        status = brain.get_full_status()
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize Unified Brain API

        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Hub System - التنسيق والوزراء
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            from integration.unified_brain_hub import (
                UnifiedBrainHub,
                HAS_DEEP_COGNITION,
                HAS_AGENT_BRAIN
            )
            self.hub = UnifiedBrainHub()
            self._hub_available = True

            # Store constants for later use
            self._has_deep_cognition = HAS_DEEP_COGNITION
            self._has_agent_brain = HAS_AGENT_BRAIN

            if self.verbose:
                logger.info("✅ UnifiedBrainHub loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load UnifiedBrainHub: {e}")
            self.hub = None
            self._hub_available = False
            self._has_deep_cognition = False
            self._has_agent_brain = False

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Neural System - التدريب والنماذج (Lazy Loaded)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._neural_brain = None
        self._neural_available = False

        if self.verbose:
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info("🧠 Unified Brain API - Ready")
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"Hub System: {'✅ Active' if self._hub_available else '❌ Unavailable'}")
            logger.info(f"Neural System: 💤 Lazy loaded (call init_neural_brain() when needed)")
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROPERTIES - الخصائص
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def neural_brain(self):
        """Get neural brain (auto-init if needed)"""
        if self._neural_brain is None:
            logger.warning("⚠️  Neural brain not initialized. Call init_neural_brain() first.")
        return self._neural_brain

    @property
    def is_hub_ready(self) -> bool:
        """Is hub system ready?"""
        return self._hub_available and self.hub is not None

    @property
    def is_neural_ready(self) -> bool:
        """Is neural system ready?"""
        return self._neural_available and self._neural_brain is not None

    @property
    def is_ready(self) -> bool:
        """Is entire system ready?"""
        return self.is_hub_ready or self.is_neural_ready

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HUB CAPABILITIES - قدرات المنسق والوزراء
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def ask_minister(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🎯 طلب من وزير
        Delegate request to appropriate minister

        Args:
            request: المطلوب (e.g., "أريد تعليم الطلاب")
            context: سياق إضافي

        Returns:
            Minister response with result and confidence

        Example:
            result = brain.ask_minister("أريد تعليم الطلاب عن الذكاء الاصطناعي")
            print(result['minister'])    # "وزير التعليم"
            print(result['response'])    # Response text
            print(result['confidence'])  # 0.95
        """
        if not self.is_hub_ready:
            raise RuntimeError("❌ Hub system not available")

        try:
            from dataclasses import asdict
            result = self.hub.process_request(
                request=request,
                context=context or {}
            )

            # Convert ProcessingResult to dict
            result_dict = asdict(result)

            if self.verbose:
                logger.info(f"✅ Minister delegation successful: {result_dict.get('minister_used', 'Unknown')}")

            return result_dict
        except Exception as e:
            logger.error(f"❌ Minister delegation failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def get_cognition_score(self) -> float:
        """
        🧠 الحصول على درجة الإدراك
        Get cognition score (0.0 - 1.0)

        Returns:
            Cognition score (e.g., 0.975 = 97.5% TRANSCENDENT)
        """
        if not self.is_hub_ready:
            logger.warning("⚠️  Hub not available, returning default score")
            return 0.0

        return self.hub.cognition_score

    def list_ministers(self) -> List[Dict[str, Any]]:
        """
        👔 قائمة الوزراء النشطين
        Get list of active ministers

        Returns:
            List of minister info dicts
        """
        if not self.is_hub_ready or not self.hub.ministers_system:
            return []

        return [
            minister.get_stats()
            for minister in self.hub.ministers_system.active_ministers.values()
        ]

    def get_hub_status(self) -> Dict[str, Any]:
        """
        📊 حالة نظام المنسق
        Get hub system status

        Returns:
            Hub status dict
        """
        if not self.is_hub_ready:
            return {"error": "Hub not available"}

        from dataclasses import asdict
        status = self.hub.get_status()
        return asdict(status)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEURAL CAPABILITIES - قدرات التدريب والنماذج
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def init_neural_brain(
        self,
        config: str = 'small',
        device: str = 'auto',
        use_autonomous: bool = True
    ):
        """
        🔧 تهيئة الدماغ العصبوني
        Initialize neural brain system

        Args:
            config: Model config (micro, tiny, small, base, medium, large, ultra, insane)
            device: Device ('auto', 'cuda', 'cpu')
            use_autonomous: Enable autonomous system integration

        Returns:
            self for chaining

        Example:
            brain.init_neural_brain(config='small')
            brain.train_model(data, labels)
        """
        if self._neural_brain is not None:
            if self.verbose:
                logger.info("ℹ️  Neural brain already initialized")
            return self

        try:
            from brain.unified_brain import create_mega_brain

            if self.verbose:
                logger.info(f"🔧 Initializing neural brain (config={config}, device={device})...")

            self._neural_brain = create_mega_brain(
                config=config,
                device=device,
                verbose=self.verbose,
                use_autonomous=use_autonomous
            )

            self._neural_available = True

            if self.verbose:
                logger.info("✅ Neural brain initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize neural brain: {e}")
            self._neural_brain = None
            self._neural_available = False

        return self

    def train_model(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs: int = 10,
        batch_size: Optional[int] = None,
        save_path: Optional[str] = None,
        auto_prepare: bool = True,
        estimated_vram: float = 4.0
    ):
        """
        🏋️ تدريب نموذج
        Train a model

        Args:
            train_data: Training data
            train_labels: Training labels
            test_data: Test data
            test_labels: Test labels
            epochs: Number of epochs
            batch_size: Batch size (None = auto)
            save_path: Path to save model
            auto_prepare: Auto-prepare system (pause ministers, free VRAM)
            estimated_vram: Estimated VRAM needed (GB)

        Returns:
            Training history

        Example:
            brain.init_neural_brain(config='small')
            history = brain.train_model(
                train_data, train_labels,
                test_data, test_labels,
                epochs=10
            )
        """
        if not self.is_neural_ready:
            raise RuntimeError("❌ Neural brain not initialized. Call init_neural_brain() first.")

        try:
            history = self._neural_brain.train(
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                epochs=epochs,
                batch_size=batch_size,
                save_path=save_path,
                auto_prepare=auto_prepare,
                estimated_vram=estimated_vram
            )

            if self.verbose:
                logger.info("✅ Training completed successfully")

            return history

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def predict(self, data):
        """
        🔮 استدلال
        Make predictions

        Args:
            data: Input data

        Returns:
            Predictions
        """
        if not self.is_neural_ready:
            raise RuntimeError("❌ Neural brain not initialized")

        return self._neural_brain.predict(data)

    def check_gpu_resources(self) -> Optional[Dict[str, Any]]:
        """
        🎮 فحص موارد GPU
        Check GPU resources

        Returns:
            GPU resource info or None
        """
        if not self.is_neural_ready:
            logger.warning("⚠️  Neural brain not initialized")
            return None

        return self._neural_brain.check_resources()

    def is_ready_for_training(self, estimated_vram: float = 4.0) -> bool:
        """
        ✅ هل النظام جاهز للتدريب؟
        Check if system is ready for training

        Args:
            estimated_vram: Required VRAM (GB)

        Returns:
            True if ready, False otherwise
        """
        if not self.is_neural_ready:
            return False

        return self._neural_brain.is_ready_for_training(estimated_vram)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UNIFIED CAPABILITIES - القدرات الموحدة
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_full_status(self) -> Dict[str, Any]:
        """
        📊 الحصول على حالة كاملة للنظام
        Get complete system status

        Returns:
            Complete status including hub and neural systems
        """
        status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "api_version": "1.0.0",
            "systems": {}
        }

        # Hub system status
        if self.is_hub_ready:
            try:
                cognition_score = self.hub.cognition_score
                status["systems"]["hub"] = {
                    "active": True,
                    "cognition_score": cognition_score,
                    "cognition_level": self._get_cognition_level(cognition_score),
                    "ministers_active": len(self.hub.ministers_system.active_ministers) if self.hub.ministers_system else 0,
                    "deep_cognition_available": self._has_deep_cognition,
                    "agent_brain_available": self._has_agent_brain
                }
            except Exception as e:
                status["systems"]["hub"] = {"active": False, "error": str(e)}
        else:
            status["systems"]["hub"] = {"active": False}

        # Neural system status
        if self.is_neural_ready:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                status["systems"]["neural"] = {
                    "active": True,
                    "initialized": True,
                    "model_loaded": self._neural_brain.is_ready,
                    "device": self._neural_brain.device.type,
                    "gpu_available": gpu_available
                }

                if gpu_available:
                    status["systems"]["neural"]["gpu_name"] = torch.cuda.get_device_name(0)
                    status["systems"]["neural"]["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024

            except Exception as e:
                status["systems"]["neural"] = {"active": True, "error": str(e)}
        else:
            status["systems"]["neural"] = {
                "active": False,
                "initialized": False,
                "message": "Call init_neural_brain() to initialize"
            }

        # Overall status
        status["overall"] = {
            "ready": self.is_ready,
            "hub_ready": self.is_hub_ready,
            "neural_ready": self.is_neural_ready,
            "capabilities": []
        }

        if self.is_hub_ready:
            status["overall"]["capabilities"].extend([
                "minister_delegation",
                "cognition_scoring",
                "deep_cognition",
                "agent_brain"
            ])

        if self.is_neural_ready:
            status["overall"]["capabilities"].extend([
                "neural_training",
                "model_inference",
                "gpu_acceleration"
            ])

        return status

    def _get_cognition_level(self, score: float) -> str:
        """Get cognition level label"""
        if score >= 0.95:
            return "TRANSCENDENT"
        elif score >= 0.85:
            return "SUPERIOR"
        elif score >= 0.75:
            return "ADVANCED"
        elif score >= 0.60:
            return "COMPETENT"
        else:
            return "DEVELOPING"

    def print_status(self):
        """
        🖨️ طباعة حالة النظام
        Print system status
        """
        status = self.get_full_status()

        print("\n" + "━" * 60)
        print("🧠 UNIFIED BRAIN API - SYSTEM STATUS")
        print("━" * 60)

        # Hub System
        print("\n📡 HUB SYSTEM:")
        if status["systems"]["hub"]["active"]:
            hub = status["systems"]["hub"]
            print(f"   Status: ✅ Active")
            print(f"   Cognition Score: {hub['cognition_score']:.1%} ({hub['cognition_level']})")
            print(f"   Active Ministers: {hub['ministers_active']}")
            print(f"   Deep Cognition: {'✅' if hub['deep_cognition_available'] else '❌'}")
            print(f"   Agent Brain: {'✅' if hub['agent_brain_available'] else '❌'}")
        else:
            print(f"   Status: ❌ Unavailable")

        # Neural System
        print("\n🧠 NEURAL SYSTEM:")
        if status["systems"]["neural"]["initialized"]:
            neural = status["systems"]["neural"]
            print(f"   Status: ✅ Initialized")
            print(f"   Model Loaded: {'✅' if neural.get('model_loaded', False) else '❌'}")
            print(f"   Device: {neural.get('device', 'Unknown').upper()}")
            if neural.get('gpu_available'):
                print(f"   GPU: ✅ {neural.get('gpu_name', 'Unknown')}")
                print(f"   VRAM Used: {neural.get('gpu_memory_allocated_mb', 0):.2f} MB")
            else:
                print(f"   GPU: ❌ Not available")
        else:
            print(f"   Status: 💤 Not initialized")
            print(f"   Message: {status['systems']['neural'].get('message', '')}")

        # Overall
        print("\n🎯 CAPABILITIES:")
        for cap in status["overall"]["capabilities"]:
            print(f"   ✅ {cap.replace('_', ' ').title()}")

        print("\n" + "━" * 60)
        print(f"⏰ Timestamp: {status['timestamp']}")
        print("━" * 60 + "\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONVENIENCE METHODS - طرق سهلة
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def quick_test(self):
        """
        ⚡ اختبار سريع للنظام
        Quick system test
        """
        print("\n" + "━" * 60)
        print("⚡ QUICK SYSTEM TEST")
        print("━" * 60)

        # Test 1: Hub
        if self.is_hub_ready:
            print("\n✅ Test 1: Hub System")
            score = self.get_cognition_score()
            print(f"   Cognition Score: {score:.1%}")

            ministers = self.list_ministers()
            print(f"   Active Ministers: {len(ministers)}")
        else:
            print("\n❌ Test 1: Hub System - Not Available")

        # Test 2: Neural (if initialized)
        if self.is_neural_ready:
            print("\n✅ Test 2: Neural System")
            print(f"   Device: {self._neural_brain.device.type.upper()}")
            print(f"   Model Ready: {self._neural_brain.is_ready}")

            try:
                import torch
                if torch.cuda.is_available():
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
            except ImportError:
                pass
        else:
            print("\n💤 Test 2: Neural System - Not Initialized")
            print("   (Call init_neural_brain() to initialize)")

        print("\n" + "━" * 60)
        print("✅ Quick test completed")
        print("━" * 60 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS - دوال مساعدة
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_unified_brain(verbose: bool = True) -> UnifiedBrainAPI:
    """
    إنشاء واجهة دماغ موحدة
    Create unified brain API

    Args:
        verbose: Enable detailed logging

    Returns:
        UnifiedBrainAPI instance

    Usage:
        brain = create_unified_brain()
        brain.print_status()

        # Use hub
        result = brain.ask_minister("أريد تعليم الطلاب")

        # Use neural (when needed)
        brain.init_neural_brain(config='small')
        brain.train_model(data, labels, test_data, test_labels)
    """
    return UnifiedBrainAPI(verbose=verbose)


if __name__ == "__main__":
    # اختبار سريع
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("🧪 TESTING UNIFIED BRAIN API")
    print("=" * 60 + "\n")

    # Create unified brain
    brain = create_unified_brain(verbose=True)

    # Print status
    brain.print_status()

    # Quick test
    brain.quick_test()

    print("\n✅ Unified Brain API ready for use!\n")
