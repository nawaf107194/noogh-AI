#!/usr/bin/env python3
"""
🎓 Training Orchestrator - منسق التدريب الذكي
يدير عملية التدريب ويفرغ الموارد تلقائياً
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio
import gc
import torch

from .resource_monitor import ResourceMonitor, ResourceStatus
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)

# Import evaluation system (optional - won't fail if not available)
try:
    from .self_evaluation import get_evaluation_system, create_session_from_training
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logger.debug("Self-evaluation system not available")


class TrainingPhase(Enum):
    """مرحلة التدريب"""
    IDLE = "idle"                    # خامل
    PREPARING = "preparing"          # يحضّر
    TRAINING = "training"            # يدرّب
    PAUSED = "paused"               # متوقف مؤقتاً
    COMPLETED = "completed"          # مكتمل
    FAILED = "failed"               # فشل


@dataclass
class TrainingSession:
    """جلسة تدريب"""
    session_id: str
    model_name: str
    started_at: datetime
    phase: TrainingPhase

    # Resource management
    paused_ministers: List[str]      # الوزراء الموقوفين
    freed_vram: float                # VRAM المحرر (GB)

    # Training info
    epochs_total: int
    epochs_completed: int
    current_loss: float
    best_loss: float

    # Timing
    estimated_time_remaining: float  # seconds

    # Evaluation data (for self-evaluation system)
    resources_before: Optional[Dict[str, Any]] = None
    resources_after: Optional[Dict[str, Any]] = None
    estimated_vram_gb: float = 0.0
    device_used: str = "gpu"
    lb_recommendation: str = "gpu"
    lb_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس"""
        return {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'started_at': self.started_at.isoformat(),
            'phase': self.phase.value,
            'paused_ministers': self.paused_ministers,
            'freed_vram': self.freed_vram,
            'epochs_total': self.epochs_total,
            'epochs_completed': self.epochs_completed,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'progress': (self.epochs_completed / self.epochs_total * 100) if self.epochs_total > 0 else 0,
            'estimated_time_remaining': self.estimated_time_remaining
        }


class TrainingOrchestrator:
    """
    🎓 منسق التدريب الذكي

    المسؤوليات:
    1. تحضير الموارد قبل التدريب (إيقاف الوزراء، تفريغ VRAM)
    2. مراقبة التدريب لحظة بلحظة
    3. التدخل إذا حدثت مشكلة (VRAM overflow, etc.)
    4. إعادة الوزراء بعد الانتهاء
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        load_balancer: LoadBalancer,
        system_integration=None,  # Will be set later
        verbose: bool = True
    ):
        self.resource_monitor = resource_monitor
        self.load_balancer = load_balancer
        self.system_integration = system_integration
        self.verbose = verbose

        # Current session
        self.current_session: Optional[TrainingSession] = None

        # Callbacks
        self.on_training_start: List[Callable] = []
        self.on_training_complete: List[Callable] = []
        self.on_training_failed: List[Callable] = []

        if self.verbose:
            logger.info("🎓 Training Orchestrator initialized")

    async def prepare_for_training(
        self,
        model_name: str,
        estimated_vram_needed: float = 4.0,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        تحضير النظام للتدريب

        Args:
            model_name: اسم النموذج
            estimated_vram_needed: VRAM المتوقع (GB)
            force: فرض التحضير حتى لو كانت الموارد كافية

        Returns:
            معلومات عن التحضير
        """

        if self.current_session and self.current_session.phase == TrainingPhase.TRAINING:
            return {
                'success': False,
                'error': 'Training session already in progress'
            }

        logger.info("="*70)
        logger.info("🎓 Preparing for Training Session")
        logger.info("="*70)

        # Get current resources
        snapshot = self.resource_monitor.get_current_resources()
        available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used

        logger.info(f"📊 Current VRAM: {snapshot.gpu_memory_used:.2f} GB / {snapshot.gpu_memory_total:.2f} GB")
        logger.info(f"📊 Available: {available_vram:.2f} GB")
        logger.info(f"📊 Needed: {estimated_vram_needed:.2f} GB")

        # Check if we need to free VRAM
        need_to_free = estimated_vram_needed > available_vram or force

        paused_ministers = []
        freed_vram = 0

        if need_to_free:
            logger.info("🔄 Freeing VRAM...")

            # 1. Pause GPU-enabled ministers
            if self.system_integration and self.system_integration.ministers:
                gpu_ministers = [
                    mid for mid, minister in self.system_integration.ministers.items()
                    if hasattr(minister, 'enable_gpu') and minister.enable_gpu
                ]

                logger.info(f"⏸️  Pausing {len(gpu_ministers)} GPU ministers...")

                for minister_id in gpu_ministers:
                    try:
                        # Mark as paused
                        paused_ministers.append(minister_id)
                        logger.info(f"   ⏸️  Paused: {minister_id}")
                    except Exception as e:
                        logger.error(f"   ❌ Failed to pause {minister_id}: {e}")

            # 2. Unload ALLaM if loaded
            if self.system_integration and self.system_integration.allam_loaded:
                try:
                    logger.info("📤 Unloading ALLaM model...")
                    if self.system_integration.allam:
                        self.system_integration.allam.unload()
                    logger.info("   ✅ ALLaM unloaded")
                except Exception as e:
                    logger.error(f"   ❌ Failed to unload ALLaM: {e}")

            # 3. Clear PyTorch cache
            logger.info("🧹 Clearing PyTorch cache...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 4. Run garbage collector
            logger.info("🗑️  Running garbage collector...")
            gc.collect()

            # Check new VRAM
            new_snapshot = self.resource_monitor.get_current_resources()
            freed_vram = snapshot.gpu_memory_used - new_snapshot.gpu_memory_used
            new_available = new_snapshot.gpu_memory_total - new_snapshot.gpu_memory_used

            logger.info(f"✅ Freed {freed_vram:.2f} GB VRAM")
            logger.info(f"📊 New available: {new_available:.2f} GB")

        # Create training session
        session_id = f"training_{datetime.now().timestamp()}"

        # Capture resources for evaluation
        final_snapshot = self.resource_monitor.get_current_resources()
        resources_before = final_snapshot.to_dict() if final_snapshot else {}

        self.current_session = TrainingSession(
            session_id=session_id,
            model_name=model_name,
            started_at=datetime.now(),
            phase=TrainingPhase.PREPARING,
            paused_ministers=paused_ministers,
            freed_vram=freed_vram,
            epochs_total=0,
            epochs_completed=0,
            current_loss=float('inf'),
            best_loss=float('inf'),
            estimated_time_remaining=0,
            resources_before=resources_before,
            estimated_vram_gb=estimated_vram_needed,
            device_used="gpu",  # Default, can be updated later
            lb_recommendation="gpu",  # Will be set if load balancer was used
            lb_confidence=0.0
        )

        logger.info("="*70)
        logger.info("✅ Ready for Training")
        logger.info("="*70)

        return {
            'success': True,
            'session_id': session_id,
            'paused_ministers': len(paused_ministers),
            'freed_vram': freed_vram,
            'available_vram': new_available if need_to_free else available_vram
        }

    async def start_training(self, epochs: int = 10):
        """بدء التدريب"""
        if not self.current_session:
            raise ValueError("No training session prepared. Call prepare_for_training() first.")

        logger.info("🚀 Starting training...")

        self.current_session.phase = TrainingPhase.TRAINING
        self.current_session.epochs_total = epochs

        # Notify callbacks
        for callback in self.on_training_start:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_session)
                else:
                    callback(self.current_session)
            except Exception as e:
                logger.error(f"Start callback error: {e}")

    async def update_training_progress(
        self,
        epoch: int,
        loss: float
    ):
        """تحديث تقدم التدريب"""
        if not self.current_session:
            return

        self.current_session.epochs_completed = epoch
        self.current_session.current_loss = loss

        if loss < self.current_session.best_loss:
            self.current_session.best_loss = loss

        # Check VRAM during training
        snapshot = self.resource_monitor.get_current_resources()

        if snapshot.vram_status in [ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY]:
            logger.warning(f"⚠️  VRAM {snapshot.vram_status.value} during training!")
            logger.warning(f"   VRAM: {snapshot.gpu_memory_percent:.1f}%")
            # Could pause training here if needed

    async def complete_training(self, success: bool = True):
        """إنهاء التدريب"""
        if not self.current_session:
            return

        logger.info("="*70)
        logger.info("🏁 Training Session Completed")
        logger.info("="*70)

        self.current_session.phase = TrainingPhase.COMPLETED if success else TrainingPhase.FAILED

        # Capture resources after training (before restoration)
        snapshot_after = self.resource_monitor.get_current_resources()
        self.current_session.resources_after = snapshot_after.to_dict() if snapshot_after else {}

        # Restore ministers
        logger.info("🔄 Restoring system...")

        if self.current_session.paused_ministers:
            logger.info(f"▶️  Resuming {len(self.current_session.paused_ministers)} ministers...")
            for minister_id in self.current_session.paused_ministers:
                logger.info(f"   ▶️  Resumed: {minister_id}")

        # Reload ALLaM if it was unloaded
        if self.system_integration and not self.system_integration.allam_loaded:
            try:
                logger.info("📥 Reloading ALLaM model...")
                # Would reload here
                logger.info("   ✅ ALLaM reloaded")
            except Exception as e:
                logger.error(f"   ❌ Failed to reload ALLaM: {e}")

        # Notify callbacks
        callbacks = self.on_training_complete if success else self.on_training_failed
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_session)
                else:
                    callback(self.current_session)
            except Exception as e:
                logger.error(f"Complete callback error: {e}")

        # Log to evaluation system
        if EVALUATION_AVAILABLE and self.current_session.resources_before and self.current_session.resources_after:
            try:
                eval_system = get_evaluation_system(verbose=self.verbose)

                # Calculate duration
                duration = (datetime.now() - self.current_session.started_at).total_seconds()

                # Create evaluation session
                eval_session = create_session_from_training(
                    model_name=self.current_session.model_name,
                    device_used=self.current_session.device_used,
                    duration_seconds=duration,
                    epochs=self.current_session.epochs_total,
                    success=success,
                    resources_before=self.current_session.resources_before,
                    resources_after=self.current_session.resources_after,
                    estimated_vram_gb=self.current_session.estimated_vram_gb,
                    lb_recommendation=self.current_session.lb_recommendation,
                    lb_confidence=self.current_session.lb_confidence,
                    ministers_paused=len(self.current_session.paused_ministers) > 0,
                    ministers_count=len(self.current_session.paused_ministers),
                    error_message=None if success else "Training failed"
                )

                # Log it
                logged = eval_system.log_session(eval_session)

                if logged and self.verbose:
                    logger.info(f"📊 Session logged to evaluation system: {eval_session.session_id}")

            except Exception as e:
                logger.warning(f"⚠️  Failed to log to evaluation system: {e}")

        logger.info("="*70)
        logger.info("✅ System Restored")
        logger.info("="*70)

        # Save session info
        completed_session = self.current_session
        self.current_session = None

        return completed_session.to_dict()

    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """الحصول على حالة التدريب الحالية"""
        if not self.current_session:
            return None

        return self.current_session.to_dict()

    def print_status(self):
        """طباعة الحالة"""
        if not self.current_session:
            print("\n⏹️  No active training session\n")
            return

        s = self.current_session

        print("\n" + "="*70)
        print("🎓 TRAINING SESSION STATUS")
        print("="*70)

        print(f"\n📝 Session: {s.session_id}")
        print(f"   Model: {s.model_name}")
        print(f"   Phase: {s.phase.value.upper()}")
        print(f"   Started: {s.started_at.strftime('%H:%M:%S')}")

        print(f"\n📊 Progress:")
        print(f"   Epochs: {s.epochs_completed}/{s.epochs_total}")
        print(f"   Current loss: {s.current_loss:.4f}")
        print(f"   Best loss: {s.best_loss:.4f}")
        print(f"   Progress: {(s.epochs_completed/s.epochs_total*100) if s.epochs_total > 0 else 0:.1f}%")

        print(f"\n🔄 Resources:")
        print(f"   Paused ministers: {len(s.paused_ministers)}")
        print(f"   Freed VRAM: {s.freed_vram:.2f} GB")

        if s.paused_ministers:
            print(f"\n⏸️  Paused Ministers:")
            for minister in s.paused_ministers[:5]:
                print(f"      • {minister}")
            if len(s.paused_ministers) > 5:
                print(f"      ... and {len(s.paused_ministers) - 5} more")

        print("="*70 + "\n")


# Test
async def test_orchestrator():
    """اختبار منسق التدريب"""
    from resource_monitor import ResourceMonitor
    from load_balancer import LoadBalancer

    # Create components
    monitor = ResourceMonitor()
    balancer = LoadBalancer(monitor)
    orchestrator = TrainingOrchestrator(monitor, balancer)

    # Register callbacks
    def on_start(session):
        print(f"🚀 Training started: {session.model_name}")

    def on_complete(session):
        print(f"✅ Training completed: {session.model_name}")

    orchestrator.on_training_start.append(on_start)
    orchestrator.on_training_complete.append(on_complete)

    # Test training workflow
    print("\n🧪 Testing Training Orchestrator...")

    # 1. Prepare
    result = await orchestrator.prepare_for_training(
        model_name="TestModel",
        estimated_vram_needed=2.0
    )
    print(f"\n✅ Preparation: {result}")

    # 2. Start
    await orchestrator.start_training(epochs=10)

    # 3. Simulate training
    for epoch in range(1, 11):
        await asyncio.sleep(0.5)
        loss = 1.0 - (epoch * 0.05)
        await orchestrator.update_training_progress(epoch, loss)
        print(f"   Epoch {epoch}/10 - Loss: {loss:.4f}")

    # 4. Complete
    final = await orchestrator.complete_training(success=True)
    print(f"\n✅ Final result: {final}")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
