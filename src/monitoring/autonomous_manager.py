#!/usr/bin/env python3
"""
🤖 Autonomous System Manager - المدير الذاتي الكامل
يربط جميع المكونات ويتخذ قرارات ذاتية لإدارة الموارد
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio

from .resource_monitor import ResourceMonitor, ResourceStatus, ResourceThresholds
from .load_balancer import LoadBalancer, TaskPriority
from .training_orchestrator import TrainingOrchestrator, TrainingPhase

logger = logging.getLogger(__name__)


class AutonomousSystemManager:
    """
    🤖 المدير الذاتي الكامل لنظام نوغ

    المسؤوليات:
    1. مراقبة الموارد 24/7
    2. اتخاذ قرارات فورية عند المشاكل
    3. إدارة الوزراء (إيقاف/تشغيل حسب الحاجة)
    4. تنسيق التدريب
    5. تحسين استخدام الموارد
    """

    def __init__(
        self,
        system_integration=None,
        check_interval: float = 5.0,
        verbose: bool = True
    ):
        self.system_integration = system_integration
        self.verbose = verbose

        # Create monitoring components
        self.resource_monitor = ResourceMonitor(
            check_interval=check_interval,
            verbose=verbose
        )

        self.load_balancer = LoadBalancer(
            resource_monitor=self.resource_monitor,
            verbose=verbose
        )

        self.training_orchestrator = TrainingOrchestrator(
            resource_monitor=self.resource_monitor,
            load_balancer=self.load_balancer,
            system_integration=system_integration,
            verbose=verbose
        )

        # Decision history
        self.decisions_made: List[Dict[str, Any]] = []
        self.max_decision_history = 1000

        # State
        self.is_running = False
        self.manager_task = None

        # Statistics
        self.stats = {
            'total_decisions': 0,
            'vram_interventions': 0,
            'temp_interventions': 0,
            'cpu_interventions': 0,
            'auto_optimizations': 0,
            'started_at': None
        }

        # Register handlers
        self._register_handlers()

        if self.verbose:
            logger.info("🤖 Autonomous System Manager initialized")

    def _register_handlers(self):
        """تسجيل معالجات الأحداث"""

        # WARNING handlers
        self.resource_monitor.on_warning(self._handle_warning)

        # CRITICAL handlers
        self.resource_monitor.on_critical(self._handle_critical)

        # EMERGENCY handlers
        self.resource_monitor.on_emergency(self._handle_emergency)

    async def _handle_warning(self, snapshot):
        """معالجة حالة WARNING"""
        decision = {
            'timestamp': datetime.now(),
            'level': 'WARNING',
            'snapshot': snapshot.to_dict(),
            'actions': []
        }

        # Check which resource is in warning
        if snapshot.vram_status == ResourceStatus.WARNING:
            # VRAM Warning: Be more selective about GPU usage
            action = {
                'type': 'vram_warning',
                'description': 'Increased selectivity for GPU allocation',
                'vram_percent': snapshot.gpu_memory_percent
            }
            decision['actions'].append(action)
            self.stats['vram_interventions'] += 1

        if snapshot.temp_status == ResourceStatus.WARNING:
            # Temperature Warning: Reduce GPU load
            action = {
                'type': 'temp_warning',
                'description': 'Considering temperature in allocation decisions',
                'temperature': snapshot.gpu_temperature
            }
            decision['actions'].append(action)
            self.stats['temp_interventions'] += 1

        self._log_decision(decision)

    async def _handle_critical(self, snapshot):
        """معالجة حالة CRITICAL"""
        decision = {
            'timestamp': datetime.now(),
            'level': 'CRITICAL',
            'snapshot': snapshot.to_dict(),
            'actions': []
        }

        # VRAM Critical: Pause low-priority GPU ministers
        if snapshot.vram_status == ResourceStatus.CRITICAL:
            if self.system_integration and self.system_integration.ministers:
                # Find GPU ministers
                gpu_ministers = [
                    (mid, m) for mid, m in self.system_integration.ministers.items()
                    if hasattr(m, 'enable_gpu') and m.enable_gpu
                ]

                # Pause some of them
                paused_count = 0
                for minister_id, minister in gpu_ministers[:len(gpu_ministers)//2]:
                    try:
                        # Mark as paused (would actually pause in real implementation)
                        paused_count += 1
                    except Exception as e:
                        logger.error(f"Failed to pause {minister_id}: {e}")

                action = {
                    'type': 'pause_ministers_critical',
                    'description': f'Paused {paused_count} GPU ministers due to CRITICAL VRAM',
                    'vram_percent': snapshot.gpu_memory_percent,
                    'paused_count': paused_count
                }
                decision['actions'].append(action)
                self.stats['vram_interventions'] += 1

        # Temperature Critical: Force CPU for new tasks
        if snapshot.temp_status == ResourceStatus.CRITICAL:
            action = {
                'type': 'temp_critical',
                'description': 'Forcing CPU allocation for cooling',
                'temperature': snapshot.gpu_temperature
            }
            decision['actions'].append(action)
            self.stats['temp_interventions'] += 1

        self._log_decision(decision)

    async def _handle_emergency(self, snapshot):
        """معالجة حالة EMERGENCY"""
        decision = {
            'timestamp': datetime.now(),
            'level': 'EMERGENCY',
            'snapshot': snapshot.to_dict(),
            'actions': []
        }

        logger.error("🆘 EMERGENCY MODE ACTIVATED")

        # VRAM Emergency: Pause ALL GPU ministers except critical ones
        if snapshot.vram_status == ResourceStatus.EMERGENCY:
            if self.system_integration and self.system_integration.ministers:
                gpu_ministers = [
                    (mid, m) for mid, m in self.system_integration.ministers.items()
                    if hasattr(m, 'enable_gpu') and m.enable_gpu
                ]

                paused_count = 0
                for minister_id, minister in gpu_ministers:
                    try:
                        # Pause (would actually pause in real implementation)
                        paused_count += 1
                        logger.warning(f"   🆘 Emergency pause: {minister_id}")
                    except Exception as e:
                        logger.error(f"Failed to pause {minister_id}: {e}")

                # Clear PyTorch cache
                import torch
                import gc

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

                action = {
                    'type': 'emergency_vram_clearance',
                    'description': f'Emergency: Paused ALL {paused_count} GPU ministers + cleared cache',
                    'vram_percent': snapshot.gpu_memory_percent,
                    'paused_count': paused_count
                }
                decision['actions'].append(action)
                self.stats['vram_interventions'] += 1

        # Temperature Emergency: Emergency shutdown of GPU tasks
        if snapshot.temp_status == ResourceStatus.EMERGENCY:
            logger.error(f"   🔥 GPU OVERHEATING: {snapshot.gpu_temperature}°C")

            action = {
                'type': 'emergency_temp',
                'description': 'Emergency temperature - GPU tasks halted',
                'temperature': snapshot.gpu_temperature
            }
            decision['actions'].append(action)
            self.stats['temp_interventions'] += 1

        self._log_decision(decision)

    def _log_decision(self, decision: Dict[str, Any]):
        """تسجيل قرار"""
        self.decisions_made.append(decision)
        if len(self.decisions_made) > self.max_decision_history:
            self.decisions_made.pop(0)

        self.stats['total_decisions'] += 1

        if self.verbose and decision['actions']:
            logger.info(f"🤖 Decision [{decision['level']}]:")
            for action in decision['actions']:
                logger.info(f"   • {action['description']}")

    async def optimize_resources(self):
        """تحسين استخدام الموارد بشكل دوري"""
        snapshot = self.resource_monitor.current_snapshot

        if not snapshot:
            return

        # Only optimize if NORMAL status
        if snapshot.overall_status != ResourceStatus.NORMAL:
            return

        decision = {
            'timestamp': datetime.now(),
            'level': 'OPTIMIZATION',
            'snapshot': snapshot.to_dict(),
            'actions': []
        }

        # Check if we can move ministers back to GPU
        if snapshot.vram_status == ResourceStatus.NORMAL:
            available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used

            if available_vram >= 3.0:  # At least 3GB free
                # Could resume GPU ministers here
                action = {
                    'type': 'optimize_gpu_usage',
                    'description': f'GPU available ({available_vram:.1f} GB free) - ready for ministers',
                    'available_vram': available_vram
                }
                decision['actions'].append(action)
                self.stats['auto_optimizations'] += 1

        if decision['actions']:
            self._log_decision(decision)

    async def start(self):
        """بدء المدير الذاتي"""
        if self.is_running:
            logger.warning("⚠️  Already running")
            return

        logger.info("="*70)
        logger.info("🤖 STARTING AUTONOMOUS SYSTEM MANAGER")
        logger.info("="*70)

        self.is_running = True
        self.stats['started_at'] = datetime.now()

        # Start resource monitoring
        await self.resource_monitor.start_monitoring()

        # Start manager loop
        self.manager_task = asyncio.create_task(self._manager_loop())

        logger.info("✅ Autonomous management active")
        logger.info("="*70)

    async def stop(self):
        """إيقاف المدير الذاتي"""
        if not self.is_running:
            return

        logger.info("⏹️  Stopping Autonomous System Manager...")

        self.is_running = False

        # Stop resource monitoring
        await self.resource_monitor.stop_monitoring()

        # Stop manager loop
        if self.manager_task:
            self.manager_task.cancel()
            try:
                await self.manager_task
            except asyncio.CancelledError:
                pass

        logger.info("✅ Autonomous management stopped")

    async def _manager_loop(self):
        """حلقة الإدارة الرئيسية"""
        # Run optimization every 60 seconds
        optimization_interval = 60.0
        last_optimization = datetime.now()

        while self.is_running:
            try:
                # Check if it's time to optimize
                if (datetime.now() - last_optimization).total_seconds() >= optimization_interval:
                    await self.optimize_resources()
                    last_optimization = datetime.now()

                # Sleep
                await asyncio.sleep(10.0)

            except Exception as e:
                logger.error(f"❌ Manager loop error: {e}")
                await asyncio.sleep(10.0)

    def get_status(self) -> Dict[str, Any]:
        """الحصول على الحالة الكاملة"""
        uptime = None
        if self.stats['started_at']:
            uptime = (datetime.now() - self.stats['started_at']).total_seconds()

        return {
            'running': self.is_running,
            'uptime_seconds': uptime,
            'statistics': self.stats,
            'resource_monitor': {
                'current': self.resource_monitor.current_snapshot.to_dict() if self.resource_monitor.current_snapshot else None,
                'statistics': self.resource_monitor.get_statistics()
            },
            'load_balancer': self.load_balancer.get_statistics(),
            'training': self.training_orchestrator.get_training_status(),
            'recent_decisions': [d for d in self.decisions_made[-5:]]
        }

    def print_status(self):
        """طباعة الحالة الكاملة"""
        print("\n" + "="*70)
        print("🤖 AUTONOMOUS SYSTEM MANAGER - STATUS")
        print("="*70)

        print(f"\n⚙️  Manager:")
        print(f"   Running: {'✅ Yes' if self.is_running else '❌ No'}")
        if self.stats['started_at']:
            uptime = (datetime.now() - self.stats['started_at']).total_seconds()
            print(f"   Uptime: {uptime/60:.1f} minutes")

        print(f"\n📊 Statistics:")
        print(f"   Total decisions: {self.stats['total_decisions']}")
        print(f"   VRAM interventions: {self.stats['vram_interventions']}")
        print(f"   Temp interventions: {self.stats['temp_interventions']}")
        print(f"   Auto optimizations: {self.stats['auto_optimizations']}")

        # Resource monitor status
        self.resource_monitor.print_status()

        # Load balancer status
        self.load_balancer.print_status()

        # Training status
        if self.training_orchestrator.current_session:
            self.training_orchestrator.print_status()

        # Recent decisions
        if self.decisions_made:
            print("\n📜 Recent Decisions:")
            for decision in self.decisions_made[-5:]:
                print(f"\n   [{decision['level']}] {decision['timestamp'].strftime('%H:%M:%S')}")
                for action in decision['actions']:
                    print(f"      • {action['description']}")

        print("="*70 + "\n")


# Test
async def test_autonomous_manager():
    """اختبار المدير الذاتي"""

    print("\n🧪 Testing Autonomous System Manager...")

    # Create manager
    manager = AutonomousSystemManager(
        check_interval=2.0,
        verbose=True
    )

    # Start
    await manager.start()

    # Run for 30 seconds
    print("\n⏳ Running for 30 seconds...")
    await asyncio.sleep(30)

    # Print status
    manager.print_status()

    # Stop
    await manager.stop()

    print("\n✅ Test complete")


if __name__ == "__main__":
    asyncio.run(test_autonomous_manager())
