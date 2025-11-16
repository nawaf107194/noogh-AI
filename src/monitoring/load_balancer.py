#!/usr/bin/env python3
"""
⚖️ Load Balancer - مدير الحمل الذكي
يوزع المهام بين CPU و GPU ديناميكياً بناءً على حالة الموارد
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio

from .resource_monitor import ResourceMonitor, ResourceStatus, ResourceSnapshot

logger = logging.getLogger(__name__)


class ProcessingDevice(Enum):
    """جهاز المعالجة"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class TaskPriority(Enum):
    """أولوية المهمة"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskAllocation:
    """توزيع المهمة"""
    task_id: str
    task_type: str
    device: ProcessingDevice
    priority: TaskPriority
    estimated_vram: float  # GB
    estimated_time: float  # seconds
    timestamp: datetime
    reason: str


class LoadBalancer:
    """
    ⚖️ مدير الحمل الذكي

    يقرر أين تُنفذ كل مهمة (CPU أو GPU) بناءً على:
    - حالة الموارد الحالية
    - أولوية المهمة
    - حجم المهمة المتوقع
    - الحمل الحالي على كل جهاز
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        verbose: bool = True
    ):
        self.resource_monitor = resource_monitor
        self.verbose = verbose

        # Task tracking
        self.active_tasks: Dict[str, TaskAllocation] = {}
        self.completed_tasks: List[TaskAllocation] = []

        # Device capabilities (which ministers can use which device)
        self.gpu_capable_ministers = [
            'education', 'research', 'knowledge',
            'security', 'privacy', 'creativity',
            'analysis', 'strategy', 'training',
            'reasoning', 'finance'
        ]

        self.cpu_only_ministers = [
            'development', 'communication', 'resources'
        ]

        # Load balancing policy
        self.vram_reserve = 1.0  # GB - always keep 1GB free
        self.max_vram_percent = 85.0  # Never exceed 85% VRAM

        if self.verbose:
            logger.info("⚖️  Load Balancer initialized")

    def decide_device(
        self,
        task_type: str,
        minister_id: Optional[str] = None,
        estimated_vram: float = 0.5,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> TaskAllocation:
        """
        تحديد الجهاز المناسب لتنفيذ المهمة

        Args:
            task_type: نوع المهمة
            minister_id: معرف الوزير (إن وجد)
            estimated_vram: التقدير المتوقع للـ VRAM (GB)
            priority: أولوية المهمة

        Returns:
            TaskAllocation
        """

        task_id = f"task_{datetime.now().timestamp()}"
        snapshot = self.resource_monitor.current_snapshot

        if not snapshot:
            # No snapshot available, default to CPU for safety
            return TaskAllocation(
                task_id=task_id,
                task_type=task_type,
                device=ProcessingDevice.CPU,
                priority=priority,
                estimated_vram=estimated_vram,
                estimated_time=0,
                timestamp=datetime.now(),
                reason="No resource snapshot available"
            )

        # Check if minister can use GPU
        if minister_id and minister_id in self.cpu_only_ministers:
            return TaskAllocation(
                task_id=task_id,
                task_type=task_type,
                device=ProcessingDevice.CPU,
                priority=priority,
                estimated_vram=0,
                estimated_time=0,
                timestamp=datetime.now(),
                reason=f"Minister {minister_id} is CPU-only"
            )

        # Decision logic
        device, reason = self._decide_device_logic(
            snapshot, estimated_vram, priority, task_type
        )

        allocation = TaskAllocation(
            task_id=task_id,
            task_type=task_type,
            device=device,
            priority=priority,
            estimated_vram=estimated_vram if device == ProcessingDevice.GPU else 0,
            estimated_time=0,
            timestamp=datetime.now(),
            reason=reason
        )

        # Register task
        self.active_tasks[task_id] = allocation

        if self.verbose:
            logger.info(f"⚖️  Task {task_id[:8]}... → {device.value.upper()}: {reason}")

        return allocation

    def _decide_device_logic(
        self,
        snapshot: ResourceSnapshot,
        estimated_vram: float,
        priority: TaskPriority,
        task_type: str
    ) -> tuple[ProcessingDevice, str]:
        """منطق اتخاذ القرار"""

        # EMERGENCY: Force CPU for everything except CRITICAL tasks
        if snapshot.vram_status == ResourceStatus.EMERGENCY:
            if priority == TaskPriority.CRITICAL:
                return ProcessingDevice.GPU, "CRITICAL priority overrides EMERGENCY"
            else:
                return ProcessingDevice.CPU, "VRAM EMERGENCY - all non-critical to CPU"

        # CRITICAL VRAM: Very selective about GPU
        if snapshot.vram_status == ResourceStatus.CRITICAL:
            # Only HIGH and CRITICAL tasks can use GPU
            if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                # Check if there's enough VRAM
                available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used
                if available_vram >= (estimated_vram + self.vram_reserve):
                    return ProcessingDevice.GPU, "HIGH priority + enough VRAM"
                else:
                    return ProcessingDevice.CPU, "Not enough VRAM even for HIGH priority"
            else:
                return ProcessingDevice.CPU, "VRAM CRITICAL - only HIGH+ to GPU"

        # WARNING VRAM: Be cautious
        if snapshot.vram_status == ResourceStatus.WARNING:
            available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used

            # Check if task fits
            if available_vram >= (estimated_vram + self.vram_reserve):
                # HIGH and CRITICAL always get GPU
                if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                    return ProcessingDevice.GPU, "HIGH priority + available VRAM"

                # MEDIUM gets GPU if there's plenty of space
                if priority == TaskPriority.MEDIUM:
                    if available_vram >= (estimated_vram + 2.0):  # Extra buffer
                        return ProcessingDevice.GPU, "MEDIUM + good VRAM buffer"
                    else:
                        return ProcessingDevice.CPU, "VRAM WARNING - save space for higher priority"

                # LOW always goes to CPU when WARNING
                return ProcessingDevice.CPU, "VRAM WARNING - LOW priority to CPU"
            else:
                return ProcessingDevice.CPU, "Not enough VRAM"

        # NORMAL: Smart allocation based on task type and resources
        if snapshot.vram_status == ResourceStatus.NORMAL:
            available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used

            # Check temperature
            if snapshot.temp_status in [ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY]:
                return ProcessingDevice.CPU, f"GPU temp {snapshot.temp_status.value} - cooling down"

            # Check if task fits
            if available_vram >= (estimated_vram + self.vram_reserve):
                # Training tasks always get GPU if available
                if task_type in ['training', 'fine_tune', 'model_training']:
                    return ProcessingDevice.GPU, "Training task + VRAM available"

                # Analysis and reasoning tasks prefer GPU
                if task_type in ['analysis', 'reasoning', 'research']:
                    return ProcessingDevice.GPU, "AI task + VRAM available"

                # Other tasks: use GPU if priority is MEDIUM or higher
                if priority != TaskPriority.LOW:
                    return ProcessingDevice.GPU, "Normal conditions + MEDIUM+ priority"
                else:
                    # LOW priority: check CPU load
                    if snapshot.cpu_percent < 50.0:
                        return ProcessingDevice.CPU, "LOW priority + CPU available"
                    else:
                        return ProcessingDevice.GPU, "LOW priority but CPU busy"
            else:
                return ProcessingDevice.CPU, "Not enough VRAM"

        # Default fallback
        return ProcessingDevice.CPU, "Default fallback to CPU"

    def should_migrate_to_cpu(self, minister_id: str) -> bool:
        """
        هل يجب نقل وزير إلى CPU؟

        Args:
            minister_id: معرف الوزير

        Returns:
            True إذا يجب النقل
        """
        snapshot = self.resource_monitor.current_snapshot

        if not snapshot:
            return False

        # CPU-only ministers can't migrate
        if minister_id in self.cpu_only_ministers:
            return False

        # EMERGENCY: migrate all non-critical
        if snapshot.vram_status == ResourceStatus.EMERGENCY:
            return True

        # CRITICAL: migrate low-priority ministers
        if snapshot.vram_status == ResourceStatus.CRITICAL:
            # Check if this minister is currently critical
            # For now, migrate all when critical
            return True

        return False

    def should_migrate_to_gpu(self, minister_id: str) -> bool:
        """
        هل يجب نقل وزير إلى GPU؟

        Args:
            minister_id: معرف الوزير

        Returns:
            True إذا يجب النقل
        """
        snapshot = self.resource_monitor.current_snapshot

        if not snapshot:
            return False

        # CPU-only ministers can't use GPU
        if minister_id in self.cpu_only_ministers:
            return False

        # Only migrate to GPU if NORMAL and plenty of VRAM
        if snapshot.vram_status == ResourceStatus.NORMAL:
            available_vram = snapshot.gpu_memory_total - snapshot.gpu_memory_used
            # Need at least 3GB free to migrate
            if available_vram >= 3.0:
                return True

        return False

    def complete_task(self, task_id: str):
        """تسجيل اكتمال مهمة"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            self.completed_tasks.append(task)

            if self.verbose:
                logger.info(f"✅ Task {task_id[:8]}... completed on {task.device.value.upper()}")

    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات توزيع المهام"""
        total_tasks = len(self.completed_tasks)

        if total_tasks == 0:
            return {
                'total_tasks': 0,
                'active_tasks': len(self.active_tasks),
                'gpu_tasks': 0,
                'cpu_tasks': 0,
                'gpu_percentage': 0,
                'cpu_percentage': 0
            }

        gpu_tasks = sum(1 for t in self.completed_tasks if t.device == ProcessingDevice.GPU)
        cpu_tasks = total_tasks - gpu_tasks

        return {
            'total_tasks': total_tasks,
            'active_tasks': len(self.active_tasks),
            'gpu_tasks': gpu_tasks,
            'cpu_tasks': cpu_tasks,
            'gpu_percentage': (gpu_tasks / total_tasks) * 100,
            'cpu_percentage': (cpu_tasks / total_tasks) * 100
        }

    def print_status(self):
        """طباعة الحالة"""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("⚖️  LOAD BALANCER STATUS")
        print("="*70)

        print(f"\n📊 Task Statistics:")
        print(f"   Total completed: {stats['total_tasks']}")
        print(f"   Active tasks: {stats['active_tasks']}")
        print(f"   GPU tasks: {stats['gpu_tasks']} ({stats['gpu_percentage']:.1f}%)")
        print(f"   CPU tasks: {stats['cpu_tasks']} ({stats['cpu_percentage']:.1f}%)")

        if self.active_tasks:
            print(f"\n🔄 Active Tasks:")
            for task_id, task in list(self.active_tasks.items())[:5]:
                print(f"   • {task_id[:8]}... ({task.task_type}) → {task.device.value.upper()}")

        print("="*70 + "\n")

    def decide_execution_mode(
        self,
        task_type: str,
        model_id: Optional[str] = None,
        estimated_vram: float = 0.5,
        priority: TaskPriority = TaskPriority.MEDIUM,
        allow_cloud: bool = True
    ) -> Dict[str, Any]:
        """
        🌐 Decide execution mode: local (GPU/CPU) or cloud (Inference API)

        This is the smart fallback mechanism - when local GPU is too busy,
        we automatically fall back to HF Inference API.

        Args:
            task_type: نوع المهمة
            model_id: HF model ID (required for cloud fallback)
            estimated_vram: التقدير المتوقع للـ VRAM (GB)
            priority: أولوية المهمة
            allow_cloud: السماح باستخدام السحابة

        Returns:
            Dict with mode, device, reason, model_id

        Example:
            result = balancer.decide_execution_mode(
                task_type="text_generation",
                model_id="google/gemma-2-2b-it",
                estimated_vram=2.0,
                allow_cloud=True
            )

            if result['mode'] == 'cloud':
                # Use HF Inference API
                output = inference_client.generate(model=result['model_id'], ...)
            else:
                # Use local GPU/CPU
                output = local_model.generate(...)
        """
        snapshot = self.resource_monitor.current_snapshot

        # Try local first
        allocation = self.decide_device(
            task_type=task_type,
            estimated_vram=estimated_vram,
            priority=priority
        )

        # Check if we should use cloud instead
        use_cloud = False
        cloud_reason = None

        if allow_cloud and model_id:
            # Cloud fallback conditions:
            # 1. GPU assigned but VRAM is in WARNING/CRITICAL/EMERGENCY
            # 2. GPU assigned but not enough VRAM for the task
            # 3. Task is LOW priority and VRAM > 50%

            if allocation.device == ProcessingDevice.GPU:
                if snapshot and snapshot.vram_status in [
                    ResourceStatus.WARNING,
                    ResourceStatus.CRITICAL,
                    ResourceStatus.EMERGENCY
                ]:
                    use_cloud = True
                    cloud_reason = f"GPU VRAM {snapshot.vram_status.value} - using cloud to preserve local resources"

                elif snapshot and snapshot.gpu_memory_percent > 50 and priority == TaskPriority.LOW:
                    use_cloud = True
                    cloud_reason = f"GPU at {snapshot.gpu_memory_percent:.1f}% and LOW priority - offloading to cloud"

            # CPU fallback could also benefit from cloud for certain tasks
            elif allocation.device == ProcessingDevice.CPU and task_type in [
                'text_generation', 'text_classification', 'question_answering'
            ]:
                if priority == TaskPriority.LOW:
                    use_cloud = True
                    cloud_reason = "CPU execution for LOW priority - using cloud for better performance"

        # Build result
        if use_cloud:
            result = {
                'mode': 'cloud',
                'device': 'inference_api',
                'model_id': model_id,
                'reason': cloud_reason,
                'task_id': allocation.task_id,
                'estimated_cost': 'free (rate-limited)',
                'fallback_device': allocation.device.value  # Fallback if cloud fails
            }

            if self.verbose:
                logger.info(f"🌐 Task {allocation.task_id[:8]}... → CLOUD: {cloud_reason}")

        else:
            result = {
                'mode': 'local',
                'device': allocation.device.value,
                'model_id': None,
                'reason': allocation.reason,
                'task_id': allocation.task_id,
                'estimated_vram': allocation.estimated_vram
            }

        return result


# Test
async def test_load_balancer():
    """اختبار مدير الحمل"""
    from resource_monitor import ResourceMonitor

    # Create monitor
    monitor = ResourceMonitor()
    monitor.current_snapshot = monitor.get_current_resources()

    # Create load balancer
    balancer = LoadBalancer(monitor)

    # Test different scenarios
    print("\n🧪 Testing Load Balancer...")

    # Test 1: Normal task
    print("\n1️⃣  Testing normal task:")
    allocation = balancer.decide_device(
        task_type='analysis',
        minister_id='analysis',
        estimated_vram=0.5,
        priority=TaskPriority.MEDIUM
    )
    print(f"   Device: {allocation.device.value}")
    print(f"   Reason: {allocation.reason}")

    # Test 2: Training task (high VRAM)
    print("\n2️⃣  Testing training task:")
    allocation = balancer.decide_device(
        task_type='training',
        minister_id='training',
        estimated_vram=3.0,
        priority=TaskPriority.HIGH
    )
    print(f"   Device: {allocation.device.value}")
    print(f"   Reason: {allocation.reason}")

    # Test 3: CPU-only minister
    print("\n3️⃣  Testing CPU-only minister:")
    allocation = balancer.decide_device(
        task_type='development',
        minister_id='development',
        estimated_vram=0,
        priority=TaskPriority.MEDIUM
    )
    print(f"   Device: {allocation.device.value}")
    print(f"   Reason: {allocation.reason}")

    # Complete tasks
    for task_id in list(balancer.active_tasks.keys()):
        balancer.complete_task(task_id)

    # Print statistics
    balancer.print_status()


if __name__ == "__main__":
    asyncio.run(test_load_balancer())
