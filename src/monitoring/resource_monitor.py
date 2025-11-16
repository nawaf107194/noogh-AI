#!/usr/bin/env python3
"""
🔍 Resource Monitor - مراقب الموارد الذاتي
يراقب GPU, CPU, RAM, Temperature ويتخذ قرارات فورية
"""

import psutil
import GPUtil
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """حالة الموارد"""
    NORMAL = "normal"           # طبيعي
    WARNING = "warning"         # تحذير
    CRITICAL = "critical"       # حرج
    EMERGENCY = "emergency"     # طوارئ


@dataclass
class ResourceThresholds:
    """عتبات الموارد"""
    # GPU VRAM
    vram_warning: float = 75.0      # %
    vram_critical: float = 85.0     # %
    vram_emergency: float = 95.0    # %

    # GPU Temperature
    temp_warning: float = 70.0      # °C
    temp_critical: float = 80.0     # °C
    temp_emergency: float = 85.0    # °C

    # CPU
    cpu_warning: float = 70.0       # %
    cpu_critical: float = 85.0      # %
    cpu_emergency: float = 95.0     # %

    # RAM
    ram_warning: float = 75.0       # %
    ram_critical: float = 85.0      # %
    ram_emergency: float = 95.0     # %


@dataclass
class ResourceSnapshot:
    """لقطة من حالة الموارد"""
    timestamp: datetime

    # GPU
    gpu_memory_used: float          # GB
    gpu_memory_total: float         # GB
    gpu_memory_percent: float       # %
    gpu_temperature: float          # °C
    gpu_utilization: float          # %

    # CPU
    cpu_percent: float              # %
    cpu_cores: int
    cpu_freq_current: float         # MHz

    # RAM
    ram_used: float                 # GB
    ram_total: float                # GB
    ram_percent: float              # %
    ram_available: float            # GB

    # Status
    overall_status: ResourceStatus
    vram_status: ResourceStatus
    temp_status: ResourceStatus
    cpu_status: ResourceStatus
    ram_status: ResourceStatus

    # Warnings
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'gpu': {
                'memory_used': self.gpu_memory_used,
                'memory_total': self.gpu_memory_total,
                'memory_percent': self.gpu_memory_percent,
                'temperature': self.gpu_temperature,
                'utilization': self.gpu_utilization,
                'status': self.vram_status.value
            },
            'cpu': {
                'percent': self.cpu_percent,
                'cores': self.cpu_cores,
                'freq_current': self.cpu_freq_current,
                'status': self.cpu_status.value
            },
            'ram': {
                'used': self.ram_used,
                'total': self.ram_total,
                'percent': self.ram_percent,
                'available': self.ram_available,
                'status': self.ram_status.value
            },
            'temperature': {
                'value': self.gpu_temperature,
                'status': self.temp_status.value
            },
            'overall_status': self.overall_status.value,
            'warnings': self.warnings
        }


class ResourceMonitor:
    """
    🔍 مراقب الموارد الذكي

    يراقب:
    - GPU VRAM
    - GPU Temperature
    - CPU Usage
    - RAM Usage

    يتخذ قرارات فورية عند تجاوز العتبات
    """

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        check_interval: float = 5.0,
        verbose: bool = True
    ):
        self.thresholds = thresholds or ResourceThresholds()
        self.check_interval = check_interval
        self.verbose = verbose

        # History
        self.history: List[ResourceSnapshot] = []
        self.max_history = 1000

        # Event handlers
        self.warning_handlers: List[Callable] = []
        self.critical_handlers: List[Callable] = []
        self.emergency_handlers: List[Callable] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitor_task = None

        # Current snapshot
        self.current_snapshot: Optional[ResourceSnapshot] = None

        if self.verbose:
            logger.info("🔍 Resource Monitor initialized")
            logger.info(f"   Check interval: {self.check_interval}s")

    def get_current_resources(self) -> ResourceSnapshot:
        """الحصول على حالة الموارد الحالية"""

        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory_used = gpu.memoryUsed / 1024  # GB
                gpu_memory_total = gpu.memoryTotal / 1024  # GB
                gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                gpu_temperature = gpu.temperature
                gpu_utilization = gpu.load * 100
            else:
                gpu_memory_used = 0
                gpu_memory_total = 0
                gpu_memory_percent = 0
                gpu_temperature = 0
                gpu_utilization = 0
        except:
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_memory_percent = 0
            gpu_temperature = 0
            gpu_utilization = 0

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_cores = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else 0

        # RAM
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # GB
        ram_total = ram.total / (1024**3)  # GB
        ram_percent = ram.percent
        ram_available = ram.available / (1024**3)  # GB

        # Determine status for each resource
        vram_status = self._get_status(gpu_memory_percent, 'vram')
        temp_status = self._get_status(gpu_temperature, 'temp')
        cpu_status = self._get_status(cpu_percent, 'cpu')
        ram_status = self._get_status(ram_percent, 'ram')

        # Overall status (worst of all)
        statuses = [vram_status, temp_status, cpu_status, ram_status]
        status_priority = {
            ResourceStatus.NORMAL: 0,
            ResourceStatus.WARNING: 1,
            ResourceStatus.CRITICAL: 2,
            ResourceStatus.EMERGENCY: 3
        }
        overall_status = max(statuses, key=lambda s: status_priority[s])

        # Generate warnings
        warnings = []
        if vram_status != ResourceStatus.NORMAL:
            warnings.append(f"VRAM {vram_status.value}: {gpu_memory_percent:.1f}%")
        if temp_status != ResourceStatus.NORMAL:
            warnings.append(f"Temperature {temp_status.value}: {gpu_temperature:.1f}°C")
        if cpu_status != ResourceStatus.NORMAL:
            warnings.append(f"CPU {cpu_status.value}: {cpu_percent:.1f}%")
        if ram_status != ResourceStatus.NORMAL:
            warnings.append(f"RAM {ram_status.value}: {ram_percent:.1f}%")

        # Create snapshot
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature,
            gpu_utilization=gpu_utilization,
            cpu_percent=cpu_percent,
            cpu_cores=cpu_cores,
            cpu_freq_current=cpu_freq_current,
            ram_used=ram_used,
            ram_total=ram_total,
            ram_percent=ram_percent,
            ram_available=ram_available,
            overall_status=overall_status,
            vram_status=vram_status,
            temp_status=temp_status,
            cpu_status=cpu_status,
            ram_status=ram_status,
            warnings=warnings
        )

        return snapshot

    def _get_status(self, value: float, resource_type: str) -> ResourceStatus:
        """تحديد حالة المورد بناءً على القيمة"""
        if resource_type == 'vram':
            if value >= self.thresholds.vram_emergency:
                return ResourceStatus.EMERGENCY
            elif value >= self.thresholds.vram_critical:
                return ResourceStatus.CRITICAL
            elif value >= self.thresholds.vram_warning:
                return ResourceStatus.WARNING

        elif resource_type == 'temp':
            if value >= self.thresholds.temp_emergency:
                return ResourceStatus.EMERGENCY
            elif value >= self.thresholds.temp_critical:
                return ResourceStatus.CRITICAL
            elif value >= self.thresholds.temp_warning:
                return ResourceStatus.WARNING

        elif resource_type == 'cpu':
            if value >= self.thresholds.cpu_emergency:
                return ResourceStatus.EMERGENCY
            elif value >= self.thresholds.cpu_critical:
                return ResourceStatus.CRITICAL
            elif value >= self.thresholds.cpu_warning:
                return ResourceStatus.WARNING

        elif resource_type == 'ram':
            if value >= self.thresholds.ram_emergency:
                return ResourceStatus.EMERGENCY
            elif value >= self.thresholds.ram_critical:
                return ResourceStatus.CRITICAL
            elif value >= self.thresholds.ram_warning:
                return ResourceStatus.WARNING

        return ResourceStatus.NORMAL

    async def _monitor_loop(self):
        """حلقة المراقبة الرئيسية"""
        logger.info("🔍 Resource monitoring started")

        while self.is_monitoring:
            try:
                # Get current resources
                snapshot = self.get_current_resources()
                self.current_snapshot = snapshot

                # Add to history
                self.history.append(snapshot)
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                # Handle status changes
                await self._handle_status(snapshot)

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _handle_status(self, snapshot: ResourceSnapshot):
        """معالجة حالة الموارد"""

        # Emergency handlers
        if snapshot.overall_status == ResourceStatus.EMERGENCY:
            if self.verbose:
                logger.error(f"🚨 EMERGENCY: {', '.join(snapshot.warnings)}")
            for handler in self.emergency_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(snapshot)
                    else:
                        handler(snapshot)
                except Exception as e:
                    logger.error(f"Emergency handler error: {e}")

        # Critical handlers
        elif snapshot.overall_status == ResourceStatus.CRITICAL:
            if self.verbose:
                logger.warning(f"⚠️  CRITICAL: {', '.join(snapshot.warnings)}")
            for handler in self.critical_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(snapshot)
                    else:
                        handler(snapshot)
                except Exception as e:
                    logger.error(f"Critical handler error: {e}")

        # Warning handlers
        elif snapshot.overall_status == ResourceStatus.WARNING:
            if self.verbose:
                logger.warning(f"⚠️  WARNING: {', '.join(snapshot.warnings)}")
            for handler in self.warning_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(snapshot)
                    else:
                        handler(snapshot)
                except Exception as e:
                    logger.error(f"Warning handler error: {e}")

    def on_warning(self, handler: Callable):
        """تسجيل معالج للتحذيرات"""
        self.warning_handlers.append(handler)

    def on_critical(self, handler: Callable):
        """تسجيل معالج للحالات الحرجة"""
        self.critical_handlers.append(handler)

    def on_emergency(self, handler: Callable):
        """تسجيل معالج للطوارئ"""
        self.emergency_handlers.append(handler)

    async def start_monitoring(self):
        """بدء المراقبة"""
        if self.is_monitoring:
            logger.warning("⚠️  Already monitoring")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())

        if self.verbose:
            logger.info("✅ Resource monitoring started")

    async def stop_monitoring(self):
        """إيقاف المراقبة"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.verbose:
            logger.info("⏹️  Resource monitoring stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات من السجل"""
        if not self.history:
            return {}

        vram_values = [s.gpu_memory_percent for s in self.history]
        temp_values = [s.gpu_temperature for s in self.history]
        cpu_values = [s.cpu_percent for s in self.history]
        ram_values = [s.ram_percent for s in self.history]

        return {
            'samples': len(self.history),
            'vram': {
                'avg': sum(vram_values) / len(vram_values),
                'min': min(vram_values),
                'max': max(vram_values),
                'current': vram_values[-1]
            },
            'temperature': {
                'avg': sum(temp_values) / len(temp_values),
                'min': min(temp_values),
                'max': max(temp_values),
                'current': temp_values[-1]
            },
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'current': cpu_values[-1]
            },
            'ram': {
                'avg': sum(ram_values) / len(ram_values),
                'min': min(ram_values),
                'max': max(ram_values),
                'current': ram_values[-1]
            }
        }

    def print_status(self):
        """طباعة الحالة الحالية"""
        if not self.current_snapshot:
            logger.info("No snapshot available")
            return

        s = self.current_snapshot

        print("\n" + "="*70)
        print("🔍 RESOURCE MONITOR STATUS")
        print("="*70)

        # GPU
        print(f"\n🎮 GPU (Status: {s.vram_status.value.upper()})")
        print(f"   VRAM: {s.gpu_memory_used:.2f} GB / {s.gpu_memory_total:.2f} GB ({s.gpu_memory_percent:.1f}%)")
        print(f"   Temperature: {s.gpu_temperature:.1f}°C ({s.temp_status.value})")
        print(f"   Utilization: {s.gpu_utilization:.1f}%")

        # CPU
        print(f"\n🖥️  CPU (Status: {s.cpu_status.value.upper()})")
        print(f"   Usage: {s.cpu_percent:.1f}%")
        print(f"   Cores: {s.cpu_cores}")
        print(f"   Frequency: {s.cpu_freq_current:.0f} MHz")

        # RAM
        print(f"\n💾 RAM (Status: {s.ram_status.value.upper()})")
        print(f"   Used: {s.ram_used:.2f} GB / {s.ram_total:.2f} GB ({s.ram_percent:.1f}%)")
        print(f"   Available: {s.ram_available:.2f} GB")

        # Overall
        print(f"\n📊 Overall Status: {s.overall_status.value.upper()}")

        if s.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in s.warnings:
                print(f"   • {warning}")

        print("="*70 + "\n")


# Test
async def test_monitor():
    """اختبار المراقب"""

    monitor = ResourceMonitor(check_interval=2.0)

    # Register handlers
    def on_warning(snapshot):
        print(f"⚠️  WARNING triggered: {snapshot.warnings}")

    def on_critical(snapshot):
        print(f"🚨 CRITICAL triggered: {snapshot.warnings}")

    async def on_emergency(snapshot):
        print(f"🆘 EMERGENCY triggered: {snapshot.warnings}")
        print("   Taking immediate action!")

    monitor.on_warning(on_warning)
    monitor.on_critical(on_critical)
    monitor.on_emergency(on_emergency)

    # Start monitoring
    await monitor.start_monitoring()

    # Monitor for 30 seconds
    for i in range(15):
        await asyncio.sleep(2)
        monitor.print_status()

    # Stop
    await monitor.stop_monitoring()

    # Print statistics
    stats = monitor.get_statistics()
    print("\n📊 Statistics:")
    print(f"   Samples: {stats['samples']}")
    print(f"   VRAM avg: {stats['vram']['avg']:.1f}%")
    print(f"   CPU avg: {stats['cpu']['avg']:.1f}%")
    print(f"   RAM avg: {stats['ram']['avg']:.1f}%")


if __name__ == "__main__":
    asyncio.run(test_monitor())
