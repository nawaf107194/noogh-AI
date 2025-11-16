"""
System Monitor - Real Hardware Monitoring
GPU, CPU, RAM, VRAM, Temperature tracking
"""

import psutil
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

@dataclass
class SystemHealth:
    timestamp: float
    cpu_percent: float
    cpu_count: int
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_temp: Optional[float] = None
    gpu_utilization: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_percent: Optional[float] = None

    def is_healthy(self) -> bool:
        """Overall health check"""
        issues = []

        if self.cpu_percent > 90:
            issues.append("CPU")
        if self.ram_percent > 90:
            issues.append("RAM")
        if self.disk_percent > 90:
            issues.append("DISK")
        if self.gpu_available:
            if self.gpu_temp and self.gpu_temp > 85:
                issues.append("GPU_TEMP")
            if self.vram_percent and self.vram_percent > 90:
                issues.append("VRAM")

        return len(issues) == 0

    def get_issues(self) -> list:
        """Get list of health issues"""
        issues = []

        if self.cpu_percent > 90:
            issues.append(f"High CPU usage: {self.cpu_percent:.1f}%")
        if self.cpu_percent > 80:
            issues.append(f"Elevated CPU usage: {self.cpu_percent:.1f}%")

        if self.ram_percent > 90:
            issues.append(f"High RAM usage: {self.ram_percent:.1f}%")
        if self.ram_percent > 80:
            issues.append(f"Elevated RAM usage: {self.ram_percent:.1f}%")

        if self.disk_percent > 90:
            issues.append(f"Disk almost full: {self.disk_percent:.1f}%")

        if self.gpu_available:
            if self.gpu_temp and self.gpu_temp > 85:
                issues.append(f"GPU overheating: {self.gpu_temp:.1f}°C")
            if self.gpu_temp and self.gpu_temp > 80:
                issues.append(f"GPU running hot: {self.gpu_temp:.1f}°C")

            if self.vram_percent and self.vram_percent > 90:
                issues.append(f"VRAM critical: {self.vram_percent:.1f}%")
            if self.vram_percent and self.vram_percent > 80:
                issues.append(f"VRAM high: {self.vram_percent:.1f}%")

        return issues

class SystemMonitor:
    """Real-time system monitoring"""

    def __init__(self):
        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass

    def get_cpu_stats(self) -> Dict:
        """Get CPU statistics"""
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }

    def get_ram_stats(self) -> Dict:
        """Get RAM statistics"""
        mem = psutil.virtual_memory()
        return {
            "percent": mem.percent,
            "used_gb": mem.used / (1024**3),
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3)
        }

    def get_disk_stats(self) -> Dict:
        """Get Disk statistics"""
        disk = psutil.disk_usage('/')
        return {
            "percent": disk.percent,
            "used_gb": disk.used / (1024**3),
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3)
        }

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics (NVIDIA only)"""
        if not GPU_AVAILABLE or not self.gpu_handle:
            return None

        try:
            name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            vram_used = mem_info.used / (1024**3)
            vram_total = mem_info.total / (1024**3)
            vram_percent = (mem_info.used / mem_info.total) * 100

            return {
                "name": name,
                "temp": temp,
                "utilization": utilization.gpu,
                "vram_used_gb": vram_used,
                "vram_total_gb": vram_total,
                "vram_percent": vram_percent
            }
        except Exception as e:
            return None

    def capture_snapshot(self) -> SystemHealth:
        """Capture complete system snapshot"""
        cpu = self.get_cpu_stats()
        ram = self.get_ram_stats()
        disk = self.get_disk_stats()
        gpu = self.get_gpu_stats()

        return SystemHealth(
            timestamp=time.time(),
            cpu_percent=cpu["percent"],
            cpu_count=cpu["count"],
            ram_percent=ram["percent"],
            ram_used_gb=ram["used_gb"],
            ram_total_gb=ram["total_gb"],
            disk_percent=disk["percent"],
            disk_used_gb=disk["used_gb"],
            disk_total_gb=disk["total_gb"],
            gpu_available=gpu is not None,
            gpu_name=gpu["name"] if gpu else None,
            gpu_temp=gpu["temp"] if gpu else None,
            gpu_utilization=gpu["utilization"] if gpu else None,
            vram_used_gb=gpu["vram_used_gb"] if gpu else None,
            vram_total_gb=gpu["vram_total_gb"] if gpu else None,
            vram_percent=gpu["vram_percent"] if gpu else None
        )

    def get_health_report(self) -> Dict:
        """Get formatted health report"""
        snapshot = self.capture_snapshot()

        return {
            "timestamp": snapshot.timestamp,
            "healthy": snapshot.is_healthy(),
            "issues": snapshot.get_issues(),
            "metrics": {
                "cpu": {
                    "percent": snapshot.cpu_percent,
                    "cores": snapshot.cpu_count
                },
                "ram": {
                    "percent": snapshot.ram_percent,
                    "used_gb": round(snapshot.ram_used_gb, 2),
                    "total_gb": round(snapshot.ram_total_gb, 2)
                },
                "disk": {
                    "percent": snapshot.disk_percent,
                    "used_gb": round(snapshot.disk_used_gb, 2),
                    "total_gb": round(snapshot.disk_total_gb, 2)
                },
                "gpu": {
                    "available": snapshot.gpu_available,
                    "name": snapshot.gpu_name,
                    "temp": snapshot.gpu_temp,
                    "utilization": snapshot.gpu_utilization,
                    "vram": {
                        "percent": snapshot.vram_percent,
                        "used_gb": round(snapshot.vram_used_gb, 2) if snapshot.vram_used_gb else None,
                        "total_gb": round(snapshot.vram_total_gb, 2) if snapshot.vram_total_gb else None
                    } if snapshot.gpu_available else None
                } if snapshot.gpu_available else None
            }
        }


# Test
if __name__ == "__main__":
    monitor = SystemMonitor()

    print("=" * 70)
    print("🔍 System Health Monitor Test")
    print("=" * 70)

    report = monitor.get_health_report()

    print(f"\n✅ Healthy: {report['healthy']}")
    print(f"\n⚠️  Issues: {len(report['issues'])}")
    for issue in report['issues']:
        print(f"   • {issue}")

    metrics = report['metrics']
    print(f"\n💻 CPU: {metrics['cpu']['percent']:.1f}% ({metrics['cpu']['cores']} cores)")
    print(f"🧠 RAM: {metrics['ram']['percent']:.1f}% ({metrics['ram']['used_gb']:.1f}/{metrics['ram']['total_gb']:.1f} GB)")
    print(f"💾 Disk: {metrics['disk']['percent']:.1f}% ({metrics['disk']['used_gb']:.1f}/{metrics['disk']['total_gb']:.1f} GB)")

    if metrics['gpu']:
        gpu = metrics['gpu']
        print(f"\n🎮 GPU: {gpu['name']}")
        print(f"   Temperature: {gpu['temp']}°C")
        print(f"   Utilization: {gpu['utilization']}%")
        if gpu['vram']:
            print(f"   VRAM: {gpu['vram']['percent']:.1f}% ({gpu['vram']['used_gb']:.1f}/{gpu['vram']['total_gb']:.1f} GB)")
    else:
        print("\n🎮 GPU: Not available")

    print("\n" + "=" * 70)
