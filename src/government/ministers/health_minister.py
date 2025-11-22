#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Minister - Hardware Warlord & System Guardian
=====================================================

Monitors GPU/CPU and NOW controls system resources aggressively.
Detects peripherals, optimizes resources, AI-powered process management.
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class HealthMinister(BaseMinister):
    """
    Minister of Health - Hardware Guardian and Resource Warlord.
    
    NEW Powers:
    - Peripheral device detection
    - Aggressive resource optimization
    - AI-powered process killing recommendations
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Health Minister."""
        super().__init__(
            name="Health Minister (Hardware Warlord)",
            description="Hardware monitor, peripheral detector, resource optimizer.",
            brain=brain
        )
        
        self.system_prompt = """You are a System Health Specialist and Resource Manager.
Analyze hardware and make AGGRESSIVE recommendations:
- Identify resource hogs that can be terminated
- Recommend process kills for optimization
- Assess thermal/power issues
- Suggest cooling strategies

Be decisive and prioritize system performance."""
        
        # Initialize monitoring
        self.gpu_available = self._init_gpu_monitoring()
        self.pyudev_available = self._init_udev()
    
    
    def _init_gpu_monitoring(self) -> bool:
        """Initialize NVIDIA GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            logger.info("✅ GPU monitoring initialized")
            return True
        except ImportError:
            logger.warning("⚠️ pynvml not installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ GPU monitoring unavailable: {e}")
            return False
    
    def _init_udev(self) -> bool:
        """Initialize pyudev for device detection."""
        try:
            import pyudev
            logger.info("✅ Device detection (pyudev) initialized")
            return True
        except ImportError:
            logger.warning("⚠️ pyudev not installed. Install with: pip install pyudev")
            return False
    
    def monitor_peripherals(self) -> Dict[str, Any]:
        """
        Monitor connected peripheral devices (USB, HDD, etc.).
        
        Returns:
            List of detected devices
        """
        if not self.pyudev_available:
            return {"success": False, "error": "pyudev not available"}
        
        try:
            import pyudev
            
            context = pyudev.Context()
            devices = []
            
            # List block devices
            for device in context.list_devices(subsystem='block'):
                if device.device_type == 'disk':
                    devices.append({
                        "name": device.sys_name,
                        "type": "disk",
                        "path": device.device_node,
                        "size": device.attributes.get('size'),
                        "model": device.properties.get('ID_MODEL', 'Unknown')
                    })
            
            # List USB devices
            for device in context.list_devices(subsystem='usb'):
                if 'ID_MODEL' in device.properties:
                    devices.append({
                        "name": device.properties.get('ID_MODEL', 'Unknown'),
                        "type": "usb",
                        "vendor": device.properties.get('ID_VENDOR', 'Unknown')
                    })
            
            return {
                "success": True,
                "devices": devices,
                "count": len(devices)
            }
        
        except Exception as e:
            logger.error(f"Error monitoring peripherals: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_resources(self) -> Dict[str, Any]:
        """
        Identify resource hogs and generate AI recommendations.
        
        Returns:
            Optimization recommendations
        """
        try:
            import psutil
            
            # Get top processes by memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    info = proc.info
                    if info['memory_percent'] > 1.0:  # More than 1% RAM
                        processes.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by memory
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            top_processes = processes[:10]
            
            return {
                "success": True,
                "top_processes": top_processes,
                "total_processes": len(processes)
            }
        
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
            return {"success": False, "error": str(e)}
    
    def check_vital_signs(self) -> Dict[str, Any]:
        """Check all system vital signs (original method)."""
        vitals = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self._get_gpu_stats(),
            "cpu": self._get_cpu_stats(),
            "memory": self._get_memory_stats(),
            "disk": self._get_disk_stats()
        }
        
        return vitals
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get simplified system health for trading decisions.
        
        Returns flattened format with gpu_temp, cpu_usage, ram_usage.
        Used by FinanceMinister for hardware safety checks.
        """
        vitals = self.check_vital_signs()
        
        # Extract and flatten key metrics
        gpu = vitals.get("gpu", {})
        cpu = vitals.get("cpu", {})
        memory = vitals.get("memory", {})
        
        return {
            "gpu_temp": gpu.get("temperature_c", 0),
            "gpu_status": gpu.get("status", "unknown"),
            "cpu_usage": cpu.get("percent", 0),
            "ram_usage": memory.get("percent", 0),
            "timestamp": vitals.get("timestamp")
        }
    
    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        if not self.gpu_available:
            return {"status": "unavailable"}
        
        try:
            import pynvml
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            
            return {
                "name": name.decode() if isinstance(name, bytes) else name,
                "temperature_c": temp,
                "vram_used_mb": mem_info.used / (1024 ** 2),
                "vram_total_mb": mem_info.total / (1024 ** 2),
                "vram_percent": (mem_info.used / mem_info.total) * 100,
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "status": "healthy" if temp < 80 else "warning" if temp < 90 else "critical"
            }
        
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics."""
        try:
            import psutil
            
            return {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get RAM statistics."""
        try:
            import psutil
            
            mem = psutil.virtual_memory()
            
            return {
                "total_gb": mem.total / (1024 ** 3),
                "used_gb": mem.used / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "percent": mem.percent
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_disk_stats(self) -> Dict[str, Any]:
        """Get disk statistics."""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            
            return {
                "total_gb": disk.total / (1024 ** 3),
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
                "percent": disk.percent
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute health monitoring task with NEW resource optimization."""
        self.tasks_processed += 1
        
        try:
            # Parse task
            task_lower = task.lower()
            
            if "peripheral" in task_lower or "device" in task_lower:
                # Monitor peripherals
                result = self.monitor_peripherals()
                
                if result.get("success"):
                    devices = result.get("devices", [])
                    device_list = "\n".join([f"- {d['name']} ({d['type']})" for d in devices])
                    
                    analysis = f"📱 **Detected {len(devices)} Peripheral Devices:**\n{device_list}"
                    
                    self.tasks_successful += 1
                    return {
                        "success": True,
                        "response": analysis,
                        "minister": self.name,
                        "domain": "health",
                        "metadata": result
                    }
            
            elif "optimize" in task_lower or "resource" in task_lower:
                # Resource optimization
                vitals = self.check_vital_signs()
                resource_result = self.optimize_resources()
                
                if resource_result.get("success"):
                    top_procs = resource_result.get("top_processes", [])
                    
                    optimization_prompt = f"""System Resource Analysis:

GPU: {vitals['gpu'].get('temperature_c', 'N/A')}°C | VRAM: {vitals['gpu'].get('vram_percent', 0):.1f}%
CPU: {vitals['cpu'].get('percent', 'N/A')}%
RAM: {vitals['memory'].get('percent', 'N/A')}%

Top Resource Consumers:
{chr(10).join([f"- {p['name']}: {p['memory_percent']:.1f}% RAM, {p['cpu_percent']:.1f}% CPU" for p in top_procs[:5]])}

**Make aggressive recommendations:**
1. Which processes should be killed to free resources?
2. Is the system healthy or overloaded?
3. What optimizations are needed?"""
                    
                    analysis = await self._think_with_prompt(
                        system_prompt=self.system_prompt,
                        user_message=optimization_prompt,
                        max_tokens=500
                    )
                    
                    self.tasks_successful += 1
                    return {
                        "success": True,
                        "response": analysis,
                        "minister": self.name,
                        "domain": "health",
                        "metadata": {
                            "vitals": vitals,
                            "top_processes": top_procs
                        }
                    }
            
            # Default: Check vital signs
            vitals = self.check_vital_signs()
            
            gpu_stats = vitals.get("gpu", {})
            cpu_stats = vitals.get("cpu", {})
            mem_stats = vitals.get("memory", {})
            
            health_report = f"""System Health Report:

GPU ({gpu_stats.get('name', 'N/A')}):
- Temperature: {gpu_stats.get('temperature_c', 'N/A')}°C
- VRAM: {gpu_stats.get('vram_used_mb', 0):.0f}/{gpu_stats.get('vram_total_mb', 0):.0f} MB ({gpu_stats.get('vram_percent', 0):.1f}%)
- Utilization: {gpu_stats.get('gpu_utilization', 'N/A')}%

CPU: {cpu_stats.get('percent', 'N/A')}%
RAM: {mem_stats.get('used_gb', 0):.1f}/{mem_stats.get('total_gb', 0):.1f} GB ({mem_stats.get('percent', 0)}%)

Provide health assessment and recommendations."""
            
            analysis = await self._think_with_prompt(
                system_prompt=self.system_prompt,
                user_message=health_report,
                max_tokens=400
            )
            
            self.tasks_successful += 1
            
            return {
                "success": True,
                "response": analysis,
                "minister": self.name,
                "domain": "health",
                "metadata": {"vitals": vitals}
            }
        
        except Exception as e:
            logger.error(f"Health Minister error: {e}")
            return {
                "success": False,
                "response": f"Health check failed: {str(e)}",
                "minister": self.name,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["HealthMinister"]
