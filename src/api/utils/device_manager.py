# -*- coding: utf-8 -*-
"""
Device Manager - إدارة الأجهزة
GPU/CPU device management for Noogh Unified AI System

This module provides intelligent device management with automatic detection,
resource monitoring, and optimal configuration for different hardware setups.
"""

import torch
import logging
import psutil
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Intelligent Device Manager
    إدارة ذكية للأجهزة مع كشف تلقائي ومراقبة موارد
    """

    def __init__(self):
        self.device = self._detect_optimal_device()
        self.device_info = self._get_device_info()
        self.capabilities = self._assess_capabilities()

    def _detect_optimal_device(self) -> torch.device:
        """
        Detect the optimal device based on availability and performance
        كشف الجهاز الأمثل بناءً على التوفر والأداء
        """
        # Priority: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🎮 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return device

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Apple Silicon MPS detected")
            return device

        else:
            device = torch.device("cpu")
            logger.info("💻 CPU detected (no GPU available)")
            return device

    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            "type": self.device.type,
            "name": "Unknown",
            "memory_total_gb": 0,
            "memory_available_gb": 0,
            "compute_capability": None,
            "driver_version": None,
            "cuda_version": None
        }

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            info.update({
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": props.total_memory / (1024**3),
                "memory_available_gb": torch.cuda.mem_get_info()[0] / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
            
            # Handle max_threads_per_block (may not exist in newer PyTorch versions)
            try:
                info["max_threads_per_block"] = props.max_threads_per_block
            except AttributeError:
                # Fallback for newer PyTorch versions
                info["max_threads_per_block"] = getattr(props, "max_threads_per_block", 1024)

        elif self.device.type == "cpu":
            # Get system memory info
            memory = psutil.virtual_memory()
            info.update({
                "name": "CPU",
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True)
            })

        return info

    def _assess_capabilities(self) -> Dict[str, bool]:
        """Assess device capabilities"""
        return {
            "cuda": torch.cuda.is_available(),
            "mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Ampere+
            "tf32": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Ampere+
            "float16": torch.cuda.is_available(),
            "distributed": torch.cuda.device_count() > 1,
            "memory_efficient_attention": torch.cuda.is_available()
        }

    def get_device(self) -> torch.device:
        """Get the current device"""
        return self.device

    def get_optimal_batch_size(self, model_size_mb: float = 100) -> int:
        """
        Calculate optimal batch size based on device memory
        حساب حجم الدفعة الأمثل بناءً على ذاكرة الجهاز
        """
        if self.device.type == "cuda":
            available_memory_gb = self.device_info["memory_available_gb"]

            # Reserve 20% for overhead
            usable_memory_gb = available_memory_gb * 0.8

            # Estimate batch size (rough calculation)
            # Assuming model uses ~2x its size in memory during training
            memory_per_sample_mb = model_size_mb * 2

            optimal_batch = int((usable_memory_gb * 1024) / memory_per_sample_mb)

            # Clamp between reasonable bounds
            optimal_batch = max(1, min(optimal_batch, 128))

            return optimal_batch

        else:
            # CPU: smaller batches
            return 8

    def get_recommended_workers(self) -> int:
        """Get recommended number of data loading workers"""
        if self.device.type == "cuda":
            return 4  # GPU benefits from parallel loading
        else:
            return 2  # CPU: fewer workers to avoid overhead

    def print_info(self):
        """Print comprehensive device information"""
        logger.info("=" * 60)
        logger.info("🖥️  Device Information")
        logger.info("=" * 60)

        logger.info(f"Device Type: {self.device_info['type'].upper()}")
        logger.info(f"Device Name: {self.device_info['name']}")

        if self.device_info['memory_total_gb'] > 0:
            logger.info(f"Memory Total: {self.device_info['memory_total_gb']:.2f} GB")
            logger.info(f"Memory Available: {self.device_info['memory_available_gb']:.2f} GB")

        if self.device.type == "cuda":
            logger.info(f"Compute Capability: {self.device_info.get('compute_capability', 'Unknown')}")
            logger.info(f"Multi-Processors: {self.device_info.get('multi_processor_count', 'Unknown')}")

        # Capabilities
        logger.info("\nCapabilities:")
        for cap, available in self.capabilities.items():
            status = "✅" if available else "❌"
            logger.info(f"  {cap}: {status}")

        logger.info("=" * 60)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization_percent": (allocated / self.device_info["memory_total_gb"]) * 100
            }
        else:
            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "utilization_percent": memory.percent
            }

    def optimize_for_inference(self):
        """Apply optimizations for inference"""
        if self.device.type == "cuda":
            # Enable TF32 for faster inference on Ampere+
            if self.capabilities.get("tf32", False):
                torch.set_float32_matmul_precision("high")

            # Disable gradient computation globally
            torch.set_grad_enabled(False)

            logger.info("🚀 Inference optimizations applied")

    def optimize_for_training(self):
        """Apply optimizations for training"""
        if self.device.type == "cuda":
            # Enable TF32 for faster training
            if self.capabilities.get("tf32", False):
                torch.set_float32_matmul_precision("high")

            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            logger.info("🏋️  Training optimizations applied")

    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleaned")

    @property
    def device_info_summary(self) -> Dict[str, Any]:
        """Get a summary of device information"""
        return {
            "device": self.device_info,
            "capabilities": self.capabilities,
            "memory_stats": self.get_memory_stats()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DI Container Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_device_manager() -> DeviceManager:
    """
    Factory function to get device manager instance
    دالة المصنع للحصول على مثيل إدارة الأجهزة
    
    Note: Creates a new instance each time (not singleton)
    """
    return DeviceManager()


# Global instance for convenience
_device_manager = None

def get_global_device_manager() -> DeviceManager:
    """
    Get global device manager instance from DI container
    
    Returns:
        DeviceManager instance (singleton)
    """
    try:
        from src.core.di import Container
        device = Container.resolve("device_manager")
        if device is not None:
            return device
    except ImportError:
        pass
    
    # Fallback to manual singleton for backward compatibility
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


if __name__ == "__main__":
    # Test the device manager
    manager = get_device_manager()
    manager.print_info()

    print("\nMemory Stats:")
    print(manager.get_memory_stats())

    print(f"\nOptimal batch size: {manager.get_optimal_batch_size()}")
    print(f"Recommended workers: {manager.get_recommended_workers()}")
