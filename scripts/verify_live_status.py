#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 NOOGH LIVE STATUS VERIFICATION
==================================
Chief SRE: Production Readiness Check
"""

import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def check_logs():
    """Verify critical log files exist"""
    core_log = PROJECT_ROOT / "logs" / "core.log"
    
    if not core_log.exists():
        logger.error("❌ Core API log not found - System may not be running")
        return False
    
    # Check if log is being written to (has recent content)
    if core_log.stat().st_size == 0:
        logger.error("❌ Core API log is empty - System may have failed to start")
        return False
    
    logger.info("✅ Core API logs active")
    return True


def check_gpu():
    """Verify GPU is being utilized"""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            logger.warning("⚠️  No NVIDIA GPUs detected")
            return False
        
        # Check first GPU (RTX 5070)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        
        # Get utilization
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem_info.used / 1024**3
            mem_total_gb = mem_info.total / 1024**3
            
            logger.info(f"✅ GPU Active: {name.decode() if isinstance(name, bytes) else name}")
            logger.info(f"   • GPU Utilization: {gpu_util}%")
            logger.info(f"   • VRAM Usage: {mem_used_gb:.2f} GB / {mem_total_gb:.2f} GB")
            
            if mem_used_gb > 1.0:  # Brain loaded if using >1GB VRAM
                logger.info("   • 🧠 Neural Brain (Llama-3) likely loaded in VRAM")
            else:
                logger.warning("   • ⚠️  Low VRAM usage - Brain may not be loaded yet")
            
            pynvml.nvmlShutdown()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not read GPU metrics: {e}")
            pynvml.nvmlShutdown()
            return False
            
    except ImportError:
        logger.warning("⚠️  pynvml not installed - Cannot verify GPU status")
        logger.info("   Install with: pip install nvidia-ml-py")
        return False
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False


def check_api_health():
    """Verify API is responding"""
    try:
        import requests
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ API Health Check passed")
            logger.info(f"   • Status: {data.get('status', 'unknown')}")
            logger.info(f"   • Components: {len(data.get('components', {}))} active")
            return True
        else:
            logger.error(f"❌ API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to API on localhost:8000")
        logger.info("   System may still be starting up...")
        return False
    except ImportError:
        logger.warning("⚠️  requests module not installed - skipping API check")
        return False
    except Exception as e:
        logger.error(f"❌ API health check failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 70)
    print("🔍 NOOGH SOVEREIGN SYSTEM - LIVE STATUS VERIFICATION")
    print("=" * 70)
    print()
    
    checks = {
        "Logs": check_logs(),
        "GPU": check_gpu(),
        "API Health": check_api_health()
    }
    
    print()
    print("=" * 70)
    
    passed = sum(checks.values())
    total = len(checks)
    
    if passed == total:
        print("✅ SYSTEM READY FOR COMMAND")
        print("=" * 70)
        print()
        print("🎯 All systems operational - Live mode confirmed")
        print()
        print("Access Points:")
        print("   • Dashboard: http://localhost:8501")
        print("   • API Docs:  http://localhost:8000/docs")
        print("=" * 70)
        return 0
    else:
        print(f"⚠️  PARTIAL READINESS: {passed}/{total} checks passed")
        print("=" * 70)
        print()
        print("Some systems may be starting up or degraded.")
        print("Check logs: tail -f logs/core.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
