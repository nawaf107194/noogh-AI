#!/usr/bin/env python3
"""
Health Check Script - Monitors system health and triggers alerts
"""
import sys
import os
import requests
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configuration
API_HOST = os.getenv('API_HOST', 'localhost')
API_PORT = os.getenv('API_PORT', '8000')
MCP_PORT = os.getenv('MCP_PORT', '8001')
CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
MAX_FAILURES = int(os.getenv('MAX_FAILURES', '3'))
LOG_FILE = os.getenv('HEALTH_LOG', 'logs/healthcheck.log')

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self):
        self.failures = {'api': 0, 'mcp': 0}
        self.last_check = {}
        self.alerts_sent = set()

    def check_api_health(self):
        """Check FastAPI health endpoint"""
        try:
            response = requests.get(
                f'http://{API_HOST}:{API_PORT}/health',
                timeout=5
            )
            if response.status_code == 200:
                self.failures['api'] = 0
                logger.info(f"✅ API Health OK - {response.json()}")
                return True
            else:
                logger.warning(f"⚠️ API returned status {response.status_code}")
                self.failures['api'] += 1
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API Health Check Failed: {e}")
            self.failures['api'] += 1
            return False

    def check_mcp_health(self):
        """Check MCP server health"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', int(MCP_PORT)))
            sock.close()

            if result == 0:
                self.failures['mcp'] = 0
                logger.info(f"✅ MCP Health OK - Port {MCP_PORT} listening")
                return True
            else:
                logger.warning(f"⚠️ MCP port {MCP_PORT} not responding")
                self.failures['mcp'] += 1
                return False
        except Exception as e:
            logger.error(f"❌ MCP Health Check Failed: {e}")
            self.failures['mcp'] += 1
            return False

    def check_disk_space(self):
        """Check available disk space"""
        import shutil
        try:
            total, used, free = shutil.disk_usage("/")
            percent_used = (used / total) * 100

            if percent_used > 90:
                logger.error(f"❌ CRITICAL: Disk usage at {percent_used:.1f}%")
                return False
            elif percent_used > 80:
                logger.warning(f"⚠️ WARNING: Disk usage at {percent_used:.1f}%")
                return True
            else:
                logger.info(f"✅ Disk usage OK ({percent_used:.1f}%)")
                return True
        except Exception as e:
            logger.error(f"❌ Disk check failed: {e}")
            return False

    def check_memory(self):
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()

            if memory.percent > 90:
                logger.error(f"❌ CRITICAL: Memory usage at {memory.percent}%")
                return False
            elif memory.percent > 80:
                logger.warning(f"⚠️ WARNING: Memory usage at {memory.percent}%")
                return True
            else:
                logger.info(f"✅ Memory usage OK ({memory.percent}%)")
                return True
        except ImportError:
            logger.warning("⚠️ psutil not installed, skipping memory check")
            return True
        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")
            return False

    def trigger_alert(self, component, failure_count):
        """Trigger alert for component failure"""
        alert_key = f"{component}_{failure_count}"

        if alert_key in self.alerts_sent:
            return

        logger.error(f"🚨 ALERT: {component.upper()} has failed {failure_count} consecutive checks!")

        # Write alert to file
        alert_file = Path('logs/alerts.log')
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - ALERT: {component} failed {failure_count} times\n")

        self.alerts_sent.add(alert_key)

    def run_checks(self):
        """Run all health checks"""
        logger.info("="*60)
        logger.info("🔍 Starting health checks...")

        # Check API
        api_ok = self.check_api_health()
        if not api_ok and self.failures['api'] >= MAX_FAILURES:
            self.trigger_alert('api', self.failures['api'])

        # Check MCP
        mcp_ok = self.check_mcp_health()
        if not mcp_ok and self.failures['mcp'] >= MAX_FAILURES:
            self.trigger_alert('mcp', self.failures['mcp'])

        # Check resources
        disk_ok = self.check_disk_space()
        memory_ok = self.check_memory()

        # Overall status
        all_ok = api_ok and mcp_ok and disk_ok and memory_ok

        status = {
            'timestamp': datetime.now().isoformat(),
            'api': api_ok,
            'mcp': mcp_ok,
            'disk': disk_ok,
            'memory': memory_ok,
            'overall': all_ok
        }

        # Write status to file
        status_file = Path('logs/health_status.json')
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        logger.info(f"Health check complete - Overall: {'✅ HEALTHY' if all_ok else '❌ DEGRADED'}")
        logger.info("="*60)

        return all_ok

    def monitor(self):
        """Continuous monitoring loop"""
        logger.info("🚀 Starting health monitoring service...")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info(f"Max failures before alert: {MAX_FAILURES}")

        while True:
            try:
                self.run_checks()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("\n👋 Stopping health monitoring...")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(CHECK_INTERVAL)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Noogh System Health Checker')
    parser.add_argument('--once', action='store_true', help='Run checks once and exit')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode')

    args = parser.parse_args()

    checker = HealthChecker()

    if args.monitor:
        checker.monitor()
    else:
        # Single check (default)
        result = checker.run_checks()
        sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
