#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitor
"""

import sys
from pathlib import Path
import time

# Add parent to path
PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.requests = 0
        self.successful_requests = 0

    def get_performance_summary(self):
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_requests': self.requests,
            'successful_requests': self.successful_requests,
            'average_response_time': 0.0
        }

    def cleanup(self):
        pass
