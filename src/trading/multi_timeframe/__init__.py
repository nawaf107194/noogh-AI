"""
🌊 Phase 6: Multi-Timeframe Analysis Module
تحليل متعدد الأطر الزمنية

Integrates multiple timeframes for better signal quality:
- 1h: Immediate action (50% weight)
- 4h: Short-term trend (30% weight)
- 1d: Long-term context (20% weight)
"""

from .config import TimeframeConfig, TIMEFRAME_CONFIGS
from .collector import MultiTimeframeCollector
from .fusion import SignalFusion
from .trend_filter import TrendAligner

__all__ = [
    'TimeframeConfig',
    'TIMEFRAME_CONFIGS',
    'MultiTimeframeCollector',
    'SignalFusion',
    'TrendAligner'
]
