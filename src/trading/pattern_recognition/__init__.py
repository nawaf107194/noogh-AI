"""
👁️ Computer Vision Pattern Recognition Module
نظام الرؤية الحاسوبية لتحليل الأنماط

Advanced pattern recognition using computer vision:
- Chart image generation
- Classical pattern detection (H&S, Triangles, Flags, etc.)
- Visual pattern matching
- 1-year historical analysis
"""

from .chart_image_generator import ChartImageGenerator
from .pattern_detector import (
    PatternDetector,
    PatternType,
    DetectedPattern,
    PatternSignal
)
from .visual_analyzer import VisualPatternAnalyzer
from .pattern_database import PatternDatabase

__all__ = [
    'ChartImageGenerator',
    'PatternDetector',
    'PatternType',
    'DetectedPattern',
    'PatternSignal',
    'VisualPatternAnalyzer',
    'PatternDatabase'
]
