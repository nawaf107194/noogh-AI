#!/usr/bin/env python3
"""
👁️ Visual Pattern Analyzer
محلل الأنماط البصرية

Computer vision-based pattern analysis
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not available - visual analysis disabled")

from .pattern_detector import DetectedPattern, PatternType
from .chart_image_generator import ChartImageGenerator


@dataclass
class VisualFeatures:
    """Visual features extracted from chart image"""

    edges_density: float           # Density of detected edges
    corner_points: int             # Number of corner points
    line_segments: List[Tuple]     # Detected line segments
    symmetry_score: float          # Horizontal symmetry
    trend_angle: float             # Overall trend angle
    volatility_visual: float       # Visual volatility measure
    pattern_complexity: float      # Complexity score


class VisualPatternAnalyzer:
    """
    Analyze chart patterns using computer vision techniques

    Uses OpenCV for:
    - Edge detection
    - Line detection (Hough Transform)
    - Corner detection
    - Template matching
    - Shape analysis
    """

    def __init__(self):
        """Initialize visual pattern analyzer"""
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python is required for visual analysis")

        self.chart_generator = ChartImageGenerator(
            image_size=(800, 600),
            style='dark'
        )

    def analyze_chart_image(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> VisualFeatures:
        """
        Analyze chart using computer vision

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            VisualFeatures extracted from chart
        """
        logger.info(f"👁️ Analyzing chart image for {symbol}...")

        # Generate chart image
        img = self.chart_generator.generate_line_chart(df, symbol)

        if img is None:
            logger.warning("   ⚠️ Failed to generate chart image")
            return self._empty_features()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Extract features
        edges_density = self._detect_edges(gray)
        corner_points = self._detect_corners(gray)
        line_segments = self._detect_lines(gray)
        symmetry = self._calculate_symmetry(gray)
        trend_angle = self._calculate_trend_angle(line_segments)
        volatility = self._calculate_visual_volatility(gray)
        complexity = self._calculate_complexity(edges_density, len(line_segments))

        features = VisualFeatures(
            edges_density=edges_density,
            corner_points=corner_points,
            line_segments=line_segments,
            symmetry_score=symmetry,
            trend_angle=trend_angle,
            volatility_visual=volatility,
            pattern_complexity=complexity
        )

        logger.info(f"   ✅ Visual features extracted:")
        logger.info(f"      Edges density: {edges_density:.2f}")
        logger.info(f"      Corners: {corner_points}")
        logger.info(f"      Lines: {len(line_segments)}")
        logger.info(f"      Trend angle: {trend_angle:.1f}°")
        logger.info(f"      Complexity: {complexity:.2f}")

        return features

    def _detect_edges(self, gray_image: np.ndarray) -> float:
        """Detect edges using Canny edge detector"""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        density = edge_pixels / total_pixels
        return density

    def _detect_corners(self, gray_image: np.ndarray) -> int:
        """Detect corner points using Harris corner detector"""
        corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
        corner_threshold = 0.01 * corners.max()
        corner_points = np.count_nonzero(corners > corner_threshold)
        return int(corner_points)

    def _detect_lines(self, gray_image: np.ndarray) -> List[Tuple]:
        """Detect line segments using Hough Transform"""
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_segments.append((x1, y1, x2, y2))

        return line_segments

    def _calculate_symmetry(self, gray_image: np.ndarray) -> float:
        """Calculate horizontal symmetry score"""
        height, width = gray_image.shape
        mid = width // 2

        left_half = gray_image[:, :mid]
        right_half = gray_image[:, mid:mid + left_half.shape[1]]
        right_half_flipped = np.fliplr(right_half)

        # Calculate similarity
        difference = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
        symmetry = 1.0 - (np.mean(difference) / 255.0)

        return max(0.0, min(1.0, symmetry))

    def _calculate_trend_angle(self, line_segments: List[Tuple]) -> float:
        """Calculate average trend angle from line segments"""
        if not line_segments:
            return 0.0

        angles = []
        for x1, y1, x2, y2 in line_segments:
            if x2 - x1 != 0:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        if angles:
            return np.mean(angles)
        return 0.0

    def _calculate_visual_volatility(self, gray_image: np.ndarray) -> float:
        """Calculate visual volatility from image gradients"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude of gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        volatility = np.std(magnitude) / 255.0

        return volatility

    def _calculate_complexity(self, edges_density: float, num_lines: int) -> float:
        """Calculate pattern complexity score"""
        # Combine edges and lines
        complexity = (edges_density * 0.6) + (min(num_lines / 100, 1.0) * 0.4)
        return min(1.0, complexity)

    def _empty_features(self) -> VisualFeatures:
        """Return empty features"""
        return VisualFeatures(
            edges_density=0.0,
            corner_points=0,
            line_segments=[],
            symmetry_score=0.0,
            trend_angle=0.0,
            volatility_visual=0.0,
            pattern_complexity=0.0
        )

    def enhance_pattern_detection(
        self,
        patterns: List[DetectedPattern],
        visual_features: VisualFeatures
    ) -> List[DetectedPattern]:
        """
        Enhance pattern detection confidence using visual features

        Args:
            patterns: Algorithmically detected patterns
            visual_features: Visual features from chart

        Returns:
            Enhanced patterns with adjusted confidence
        """
        logger.info(f"🔍 Enhancing {len(patterns)} patterns with visual analysis...")

        enhanced = []

        for pattern in patterns:
            # Adjust confidence based on visual features
            confidence_adjustment = 0.0

            # Head & Shoulders: High symmetry increases confidence
            if pattern.pattern_type in [PatternType.HEAD_AND_SHOULDERS,
                                       PatternType.INVERSE_HEAD_AND_SHOULDERS]:
                confidence_adjustment += visual_features.symmetry_score * 0.1

            # Triangles: Line detection confirms pattern
            if 'TRIANGLE' in pattern.pattern_type.value:
                line_score = min(len(visual_features.line_segments) / 50, 1.0)
                confidence_adjustment += line_score * 0.1

            # Channels: Trend angle consistency
            if 'CHANNEL' in pattern.pattern_type.value:
                if pattern.direction == 'BULLISH' and visual_features.trend_angle > 0:
                    confidence_adjustment += 0.1
                elif pattern.direction == 'BEARISH' and visual_features.trend_angle < 0:
                    confidence_adjustment += 0.1

            # Apply adjustment
            new_confidence = min(0.99, pattern.confidence + confidence_adjustment)

            # Create enhanced pattern
            enhanced_pattern = DetectedPattern(
                pattern_type=pattern.pattern_type,
                confidence=new_confidence,
                start_idx=pattern.start_idx,
                end_idx=pattern.end_idx,
                key_points=pattern.key_points,
                direction=pattern.direction,
                target_price=pattern.target_price,
                stop_loss=pattern.stop_loss,
                description=f"{pattern.description} (CV-enhanced: +{confidence_adjustment:.1%})",
                strength=pattern.strength
            )

            enhanced.append(enhanced_pattern)

        logger.info(f"   ✅ Patterns enhanced with visual features")

        return enhanced


# TODO: Add more CV techniques
# - Template matching for known patterns
# - Contour analysis
# - Fourier descriptors
# - Deep learning-based pattern recognition
