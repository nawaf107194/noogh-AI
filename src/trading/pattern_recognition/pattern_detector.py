#!/usr/bin/env python3
"""
🔍 Pattern Detector
كاشف الأنماط التقنية

Detects classical chart patterns using algorithmic analysis
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Chart pattern types"""

    # Reversal patterns
    HEAD_AND_SHOULDERS = "Head and Shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "Inverse Head and Shoulders"
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    TRIPLE_TOP = "Triple Top"
    TRIPLE_BOTTOM = "Triple Bottom"

    # Continuation patterns
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    FLAG = "Flag"
    PENNANT = "Pennant"
    WEDGE_RISING = "Rising Wedge"
    WEDGE_FALLING = "Falling Wedge"

    # Trend patterns
    UPTREND_CHANNEL = "Uptrend Channel"
    DOWNTREND_CHANNEL = "Downtrend Channel"
    HORIZONTAL_CHANNEL = "Horizontal Channel"

    # Other
    ROUNDING_BOTTOM = "Rounding Bottom"
    ROUNDING_TOP = "Rounding Top"


@dataclass
class DetectedPattern:
    """Detected chart pattern"""

    pattern_type: PatternType
    confidence: float              # 0.0 - 1.0
    start_idx: int                 # Start candle index
    end_idx: int                   # End candle index
    key_points: List[Tuple[int, float]]  # Key price levels
    direction: str                 # 'BULLISH', 'BEARISH', 'NEUTRAL'
    target_price: Optional[float]  # Projected target
    stop_loss: Optional[float]     # Suggested stop-loss
    description: str               # Pattern description
    strength: float                # Pattern strength (0.0 - 1.0)


@dataclass
class PatternSignal:
    """Trading signal from pattern"""

    pattern: DetectedPattern
    action: str                    # 'BUY', 'SELL', 'HOLD'
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    confidence: float
    reasoning: str


class PatternDetector:
    """
    Detect classical chart patterns using algorithmic analysis

    Implements pattern detection for:
    - Reversal patterns (H&S, Double Top/Bottom, etc.)
    - Continuation patterns (Triangles, Flags, etc.)
    - Trend channels
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        lookback_period: int = 100
    ):
        """
        Initialize pattern detector

        Args:
            min_confidence: Minimum confidence threshold
            lookback_period: Number of candles to analyze
        """
        self.min_confidence = min_confidence
        self.lookback_period = lookback_period

    def detect_patterns(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[DetectedPattern]:
        """
        Detect all patterns in the given data

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            List of detected patterns
        """
        logger.info(f"🔍 Detecting patterns in {symbol} ({len(df)} candles)...")

        patterns = []

        # Use recent data
        df_recent = df.tail(self.lookback_period).copy()

        if len(df_recent) < 50:
            logger.warning("   ⚠️ Insufficient data for pattern detection")
            return patterns

        # Detect different pattern types
        patterns.extend(self._detect_head_and_shoulders(df_recent))
        patterns.extend(self._detect_double_tops_bottoms(df_recent))
        patterns.extend(self._detect_triangles(df_recent))
        patterns.extend(self._detect_channels(df_recent))
        patterns.extend(self._detect_flags_pennants(df_recent))

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]

        logger.info(f"   ✅ Found {len(patterns)} patterns (min confidence: {self.min_confidence:.0%})")

        return patterns

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect Head & Shoulders patterns"""
        patterns = []

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Find peaks and troughs
        peaks, _ = find_peaks(highs, distance=10)
        troughs, _ = find_peaks(-lows, distance=10)

        if len(peaks) < 3 or len(troughs) < 2:
            return patterns

        # Check for H&S pattern (bearish)
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]

            # Check if middle is highest (head)
            if highs[head] > highs[left_shoulder] and highs[head] > highs[right_shoulder]:
                # Check if shoulders are roughly equal
                shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[head]

                if shoulder_diff < 0.05:  # 5% tolerance
                    # Find neckline (troughs between shoulders)
                    neckline_troughs = [t for t in troughs if left_shoulder < t < right_shoulder]

                    if len(neckline_troughs) >= 1:
                        neckline = np.mean([lows[t] for t in neckline_troughs])

                        # Calculate confidence
                        symmetry = 1.0 - shoulder_diff
                        head_prominence = (highs[head] - neckline) / highs[head]
                        confidence = min(0.95, symmetry * 0.5 + head_prominence * 0.5)

                        if confidence >= 0.60:
                            # Calculate target
                            height = highs[head] - neckline
                            target = neckline - height

                            pattern = DetectedPattern(
                                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                                confidence=confidence,
                                start_idx=left_shoulder,
                                end_idx=right_shoulder,
                                key_points=[
                                    (left_shoulder, highs[left_shoulder]),
                                    (head, highs[head]),
                                    (right_shoulder, highs[right_shoulder])
                                ],
                                direction='BEARISH',
                                target_price=target,
                                stop_loss=highs[head] * 1.02,
                                description=f"Head & Shoulders pattern detected with {confidence:.1%} confidence",
                                strength=head_prominence
                            )
                            patterns.append(pattern)

        # Check for inverse H&S (bullish)
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]

            if lows[head] < lows[left_shoulder] and lows[head] < lows[right_shoulder]:
                shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder]) / lows[head]

                if shoulder_diff < 0.05:
                    neckline_peaks = [p for p in peaks if left_shoulder < p < right_shoulder]

                    if len(neckline_peaks) >= 1:
                        neckline = np.mean([highs[p] for p in neckline_peaks])

                        symmetry = 1.0 - shoulder_diff
                        head_prominence = (neckline - lows[head]) / lows[head]
                        confidence = min(0.95, symmetry * 0.5 + head_prominence * 0.5)

                        if confidence >= 0.60:
                            height = neckline - lows[head]
                            target = neckline + height

                            pattern = DetectedPattern(
                                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                                confidence=confidence,
                                start_idx=left_shoulder,
                                end_idx=right_shoulder,
                                key_points=[
                                    (left_shoulder, lows[left_shoulder]),
                                    (head, lows[head]),
                                    (right_shoulder, lows[right_shoulder])
                                ],
                                direction='BULLISH',
                                target_price=target,
                                stop_loss=lows[head] * 0.98,
                                description=f"Inverse Head & Shoulders with {confidence:.1%} confidence",
                                strength=head_prominence
                            )
                            patterns.append(pattern)

        return patterns

    def _detect_double_tops_bottoms(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect Double Top/Bottom patterns"""
        patterns = []

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        peaks, _ = find_peaks(highs, distance=15, prominence=np.std(highs) * 0.5)
        troughs, _ = find_peaks(-lows, distance=15, prominence=np.std(lows) * 0.5)

        # Double Top (bearish)
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]

            price_diff = abs(highs[peak1] - highs[peak2]) / highs[peak1]

            if price_diff < 0.02:  # Peaks within 2%
                trough_between = [t for t in troughs if peak1 < t < peak2]

                if len(trough_between) > 0:
                    support = min([lows[t] for t in trough_between])

                    confidence = 1.0 - price_diff
                    height = highs[peak1] - support
                    target = support - height

                    pattern = DetectedPattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=confidence,
                        start_idx=peak1,
                        end_idx=peak2,
                        key_points=[(peak1, highs[peak1]), (peak2, highs[peak2])],
                        direction='BEARISH',
                        target_price=target,
                        stop_loss=max(highs[peak1], highs[peak2]) * 1.02,
                        description=f"Double Top pattern ({confidence:.1%} confidence)",
                        strength=height / highs[peak1]
                    )
                    patterns.append(pattern)

        # Double Bottom (bullish)
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]

            price_diff = abs(lows[trough1] - lows[trough2]) / lows[trough1]

            if price_diff < 0.02:
                peak_between = [p for p in peaks if trough1 < p < trough2]

                if len(peak_between) > 0:
                    resistance = max([highs[p] for p in peak_between])

                    confidence = 1.0 - price_diff
                    height = resistance - lows[trough1]
                    target = resistance + height

                    pattern = DetectedPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=confidence,
                        start_idx=trough1,
                        end_idx=trough2,
                        key_points=[(trough1, lows[trough1]), (trough2, lows[trough2])],
                        direction='BULLISH',
                        target_price=target,
                        stop_loss=min(lows[trough1], lows[trough2]) * 0.98,
                        description=f"Double Bottom pattern ({confidence:.1%} confidence)",
                        strength=height / lows[trough1]
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_triangles(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect Triangle patterns"""
        patterns = []

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Use last 50 candles for triangle detection
        window = min(50, len(df))
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        x = np.arange(window)

        # Fit trendlines
        upper_slope, upper_intercept, upper_r, _, _ = linregress(x, recent_highs)
        lower_slope, lower_intercept, lower_r, _, _ = linregress(x, recent_lows)

        # Check for triangle patterns
        if abs(upper_r) > 0.7 and abs(lower_r) > 0.7:  # Strong correlation

            # Ascending Triangle (bullish)
            if abs(upper_slope) < 0.01 and lower_slope > 0.01:
                confidence = min(abs(upper_r), abs(lower_r))

                pattern = DetectedPattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    confidence=confidence,
                    start_idx=len(df) - window,
                    end_idx=len(df) - 1,
                    key_points=[],
                    direction='BULLISH',
                    target_price=recent_highs[-1] * 1.05,
                    stop_loss=recent_lows[-1] * 0.98,
                    description=f"Ascending Triangle ({confidence:.1%} confidence)",
                    strength=0.7
                )
                patterns.append(pattern)

            # Descending Triangle (bearish)
            elif abs(lower_slope) < 0.01 and upper_slope < -0.01:
                confidence = min(abs(upper_r), abs(lower_r))

                pattern = DetectedPattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    confidence=confidence,
                    start_idx=len(df) - window,
                    end_idx=len(df) - 1,
                    key_points=[],
                    direction='BEARISH',
                    target_price=recent_lows[-1] * 0.95,
                    stop_loss=recent_highs[-1] * 1.02,
                    description=f"Descending Triangle ({confidence:.1%} confidence)",
                    strength=0.7
                )
                patterns.append(pattern)

            # Symmetrical Triangle (continuation)
            elif upper_slope < -0.01 and lower_slope > 0.01:
                confidence = min(abs(upper_r), abs(lower_r))

                pattern = DetectedPattern(
                    pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    confidence=confidence,
                    start_idx=len(df) - window,
                    end_idx=len(df) - 1,
                    key_points=[],
                    direction='NEUTRAL',
                    target_price=None,
                    stop_loss=None,
                    description=f"Symmetrical Triangle ({confidence:.1%} confidence)",
                    strength=0.6
                )
                patterns.append(pattern)

        return patterns

    def _detect_channels(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect trend channels"""
        patterns = []

        closes = df['close'].values

        # Use last 60 candles
        window = min(60, len(df))
        recent_closes = closes[-window:]
        x = np.arange(window)

        # Fit trendline
        slope, intercept, r_value, _, _ = linregress(x, recent_closes)

        if abs(r_value) > 0.75:  # Strong trend

            # Uptrend Channel
            if slope > 0:
                pattern = DetectedPattern(
                    pattern_type=PatternType.UPTREND_CHANNEL,
                    confidence=abs(r_value),
                    start_idx=len(df) - window,
                    end_idx=len(df) - 1,
                    key_points=[],
                    direction='BULLISH',
                    target_price=None,
                    stop_loss=None,
                    description=f"Uptrend Channel (R²={r_value**2:.2f})",
                    strength=abs(slope)
                )
                patterns.append(pattern)

            # Downtrend Channel
            else:
                pattern = DetectedPattern(
                    pattern_type=PatternType.DOWNTREND_CHANNEL,
                    confidence=abs(r_value),
                    start_idx=len(df) - window,
                    end_idx=len(df) - 1,
                    key_points=[],
                    direction='BEARISH',
                    target_price=None,
                    stop_loss=None,
                    description=f"Downtrend Channel (R²={r_value**2:.2f})",
                    strength=abs(slope)
                )
                patterns.append(pattern)

        return patterns

    def _detect_flags_pennants(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect Flag and Pennant patterns"""
        patterns = []

        # Flags are short consolidation patterns after strong moves
        # Implementation placeholder - requires more sophisticated analysis

        return patterns

    def generate_signal(
        self,
        pattern: DetectedPattern,
        current_price: float
    ) -> Optional[PatternSignal]:
        """
        Generate trading signal from detected pattern

        Args:
            pattern: Detected pattern
            current_price: Current market price

        Returns:
            PatternSignal or None
        """
        if pattern.direction == 'BULLISH' and pattern.target_price:
            action = 'BUY'
            entry = current_price
            target = pattern.target_price
            stop = pattern.stop_loss or current_price * 0.95

        elif pattern.direction == 'BEARISH' and pattern.target_price:
            action = 'SELL'
            entry = current_price
            target = pattern.target_price
            stop = pattern.stop_loss or current_price * 1.05

        else:
            return None

        # Calculate risk/reward
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 1.5:  # Minimum R/R
            return None

        signal = PatternSignal(
            pattern=pattern,
            action=action,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            risk_reward_ratio=rr_ratio,
            confidence=pattern.confidence,
            reasoning=f"{pattern.pattern_type.value} detected with {pattern.confidence:.1%} confidence"
        )

        return signal


# TODO: Add more patterns
# - Cup and Handle
# - Rounding patterns
# - Gaps analysis
# - Fibonacci retracements
