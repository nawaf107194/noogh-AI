#!/usr/bin/env python3
"""
🎯 Phase 6: Trend Alignment Filter
مرشح اتجاه الترند
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd

from .config import TIMEFRAME_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Trend analysis for a single timeframe"""

    timeframe: str           # '1h', '4h', '1d'
    trend_direction: str     # 'UPTREND', 'DOWNTREND', 'SIDEWAYS'
    trend_strength: float    # 0.0 - 1.0
    ma_alignment: bool       # Are moving averages aligned?
    price_position: str      # 'ABOVE_MA', 'BELOW_MA', 'AT_MA'
    reasoning: str           # Trend reasoning


@dataclass
class MultiTimeframeTrend:
    """Combined trend analysis across timeframes"""

    timeframe_trends: Dict[str, TrendAnalysis]
    overall_trend: str                    # 'UPTREND', 'DOWNTREND', 'SIDEWAYS', 'CONFLICTED'
    trend_alignment: str                  # 'ALIGNED', 'PARTIAL', 'CONFLICTED'
    should_filter: bool                   # Should this signal be filtered out?
    reasoning: str                        # Filter reasoning


class TrendAligner:
    """
    Ensure signals align with multi-timeframe trend

    Filter rules:
    - LONG signals: Only in uptrend or sideways with bullish momentum
    - SHORT signals: Only in downtrend or sideways with bearish momentum
    - CONFLICTED trend: No trades
    """

    def __init__(self):
        """Initialize trend aligner"""
        self.configs = TIMEFRAME_CONFIGS

    def analyze_trend(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> MultiTimeframeTrend:
        """
        Analyze trend across all timeframes

        Args:
            symbol: Trading symbol
            timeframe_data: Dict mapping timeframe to OHLCV DataFrame

        Returns:
            MultiTimeframeTrend with combined analysis
        """
        logger.info(f"🎯 Analyzing multi-timeframe trend for {symbol}...")

        # Analyze each timeframe
        timeframe_trends = {}

        for tf, df in timeframe_data.items():
            if df.empty:
                logger.warning(f"⚠️ Empty data for {tf}, skipping")
                continue

            trend = self._analyze_single_timeframe(tf, df)
            timeframe_trends[tf] = trend

            logger.info(f"   {tf}: {trend.trend_direction} (strength: {trend.trend_strength:.1%})")

        # Determine overall trend
        overall_trend = self._determine_overall_trend(timeframe_trends)

        # Check alignment
        trend_alignment = self._check_trend_alignment(timeframe_trends)

        # Should filter?
        should_filter = (
            trend_alignment == 'CONFLICTED' or
            overall_trend == 'CONFLICTED'
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            timeframe_trends,
            overall_trend,
            trend_alignment
        )

        multi_trend = MultiTimeframeTrend(
            timeframe_trends=timeframe_trends,
            overall_trend=overall_trend,
            trend_alignment=trend_alignment,
            should_filter=should_filter,
            reasoning=reasoning
        )

        logger.info(f"   ✅ Overall trend: {overall_trend}")
        logger.info(f"      Alignment: {trend_alignment}")
        logger.info(f"      Filter: {should_filter}")

        return multi_trend

    def _analyze_single_timeframe(
        self,
        timeframe: str,
        df: pd.DataFrame
    ) -> TrendAnalysis:
        """
        Analyze trend for a single timeframe

        Uses simple moving averages:
        - MA20, MA50, MA200
        """
        # Calculate moving averages
        close = df['close']

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean() if len(df) >= 200 else None

        # Current price
        current_price = close.iloc[-1]

        # Check MA alignment
        ma_aligned = False
        if ma200 is not None:
            ma_aligned = (
                ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1] or
                ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]
            )

        # Determine trend direction
        if ma20.iloc[-1] > ma50.iloc[-1]:
            if ma200 is not None and ma50.iloc[-1] > ma200.iloc[-1]:
                trend_direction = 'UPTREND'
                trend_strength = 0.8
            else:
                trend_direction = 'UPTREND'
                trend_strength = 0.6
        elif ma20.iloc[-1] < ma50.iloc[-1]:
            if ma200 is not None and ma50.iloc[-1] < ma200.iloc[-1]:
                trend_direction = 'DOWNTREND'
                trend_strength = 0.8
            else:
                trend_direction = 'DOWNTREND'
                trend_strength = 0.6
        else:
            trend_direction = 'SIDEWAYS'
            trend_strength = 0.4

        # Price position
        if current_price > ma20.iloc[-1]:
            price_position = 'ABOVE_MA'
        elif current_price < ma20.iloc[-1]:
            price_position = 'BELOW_MA'
        else:
            price_position = 'AT_MA'

        reasoning = f"{trend_direction} ({price_position}, MA aligned: {ma_aligned})"

        return TrendAnalysis(
            timeframe=timeframe,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            ma_alignment=ma_aligned,
            price_position=price_position,
            reasoning=reasoning
        )

    def _determine_overall_trend(
        self,
        timeframe_trends: Dict[str, TrendAnalysis]
    ) -> str:
        """
        Determine overall trend from all timeframes

        Priority: 1d > 4h > 1h
        """
        if not timeframe_trends:
            return 'SIDEWAYS'

        # Count trend directions
        trend_counts = {}
        for trend in timeframe_trends.values():
            direction = trend.trend_direction
            trend_counts[direction] = trend_counts.get(direction, 0) + 1

        # Check for clear trend
        if trend_counts.get('UPTREND', 0) >= 2:
            return 'UPTREND'
        elif trend_counts.get('DOWNTREND', 0) >= 2:
            return 'DOWNTREND'
        elif trend_counts.get('SIDEWAYS', 0) >= 2:
            return 'SIDEWAYS'
        else:
            return 'CONFLICTED'

    def _check_trend_alignment(
        self,
        timeframe_trends: Dict[str, TrendAnalysis]
    ) -> str:
        """
        Check if timeframe trends are aligned

        Returns:
            'ALIGNED': All trends agree
            'PARTIAL': Majority agrees
            'CONFLICTED': No agreement
        """
        if not timeframe_trends:
            return 'CONFLICTED'

        directions = [t.trend_direction for t in timeframe_trends.values()]
        unique_directions = set(directions)

        if len(unique_directions) == 1:
            return 'ALIGNED'
        elif len(unique_directions) == 2:
            return 'PARTIAL'
        else:
            return 'CONFLICTED'

    def _build_reasoning(
        self,
        timeframe_trends: Dict[str, TrendAnalysis],
        overall_trend: str,
        trend_alignment: str
    ) -> str:
        """Build human-readable reasoning"""

        lines = []

        # Individual trends
        for tf, trend in timeframe_trends.items():
            lines.append(f"{tf}: {trend.trend_direction}")

        # Overall
        lines.append(f"Overall: {overall_trend} ({trend_alignment})")

        return " | ".join(lines)

    def should_allow_trade(
        self,
        signal_direction: str,
        multi_trend: MultiTimeframeTrend
    ) -> bool:
        """
        Check if a signal should be allowed based on trend

        Args:
            signal_direction: 'BUY' or 'SELL'
            multi_trend: Multi-timeframe trend analysis

        Returns:
            True if trade is allowed
        """
        if multi_trend.trend_alignment == 'CONFLICTED':
            return False

        overall = multi_trend.overall_trend

        if signal_direction == 'BUY':
            return overall in ['UPTREND', 'SIDEWAYS']
        elif signal_direction == 'SELL':
            return overall in ['DOWNTREND', 'SIDEWAYS']
        else:
            return False


# TODO: Implement in Phase 6 Week 2
# - Add ADX for trend strength
# - Add volume confirmation
# - Add custom trend indicators
# - Add trend reversal detection
