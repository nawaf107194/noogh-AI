#!/usr/bin/env python3
"""
📊 Phase 6: Multi-Timeframe Data Collector
جامع البيانات متعدد الأطر الزمنية
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .config import TIMEFRAME_CONFIGS, TimeframeConfig
from ..binance_data_collector import BinanceDataCollector

logger = logging.getLogger(__name__)


class MultiTimeframeCollector:
    """
    Collect market data across multiple timeframes

    Collects data for 1h, 4h, and 1d timeframes simultaneously
    for comprehensive market analysis.
    """

    def __init__(self, data_collector: Optional[BinanceDataCollector] = None):
        """
        Initialize multi-timeframe collector

        Args:
            data_collector: Optional BinanceDataCollector instance
        """
        self.data_collector = data_collector or BinanceDataCollector()
        self.configs = TIMEFRAME_CONFIGS

    async def collect_all_timeframes(
        self,
        symbol: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all configured timeframes

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')

        Returns:
            Dictionary mapping timeframe to DataFrame

        Example:
            {
                '1h': DataFrame with 168 rows (7 days),
                '4h': DataFrame with 180 rows (30 days),
                '1d': DataFrame with 90 rows (90 days)
            }
        """
        logger.info(f"📊 Collecting multi-timeframe data for {symbol}...")

        # Collect all timeframes in parallel
        tasks = []
        for tf, config in self.configs.items():
            task = self._collect_single_timeframe(symbol, config)
            tasks.append((tf, task))

        # Wait for all collections
        results = {}
        for tf, task in tasks:
            try:
                df = await task
                results[tf] = df
                logger.info(f"   ✅ {tf}: {len(df)} candles")
            except Exception as e:
                logger.error(f"   ❌ {tf}: {e}")
                results[tf] = pd.DataFrame()

        return results

    async def _collect_single_timeframe(
        self,
        symbol: str,
        config: TimeframeConfig
    ) -> pd.DataFrame:
        """
        Collect data for a single timeframe

        Args:
            symbol: Trading symbol
            config: Timeframe configuration

        Returns:
            DataFrame with OHLCV data
        """
        df = await self.data_collector.fetch_ohlcv(
            symbol=symbol,
            interval=config.interval,
            days=config.lookback_days
        )

        return df

    async def collect_batch_multi_timeframe(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect multi-timeframe data for multiple symbols

        Args:
            symbols: List of trading symbols

        Returns:
            Nested dictionary: {symbol: {timeframe: DataFrame}}

        Example:
            {
                'BTC/USDT': {
                    '1h': DataFrame,
                    '4h': DataFrame,
                    '1d': DataFrame
                },
                'ETH/USDT': {...}
            }
        """
        logger.info(f"📊 Collecting multi-timeframe data for {len(symbols)} symbols...")

        # Collect all symbols in parallel
        tasks = [(symbol, self.collect_all_timeframes(symbol)) for symbol in symbols]

        results = {}
        for symbol, task in tasks:
            try:
                timeframe_data = await task
                results[symbol] = timeframe_data
                logger.info(f"   ✅ {symbol}: {len(timeframe_data)} timeframes")
            except Exception as e:
                logger.error(f"   ❌ {symbol}: {e}")
                results[symbol] = {}

        return results

    def validate_data(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Validate that all timeframes have sufficient data

        Args:
            timeframe_data: Dictionary of timeframe DataFrames

        Returns:
            True if all timeframes have valid data
        """
        for tf, df in timeframe_data.items():
            if df.empty:
                logger.warning(f"⚠️ Empty data for timeframe {tf}")
                return False

            config = self.configs[tf]
            min_rows = config.lookback_days // 30  # Rough estimate

            if len(df) < min_rows:
                logger.warning(f"⚠️ Insufficient data for {tf}: {len(df)} rows")
                return False

        return True


# TODO: Implement in Phase 6 Week 1
# - Add caching for repeated requests
# - Add data quality checks (gaps, outliers)
# - Add automatic retry on failures
# - Add progress tracking for large batches
