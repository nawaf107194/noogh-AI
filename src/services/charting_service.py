#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charting Service - Financial Chart Generation
==============================================

Generates clean, readable candlestick charts for vision AI analysis.
Fixed: Now uses ABSOLUTE paths for Streamlit compatibility.
"""

import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

logger = logging.getLogger(__name__)


class ChartingService:
    """
    Service for generating financial charts.

    Creates candlestick charts optimized for AI vision analysis.
    Now uses ABSOLUTE paths to ensure Streamlit can load images.
    """

    def __init__(self, output_dir: str = "data/charts"):
        """
        Initialize charting service.

        Args:
            output_dir: Relative path from project root (default: "data/charts")
        """
        # Establish ABSOLUTE path
        project_root = Path(os.getcwd())
        self.output_dir = project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ Charting Service initialized")
        logger.info(f"   Output: {self.output_dir.absolute()}")

    def generate_chart_image(
        self,
        df,
        symbol: str,
        timeframe: str = "1h",
        style: str = "charles"
    ) -> Dict[str, Any]:
        """
        Generate candlestick chart image.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe string
            style: Chart style

        Returns:
            Dictionary with ABSOLUTE file path and metadata
        """
        try:
            import mplfinance as mpf
            import pandas as pd

            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Clean symbol for filename (replace "/" and other illegal chars)
            clean_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
            filename = f"chart_{clean_symbol}_{timeframe}.png"
            output_path = self.output_dir / filename

            # Configure chart style
            mc = mpf.make_marketcolors(
                up='g', down='r',
                edge='inherit',
                wick={'up':'g', 'down':'r'},
                volume='in'
            )

            s = mpf.make_mpf_style(
                base_mpf_style=style,
                marketcolors=mc,
                gridstyle='',
                y_on_right=False
            )

            # Generate chart
            mpf.plot(
                df,
                type='candle',
                style=s,
                volume=True,
                title=f"{symbol} {timeframe}",
                ylabel='Price (USDT)',
                ylabel_lower='Volume',
                savefig=dict(
                    fname=str(output_path.absolute()),  # Use absolute path
                    dpi=150,
                    bbox_inches='tight'
                ),
                figsize=(12, 8),
                tight_layout=True
            )

            logger.info(f"📊 Chart generated: {filename}")
            logger.info(f"   Path: {output_path.absolute()}")

            return {
                "success": True,
                "path": str(output_path.absolute()),  # Return ABSOLUTE path
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(df)
            }

        except ImportError as e:
            logger.error("mplfinance not installed. Install with: pip install mplfinance")
            return {
                "success": False,
                "error": "mplfinance not installed"
            }
        except Exception as e:
            logger.error(f"❌ Error generating chart: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["ChartingService"]
