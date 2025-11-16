#!/usr/bin/env python3
"""
📊 Chart Image Generator
مولد صور الشارتات

Converts OHLCV data to visual chart images for pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - chart generation disabled")


class ChartImageGenerator:
    """
    Generate chart images from OHLCV data

    Supports:
    - Candlestick charts
    - Line charts
    - Volume overlay
    - Multiple timeframes
    - Custom styling
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 600),
        dpi: int = 100,
        style: str = 'dark'
    ):
        """
        Initialize chart image generator

        Args:
            image_size: Image dimensions (width, height)
            dpi: Dots per inch for image quality
            style: Chart style ('dark', 'light')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for chart generation")

        self.image_size = image_size
        self.dpi = dpi
        self.style = style

        # Style configuration
        if style == 'dark':
            self.bg_color = '#0a0e27'
            self.text_color = '#ffffff'
            self.grid_color = '#1e2642'
            self.up_color = '#00ff88'
            self.down_color = '#ff0055'
        else:
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.grid_color = '#e0e0e0'
            self.up_color = '#26a69a'
            self.down_color = '#ef5350'

    def generate_candlestick_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[str] = None,
        show_volume: bool = True,
        indicators: Optional[dict] = None
    ) -> Optional[np.ndarray]:
        """
        Generate candlestick chart

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            output_path: Optional path to save image
            show_volume: Show volume subplot
            indicators: Optional indicators to overlay (e.g., {'MA20': series, 'MA50': series})

        Returns:
            Image as numpy array (or None if saved to file)
        """
        logger.info(f"📊 Generating chart for {symbol} ({len(df)} candles)...")

        # Create figure
        fig = plt.figure(figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi), dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)

        # Create subplots
        if show_volume:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_price = fig.add_subplot(gs[0])
            ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
        else:
            ax_price = fig.add_subplot(111)
            ax_volume = None

        # Style axes
        for ax in [ax_price, ax_volume]:
            if ax:
                ax.set_facecolor(self.bg_color)
                ax.spines['bottom'].set_color(self.grid_color)
                ax.spines['top'].set_color(self.grid_color)
                ax.spines['left'].set_color(self.grid_color)
                ax.spines['right'].set_color(self.grid_color)
                ax.tick_params(colors=self.text_color)
                ax.grid(True, color=self.grid_color, alpha=0.3)

        # Prepare data
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None

        # Draw candlesticks
        width = 0.6
        for i in range(len(df)):
            date = mdates.date2num(dates[i])
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]

            # Candle color
            color = self.up_color if close_price >= open_price else self.down_color

            # High-Low line
            ax_price.plot([date, date], [low_price, high_price], color=color, linewidth=1)

            # Body rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle(
                (date - width/2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=1
            )
            ax_price.add_patch(rect)

        # Draw indicators if provided
        if indicators:
            for name, series in indicators.items():
                if len(series) == len(dates):
                    ax_price.plot(dates, series, label=name, linewidth=1.5, alpha=0.7)
            ax_price.legend(loc='upper left', facecolor=self.bg_color,
                          edgecolor=self.text_color, labelcolor=self.text_color)

        # Draw volume
        if show_volume and ax_volume and volumes is not None:
            colors = [self.up_color if closes[i] >= opens[i] else self.down_color
                     for i in range(len(df))]
            ax_volume.bar(dates, volumes, color=colors, width=width, alpha=0.5)
            ax_volume.set_ylabel('Volume', color=self.text_color)

        # Format x-axis
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Labels
        ax_price.set_ylabel('Price (USDT)', color=self.text_color)
        ax_price.set_title(f'{symbol} - {len(df)} Candles',
                          color=self.text_color, fontsize=14, fontweight='bold')

        # Tight layout
        plt.tight_layout()

        # Save or return
        if output_path:
            plt.savefig(output_path, facecolor=self.bg_color, dpi=self.dpi)
            logger.info(f"   ✅ Chart saved to {output_path}")
            plt.close()
            return None
        else:
            # Convert to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            logger.info(f"   ✅ Chart generated as array")
            return img

    def generate_line_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate simple line chart (closing prices)

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            output_path: Optional path to save image

        Returns:
            Image as numpy array (or None if saved to file)
        """
        logger.info(f"📈 Generating line chart for {symbol}...")

        fig, ax = plt.subplots(figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi),
                               dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        # Prepare data
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index

        closes = df['close'].values

        # Plot line
        ax.plot(dates, closes, color=self.up_color, linewidth=2)
        ax.fill_between(dates, closes, alpha=0.2, color=self.up_color)

        # Style
        ax.spines['bottom'].set_color(self.grid_color)
        ax.spines['top'].set_color(self.grid_color)
        ax.spines['left'].set_color(self.grid_color)
        ax.spines['right'].set_color(self.grid_color)
        ax.tick_params(colors=self.text_color)
        ax.grid(True, color=self.grid_color, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Labels
        ax.set_ylabel('Price (USDT)', color=self.text_color)
        ax.set_title(f'{symbol} - Closing Prices',
                    color=self.text_color, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save or return
        if output_path:
            plt.savefig(output_path, facecolor=self.bg_color, dpi=self.dpi)
            logger.info(f"   ✅ Line chart saved to {output_path}")
            plt.close()
            return None
        else:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            logger.info(f"   ✅ Line chart generated")
            return img

    def generate_multi_timeframe_chart(
        self,
        timeframe_data: dict,
        symbol: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate chart showing multiple timeframes

        Args:
            timeframe_data: Dict mapping timeframe to DataFrame
            symbol: Trading symbol
            output_path: Optional path to save image

        Returns:
            Image as numpy array (or None if saved to file)
        """
        logger.info(f"📊 Generating multi-timeframe chart for {symbol}...")

        n_timeframes = len(timeframe_data)
        fig, axes = plt.subplots(n_timeframes, 1,
                                figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi),
                                dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)

        if n_timeframes == 1:
            axes = [axes]

        for idx, (tf, df) in enumerate(timeframe_data.items()):
            ax = axes[idx]
            ax.set_facecolor(self.bg_color)

            # Prepare data
            if 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp'])
            else:
                dates = df.index

            closes = df['close'].values

            # Plot
            ax.plot(dates, closes, color=self.up_color, linewidth=1.5)

            # Style
            ax.spines['bottom'].set_color(self.grid_color)
            ax.spines['top'].set_color(self.grid_color)
            ax.spines['left'].set_color(self.grid_color)
            ax.spines['right'].set_color(self.grid_color)
            ax.tick_params(colors=self.text_color)
            ax.grid(True, color=self.grid_color, alpha=0.3)

            # Labels
            ax.set_ylabel(f'{tf}', color=self.text_color, fontweight='bold')

            if idx == 0:
                ax.set_title(f'{symbol} - Multi-Timeframe View',
                           color=self.text_color, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save or return
        if output_path:
            plt.savefig(output_path, facecolor=self.bg_color, dpi=self.dpi)
            logger.info(f"   ✅ Multi-timeframe chart saved")
            plt.close()
            return None
        else:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            logger.info(f"   ✅ Multi-timeframe chart generated")
            return img


# TODO: Add more chart types
# - Heatmaps
# - Renko charts
# - Point & Figure
# - Custom overlays
