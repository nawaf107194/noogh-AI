#!/usr/bin/env python3
"""
📊 Live Market Data Collector - جامع بيانات السوق الحية
يجمع البيانات المالية الحية لتدريب نماذج التنبؤ
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import pandas as pd
import numpy as np

from .crypto_symbols import (
    TOP_100_CRYPTO_SYMBOLS,
    TOP_20_CRYPTO_SYMBOLS,
    get_top_n_symbols,
    to_yfinance_ticker
)

logger = logging.getLogger(__name__)


@dataclass
class MarketCandle:
    """شمعة سوق واحدة"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


@dataclass
class TechnicalIndicators:
    """المؤشرات الفنية"""
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    ema_9: float
    ema_21: float
    ema_50: float


class LiveMarketDataCollector:
    """
    📊 جامع بيانات السوق الحية

    القدرات:
    - جمع البيانات الحية من API
    - حساب المؤشرات الفنية
    - حفظ البيانات للتدريب
    - تنظيف ومعالجة البيانات
    """

    def __init__(
        self,
        work_dir: str = "/home/noogh/projects/noogh_unified_system",
        symbols: List[str] = None,
        timeframe: str = "1h",
        use_top_100: bool = False
    ):
        self.work_dir = Path(work_dir)
        self.data_dir = self.work_dir / "data" / "trading"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Default symbols - support for 100 coins
        if symbols is not None:
            self.symbols = symbols
        elif use_top_100:
            self.symbols = TOP_100_CRYPTO_SYMBOLS.copy()
        else:
            # Default: top 20 for reasonable training time
            self.symbols = TOP_20_CRYPTO_SYMBOLS.copy()

        self.timeframe = timeframe

        # Stats
        self.candles_collected = 0
        self.datasets_created = 0

        logger.info(f"📊 LiveMarketDataCollector initialized")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   Timeframe: {self.timeframe}")
        logger.info(f"   Data dir: {self.data_dir}")

    async def collect_historical_data(
        self,
        symbol: str,
        days: int = 30,
        limit: int = 1000
    ) -> List[MarketCandle]:
        """
        جمع البيانات التاريخية

        في الإنتاج: سيستخدم API حقيقي (Binance, etc)
        حالياً: بيانات محاكاة للتطوير
        """
        logger.info(f"📥 Collecting {days} days of data for {symbol}...")

        try:
            # محاولة استخدام yfinance أولاً
            candles = await self._collect_from_yfinance(symbol, days)

            if candles:
                logger.info(f"   ✅ Collected {len(candles)} candles from yfinance")
                return candles
        except Exception as e:
            logger.warning(f"   ⚠️ yfinance failed: {e}")

        # Fallback: بيانات محاكاة
        logger.info(f"   Using simulated data for development")
        candles = self._generate_simulated_data(symbol, days, limit)
        logger.info(f"   ✅ Generated {len(candles)} simulated candles")

        self.candles_collected += len(candles)
        return candles

    async def _collect_from_yfinance(
        self,
        symbol: str,
        days: int
    ) -> List[MarketCandle]:
        """جمع بيانات حقيقية من yfinance"""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("   yfinance not installed, using simulation")
            return []

        # تحويل رمز العملة لصيغة yfinance
        # BTC/USDT -> BTC-USD
        yf_symbol = symbol.replace("/USDT", "-USD").replace("/", "-")

        # جمع البيانات
        ticker = yf.Ticker(yf_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = ticker.history(start=start_date, end=end_date, interval="1h")

        if df.empty:
            return []

        # تحويل لـ MarketCandle
        candles = []
        for idx, row in df.iterrows():
            candle = MarketCandle(
                timestamp=idx.isoformat(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                symbol=symbol
            )
            candles.append(candle)

        return candles

    def _generate_simulated_data(
        self,
        symbol: str,
        days: int,
        limit: int
    ) -> List[MarketCandle]:
        """توليد بيانات محاكاة للتطوير"""

        candles = []
        base_price = {
            "BTC/USDT": 40000.0,
            "ETH/USDT": 2500.0,
            "BNB/USDT": 300.0
        }.get(symbol, 100.0)

        # عدد الشموع (ساعات)
        num_candles = min(days * 24, limit)

        # توليد مسار عشوائي واقعي (Random Walk with Trend)
        np.random.seed(42)
        trend = np.random.randn() * 0.0001  # اتجاه طفيف
        volatility = 0.02  # تقلب 2%

        price = base_price
        start_time = datetime.now() - timedelta(hours=num_candles)

        for i in range(num_candles):
            # حركة السعر
            change = np.random.randn() * volatility + trend
            price = price * (1 + change)

            # شمعة واحدة
            open_price = price
            close_price = price * (1 + np.random.randn() * 0.01)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.005)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.005)
            volume = abs(np.random.randn() * 1000000)

            candle = MarketCandle(
                timestamp=(start_time + timedelta(hours=i)).isoformat(),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                symbol=symbol
            )
            candles.append(candle)

            price = close_price

        return candles

    def calculate_technical_indicators(
        self,
        candles: List[MarketCandle]
    ) -> List[TechnicalIndicators]:
        """حساب المؤشرات الفنية"""

        logger.info(f"📊 Calculating technical indicators...")

        # تحويل لـ DataFrame
        df = pd.DataFrame([asdict(c) for c in candles])

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=14)

        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # Fill NaN values with forward fill then backward fill
        df['rsi'] = df['rsi'].ffill().bfill().fillna(50.0)
        df['macd'] = df['macd'].ffill().bfill().fillna(0.0)
        df['macd_signal'] = df['macd_signal'].ffill().bfill().fillna(0.0)
        df['bb_upper'] = df['bb_upper'].ffill().bfill().fillna(df['close'])
        df['bb_middle'] = df['bb_middle'].ffill().bfill().fillna(df['close'])
        df['bb_lower'] = df['bb_lower'].ffill().bfill().fillna(df['close'])
        df['ema_9'] = df['ema_9'].ffill().bfill().fillna(df['close'])
        df['ema_21'] = df['ema_21'].ffill().bfill().fillna(df['close'])
        df['ema_50'] = df['ema_50'].ffill().bfill().fillna(df['close'])

        # تحويل لـ TechnicalIndicators
        indicators = []
        for _, row in df.iterrows():
            ind = TechnicalIndicators(
                rsi=float(row['rsi']),
                macd=float(row['macd']),
                macd_signal=float(row['macd_signal']),
                bb_upper=float(row['bb_upper']),
                bb_middle=float(row['bb_middle']),
                bb_lower=float(row['bb_lower']),
                ema_9=float(row['ema_9']),
                ema_21=float(row['ema_21']),
                ema_50=float(row['ema_50'])
            )
            indicators.append(ind)

        logger.info(f"   ✅ Calculated indicators for {len(indicators)} candles")
        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """حساب RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def prepare_training_dataset(
        self,
        candles: List[MarketCandle],
        indicators: List[TechnicalIndicators],
        symbol: Optional[str] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> Dict:
        """
        تحضير dataset للتدريب

        Args:
            candles: الشموع
            indicators: المؤشرات
            symbol: رمز العملة (optional)
            sequence_length: طول التسلسل (60 شموع = 60 ساعة)
            prediction_horizon: كم شمعة للتنبؤ (1 = ساعة واحدة)

        Returns:
            Dict مع X (features) و y (target)
        """
        logger.info(f"📦 Preparing training dataset...")
        logger.info(f"   Sequence length: {sequence_length}")
        logger.info(f"   Prediction horizon: {prediction_horizon}")

        # دمج البيانات
        data = []
        for candle, ind in zip(candles, indicators):
            features = {
                # OHLCV
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                # Technical Indicators
                'rsi': ind.rsi,
                'macd': ind.macd,
                'macd_signal': ind.macd_signal,
                'bb_upper': ind.bb_upper,
                'bb_middle': ind.bb_middle,
                'bb_lower': ind.bb_lower,
                'ema_9': ind.ema_9,
                'ema_21': ind.ema_21,
                'ema_50': ind.ema_50
            }
            data.append(features)

        df = pd.DataFrame(data)

        # CRITICAL: Remove any remaining NaN values before normalization
        # Fill NaN with forward/backward fill, then use median as fallback
        for col in df.columns:
            df[col] = df[col].ffill().bfill()
            if df[col].isna().any():
                # If still NaN, fill with median or 0
                median_val = df[col].median() if not df[col].isna().all() else 0.0
                df[col] = df[col].fillna(median_val)

        # Verify no NaN values remain
        assert not df.isna().any().any(), f"NaN values found in dataset after cleaning: {df.isna().sum()}"

        # Normalize with epsilon to avoid division by zero
        mean = df.mean()
        std = df.std()
        std = std.replace(0, 1e-8)  # Replace 0 with small value
        df_normalized = (df - mean) / std

        # Create sequences
        X, y = [], []

        for i in range(len(df_normalized) - sequence_length - prediction_horizon):
            # Input: sequence_length candles
            sequence = df_normalized.iloc[i:i+sequence_length].values
            X.append(sequence)

            # Target: السعر بعد prediction_horizon
            future_price = df.iloc[i+sequence_length+prediction_horizon]['close']
            current_price = df.iloc[i+sequence_length]['close']

            # Target: 1 للشراء (سيرتفع)، 0 للبيع (سينخفض)
            target = 1 if future_price > current_price else 0
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"   ✅ Dataset prepared:")
        logger.info(f"      X shape: {X.shape}")
        logger.info(f"      y shape: {y.shape}")
        logger.info(f"      Buy signals: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        logger.info(f"      Sell signals: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")

        self.datasets_created += 1

        return {
            'X': X,
            'y': y,
            'symbol': symbol if symbol else 'unknown',
            'feature_names': list(df.columns),
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'num_features': X.shape[2],
            'num_samples': len(X)
        }

    async def collect_and_prepare_all_symbols(
        self,
        days: int = 30,
        sequence_length: int = 60
    ) -> List[Dict]:
        """جمع وتحضير بيانات كل الرموز"""

        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"📊 Collecting data for {len(self.symbols)} symbols...")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        all_datasets = []

        for symbol in self.symbols:
            logger.info(f"\n📈 Processing {symbol}...")

            # 1. جمع البيانات
            candles = await self.collect_historical_data(symbol, days=days)

            if len(candles) < sequence_length + 10:
                logger.warning(f"   ⚠️ Not enough data for {symbol}, skipping")
                continue

            # 2. حساب المؤشرات
            indicators = self.calculate_technical_indicators(candles)

            # 3. تحضير dataset
            dataset = self.prepare_training_dataset(
                candles,
                indicators,
                sequence_length=sequence_length
            )

            dataset['symbol'] = symbol
            all_datasets.append(dataset)

            # 4. حفظ
            self._save_dataset(dataset, symbol)

        logger.info(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"✅ Data collection complete!")
        logger.info(f"   Symbols processed: {len(all_datasets)}")
        logger.info(f"   Total candles: {self.candles_collected}")
        logger.info(f"   Datasets created: {self.datasets_created}")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return all_datasets

    def _save_dataset(self, dataset: Dict, symbol: str):
        """حفظ dataset"""
        symbol_clean = symbol.replace("/", "_")
        filepath = self.data_dir / f"{symbol_clean}_dataset.npz"

        np.savez(
            filepath,
            X=dataset['X'],
            y=dataset['y'],
            feature_names=dataset['feature_names'],
            sequence_length=dataset['sequence_length'],
            prediction_horizon=dataset['prediction_horizon']
        )

        logger.info(f"   💾 Saved dataset to {filepath}")

    def load_dataset(self, symbol: str) -> Optional[Dict]:
        """تحميل dataset محفوظ"""
        symbol_clean = symbol.replace("/", "_")
        filepath = self.data_dir / f"{symbol_clean}_dataset.npz"

        if not filepath.exists():
            return None

        data = np.load(filepath, allow_pickle=True)

        return {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names'].tolist(),
            'sequence_length': int(data['sequence_length']),
            'prediction_horizon': int(data['prediction_horizon']),
            'symbol': symbol
        }

    def get_stats(self) -> Dict:
        """إحصائيات الجامع"""
        return {
            'candles_collected': self.candles_collected,
            'datasets_created': self.datasets_created,
            'symbols': self.symbols,
            'data_dir': str(self.data_dir)
        }


async def main():
    """اختبار الجامع"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("📊 Testing Live Market Data Collector")
    print("="*70 + "\n")

    # إنشاء الجامع
    collector = LiveMarketDataCollector(
        symbols=["BTC/USDT", "ETH/USDT"]
    )

    # جمع وتحضير البيانات
    datasets = await collector.collect_and_prepare_all_symbols(
        days=30,
        sequence_length=60
    )

    # عرض النتائج
    print("\n" + "="*70)
    print("📊 Results:")
    print("="*70)

    for dataset in datasets:
        print(f"\n{dataset['symbol']}:")
        print(f"  Samples: {dataset['num_samples']}")
        print(f"  Features: {dataset['num_features']}")
        print(f"  Shape: {dataset['X'].shape}")

    # الإحصائيات
    stats = collector.get_stats()
    print(f"\n📈 Stats:")
    print(f"  Total candles: {stats['candles_collected']}")
    print(f"  Datasets created: {stats['datasets_created']}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
