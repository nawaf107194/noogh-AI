#!/usr/bin/env python3
"""
📊 Binance Data Collector - جامع البيانات من بايننس
يجمع البيانات الحية من Binance للتدريب والتنبؤ
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .binance_connector import BinanceConnector, BinanceConfig
from .live_market_data_collector import MarketCandle, TechnicalIndicators

logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """
    📊 جامع البيانات من Binance

    القدرات:
    - جمع بيانات OHLCV الحية من Binance
    - حساب المؤشرات الفنية
    - تحديث البيانات بشكل دوري
    - دعم 100+ عملة رقمية
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        config: Optional[BinanceConfig] = None
    ):
        """
        تهيئة الجامع

        Args:
            symbols: قائمة الرموز (مثل ['BTC/USDT', 'ETH/USDT'])
            config: إعدادات Binance (None = استخدام الإعدادات الافتراضية)
        """
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.connector = BinanceConnector(config)

        # Cache
        self.candles_cache: Dict[str, List[MarketCandle]] = {}
        self.indicators_cache: Dict[str, List[TechnicalIndicators]] = {}
        self.last_update: Dict[str, datetime] = {}

        logger.info(f"📊 BinanceDataCollector initialized with {len(self.symbols)} symbols")

    async def collect_historical_data(
        self,
        symbol: str,
        days: int = 30,
        timeframe: str = '1h'
    ) -> List[MarketCandle]:
        """
        جمع البيانات التاريخية من Binance

        Args:
            symbol: الرمز (مثل BTC/USDT)
            days: عدد الأيام
            timeframe: الإطار الزمني (1h, 4h, 1d)

        Returns:
            قائمة من MarketCandle
        """
        try:
            logger.info(f"📥 Collecting {days} days of data for {symbol} from Binance...")

            # حساب عدد الشموع المطلوب
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = days * 24 * 60
            limit = min(total_minutes // timeframe_minutes, 1000)  # Binance limit = 1000

            # جلب البيانات من Binance
            ohlcv = await self.connector.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            if not ohlcv:
                logger.warning(f"   ⚠️ No data received for {symbol}")
                return []

            # تحويل إلى MarketCandle
            candles = []
            for row in ohlcv:
                timestamp, open_price, high, low, close, volume = row

                candle = MarketCandle(
                    timestamp=datetime.fromtimestamp(timestamp / 1000),
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                    symbol=symbol
                )
                candles.append(candle)

            # حفظ في الـ cache
            self.candles_cache[symbol] = candles
            self.last_update[symbol] = datetime.now()

            logger.info(f"   ✅ Collected {len(candles)} candles for {symbol}")
            logger.info(f"   Date range: {candles[0].timestamp.date()} to {candles[-1].timestamp.date()}")

            return candles

        except Exception as e:
            logger.error(f"   ❌ Failed to collect data for {symbol}: {e}")
            return []

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """تحويل الإطار الزمني إلى دقائق"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(timeframe, 60)

    def calculate_technical_indicators(
        self,
        candles: List[MarketCandle]
    ) -> List[TechnicalIndicators]:
        """
        حساب المؤشرات الفنية

        Args:
            candles: قائمة الشموع

        Returns:
            قائمة من TechnicalIndicators
        """
        try:
            if len(candles) < 60:
                logger.warning(f"   ⚠️ Not enough candles for indicators ({len(candles)} < 60)")
                return []

            # تحويل إلى DataFrame
            df = pd.DataFrame([{
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles])

            # حساب RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # حساب MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # حساب Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_middle'] = sma_20
            df['bb_lower'] = sma_20 - (std_20 * 2)

            # حساب EMAs
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()

            # Fill NaN values
            df['rsi'] = df['rsi'].ffill().bfill().fillna(50.0)
            df['macd'] = df['macd'].ffill().bfill().fillna(0.0)
            df['macd_signal'] = df['macd_signal'].ffill().bfill().fillna(0.0)
            df['bb_upper'] = df['bb_upper'].ffill().bfill().fillna(df['close'])
            df['bb_middle'] = df['bb_middle'].ffill().bfill().fillna(df['close'])
            df['bb_lower'] = df['bb_lower'].ffill().bfill().fillna(df['close'])
            df['ema_9'] = df['ema_9'].ffill().bfill().fillna(df['close'])
            df['ema_21'] = df['ema_21'].ffill().bfill().fillna(df['close'])
            df['ema_50'] = df['ema_50'].ffill().bfill().fillna(df['close'])

            # تحويل إلى TechnicalIndicators
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

            logger.debug(f"   ✅ Calculated indicators for {len(indicators)} candles")
            return indicators

        except Exception as e:
            logger.error(f"   ❌ Failed to calculate indicators: {e}")
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        الحصول على السعر الحالي

        Args:
            symbol: الرمز

        Returns:
            السعر الحالي
        """
        try:
            ticker = await self.connector.get_ticker(symbol)
            if ticker:
                return float(ticker['last'])
            return None

        except Exception as e:
            logger.error(f"   ❌ Failed to get current price for {symbol}: {e}")
            return None

    async def collect_batch_data(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 30,
        timeframe: str = '1h'
    ) -> Dict[str, Dict]:
        """
        جمع البيانات لعدة رموز دفعة واحدة

        Args:
            symbols: قائمة الرموز (None = استخدام self.symbols)
            days: عدد الأيام
            timeframe: الإطار الزمني

        Returns:
            قاموس من {symbol: {'candles': [...], 'indicators': [...]}}
        """
        symbols = symbols or self.symbols
        result = {}

        logger.info(f"📊 Collecting batch data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"   [{i}/{len(symbols)}] Processing {symbol}...")

                # جمع البيانات
                candles = await self.collect_historical_data(
                    symbol=symbol,
                    days=days,
                    timeframe=timeframe
                )

                if len(candles) < 60:
                    logger.warning(f"      ⚠️ Skipping {symbol}: not enough data")
                    continue

                # حساب المؤشرات
                indicators = self.calculate_technical_indicators(candles)

                if not indicators:
                    logger.warning(f"      ⚠️ Skipping {symbol}: failed to calculate indicators")
                    continue

                result[symbol] = {
                    'candles': candles,
                    'indicators': indicators
                }

                logger.info(f"      ✅ {symbol}: {len(candles)} candles")

                # تأخير صغير لتجنب rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"      ❌ Failed to process {symbol}: {e}")
                continue

        logger.info(f"✅ Batch collection complete: {len(result)}/{len(symbols)} symbols")
        return result

    async def connect(self):
        """الاتصال مع Binance"""
        await self.connector.connect()

    async def disconnect(self):
        """قطع الاتصال"""
        await self.connector.disconnect()

    def get_stats(self) -> Dict:
        """إحصائيات الجامع"""
        return {
            'total_symbols': len(self.symbols),
            'cached_symbols': len(self.candles_cache),
            'connector_stats': self.connector.get_stats()
        }


async def test_data_collection():
    """اختبار جمع البيانات من Binance"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("📊 Testing Binance Data Collection")
    print("="*70 + "\n")

    # إنشاء الجامع
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    collector = BinanceDataCollector(symbols=symbols)

    # الاتصال
    await collector.connect()

    # جمع البيانات
    print("\n🔄 Collecting data...")
    batch_data = await collector.collect_batch_data(
        symbols=symbols,
        days=30,
        timeframe='1h'
    )

    # عرض النتائج
    print("\n📊 Results:")
    for symbol, data in batch_data.items():
        candles = data['candles']
        indicators = data['indicators']

        print(f"\n{symbol}:")
        print(f"   Candles: {len(candles)}")
        print(f"   Date range: {candles[0].timestamp.date()} to {candles[-1].timestamp.date()}")
        print(f"   Latest price: ${candles[-1].close:.2f}")
        print(f"   Latest RSI: {indicators[-1].rsi:.1f}")

    # إحصائيات
    stats = collector.get_stats()
    print("\n📈 Stats:")
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Cached symbols: {stats['cached_symbols']}")
    print(f"   Connector success rate: {stats['connector_stats']['success_rate']:.1f}%")

    # قطع الاتصال
    await collector.disconnect()

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_data_collection())
