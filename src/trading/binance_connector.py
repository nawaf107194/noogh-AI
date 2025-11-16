#!/usr/bin/env python3
"""
🔗 Binance Connector - موصل بايننس
يربط النظام مع منصة Binance للحصول على بيانات حية وتنفيذ صفقات
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import ccxt.async_support as ccxt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BinanceConfig:
    """إعدادات الاتصال مع Binance"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True  # استخدام testnet افتراضياً للأمان
    read_only: bool = True  # وضع القراءة فقط (لا تداول)
    timeout: int = 30000  # 30 seconds


class BinanceConnector:
    """
    🔗 موصل Binance

    القدرات:
    - جلب بيانات السوق الحية
    - جلب تاريخ الأسعار (OHLCV)
    - جلب معلومات الحساب (رصيد، صفقات)
    - تنفيذ أوامر الشراء/البيع (في وضع التداول)
    - دعم Testnet للتجربة الآمنة
    """

    def __init__(self, config: Optional[BinanceConfig] = None):
        """
        تهيئة الموصل

        Args:
            config: إعدادات الاتصال (إذا كانت None، سيعمل بدون API keys)
        """
        self.config = config or BinanceConfig()
        self.exchange: Optional[ccxt.binance] = None
        self.connected = False

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0

        logger.info(f"🔗 BinanceConnector initialized")
        logger.info(f"   Testnet: {self.config.testnet}")
        logger.info(f"   Read-only: {self.config.read_only}")

    async def connect(self) -> bool:
        """
        الاتصال مع Binance

        Returns:
            True إذا نجح الاتصال
        """
        try:
            logger.info("🔌 Connecting to Binance...")

            # إعداد الاتصال
            options = {
                'enableRateLimit': True,
                'timeout': self.config.timeout
            }

            if self.config.api_key and self.config.api_secret:
                options['apiKey'] = self.config.api_key
                options['secret'] = self.config.api_secret

                if self.config.testnet:
                    options['options'] = {'defaultType': 'future'}
                    options['urls'] = {
                        'api': {
                            'public': 'https://testnet.binancefuture.com',
                            'private': 'https://testnet.binancefuture.com'
                        }
                    }
                    logger.info("   Using TESTNET (safe for testing)")
                else:
                    logger.warning("   ⚠️ Using LIVE TRADING (real money!)")
            else:
                logger.info("   Using public API (read-only, no API keys)")

            # إنشاء الاتصال
            self.exchange = ccxt.binance(options)

            # اختبار الاتصال
            await self.exchange.load_markets()

            self.connected = True
            logger.info(f"   ✅ Connected! Markets loaded: {len(self.exchange.markets)}")

            # عرض معلومات الحساب إذا كانت API keys متوفرة
            if self.config.api_key and self.config.api_secret:
                balance = await self.get_balance()
                logger.info(f"   Account balance loaded: {len(balance)} assets")

            return True

        except Exception as e:
            logger.error(f"   ❌ Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """قطع الاتصال"""
        if self.exchange:
            await self.exchange.close()
            self.connected = False
            logger.info("🔌 Disconnected from Binance")

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> List[List]:
        """
        جلب بيانات OHLCV (Open, High, Low, Close, Volume)

        Args:
            symbol: الرمز (مثل BTC/USDT)
            timeframe: الإطار الزمني (1m, 5m, 15m, 1h, 4h, 1d)
            limit: عدد الشموع
            since: الوقت الابتدائي (timestamp milliseconds)

        Returns:
            قائمة من [timestamp, open, high, low, close, volume]
        """
        try:
            self.total_requests += 1

            if not self.connected:
                await self.connect()

            # تحويل الرمز إلى صيغة Binance
            binance_symbol = symbol.replace('/', '')
            if not binance_symbol.endswith('USDT'):
                binance_symbol = binance_symbol.replace('USDT', '') + 'USDT'

            # جلب البيانات
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            logger.debug(f"   ✅ Fetched {len(ohlcv)} candles for {symbol}")
            return ohlcv

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to fetch OHLCV for {symbol}: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        جلب سعر الرمز الحالي

        Args:
            symbol: الرمز (مثل BTC/USDT)

        Returns:
            معلومات السعر الحالي
        """
        try:
            self.total_requests += 1

            if not self.connected:
                await self.connect()

            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to fetch ticker for {symbol}: {e}")
            return None

    async def get_balance(self) -> Dict:
        """
        جلب رصيد الحساب

        Returns:
            رصيد جميع العملات
        """
        try:
            self.total_requests += 1

            if not self.config.api_key or not self.config.api_secret:
                logger.warning("   ⚠️ API keys not configured, cannot fetch balance")
                return {}

            if not self.connected:
                await self.connect()

            balance = await self.exchange.fetch_balance()

            # تصفية العملات بدون رصيد
            filtered_balance = {
                symbol: info for symbol, info in balance['total'].items()
                if info > 0
            }

            return filtered_balance

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to fetch balance: {e}")
            return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        جلب الأوامر المفتوحة

        Args:
            symbol: الرمز (إذا كان None، يجلب جميع الأوامر)

        Returns:
            قائمة الأوامر المفتوحة
        """
        try:
            self.total_requests += 1

            if not self.config.api_key or not self.config.api_secret:
                logger.warning("   ⚠️ API keys not configured, cannot fetch orders")
                return []

            if not self.connected:
                await self.connect()

            orders = await self.exchange.fetch_open_orders(symbol)
            return orders

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to fetch open orders: {e}")
            return []

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        dry_run: bool = True
    ) -> Optional[Dict]:
        """
        إنشاء أمر سوق (تنفيذ فوري)

        Args:
            symbol: الرمز (مثل BTC/USDT)
            side: الجانب ('buy' أو 'sell')
            amount: الكمية
            dry_run: وضع التجربة (لا تنفيذ حقيقي)

        Returns:
            معلومات الأمر إذا نجح
        """
        try:
            self.total_requests += 1

            # التحقق من الأمان
            if self.config.read_only:
                logger.warning("   ⚠️ Read-only mode, order not executed!")
                logger.info(f"   DRY RUN: {side.upper()} {amount} {symbol}")
                return None

            if dry_run:
                logger.info(f"   🔍 DRY RUN: {side.upper()} {amount} {symbol}")
                return {
                    'id': 'dry_run',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'type': 'market',
                    'status': 'dry_run'
                }

            if not self.config.api_key or not self.config.api_secret:
                logger.error("   ❌ API keys required for trading!")
                return None

            if not self.connected:
                await self.connect()

            # ⚠️ تنفيذ حقيقي
            logger.warning(f"   ⚠️ EXECUTING LIVE ORDER: {side.upper()} {amount} {symbol}")
            order = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount
            )

            logger.info(f"   ✅ Order executed: {order['id']}")
            return order

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to create order: {e}")
            return None

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        dry_run: bool = True
    ) -> Optional[Dict]:
        """
        إنشاء أمر محدد (تنفيذ عند سعر معين)

        Args:
            symbol: الرمز
            side: الجانب ('buy' أو 'sell')
            amount: الكمية
            price: السعر المحدد
            dry_run: وضع التجربة

        Returns:
            معلومات الأمر إذا نجح
        """
        try:
            self.total_requests += 1

            # التحقق من الأمان
            if self.config.read_only:
                logger.warning("   ⚠️ Read-only mode, order not executed!")
                logger.info(f"   DRY RUN: {side.upper()} {amount} {symbol} @ ${price}")
                return None

            if dry_run:
                logger.info(f"   🔍 DRY RUN: {side.upper()} {amount} {symbol} @ ${price}")
                return {
                    'id': 'dry_run',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'type': 'limit',
                    'status': 'dry_run'
                }

            if not self.config.api_key or not self.config.api_secret:
                logger.error("   ❌ API keys required for trading!")
                return None

            if not self.connected:
                await self.connect()

            # ⚠️ تنفيذ حقيقي
            logger.warning(f"   ⚠️ EXECUTING LIVE ORDER: {side.upper()} {amount} {symbol} @ ${price}")
            order = await self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price
            )

            logger.info(f"   ✅ Order created: {order['id']}")
            return order

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"   ❌ Failed to create limit order: {e}")
            return None

    def get_stats(self) -> Dict:
        """إحصائيات الموصل"""
        return {
            'connected': self.connected,
            'testnet': self.config.testnet,
            'read_only': self.config.read_only,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1) * 100
        }


async def test_connection():
    """اختبار الاتصال مع Binance"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("🔗 Testing Binance Connection")
    print("="*70 + "\n")

    # اتصال بدون API keys (قراءة فقط)
    connector = BinanceConnector()

    # الاتصال
    success = await connector.connect()

    if success:
        print("\n📊 Testing data fetch...")

        # جلب بيانات BTC
        ohlcv = await connector.get_ohlcv('BTC/USDT', timeframe='1h', limit=100)
        print(f"   ✅ Fetched {len(ohlcv)} BTC candles")

        if ohlcv:
            latest = ohlcv[-1]
            print(f"   Latest: Open=${latest[1]:.2f}, Close=${latest[4]:.2f}")

        # جلب السعر الحالي
        ticker = await connector.get_ticker('BTC/USDT')
        if ticker:
            print(f"   ✅ Current BTC price: ${ticker['last']:.2f}")

        # إحصائيات
        stats = connector.get_stats()
        print("\n📈 Stats:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Failed requests: {stats['failed_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")

        # قطع الاتصال
        await connector.disconnect()

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_connection())
