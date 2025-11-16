#!/usr/bin/env python3
"""
🤖 Trade Execution Engine - محرك تنفيذ الصفقات
ينفذ الصفقات تلقائياً بناءً على الإشارات من AI
"""

import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .binance_connector import BinanceConnector, BinanceConfig
from .risk_manager import RiskManager, RiskConfig, PositionSize
from .trading_predictor import TradingSignal

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """حالة الأمر"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutedTrade:
    """صفقة منفذة"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    status: OrderStatus
    order_id: Optional[str] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    exit_timestamp: Optional[datetime] = None


class TradeExecutionEngine:
    """
    🤖 محرك تنفيذ الصفقات

    القدرات:
    - تنفيذ الصفقات تلقائياً من إشارات AI
    - مراقبة الصفقات المفتوحة
    - تنفيذ Stop Loss و Take Profit تلقائياً
    - إدارة المخاطر المتقدمة
    """

    def __init__(
        self,
        binance_config: Optional[BinanceConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        dry_run: bool = True
    ):
        """
        تهيئة محرك التنفيذ

        Args:
            binance_config: إعدادات Binance
            risk_config: إعدادات المخاطر
            dry_run: وضع التجربة (لا تنفيذ حقيقي)
        """
        self.connector = BinanceConnector(binance_config)
        self.risk_manager = RiskManager(risk_config)
        self.dry_run = dry_run

        # Open positions tracking
        self.open_positions: Dict[str, ExecutedTrade] = {}

        # Execution history
        self.trade_history: List[ExecutedTrade] = []

        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        logger.info(f"🤖 TradeExecutionEngine initialized")
        logger.info(f"   Dry run: {self.dry_run}")
        logger.info(f"   Risk config: {risk_config}")

    async def connect(self):
        """الاتصال مع Binance"""
        await self.connector.connect()

    async def disconnect(self):
        """قطع الاتصال"""
        await self.connector.disconnect()

    async def execute_signal(
        self,
        signal: TradingSignal,
        balance: float
    ) -> Optional[ExecutedTrade]:
        """
        تنفيذ إشارة تداول

        Args:
            signal: إشارة التداول من AI
            balance: الرصيد المتاح

        Returns:
            ExecutedTrade إذا نجح التنفيذ
        """
        try:
            logger.info(f"🎯 Executing signal for {signal.symbol}...")
            logger.info(f"   Action: {signal.action.upper()}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")

            # Check if we should trade
            should_trade, reason = self.risk_manager.should_trade(
                symbol=signal.symbol,
                confidence=signal.confidence,
                current_open_positions=len(self.open_positions)
            )

            if not should_trade:
                logger.warning(f"   ⚠️ Trade rejected: {reason}")
                return None

            # Skip HOLD signals
            if signal.action == 'hold':
                logger.info(f"   ⏸️ Signal is HOLD, skipping")
                return None

            # Get current price
            ticker = await self.connector.get_ticker(signal.symbol)
            if not ticker:
                logger.error(f"   ❌ Failed to get price for {signal.symbol}")
                return None

            current_price = float(ticker['last'])

            # Calculate position size
            side = 'buy' if signal.action == 'buy' else 'sell'
            position = self.risk_manager.calculate_position_size(
                symbol=signal.symbol,
                current_price=current_price,
                balance=balance,
                confidence=signal.confidence,
                side=side
            )

            if not position:
                logger.warning(f"   ⚠️ Failed to calculate position size")
                return None

            # Execute order
            self.total_executions += 1

            if self.dry_run:
                logger.info(f"   🔍 DRY RUN: {side.upper()} {position.amount} {signal.symbol} @ ${current_price}")
                order_id = f"dry_run_{self.total_executions}"
                status = OrderStatus.FILLED
                self.successful_executions += 1
            else:
                # Real execution
                logger.warning(f"   ⚠️ EXECUTING LIVE ORDER: {side.upper()} {position.amount} {signal.symbol}")
                order = await self.connector.create_market_order(
                    symbol=signal.symbol,
                    side=side,
                    amount=position.amount,
                    dry_run=False
                )

                if order and order.get('status') in ['filled', 'FILLED']:
                    order_id = order['id']
                    status = OrderStatus.FILLED
                    self.successful_executions += 1
                    logger.info(f"   ✅ Order executed: {order_id}")
                else:
                    logger.error(f"   ❌ Order failed")
                    self.failed_executions += 1
                    return None

            # Create trade record
            trade = ExecutedTrade(
                id=f"trade_{self.total_executions}",
                symbol=signal.symbol,
                side=side,
                entry_price=current_price,
                amount=position.amount,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                confidence=signal.confidence,
                timestamp=datetime.now(),
                status=status,
                order_id=order_id
            )

            # Add to open positions
            self.open_positions[signal.symbol] = trade
            self.trade_history.append(trade)

            # Update risk manager balance
            self.risk_manager.update_balance(balance - position.value_usd)

            logger.info(f"   ✅ Trade recorded:")
            logger.info(f"      ID: {trade.id}")
            logger.info(f"      Amount: {trade.amount}")
            logger.info(f"      Entry: ${trade.entry_price}")
            logger.info(f"      Stop Loss: ${trade.stop_loss}")
            logger.info(f"      Take Profit: ${trade.take_profit}")

            return trade

        except Exception as e:
            logger.error(f"   ❌ Failed to execute signal: {e}")
            self.failed_executions += 1
            return None

    async def monitor_positions(self) -> List[ExecutedTrade]:
        """
        مراقبة الصفقات المفتوحة وتنفيذ Stop Loss / Take Profit

        Returns:
            قائمة الصفقات المغلقة
        """
        closed_trades = []

        if not self.open_positions:
            return closed_trades

        logger.info(f"👁️ Monitoring {len(self.open_positions)} open positions...")

        for symbol, trade in list(self.open_positions.items()):
            try:
                # Get current price
                ticker = await self.connector.get_ticker(symbol)
                if not ticker:
                    continue

                current_price = float(ticker['last'])

                # Check Stop Loss and Take Profit
                should_close = False
                close_reason = ""

                if trade.side == 'buy':
                    if current_price <= trade.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss hit"
                    elif current_price >= trade.take_profit:
                        should_close = True
                        close_reason = "Take Profit hit"
                else:  # sell
                    if current_price >= trade.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss hit"
                    elif current_price <= trade.take_profit:
                        should_close = True
                        close_reason = "Take Profit hit"

                if should_close:
                    logger.info(f"   🔔 {symbol}: {close_reason}")
                    closed_trade = await self.close_position(trade, current_price, close_reason)
                    if closed_trade:
                        closed_trades.append(closed_trade)

            except Exception as e:
                logger.error(f"   ❌ Error monitoring {symbol}: {e}")

        return closed_trades

    async def close_position(
        self,
        trade: ExecutedTrade,
        exit_price: float,
        reason: str
    ) -> Optional[ExecutedTrade]:
        """
        إغلاق صفقة

        Args:
            trade: الصفقة المراد إغلاقها
            exit_price: سعر الخروج
            reason: سبب الإغلاق

        Returns:
            ExecutedTrade مع معلومات الإغلاق
        """
        try:
            logger.info(f"   🔒 Closing position: {trade.symbol}")
            logger.info(f"      Reason: {reason}")

            # Execute closing order (opposite side)
            close_side = 'sell' if trade.side == 'buy' else 'buy'

            if self.dry_run:
                logger.info(f"      🔍 DRY RUN: {close_side.upper()} {trade.amount} {trade.symbol} @ ${exit_price}")
            else:
                # Real execution
                order = await self.connector.create_market_order(
                    symbol=trade.symbol,
                    side=close_side,
                    amount=trade.amount,
                    dry_run=False
                )

                if not order or order.get('status') not in ['filled', 'FILLED']:
                    logger.error(f"      ❌ Failed to close position")
                    return None

            # Calculate profit/loss
            if trade.side == 'buy':
                profit_loss = (exit_price - trade.entry_price) * trade.amount
            else:  # sell
                profit_loss = (trade.entry_price - exit_price) * trade.amount

            # Update trade
            trade.exit_price = exit_price
            trade.profit_loss = profit_loss
            trade.exit_timestamp = datetime.now()
            trade.status = OrderStatus.FILLED

            # Remove from open positions
            if trade.symbol in self.open_positions:
                del self.open_positions[trade.symbol]

            # Record in risk manager
            self.risk_manager.record_trade(
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                amount=trade.amount,
                profit=profit_loss
            )

            logger.info(f"      ✅ Position closed:")
            logger.info(f"         Entry: ${trade.entry_price} → Exit: ${exit_price}")
            logger.info(f"         Profit/Loss: ${profit_loss:.2f}")

            return trade

        except Exception as e:
            logger.error(f"      ❌ Failed to close position: {e}")
            return None

    async def close_all_positions(self, reason: str = "Manual close"):
        """إغلاق جميع الصفقات المفتوحة"""
        logger.warning(f"⚠️ Closing all {len(self.open_positions)} open positions...")

        for symbol, trade in list(self.open_positions.items()):
            try:
                ticker = await self.connector.get_ticker(symbol)
                if ticker:
                    current_price = float(ticker['last'])
                    await self.close_position(trade, current_price, reason)
            except Exception as e:
                logger.error(f"   ❌ Failed to close {symbol}: {e}")

    def get_stats(self) -> Dict:
        """إحصائيات محرك التنفيذ"""
        risk_stats = self.risk_manager.get_stats()

        return {
            'execution_stats': {
                'total_executions': self.total_executions,
                'successful': self.successful_executions,
                'failed': self.failed_executions,
                'success_rate': self.successful_executions / max(self.total_executions, 1) * 100
            },
            'positions': {
                'open': len(self.open_positions),
                'total_history': len(self.trade_history)
            },
            'risk_stats': risk_stats,
            'mode': 'DRY RUN' if self.dry_run else 'LIVE TRADING'
        }


async def test_execution_engine():
    """اختبار محرك التنفيذ"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("🤖 Testing Trade Execution Engine")
    print("="*70 + "\n")

    # Create engine (dry run mode)
    engine = TradeExecutionEngine(dry_run=True)
    await engine.connect()

    # Test balance
    balance = 10000.0

    # Create mock signals
    from dataclasses import dataclass

    @dataclass
    class MockSignal:
        symbol: str
        action: str
        confidence: float
        predicted_probability: float
        reasoning: str
        features: dict

    signals = [
        MockSignal('BTC/USDT', 'buy', 0.75, 0.75, 'Test signal 1', {}),
        MockSignal('ETH/USDT', 'buy', 0.68, 0.68, 'Test signal 2', {}),
        MockSignal('BNB/USDT', 'hold', 0.52, 0.52, 'Test signal 3', {}),
    ]

    # Execute signals
    print("📊 Executing signals...")
    for signal in signals:
        trade = await engine.execute_signal(signal, balance)
        if trade:
            print(f"   ✅ Trade executed: {trade.symbol} ({trade.side})")
        await asyncio.sleep(0.5)

    # Monitor positions
    print("\n👁️ Monitoring positions...")
    await engine.monitor_positions()

    # Stats
    print("\n📈 Statistics:")
    stats = engine.get_stats()
    print(f"   Total executions: {stats['execution_stats']['total_executions']}")
    print(f"   Success rate: {stats['execution_stats']['success_rate']:.1f}%")
    print(f"   Open positions: {stats['positions']['open']}")
    print(f"   Mode: {stats['mode']}")

    # Disconnect
    await engine.disconnect()

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_execution_engine())
