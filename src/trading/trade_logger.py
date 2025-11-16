#!/usr/bin/env python3
"""
📝 Trade Logger - مسجل الصفقات
يسجل جميع الصفقات والأحداث في قاعدة بيانات SQLite
"""

import sqlite3
import logging
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    📝 مسجل الصفقات

    القدرات:
    - تسجيل جميع الصفقات في قاعدة بيانات
    - تتبع الأرباح والخسائر
    - إحصائيات تفصيلية
    - تصدير السجلات
    """

    def __init__(self, db_path: str = "data/trading/trades.db"):
        """
        تهيئة مسجل الصفقات

        Args:
            db_path: مسار قاعدة البيانات
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"📝 TradeLogger initialized")
        logger.info(f"   Database: {self.db_path}")

    def _init_database(self):
        """إنشاء جداول قاعدة البيانات"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                amount REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                profit_loss REAL,
                status TEXT NOT NULL,
                entry_timestamp TEXT NOT NULL,
                exit_timestamp TEXT,
                order_id TEXT,
                reason TEXT
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_probability REAL,
                reasoning TEXT,
                features TEXT,
                timestamp TEXT NOT NULL,
                executed INTEGER DEFAULT 0
            )
        """)

        # Events table (for debugging and monitoring)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                symbol TEXT,
                message TEXT,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

        logger.debug("   ✅ Database initialized")

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        """
        تسجيل صفقة جديدة

        Args:
            trade_id: معرف الصفقة
            symbol: الرمز
            side: نوع الصفقة (buy/sell)
            entry_price: سعر الدخول
            amount: الكمية
            stop_loss: سعر Stop Loss
            take_profit: سعر Take Profit
            confidence: مستوى الثقة
            order_id: معرف الأمر من Binance
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO trades (
                    id, symbol, side, entry_price, amount,
                    stop_loss, take_profit, confidence,
                    status, entry_timestamp, order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, side, entry_price, amount,
                stop_loss, take_profit, confidence,
                'open', datetime.now().isoformat(), order_id
            ))

            conn.commit()
            conn.close()

            logger.info(f"   📝 Trade logged: {trade_id} ({symbol} {side})")

        except Exception as e:
            logger.error(f"   ❌ Failed to log trade: {e}")

    def update_trade(
        self,
        trade_id: str,
        exit_price: float,
        profit_loss: float,
        reason: str
    ):
        """
        تحديث صفقة عند الإغلاق

        Args:
            trade_id: معرف الصفقة
            exit_price: سعر الخروج
            profit_loss: الربح/الخسارة
            reason: سبب الإغلاق
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE trades
                SET exit_price = ?,
                    profit_loss = ?,
                    status = 'closed',
                    exit_timestamp = ?,
                    reason = ?
                WHERE id = ?
            """, (exit_price, profit_loss, datetime.now().isoformat(), reason, trade_id))

            conn.commit()
            conn.close()

            logger.info(f"   📝 Trade updated: {trade_id} (P/L: ${profit_loss:.2f})")

        except Exception as e:
            logger.error(f"   ❌ Failed to update trade: {e}")

    def log_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        predicted_probability: float,
        reasoning: str,
        features: Dict,
        executed: bool = False
    ):
        """
        تسجيل إشارة تداول

        Args:
            symbol: الرمز
            action: نوع الإشارة (buy/sell/hold)
            confidence: مستوى الثقة
            predicted_probability: احتمالية التنبؤ
            reasoning: السبب
            features: الميزات المستخدمة
            executed: هل تم تنفيذ الإشارة
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO signals (
                    symbol, action, confidence,
                    predicted_probability, reasoning, features,
                    timestamp, executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, action, confidence,
                predicted_probability, reasoning,
                json.dumps(features),
                datetime.now().isoformat(),
                1 if executed else 0
            ))

            conn.commit()
            conn.close()

            logger.debug(f"   📝 Signal logged: {symbol} {action}")

        except Exception as e:
            logger.error(f"   ❌ Failed to log signal: {e}")

    def log_event(
        self,
        event_type: str,
        message: str,
        symbol: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """
        تسجيل حدث

        Args:
            event_type: نوع الحدث (error/warning/info)
            message: الرسالة
            symbol: الرمز (اختياري)
            details: تفاصيل إضافية (اختياري)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO events (
                    event_type, symbol, message, details, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                event_type, symbol, message,
                json.dumps(details) if details else None,
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"   ❌ Failed to log event: {e}")

    def get_open_trades(self) -> List[Dict]:
        """الحصول على جميع الصفقات المفتوحة"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM trades WHERE status = 'open'
                ORDER BY entry_timestamp DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'entry_price': row[3],
                    'exit_price': row[4],
                    'amount': row[5],
                    'stop_loss': row[6],
                    'take_profit': row[7],
                    'confidence': row[8],
                    'profit_loss': row[9],
                    'status': row[10],
                    'entry_timestamp': row[11],
                    'exit_timestamp': row[12],
                    'order_id': row[13],
                    'reason': row[14]
                })

            return trades

        except Exception as e:
            logger.error(f"   ❌ Failed to get open trades: {e}")
            return []

    def get_trade_history(
        self,
        limit: int = 100,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        الحصول على تاريخ الصفقات

        Args:
            limit: عدد الصفقات
            symbol: الرمز (اختياري)

        Returns:
            قائمة الصفقات
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if symbol:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE symbol = ?
                    ORDER BY entry_timestamp DESC
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trades
                    ORDER BY entry_timestamp DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'entry_price': row[3],
                    'exit_price': row[4],
                    'amount': row[5],
                    'stop_loss': row[6],
                    'take_profit': row[7],
                    'confidence': row[8],
                    'profit_loss': row[9],
                    'status': row[10],
                    'entry_timestamp': row[11],
                    'exit_timestamp': row[12],
                    'order_id': row[13],
                    'reason': row[14]
                })

            return trades

        except Exception as e:
            logger.error(f"   ❌ Failed to get trade history: {e}")
            return []

    def get_statistics(self) -> Dict:
        """الحصول على إحصائيات الصفقات"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            # Open trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
            open_trades = cursor.fetchone()[0]

            # Winning trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND profit_loss > 0")
            winning_trades = cursor.fetchone()[0]

            # Losing trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND profit_loss < 0")
            losing_trades = cursor.fetchone()[0]

            # Total profit/loss
            cursor.execute("SELECT SUM(profit_loss) FROM trades WHERE status = 'closed'")
            total_pl = cursor.fetchone()[0] or 0.0

            # Average profit
            cursor.execute("SELECT AVG(profit_loss) FROM trades WHERE status = 'closed' AND profit_loss > 0")
            avg_profit = cursor.fetchone()[0] or 0.0

            # Average loss
            cursor.execute("SELECT AVG(profit_loss) FROM trades WHERE status = 'closed' AND profit_loss < 0")
            avg_loss = cursor.fetchone()[0] or 0.0

            conn.close()

            closed_trades = winning_trades + losing_trades
            win_rate = winning_trades / closed_trades if closed_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'open_trades': open_trades,
                'closed_trades': closed_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit_loss': total_pl,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            }

        except Exception as e:
            logger.error(f"   ❌ Failed to get statistics: {e}")
            return {}


def test_trade_logger():
    """اختبار مسجل الصفقات"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("📝 Testing Trade Logger")
    print("="*70 + "\n")

    # Create logger
    logger_instance = TradeLogger(db_path="data/trading/test_trades.db")

    # Log some trades
    print("📝 Logging test trades...")
    logger_instance.log_trade(
        trade_id="trade_001",
        symbol="BTC/USDT",
        side="buy",
        entry_price=100000.0,
        amount=0.01,
        stop_loss=98000.0,
        take_profit=104000.0,
        confidence=0.75
    )

    logger_instance.log_trade(
        trade_id="trade_002",
        symbol="ETH/USDT",
        side="buy",
        entry_price=3500.0,
        amount=0.5,
        stop_loss=3430.0,
        take_profit=3640.0,
        confidence=0.68
    )

    # Update trade
    print("\n📝 Updating trade...")
    logger_instance.update_trade(
        trade_id="trade_001",
        exit_price=104000.0,
        profit_loss=400.0,
        reason="Take Profit hit"
    )

    # Log signal
    print("\n📝 Logging signal...")
    logger_instance.log_signal(
        symbol="BNB/USDT",
        action="buy",
        confidence=0.62,
        predicted_probability=0.62,
        reasoning="Test signal",
        features={'rsi': 45.0, 'macd': 0.5},
        executed=False
    )

    # Get statistics
    print("\n📈 Statistics:")
    stats = logger_instance.get_statistics()
    print(f"   Total trades: {stats['total_trades']}")
    print(f"   Open trades: {stats['open_trades']}")
    print(f"   Win rate: {stats['win_rate']:.1%}")
    print(f"   Total P/L: ${stats['total_profit_loss']:.2f}")

    # Get history
    print("\n📜 Trade history:")
    history = logger_instance.get_trade_history(limit=10)
    for trade in history:
        print(f"   {trade['id']}: {trade['symbol']} {trade['side']} @ ${trade['entry_price']}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_trade_logger()
