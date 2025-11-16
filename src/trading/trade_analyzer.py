#!/usr/bin/env python3
"""
📊 Trade Analyzer - محلل الصفقات
يحلل أداء الصفقات ويحسب المقاييس الأساسية

Phase 4 - Skeleton Module
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """مقاييس الأداء"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    recovery_factor: float
    avg_trade_duration: float  # hours
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int


@dataclass
class SymbolPerformance:
    """أداء عملة معينة"""
    symbol: str
    total_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit_per_trade: float
    sharpe_ratio: float


class TradeAnalyzer:
    """
    📊 محلل الصفقات

    القدرات:
    - تحليل أداء الصفقات
    - حساب مقاييس المخاطر
    - تحليل أداء كل عملة
    - توليد تقارير مفصلة
    """

    def __init__(self, db_path: str = "data/trading/trades.db"):
        """
        تهيئة محلل الصفقات

        Args:
            db_path: مسار قاعدة بيانات الصفقات
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            logger.warning(f"⚠️ Database not found: {self.db_path}")

        logger.info(f"📊 TradeAnalyzer initialized")
        logger.info(f"   Database: {self.db_path}")

    def calculate_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        حساب مقاييس الأداء الشاملة

        Args:
            start_date: تاريخ البداية (اختياري)
            end_date: تاريخ النهاية (اختياري)

        Returns:
            PerformanceMetrics مع جميع المقاييس
        """
        logger.info(f"📊 Calculating performance metrics...")

        # TODO: سيتم ملء هذا بالخوارزميات بعد جمع البيانات الحقيقية

        # Placeholder: قراءة من قاعدة البيانات
        trades = self._get_closed_trades(start_date, end_date)

        if not trades:
            logger.warning("   ⚠️ No closed trades found")
            return self._empty_metrics()

        # حسابات أساسية
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        losing_trades = len([t for t in trades if t['profit_loss'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_profit_loss = sum(t['profit_loss'] for t in trades)

        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] < 0]

        avg_profit = sum(profits) / len(profits) if profits else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

        # TODO: حساب مقاييس متقدمة
        max_drawdown = self._calculate_max_drawdown(trades)
        sharpe_ratio = self._calculate_sharpe_ratio(trades)
        sortino_ratio = self._calculate_sortino_ratio(trades)
        recovery_factor = abs(total_profit_loss / max_drawdown) if max_drawdown != 0 else float('inf')

        # Trade duration
        avg_duration = self._calculate_avg_trade_duration(trades)

        # Best/Worst
        best_trade = max(t['profit_loss'] for t in trades) if trades else 0.0
        worst_trade = min(t['profit_loss'] for t in trades) if trades else 0.0

        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks(trades)

        metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit_loss=total_profit_loss,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            recovery_factor=recovery_factor,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )

        logger.info(f"   ✅ Metrics calculated")
        logger.info(f"      Total trades: {total_trades}")
        logger.info(f"      Win rate: {win_rate:.1%}")
        logger.info(f"      Total P/L: ${total_profit_loss:.2f}")
        logger.info(f"      Sharpe ratio: {sharpe_ratio:.2f}")

        return metrics

    def analyze_symbol_performance(self, symbol: str) -> SymbolPerformance:
        """
        تحليل أداء عملة معينة

        Args:
            symbol: رمز العملة

        Returns:
            SymbolPerformance
        """
        logger.info(f"📊 Analyzing performance for {symbol}...")

        # TODO: تحليل مفصل لكل عملة
        trades = self._get_trades_by_symbol(symbol)

        if not trades:
            logger.warning(f"   ⚠️ No trades found for {symbol}")
            return SymbolPerformance(
                symbol=symbol,
                total_trades=0,
                win_rate=0.0,
                total_profit_loss=0.0,
                avg_profit_per_trade=0.0,
                sharpe_ratio=0.0
            )

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pl = sum(t['profit_loss'] for t in trades)
        avg_pl = total_pl / total_trades if total_trades > 0 else 0.0

        sharpe = self._calculate_sharpe_ratio(trades)

        return SymbolPerformance(
            symbol=symbol,
            total_trades=total_trades,
            win_rate=win_rate,
            total_profit_loss=total_pl,
            avg_profit_per_trade=avg_pl,
            sharpe_ratio=sharpe
        )

    def get_top_performing_symbols(self, limit: int = 10) -> List[SymbolPerformance]:
        """
        الحصول على أفضل العملات أداءً

        Args:
            limit: عدد العملات

        Returns:
            قائمة SymbolPerformance مرتبة
        """
        logger.info(f"📊 Getting top {limit} performing symbols...")

        # TODO: تحليل جميع العملات وترتيبها

        # Placeholder
        symbols = self._get_all_traded_symbols()
        performances = []

        for symbol in symbols:
            perf = self.analyze_symbol_performance(symbol)
            if perf.total_trades > 0:
                performances.append(perf)

        # ترتيب حسب الربح الإجمالي
        performances.sort(key=lambda x: x.total_profit_loss, reverse=True)

        return performances[:limit]

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        توليد تقرير شامل

        Args:
            start_date: تاريخ البداية
            end_date: تاريخ النهاية

        Returns:
            Dict مع التقرير الكامل
        """
        logger.info(f"📊 Generating comprehensive report...")

        metrics = self.calculate_performance_metrics(start_date, end_date)
        top_symbols = self.get_top_performing_symbols()

        report = {
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'overall_metrics': metrics,
            'top_performing_symbols': top_symbols,
            'analysis': {
                'risk_level': self._assess_risk_level(metrics),
                'performance_rating': self._rate_performance(metrics),
                'recommendations': self._generate_recommendations(metrics)
            }
        }

        logger.info(f"   ✅ Report generated")

        return report

    # Helper methods

    def _get_closed_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """الحصول على الصفقات المغلقة"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE status = 'closed'"
            params = []

            if start_date:
                query += " AND exit_timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND exit_timestamp <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
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
            logger.error(f"   ❌ Error getting trades: {e}")
            return []

    def _get_trades_by_symbol(self, symbol: str) -> List[Dict]:
        """الحصول على صفقات عملة معينة"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM trades WHERE symbol = ? AND status = 'closed'",
                (symbol,)
            )

            rows = cursor.fetchall()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'profit_loss': row[9],
                    'entry_timestamp': row[11],
                    'exit_timestamp': row[12]
                })

            return trades

        except Exception as e:
            logger.error(f"   ❌ Error getting trades for {symbol}: {e}")
            return []

    def _get_all_traded_symbols(self) -> List[str]:
        """الحصول على جميع العملات المتداولة"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'closed'")
            rows = cursor.fetchall()
            conn.close()

            return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"   ❌ Error getting symbols: {e}")
            return []

    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """حساب أقصى انخفاض"""
        # TODO: خوارزمية حساب Max Drawdown الحقيقية
        if not trades:
            return 0.0

        # Placeholder
        cumulative_pl = 0.0
        max_pl = 0.0
        max_drawdown = 0.0

        for trade in sorted(trades, key=lambda x: x.get('exit_timestamp', '')):
            cumulative_pl += trade.get('profit_loss', 0)
            max_pl = max(max_pl, cumulative_pl)
            drawdown = max_pl - cumulative_pl
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """حساب Sharpe Ratio"""
        # TODO: خوارزمية Sharpe Ratio الحقيقية
        if not trades:
            return 0.0

        # Placeholder
        returns = [t.get('profit_loss', 0) for t in trades]
        avg_return = sum(returns) / len(returns) if returns else 0.0

        # Standard deviation
        if len(returns) > 1:
            variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        if std_dev == 0:
            return 0.0

        sharpe = (avg_return - risk_free_rate) / std_dev
        return sharpe

    def _calculate_sortino_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """حساب Sortino Ratio"""
        # TODO: خوارزمية Sortino Ratio الحقيقية
        if not trades:
            return 0.0

        # Placeholder - similar to Sharpe but only downside deviation
        returns = [t.get('profit_loss', 0) for t in trades]
        avg_return = sum(returns) / len(returns) if returns else 0.0

        # Downside deviation
        downside_returns = [r for r in returns if r < risk_free_rate]
        if downside_returns:
            downside_variance = sum((r - risk_free_rate) ** 2 for r in downside_returns) / len(downside_returns)
            downside_std = downside_variance ** 0.5
        else:
            return 0.0

        sortino = (avg_return - risk_free_rate) / downside_std
        return sortino

    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """حساب متوسط مدة الصفقة بالساعات"""
        # TODO: خوارزمية حساب المدة
        if not trades:
            return 0.0

        durations = []
        for trade in trades:
            if trade.get('entry_timestamp') and trade.get('exit_timestamp'):
                try:
                    entry = datetime.fromisoformat(trade['entry_timestamp'])
                    exit = datetime.fromisoformat(trade['exit_timestamp'])
                    duration = (exit - entry).total_seconds() / 3600  # hours
                    durations.append(duration)
                except Exception as e:
                    pass

        return sum(durations) / len(durations) if durations else 0.0

    def _calculate_consecutive_streaks(self, trades: List[Dict]) -> tuple:
        """حساب أطول سلسلة انتصارات/خسائر"""
        # TODO: خوارزمية Consecutive Streaks
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in sorted(trades, key=lambda x: x.get('exit_timestamp', '')):
            if trade.get('profit_loss', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _assess_risk_level(self, metrics: PerformanceMetrics) -> str:
        """تقييم مستوى المخاطر"""
        # TODO: خوارزمية تقييم المخاطر
        if metrics.max_drawdown > 0.15:
            return "High Risk"
        elif metrics.max_drawdown > 0.10:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _rate_performance(self, metrics: PerformanceMetrics) -> str:
        """تقييم الأداء"""
        # TODO: خوارزمية تقييم الأداء
        if metrics.sharpe_ratio > 2.0 and metrics.win_rate > 0.6:
            return "Excellent"
        elif metrics.sharpe_ratio > 1.0 and metrics.win_rate > 0.5:
            return "Good"
        elif metrics.win_rate > 0.4:
            return "Fair"
        else:
            return "Poor"

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """توليد توصيات"""
        # TODO: خوارزمية توليد التوصيات
        recommendations = []

        if metrics.win_rate < 0.5:
            recommendations.append("Consider reviewing entry criteria - win rate below 50%")

        if metrics.max_drawdown > 0.15:
            recommendations.append("Reduce position sizes - drawdown exceeds 15%")

        if metrics.profit_factor < 1.5:
            recommendations.append("Improve risk/reward ratio - profit factor is low")

        if metrics.avg_trade_duration > 48:
            recommendations.append("Consider shorter holding periods")

        return recommendations

    def _empty_metrics(self) -> PerformanceMetrics:
        """مقاييس فارغة"""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_profit_loss=0.0,
            avg_profit=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            recovery_factor=0.0,
            avg_trade_duration=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            consecutive_wins=0,
            consecutive_losses=0
        )


# Test function
def test_trade_analyzer():
    """اختبار محلل الصفقات"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("📊 Testing Trade Analyzer")
    print("="*70 + "\n")

    # Create analyzer
    analyzer = TradeAnalyzer(db_path="data/trading/trades.db")

    # Calculate metrics
    print("📊 Calculating performance metrics...")
    metrics = analyzer.calculate_performance_metrics()

    print(f"\n📈 Performance Metrics:")
    print(f"   Total trades: {metrics.total_trades}")
    print(f"   Win rate: {metrics.win_rate:.1%}")
    print(f"   Total P/L: ${metrics.total_profit_loss:.2f}")
    print(f"   Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max drawdown: ${metrics.max_drawdown:.2f}")

    # Get top symbols
    print(f"\n📊 Top performing symbols:")
    top_symbols = analyzer.get_top_performing_symbols(limit=5)
    for i, perf in enumerate(top_symbols, 1):
        print(f"   {i}. {perf.symbol}: ${perf.total_profit_loss:.2f} ({perf.win_rate:.1%} win rate)")

    # Generate report
    print(f"\n📊 Generating comprehensive report...")
    report = analyzer.generate_report()

    print(f"\n📈 Report Summary:")
    print(f"   Risk level: {report['analysis']['risk_level']}")
    print(f"   Performance rating: {report['analysis']['performance_rating']}")
    print(f"   Recommendations: {len(report['analysis']['recommendations'])}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_trade_analyzer()
