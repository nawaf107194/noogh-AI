#!/usr/bin/env python3
"""
💼 Portfolio Allocator - موزع المحفظة
يوزع الرأسمال بشكل ذكي بين العملات المختلفة

Phase 4 - Skeleton Module
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AllocationStrategy:
    """استراتيجية التوزيع"""
    name: str
    description: str
    risk_level: str  # 'low', 'medium', 'high'
    rebalance_frequency: str  # 'hourly', 'daily', 'weekly'


@dataclass
class SymbolAllocation:
    """توزيع رأس المال لعملة معينة"""
    symbol: str
    allocation_pct: float  # Percentage of total capital
    allocation_amount: float  # Actual amount in USD
    confidence_score: float
    expected_return: float
    risk_score: float
    sharpe_ratio: float
    reason: str


class PortfolioAllocator:
    """
    💼 موزع المحفظة

    القدرات:
    - توزيع ذكي للرأسمال
    - استراتيجيات متعددة
    - إعادة توازن تلقائية
    - إدارة المخاطر المتقدمة
    """

    def __init__(
        self,
        total_capital: float,
        strategy: str = "balanced",
        max_symbols: int = 10,
        min_allocation_pct: float = 0.05,  # 5% minimum
        max_allocation_pct: float = 0.20   # 20% maximum
    ):
        """
        تهيئة موزع المحفظة

        Args:
            total_capital: رأس المال الإجمالي
            strategy: استراتيجية التوزيع
            max_symbols: أقصى عدد من العملات
            min_allocation_pct: الحد الأدنى للتوزيع
            max_allocation_pct: الحد الأقصى للتوزيع
        """
        self.total_capital = total_capital
        self.strategy = strategy
        self.max_symbols = max_symbols
        self.min_allocation_pct = min_allocation_pct
        self.max_allocation_pct = max_allocation_pct

        self.current_allocations: Dict[str, SymbolAllocation] = {}
        self.allocation_history: List[Dict] = []

        logger.info(f"💼 PortfolioAllocator initialized")
        logger.info(f"   Total capital: ${total_capital:,.2f}")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Max symbols: {max_symbols}")

    def calculate_allocations(
        self,
        symbols_performance: Dict[str, Dict],
        market_conditions: Optional[Dict] = None
    ) -> List[SymbolAllocation]:
        """
        حساب التوزيع الأمثل للرأسمال

        Args:
            symbols_performance: أداء كل عملة
            market_conditions: ظروف السوق (اختياري)

        Returns:
            List[SymbolAllocation] التوزيع المقترح
        """
        logger.info(f"💼 Calculating optimal allocations...")
        logger.info(f"   Symbols: {len(symbols_performance)}")
        logger.info(f"   Strategy: {self.strategy}")

        # TODO: خوارزميات التوزيع الذكي

        if self.strategy == "balanced":
            allocations = self._balanced_allocation(symbols_performance)
        elif self.strategy == "aggressive":
            allocations = self._aggressive_allocation(symbols_performance)
        elif self.strategy == "conservative":
            allocations = self._conservative_allocation(symbols_performance)
        elif self.strategy == "kelly":
            allocations = self._kelly_criterion_allocation(symbols_performance)
        elif self.strategy == "risk_parity":
            allocations = self._risk_parity_allocation(symbols_performance)
        else:
            allocations = self._balanced_allocation(symbols_performance)

        # التحقق من القيود
        allocations = self._enforce_constraints(allocations)

        # حفظ التوزيع الحالي
        self.current_allocations = {alloc.symbol: alloc for alloc in allocations}

        # حفظ في التاريخ
        self.allocation_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy,
            'allocations': allocations
        })

        logger.info(f"   ✅ Allocations calculated: {len(allocations)} symbols")

        return allocations

    def _balanced_allocation(
        self,
        symbols_performance: Dict[str, Dict]
    ) -> List[SymbolAllocation]:
        """
        توزيع متوازن - يوازن بين العائد والمخاطر

        Args:
            symbols_performance: أداء العملات

        Returns:
            List[SymbolAllocation]
        """
        logger.info(f"   📊 Using balanced allocation strategy...")

        # TODO: خوارزمية التوزيع المتوازن الحقيقية

        # Placeholder: توزيع متساوي مع تعديل بسيط حسب الأداء
        allocations = []

        # ترتيب حسب Sharpe Ratio
        sorted_symbols = sorted(
            symbols_performance.items(),
            key=lambda x: x[1].get('sharpe_ratio', 0),
            reverse=True
        )[:self.max_symbols]

        if not sorted_symbols:
            return []

        # توزيع متساوي
        equal_allocation = 1.0 / len(sorted_symbols)

        for symbol, perf in sorted_symbols:
            sharpe = perf.get('sharpe_ratio', 0)
            win_rate = perf.get('win_rate', 0.5)

            # تعديل بسيط حسب الأداء
            performance_multiplier = (sharpe + win_rate) / 2 if sharpe > 0 else 0.5

            allocation_pct = equal_allocation * performance_multiplier
            allocation_amount = self.total_capital * allocation_pct

            allocations.append(SymbolAllocation(
                symbol=symbol,
                allocation_pct=allocation_pct,
                allocation_amount=allocation_amount,
                confidence_score=win_rate,
                expected_return=perf.get('avg_profit', 0),
                risk_score=perf.get('max_drawdown', 0),
                sharpe_ratio=sharpe,
                reason="Balanced allocation based on risk-adjusted returns"
            ))

        # Normalize to 100%
        total_pct = sum(a.allocation_pct for a in allocations)
        if total_pct > 0:
            for alloc in allocations:
                alloc.allocation_pct /= total_pct
                alloc.allocation_amount = self.total_capital * alloc.allocation_pct

        return allocations

    def _aggressive_allocation(
        self,
        symbols_performance: Dict[str, Dict]
    ) -> List[SymbolAllocation]:
        """
        توزيع عدواني - يركز على العملات ذات العائد الأعلى

        Args:
            symbols_performance: أداء العملات

        Returns:
            List[SymbolAllocation]
        """
        logger.info(f"   🔥 Using aggressive allocation strategy...")

        # TODO: خوارزمية التوزيع العدواني

        # Placeholder: تركيز على أعلى العوائد
        allocations = []

        sorted_symbols = sorted(
            symbols_performance.items(),
            key=lambda x: x[1].get('total_profit_loss', 0),
            reverse=True
        )[:max(3, self.max_symbols // 2)]  # Focus on fewer symbols

        if not sorted_symbols:
            return []

        # توزيع غير متساوي - الأفضل أداء يحصل على أكثر
        weights = [1 / (i + 1) for i in range(len(sorted_symbols))]
        total_weight = sum(weights)

        for (symbol, perf), weight in zip(sorted_symbols, weights):
            allocation_pct = weight / total_weight
            allocation_amount = self.total_capital * allocation_pct

            allocations.append(SymbolAllocation(
                symbol=symbol,
                allocation_pct=allocation_pct,
                allocation_amount=allocation_amount,
                confidence_score=perf.get('win_rate', 0.5),
                expected_return=perf.get('total_profit_loss', 0),
                risk_score=perf.get('max_drawdown', 0),
                sharpe_ratio=perf.get('sharpe_ratio', 0),
                reason="Aggressive allocation focusing on highest returns"
            ))

        return allocations

    def _conservative_allocation(
        self,
        symbols_performance: Dict[str, Dict]
    ) -> List[SymbolAllocation]:
        """
        توزيع محافظ - يركز على تقليل المخاطر

        Args:
            symbols_performance: أداء العملات

        Returns:
            List[SymbolAllocation]
        """
        logger.info(f"   🛡️ Using conservative allocation strategy...")

        # TODO: خوارزمية التوزيع المحافظ

        # Placeholder: تركيز على أقل مخاطر
        allocations = []

        # ترتيب حسب أقل Max Drawdown
        sorted_symbols = sorted(
            symbols_performance.items(),
            key=lambda x: (
                -x[1].get('win_rate', 0),  # Higher win rate first
                x[1].get('max_drawdown', 1)  # Lower drawdown first
            )
        )[:self.max_symbols]

        if not sorted_symbols:
            return []

        # توزيع متساوي بين العملات الأقل مخاطرة
        equal_allocation = 1.0 / len(sorted_symbols)

        for symbol, perf in sorted_symbols:
            allocations.append(SymbolAllocation(
                symbol=symbol,
                allocation_pct=equal_allocation,
                allocation_amount=self.total_capital * equal_allocation,
                confidence_score=perf.get('win_rate', 0.5),
                expected_return=perf.get('avg_profit', 0),
                risk_score=perf.get('max_drawdown', 0),
                sharpe_ratio=perf.get('sharpe_ratio', 0),
                reason="Conservative allocation minimizing drawdown"
            ))

        return allocations

    def _kelly_criterion_allocation(
        self,
        symbols_performance: Dict[str, Dict]
    ) -> List[SymbolAllocation]:
        """
        توزيع Kelly Criterion - نموذج رياضي للتوزيع الأمثل

        Args:
            symbols_performance: أداء العملات

        Returns:
            List[SymbolAllocation]
        """
        logger.info(f"   📐 Using Kelly Criterion allocation strategy...")

        # TODO: خوارزمية Kelly Criterion الحقيقية
        # f* = (p * b - q) / b
        # where:
        # f* = fraction to invest
        # p = probability of win
        # q = probability of loss (1-p)
        # b = ratio of win to loss

        allocations = []

        for symbol, perf in symbols_performance.items():
            win_rate = perf.get('win_rate', 0.5)
            avg_profit = abs(perf.get('avg_profit', 1))
            avg_loss = abs(perf.get('avg_loss', 1))

            if avg_loss == 0 or win_rate == 0:
                continue

            # Kelly fraction
            b = avg_profit / avg_loss  # profit/loss ratio
            p = win_rate
            q = 1 - p

            kelly_fraction = (p * b - q) / b

            # Apply Kelly fraction (with safety factor of 0.5)
            allocation_pct = max(0, min(kelly_fraction * 0.5, self.max_allocation_pct))

            if allocation_pct >= self.min_allocation_pct:
                allocations.append(SymbolAllocation(
                    symbol=symbol,
                    allocation_pct=allocation_pct,
                    allocation_amount=self.total_capital * allocation_pct,
                    confidence_score=win_rate,
                    expected_return=perf.get('avg_profit', 0),
                    risk_score=perf.get('max_drawdown', 0),
                    sharpe_ratio=perf.get('sharpe_ratio', 0),
                    reason=f"Kelly Criterion: {kelly_fraction:.2%} (50% safety factor)"
                ))

        # Sort by allocation and take top symbols
        allocations.sort(key=lambda x: x.allocation_pct, reverse=True)
        allocations = allocations[:self.max_symbols]

        # Normalize
        total_pct = sum(a.allocation_pct for a in allocations)
        if total_pct > 1.0:
            for alloc in allocations:
                alloc.allocation_pct /= total_pct
                alloc.allocation_amount = self.total_capital * alloc.allocation_pct

        return allocations

    def _risk_parity_allocation(
        self,
        symbols_performance: Dict[str, Dict]
    ) -> List[SymbolAllocation]:
        """
        توزيع Risk Parity - كل عملة تساهم بنفس المخاطر

        Args:
            symbols_performance: أداء العملات

        Returns:
            List[SymbolAllocation]
        """
        logger.info(f"   ⚖️ Using Risk Parity allocation strategy...")

        # TODO: خوارزمية Risk Parity الحقيقية

        allocations = []

        # حساب inverse volatility
        symbols_with_risk = []
        for symbol, perf in symbols_performance.items():
            risk = perf.get('max_drawdown', 0.01)
            if risk > 0:
                symbols_with_risk.append((symbol, perf, 1/risk))

        if not symbols_with_risk:
            return []

        # Sort and limit
        symbols_with_risk.sort(key=lambda x: x[2], reverse=True)
        symbols_with_risk = symbols_with_risk[:self.max_symbols]

        # Normalize weights
        total_inv_risk = sum(s[2] for s in symbols_with_risk)

        for symbol, perf, inv_risk in symbols_with_risk:
            allocation_pct = inv_risk / total_inv_risk

            allocations.append(SymbolAllocation(
                symbol=symbol,
                allocation_pct=allocation_pct,
                allocation_amount=self.total_capital * allocation_pct,
                confidence_score=perf.get('win_rate', 0.5),
                expected_return=perf.get('avg_profit', 0),
                risk_score=perf.get('max_drawdown', 0),
                sharpe_ratio=perf.get('sharpe_ratio', 0),
                reason="Risk Parity: Equal risk contribution"
            ))

        return allocations

    def _enforce_constraints(
        self,
        allocations: List[SymbolAllocation]
    ) -> List[SymbolAllocation]:
        """
        تطبيق القيود على التوزيع

        Args:
            allocations: التوزيع المقترح

        Returns:
            التوزيع بعد تطبيق القيود
        """
        logger.info(f"   🔒 Enforcing allocation constraints...")

        # التحقق من الحد الأدنى والأقصى
        filtered_allocations = []

        for alloc in allocations:
            if alloc.allocation_pct < self.min_allocation_pct:
                logger.debug(f"      ❌ {alloc.symbol}: below minimum ({alloc.allocation_pct:.2%})")
                continue

            if alloc.allocation_pct > self.max_allocation_pct:
                logger.debug(f"      ⚠️ {alloc.symbol}: capped at maximum ({self.max_allocation_pct:.2%})")
                alloc.allocation_pct = self.max_allocation_pct
                alloc.allocation_amount = self.total_capital * alloc.allocation_pct

            filtered_allocations.append(alloc)

        # إعادة التوزيع إذا تجاوز 100%
        total_pct = sum(a.allocation_pct for a in filtered_allocations)
        if total_pct > 1.0:
            logger.debug(f"      ⚠️ Total allocation {total_pct:.2%} > 100%, normalizing...")
            for alloc in filtered_allocations:
                alloc.allocation_pct /= total_pct
                alloc.allocation_amount = self.total_capital * alloc.allocation_pct

        logger.info(f"   ✅ Constraints applied: {len(filtered_allocations)} symbols")

        return filtered_allocations

    def suggest_rebalance(
        self,
        current_positions: Dict[str, float],
        target_allocations: List[SymbolAllocation]
    ) -> List[Dict]:
        """
        اقتراح تعديلات لإعادة التوازن

        Args:
            current_positions: المراكز الحالية {symbol: amount_usd}
            target_allocations: التوزيع المستهدف

        Returns:
            List[Dict] اقتراحات التعديل
        """
        logger.info(f"💼 Suggesting rebalance actions...")

        suggestions = []

        # حساب المراكز المستهدفة
        target_positions = {
            alloc.symbol: alloc.allocation_amount
            for alloc in target_allocations
        }

        # مقارنة الحالي مع المستهدف
        all_symbols = set(list(current_positions.keys()) + list(target_positions.keys()))

        for symbol in all_symbols:
            current = current_positions.get(symbol, 0)
            target = target_positions.get(symbol, 0)

            diff = target - current
            diff_pct = abs(diff) / self.total_capital if self.total_capital > 0 else 0

            # اقتراح التعديل فقط إذا كان الفرق كبير (> 2%)
            if diff_pct > 0.02:
                if diff > 0:
                    action = "BUY"
                else:
                    action = "SELL"

                suggestions.append({
                    'symbol': symbol,
                    'action': action,
                    'current_amount': current,
                    'target_amount': target,
                    'adjustment_amount': abs(diff),
                    'adjustment_pct': diff_pct * 100,
                    'reason': f"Rebalance to target allocation"
                })

        suggestions.sort(key=lambda x: x['adjustment_pct'], reverse=True)

        logger.info(f"   ✅ {len(suggestions)} rebalance suggestions")

        return suggestions

    def get_allocation_summary(self) -> Dict:
        """
        الحصول على ملخص التوزيع الحالي

        Returns:
            Dict مع ملخص التوزيع
        """
        if not self.current_allocations:
            return {
                'total_symbols': 0,
                'total_allocated_pct': 0.0,
                'total_allocated_amount': 0.0,
                'allocations': []
            }

        total_pct = sum(a.allocation_pct for a in self.current_allocations.values())
        total_amount = sum(a.allocation_amount for a in self.current_allocations.values())

        return {
            'total_symbols': len(self.current_allocations),
            'total_allocated_pct': total_pct,
            'total_allocated_amount': total_amount,
            'remaining_capital': self.total_capital - total_amount,
            'allocations': sorted(
                self.current_allocations.values(),
                key=lambda x: x.allocation_pct,
                reverse=True
            )
        }


# Test function
def test_portfolio_allocator():
    """اختبار موزع المحفظة"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("💼 Testing Portfolio Allocator")
    print("="*70 + "\n")

    # Create allocator
    allocator = PortfolioAllocator(
        total_capital=10000.0,
        strategy="balanced",
        max_symbols=5
    )

    # Sample performance data
    symbols_performance = {
        'BTC/USDT': {
            'sharpe_ratio': 1.5,
            'win_rate': 0.65,
            'avg_profit': 100,
            'avg_loss': -50,
            'max_drawdown': 0.10,
            'total_profit_loss': 500
        },
        'ETH/USDT': {
            'sharpe_ratio': 1.2,
            'win_rate': 0.60,
            'avg_profit': 80,
            'avg_loss': -40,
            'max_drawdown': 0.12,
            'total_profit_loss': 400
        },
        'BNB/USDT': {
            'sharpe_ratio': 0.8,
            'win_rate': 0.55,
            'avg_profit': 60,
            'avg_loss': -35,
            'max_drawdown': 0.15,
            'total_profit_loss': 200
        }
    }

    # Calculate allocations
    print("💼 Calculating allocations...")
    allocations = allocator.calculate_allocations(symbols_performance)

    print(f"\n📊 Allocation Results:")
    for alloc in allocations:
        print(f"   {alloc.symbol}:")
        print(f"      Allocation: {alloc.allocation_pct:.1%} (${alloc.allocation_amount:.2f})")
        print(f"      Sharpe: {alloc.sharpe_ratio:.2f}")
        print(f"      Reason: {alloc.reason}")

    # Get summary
    print(f"\n💼 Portfolio Summary:")
    summary = allocator.get_allocation_summary()
    print(f"   Total symbols: {summary['total_symbols']}")
    print(f"   Total allocated: {summary['total_allocated_pct']:.1%} (${summary['total_allocated_amount']:.2f})")
    print(f"   Remaining capital: ${summary['remaining_capital']:.2f}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_portfolio_allocator()
