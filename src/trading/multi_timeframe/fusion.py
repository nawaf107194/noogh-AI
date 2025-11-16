#!/usr/bin/env python3
"""
🔮 Phase 6: Signal Fusion Module
دمج الإشارات من أطر زمنية متعددة
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .config import TIMEFRAME_CONFIGS, ALIGNMENT_RULES

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe"""

    timeframe: str        # '1h', '4h', '1d'
    direction: str        # 'BUY', 'SELL', 'HOLD'
    confidence: float     # 0.0 - 1.0
    weight: float         # Timeframe weight
    reasoning: str        # Why this signal


@dataclass
class FusedSignal:
    """Fused signal from multiple timeframes"""

    final_direction: str          # Final direction after fusion
    final_confidence: float       # Weighted confidence
    timeframe_signals: List[TimeframeSignal]
    agreement_level: str          # 'ALL_AGREE', 'MAJORITY_AGREE', 'NO_AGREEMENT'
    confidence_boost: float       # Boost factor from alignment
    should_trade: bool            # Final trading decision
    reasoning: str                # Fusion reasoning


class SignalFusion:
    """
    Fuse signals from multiple timeframes into a single trading decision

    Uses weighted averaging with alignment boosting:
    - All agree: +20% confidence boost
    - Majority agrees: No boost
    - No agreement: -30% confidence (no trade)
    """

    def __init__(self):
        """Initialize signal fusion"""
        self.configs = TIMEFRAME_CONFIGS
        self.alignment_rules = ALIGNMENT_RULES

    def fuse_signals(
        self,
        timeframe_signals: Dict[str, TimeframeSignal]
    ) -> FusedSignal:
        """
        Fuse signals from multiple timeframes

        Args:
            timeframe_signals: Dict mapping timeframe to signal

        Returns:
            FusedSignal with final decision

        Example:
            Input: {
                '1h': TimeframeSignal(direction='BUY', confidence=0.70),
                '4h': TimeframeSignal(direction='BUY', confidence=0.65),
                '1d': TimeframeSignal(direction='HOLD', confidence=0.55)
            }

            Output: FusedSignal(
                final_direction='BUY',
                final_confidence=0.65,  # Weighted average
                agreement_level='MAJORITY_AGREE',
                should_trade=True
            )
        """
        logger.info("🔮 Fusing signals from multiple timeframes...")

        # Step 1: Calculate weighted average
        weighted_sum = 0.0
        weight_sum = 0.0

        for tf, signal in timeframe_signals.items():
            config = self.configs[tf]

            # Convert direction to numeric score
            score = self._direction_to_score(signal.direction, signal.confidence)

            weighted_sum += score * config.weight
            weight_sum += config.weight

        # Average score
        avg_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Step 2: Check alignment
        agreement_level = self._check_alignment(timeframe_signals)
        alignment_rule = self.alignment_rules[agreement_level]

        # Step 3: Apply confidence boost
        confidence_boost = alignment_rule['confidence_boost']
        boosted_confidence = abs(avg_score) * confidence_boost

        # Step 4: Determine final direction
        if avg_score > 0.1:
            final_direction = 'BUY'
        elif avg_score < -0.1:
            final_direction = 'SELL'
        else:
            final_direction = 'HOLD'

        # Step 5: Trading decision
        should_trade = (
            final_direction in ['BUY', 'SELL'] and
            boosted_confidence >= 0.60 and
            agreement_level != 'NO_AGREEMENT'
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            timeframe_signals,
            agreement_level,
            final_direction,
            boosted_confidence
        )

        fused = FusedSignal(
            final_direction=final_direction,
            final_confidence=boosted_confidence,
            timeframe_signals=list(timeframe_signals.values()),
            agreement_level=agreement_level,
            confidence_boost=confidence_boost,
            should_trade=should_trade,
            reasoning=reasoning
        )

        logger.info(f"   ✅ Fused signal: {fused.final_direction} ({fused.final_confidence:.1%})")
        logger.info(f"      Alignment: {agreement_level}")
        logger.info(f"      Trade: {should_trade}")

        return fused

    def _direction_to_score(self, direction: str, confidence: float) -> float:
        """
        Convert direction to numeric score

        BUY: +confidence
        SELL: -confidence
        HOLD: 0.0
        """
        if direction == 'BUY':
            return confidence
        elif direction == 'SELL':
            return -confidence
        else:
            return 0.0

    def _check_alignment(
        self,
        timeframe_signals: Dict[str, TimeframeSignal]
    ) -> str:
        """
        Check how many timeframes agree on direction

        Returns:
            'ALL_AGREE', 'MAJORITY_AGREE', or 'NO_AGREEMENT'
        """
        # Count directions
        directions = [sig.direction for sig in timeframe_signals.values()]
        direction_counts = {}

        for direction in directions:
            if direction != 'HOLD':  # Ignore HOLD for agreement
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

        if not direction_counts:
            return 'NO_AGREEMENT'

        max_count = max(direction_counts.values())

        if max_count == len(timeframe_signals):
            return 'ALL_AGREE'
        elif max_count >= 2:
            return 'MAJORITY_AGREE'
        else:
            return 'NO_AGREEMENT'

    def _build_reasoning(
        self,
        timeframe_signals: Dict[str, TimeframeSignal],
        agreement_level: str,
        final_direction: str,
        final_confidence: float
    ) -> str:
        """Build human-readable reasoning"""

        lines = []

        # Individual timeframe signals
        for tf, signal in timeframe_signals.items():
            lines.append(f"{tf}: {signal.direction} ({signal.confidence:.1%})")

        # Alignment
        rule = self.alignment_rules[agreement_level]
        lines.append(f"Alignment: {rule['description']}")

        # Final decision
        lines.append(f"Final: {final_direction} ({final_confidence:.1%})")

        return " | ".join(lines)


# TODO: Implement in Phase 6 Week 2
# - Add custom weighting schemes
# - Add timeframe disagreement handling
# - Add momentum-based weighting
# - Add volatility-adjusted confidence
