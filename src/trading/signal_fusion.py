#!/usr/bin/env python3
"""
🎯 Advanced Signal Fusion System
نظام دمج الإشارات المتقدم

Combines multiple signal sources with weighted scoring:
- ML Predictions (40%)
- Multi-Timeframe Agreement (25%)
- Pattern Recognition Boost (15%)
- Volume Confirmation (10%)
- Momentum Indicators (10%)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)


class SignalSide(Enum):
    """Signal direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class FusedSignal:
    """
    Fused signal with comprehensive scoring

    Attributes:
        symbol: Trading symbol
        side: LONG, SHORT, or HOLD
        score: Final fusion score (0-1)
        confidence: Overall confidence
        components: Individual component scores
        reasons: List of reasons for the signal
        threshold_used: Threshold that was applied
        adaptive_reduction: Amount of threshold reduction applied
        opportunistic: Whether opportunistic gate was triggered
    """
    symbol: str
    side: SignalSide
    score: float
    confidence: float
    components: Dict[str, float]
    reasons: List[str]
    threshold_used: float = 0.70
    adaptive_reduction: float = 0.0
    opportunistic: bool = False

    def is_tradeable(self, threshold: float = 0.70) -> bool:
        """Check if signal meets minimum threshold"""
        return self.score >= threshold


class AdvancedSignalFusion:
    """
    Advanced signal fusion with configurable weights

    Combines multiple signal sources into a single unified score
    """

    def __init__(
        self,
        ml_weight: float = 0.40,
        mtf_weight: float = 0.25,
        pattern_weight: float = 0.15,
        volume_weight: float = 0.10,
        momentum_weight: float = 0.10,
        config_path: Optional[str] = None
    ):
        """
        Initialize fusion system with configurable weights

        Args:
            ml_weight: Weight for ML predictions (default: 0.40)
            mtf_weight: Weight for multi-timeframe agreement (default: 0.25)
            pattern_weight: Weight for pattern recognition (default: 0.15)
            volume_weight: Weight for volume confirmation (default: 0.10)
            momentum_weight: Weight for momentum indicators (default: 0.10)
            config_path: Path to YAML config file (optional)
        """
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}

        # Apply config or use defaults
        if self.config and 'weights' in self.config:
            weights = self.config['weights']
            self.ml_weight = weights.get('ml', ml_weight)
            self.mtf_weight = weights.get('mtf', mtf_weight)
            self.pattern_weight = weights.get('pattern', pattern_weight)
            self.volume_weight = weights.get('volume', volume_weight)
            self.momentum_weight = weights.get('momentum', momentum_weight)
        else:
            self.ml_weight = ml_weight
            self.mtf_weight = mtf_weight
            self.pattern_weight = pattern_weight
            self.volume_weight = volume_weight
            self.momentum_weight = momentum_weight

        # Daily caps tracker
        self.daily_trades = {'LONG': [], 'SHORT': []}
        self.last_reset = datetime.now().date()

        # Validate weights sum to 1.0
        total_weight = sum([self.ml_weight, self.mtf_weight, self.pattern_weight,
                           self.volume_weight, self.momentum_weight])
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"⚠️ Weights sum to {total_weight}, normalizing to 1.0")
            self._normalize_weights()

        logger.info("✅ Advanced Signal Fusion initialized")
        logger.info(f"   Weights: ML={self.ml_weight:.2f}, MTF={self.mtf_weight:.2f}, "
                   f"Pattern={self.pattern_weight:.2f}, Vol={self.volume_weight:.2f}, "
                   f"Mom={self.momentum_weight:.2f}")

        if self.config:
            logger.info(f"   Config loaded: {config_path or 'default'}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}

            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Apply active profile if specified
            if 'active_profile' in config and config['active_profile'] != 'custom':
                profile_name = config['active_profile']
                if profile_name in config.get('profiles', {}):
                    profile = config['profiles'][profile_name]
                    # Merge profile into main config
                    for key, value in profile.items():
                        if isinstance(value, dict) and key in config:
                            config[key].update(value)
                        else:
                            config[key] = value

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum([self.ml_weight, self.mtf_weight, self.pattern_weight,
                    self.volume_weight, self.momentum_weight])

        self.ml_weight /= total
        self.mtf_weight /= total
        self.pattern_weight /= total
        self.volume_weight /= total
        self.momentum_weight /= total

    def _adaptive_threshold(
        self,
        side: SignalSide,
        base_threshold: float,
        mtf_agreement: float,
        volume_confirm: float,
        pattern_boost: float,
        momentum_score: float
    ) -> Tuple[float, float]:
        """
        Apply adaptive threshold reduction based on conditions

        Returns:
            Tuple of (adjusted_threshold, reduction_amount)
        """
        if not self.config.get('adaptive_threshold', {}).get('enabled', False):
            return base_threshold, 0.0

        reduction = 0.0

        if side == SignalSide.LONG:
            # LONG reduction conditions
            long_cfg = self.config.get('adaptive_threshold', {}).get('long_reduction', {})
            conditions = long_cfg.get('conditions', {})

            if (mtf_agreement >= conditions.get('mtf_min', 0.6) and
                volume_confirm >= conditions.get('volume_min', 0.6)):
                reduction = long_cfg.get('amount', 0.02)

        elif side == SignalSide.SHORT:
            # SHORT reduction conditions
            short_cfg = self.config.get('adaptive_threshold', {}).get('short_reduction', {})
            conditions = short_cfg.get('conditions', {})

            if (pattern_boost >= conditions.get('pattern_min', 0.15) and
                momentum_score <= conditions.get('momentum_max', 0.4)):
                reduction = short_cfg.get('amount', 0.02)

        # Apply safety floor
        safety_floor = self.config.get('thresholds', {}).get('safety_floor', 0.60)
        adjusted = max(base_threshold - reduction, safety_floor)

        return adjusted, reduction

    def _check_opportunistic_gate(
        self,
        ml_confidence: float,
        pattern_boost: float,
        final_score: float,
        threshold: float
    ) -> bool:
        """
        Check if signal qualifies for opportunistic gate

        Returns:
            True if gate should be opened
        """
        if not self.config.get('opportunistic_gate', {}).get('enabled', False):
            return False

        conditions = self.config.get('opportunistic_gate', {}).get('conditions', {})

        ml_min = conditions.get('ml_conf_min', 0.78)
        pattern_min = conditions.get('pattern_min', 0.20)
        gap_max = conditions.get('score_gap_max', 0.02)

        if (ml_confidence >= ml_min and
            pattern_boost >= pattern_min and
            (threshold - final_score) <= gap_max):
            return True

        return False

    def _check_daily_cap(self, side: SignalSide) -> bool:
        """
        Check if daily cap has been reached for this side

        Returns:
            True if cap not reached, False if cap reached
        """
        if not self.config.get('daily_caps', {}).get('enabled', False):
            return True

        # Reset daily trades if new day
        self._reset_daily_trades_if_needed()

        side_str = side.value
        current_count = len(self.daily_trades.get(side_str, []))

        if side_str == 'LONG':
            max_trades = self.config.get('daily_caps', {}).get('max_long_trades', 6)
        else:  # SHORT
            max_trades = self.config.get('daily_caps', {}).get('max_short_trades', 4)

        return current_count < max_trades

    def _reset_daily_trades_if_needed(self):
        """Reset daily trade counters if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_trades = {'LONG': [], 'SHORT': []}
            self.last_reset = current_date
            logger.info(f"   🔄 Daily trade counters reset")

    def fuse_signals(
        self,
        symbol: str,
        ml_confidence: float,
        ml_action: str,
        mtf_agreement: float,
        mtf_trend: str,
        pattern_boost: float = 0.0,
        pattern_type: Optional[str] = None,
        volume_confirm: float = 0.0,
        momentum_score: float = 0.5,
        **kwargs
    ) -> FusedSignal:
        """
        Fuse multiple signal sources into unified score

        Args:
            symbol: Trading symbol
            ml_confidence: ML prediction confidence (0-1)
            ml_action: ML action (buy/sell/hold)
            mtf_agreement: Multi-timeframe agreement score (0-1)
            mtf_trend: Overall trend from MTF analysis
            pattern_boost: Pattern recognition boost (0-1)
            pattern_type: Type of pattern detected
            volume_confirm: Volume confirmation score (0-1)
            momentum_score: Momentum indicator score (0-1)

        Returns:
            FusedSignal with comprehensive scoring
        """

        # Normalize ML confidence to 0-1
        ml_score = self._normalize_ml_confidence(ml_confidence, ml_action)

        # Calculate weighted components
        components = {
            'ml': ml_score * self.ml_weight,
            'mtf': mtf_agreement * self.mtf_weight,
            'pattern': pattern_boost * self.pattern_weight,
            'volume': volume_confirm * self.volume_weight,
            'momentum': momentum_score * self.momentum_weight
        }

        # Calculate final fusion score
        final_score = sum(components.values())

        # Determine signal side
        side = self._determine_side(ml_action, mtf_trend, final_score)

        # Apply trend filter penalty if trading against major trend
        final_score, trend_penalty = self._apply_trend_filter(
            final_score, side, mtf_trend
        )

        # Get base threshold from config or use defaults
        if self.config and 'thresholds' in self.config:
            thresholds_cfg = self.config['thresholds']
            base_threshold = thresholds_cfg.get('long_base', 0.70) if side == SignalSide.LONG else \
                           thresholds_cfg.get('short_base', 0.72)
        else:
            base_threshold = 0.70 if side == SignalSide.LONG else 0.72

        # Apply adaptive threshold reduction
        threshold, reduction = self._adaptive_threshold(
            side, base_threshold,
            mtf_agreement, volume_confirm, pattern_boost, momentum_score
        )

        # Check opportunistic gate
        opportunistic = self._check_opportunistic_gate(
            ml_confidence, pattern_boost, final_score, threshold
        )

        # Check daily cap
        within_daily_cap = self._check_daily_cap(side)

        # Build reasons list
        reasons = self._build_reasons(
            ml_action, ml_confidence, mtf_agreement, mtf_trend,
            pattern_boost, pattern_type, volume_confirm, trend_penalty
        )

        # Add adaptive threshold info to reasons
        if reduction > 0:
            reasons.append(f"⬇️ Threshold reduced: {base_threshold:.3f} → {threshold:.3f}")

        # Add opportunistic gate info
        if opportunistic:
            reasons.append(f"⚡ Opportunistic gate: ML={ml_confidence:.1%}, Pattern={pattern_boost:.1%}")

        # Add daily cap status
        if not within_daily_cap:
            reasons.append(f"🚫 Daily cap reached for {side.value}")

        # Create fused signal
        fused = FusedSignal(
            symbol=symbol,
            side=side,
            score=final_score,
            confidence=ml_confidence,
            components=components,
            reasons=reasons,
            threshold_used=threshold,
            adaptive_reduction=reduction,
            opportunistic=opportunistic
        )

        # Log with detailed breakdown if enabled
        if self.config.get('logging', {}).get('detailed_breakdown', False):
            logger.info(f"   🎯 {symbol}: {side.value} score={final_score:.3f}/{threshold:.3f} | "
                       f"ML={components['ml']:.3f} MTF={components['mtf']:.3f} "
                       f"Pat={components['pattern']:.3f} Vol={components['volume']:.3f} "
                       f"Mom={components['momentum']:.3f}")

            if reduction > 0 or opportunistic or not within_daily_cap:
                logger.info(f"      🔍 Adaptive: threshold_reduction={reduction:.3f}, "
                           f"opportunistic={opportunistic}, daily_cap_ok={within_daily_cap}")

        # Record trade if approved for daily cap tracking
        if final_score >= threshold and within_daily_cap:
            self.daily_trades[side.value].append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'score': final_score
            })

        return fused

    def _normalize_ml_confidence(self, confidence: float, action: str) -> float:
        """
        Normalize ML confidence to 0-1 scale

        For buy/sell actions, use confidence directly
        For hold actions, return neutral score (0.5)
        """
        if action.lower() in ['hold', 'neutral']:
            return 0.5

        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))

    def _determine_side(
        self,
        ml_action: str,
        mtf_trend: str,
        score: float
    ) -> SignalSide:
        """
        Determine signal side based on ML action ONLY

        Score threshold check happens later - don't pre-filter here!

        Args:
            ml_action: ML prediction (buy/sell/hold)
            mtf_trend: Multi-timeframe trend
            score: Fusion score (not used for side determination)

        Returns:
            SignalSide enum
        """
        action_lower = ml_action.lower()

        if action_lower in ['buy', 'long']:
            return SignalSide.LONG
        elif action_lower in ['sell', 'short']:
            return SignalSide.SHORT
        else:
            return SignalSide.HOLD

    def _apply_trend_filter(
        self,
        score: float,
        side: SignalSide,
        mtf_trend: str
    ) -> Tuple[float, bool]:
        """
        Apply multi-timeframe trend filter

        Penalize signals that go against major trend

        Args:
            score: Current fusion score
            side: Signal side (LONG/SHORT)
            mtf_trend: Overall trend (UP/DOWN/NEUTRAL)

        Returns:
            Tuple of (adjusted_score, penalty_applied)
        """
        penalty_applied = False

        # Don't trade LONG against strong downtrend
        if side == SignalSide.LONG and mtf_trend == "DOWN":
            score *= 0.75  # 25% penalty
            penalty_applied = True

        # Don't trade SHORT against strong uptrend
        elif side == SignalSide.SHORT and mtf_trend == "UP":
            score *= 0.75  # 25% penalty
            penalty_applied = True

        return score, penalty_applied

    def _build_reasons(
        self,
        ml_action: str,
        ml_confidence: float,
        mtf_agreement: float,
        mtf_trend: str,
        pattern_boost: float,
        pattern_type: Optional[str],
        volume_confirm: float,
        trend_penalty: bool
    ) -> List[str]:
        """Build list of reasons for the signal"""
        reasons = []

        # ML reason
        reasons.append(f"ML: {ml_action.upper()} ({ml_confidence:.1%})")

        # MTF reason
        if mtf_agreement > 0.7:
            reasons.append(f"MTF: Strong agreement ({mtf_agreement:.1%})")
        elif mtf_agreement > 0.5:
            reasons.append(f"MTF: Moderate agreement ({mtf_agreement:.1%})")

        reasons.append(f"Trend: {mtf_trend}")

        # Pattern reason
        if pattern_boost > 0:
            pattern_str = pattern_type if pattern_type else "detected"
            reasons.append(f"Pattern: {pattern_str} ({pattern_boost:.1%} boost)")

        # Volume reason
        if volume_confirm > 0.5:
            reasons.append(f"Volume: Confirmed ({volume_confirm:.1%})")

        # Trend penalty
        if trend_penalty:
            reasons.append("⚠️ Trend filter penalty applied")

        return reasons

    def get_thresholds(
        self,
        side: SignalSide,
        aggressive: bool = False
    ) -> float:
        """
        Get score threshold for signal acceptance

        Args:
            side: Signal side (LONG/SHORT)
            aggressive: Use aggressive (lower) thresholds

        Returns:
            Score threshold (0-1)
        """
        if aggressive:
            # Aggressive mode: lower thresholds
            return 0.65 if side == SignalSide.LONG else 0.67
        else:
            # Conservative mode: higher thresholds
            # Shorts require higher threshold due to different risk profile
            return 0.70 if side == SignalSide.LONG else 0.72


# Factory function for easy instantiation
def create_fusion_system(**kwargs) -> AdvancedSignalFusion:
    """
    Create signal fusion system with optional custom weights

    Example:
        fusion = create_fusion_system(ml_weight=0.50, pattern_weight=0.20)
    """
    return AdvancedSignalFusion(**kwargs)
