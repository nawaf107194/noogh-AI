#!/usr/bin/env python3
"""
⚙️ Phase 6: Multi-Timeframe Configuration
إعدادات الأطر الزمنية المتعددة
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe"""

    interval: str          # Timeframe interval ('1h', '4h', '1d')
    weight: float          # Signal weight (0.0 - 1.0)
    lookback_days: int     # How many days to analyze
    min_confidence: float  # Minimum confidence threshold
    description: str       # Human-readable description


# Standard multi-timeframe configurations
TIMEFRAME_CONFIGS: Dict[str, TimeframeConfig] = {
    '1h': TimeframeConfig(
        interval='1h',
        weight=0.50,           # 50% weight - immediate action
        lookback_days=7,       # 7 days of hourly data
        min_confidence=0.60,   # 60% minimum
        description='Short-term signals for immediate action'
    ),

    '4h': TimeframeConfig(
        interval='4h',
        weight=0.30,           # 30% weight - short-term trend
        lookback_days=30,      # 30 days of 4h data
        min_confidence=0.55,   # 55% minimum
        description='Medium-term trend confirmation'
    ),

    '1d': TimeframeConfig(
        interval='1d',
        weight=0.20,           # 20% weight - long-term context
        lookback_days=90,      # 90 days of daily data
        min_confidence=0.50,   # 50% minimum
        description='Long-term market context'
    )
}


def validate_weights():
    """Validate that weights sum to 1.0"""
    total_weight = sum(config.weight for config in TIMEFRAME_CONFIGS.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"


# Validate on import
validate_weights()


# Alignment rules
ALIGNMENT_RULES = {
    'ALL_AGREE': {
        'required_agreement': 3,  # All 3 timeframes agree
        'confidence_boost': 1.20,  # +20% confidence
        'description': 'All timeframes agree - HIGH confidence'
    },

    'MAJORITY_AGREE': {
        'required_agreement': 2,  # 2 out of 3 agree
        'confidence_boost': 1.00,  # No change
        'description': 'Majority agrees - MEDIUM confidence'
    },

    'NO_AGREEMENT': {
        'required_agreement': 1,  # All disagree
        'confidence_boost': 0.70,  # -30% confidence
        'description': 'No agreement - LOW confidence (NO TRADE)'
    }
}


# Conflict resolution strategies
CONFLICT_STRATEGIES = {
    'CONSERVATIVE': {
        'name': 'Conservative',
        'rule': 'Require all timeframes to agree',
        'description': 'Only trade when all timeframes agree'
    },

    'MODERATE': {
        'name': 'Moderate',
        'rule': 'Require majority agreement (2/3)',
        'description': 'Trade when at least 2 timeframes agree'
    },

    'AGGRESSIVE': {
        'name': 'Aggressive',
        'rule': 'Trade on any strong signal',
        'description': 'Trade on strong signals regardless of alignment'
    }
}
