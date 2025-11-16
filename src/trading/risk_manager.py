"""
🛡️ Adaptive Risk Manager - إدارة المخاطر الذكية
Phase 5: Intelligent risk management with dynamic position sizing and stop-loss
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters"""
    risk_per_trade: float = 0.02        # 2% per trade
    max_portfolio_heat: float = 0.06    # 6% total portfolio risk
    max_positions: int = 5              # Maximum simultaneous positions
    atr_multiplier: float = 2.0         # Stop-loss multiplier
    warning_drawdown: float = 0.10      # 10% drawdown warning
    critical_drawdown: float = 0.15     # 15% drawdown critical
    recovery_days: int = 3              # Days needed for recovery


@dataclass
class PositionRisk:
    """Risk calculation for a single position"""
    symbol: str
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    position_size: float        # USD value
    stop_loss: float
    risk_amount: float          # USD at risk
    risk_percentage: float      # % of account


@dataclass
class PortfolioRiskStatus:
    """Current portfolio risk status"""
    total_heat: float               # Current portfolio heat (%)
    open_positions: int
    available_heat: float           # Remaining heat capacity
    drawdown: float                 # Current drawdown
    status: str                     # 'NORMAL', 'WARNING', 'CRITICAL', 'RECOVERED'
    action: str                     # 'CONTINUE', 'REDUCE_SIZE', 'STOP_TRADING', 'RESUME_NORMAL'
    position_size_multiplier: float # Adjustment multiplier


class VolatilityPositionSizer:
    """Calculate position sizes based on market volatility"""

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        atr_multiplier: float = 2.0
    ):
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        atr: float,
        drawdown_multiplier: float = 1.0
    ) -> float:
        """
        Calculate optimal position size based on volatility

        Args:
            account_balance: Current account balance
            current_price: Current asset price
            atr: Average True Range (14-period)
            drawdown_multiplier: Multiplier for drawdown adjustment (0.5 = half size)

        Returns:
            Position size in USD
        """
        if atr <= 0:
            logger.warning("ATR is zero or negative, using default position size")
            return account_balance * self.risk_per_trade * drawdown_multiplier

        # Calculate risk amount
        risk_amount = account_balance * self.risk_per_trade * drawdown_multiplier

        # Calculate stop distance
        stop_distance = atr * self.atr_multiplier

        # Calculate position size in units
        position_units = risk_amount / stop_distance

        # Convert to USD value
        position_value = position_units * current_price

        # Ensure minimum and maximum position sizes
        min_position = account_balance * 0.01  # At least 1%
        max_position = account_balance * 0.20  # At most 20%

        position_value = np.clip(position_value, min_position, max_position)

        return position_value


class DynamicStopLoss:
    """Calculate dynamic stop-loss levels based on ATR"""

    def __init__(self):
        self.atr_multipliers = {
            'LOW': 1.5,      # Low volatility - tighter stops
            'NORMAL': 2.0,   # Normal volatility
            'HIGH': 3.0      # High volatility - wider stops
        }

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_regime: str = 'NORMAL'
    ) -> float:
        """
        Calculate dynamic stop-loss level

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            atr: Average True Range
            volatility_regime: 'LOW', 'NORMAL', or 'HIGH'

        Returns:
            Stop-loss price
        """
        multiplier = self.atr_multipliers.get(volatility_regime, 2.0)
        stop_distance = atr * multiplier

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit level based on risk-reward ratio

        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            direction: 'LONG' or 'SHORT'
            risk_reward_ratio: Target reward/risk ratio

        Returns:
            Take-profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if direction == 'LONG':
            take_profit = entry_price + reward
        else:  # SHORT
            take_profit = entry_price - reward

        return take_profit


class PortfolioHeatManager:
    """Manage total portfolio risk exposure"""

    def __init__(
        self,
        max_portfolio_heat: float = 0.06,
        max_positions: int = 5
    ):
        self.max_portfolio_heat = max_portfolio_heat
        self.max_positions = max_positions

    def can_open_position(
        self,
        account_balance: float,
        open_positions: List[PositionRisk],
        new_position_risk: float
    ) -> Tuple[bool, str]:
        """
        Check if new position is allowed

        Args:
            account_balance: Current account balance
            open_positions: List of currently open positions
            new_position_risk: Risk amount for new position (USD)

        Returns:
            (allowed, reason)
        """
        # Check position count
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Calculate current heat
        current_heat = self.calculate_portfolio_heat(account_balance, open_positions)

        # Calculate new total heat
        new_heat_amount = new_position_risk / account_balance
        total_heat = current_heat + new_heat_amount

        # Check if exceeds limit
        if total_heat > self.max_portfolio_heat:
            return False, (
                f"Portfolio heat too high: "
                f"{total_heat:.1%} > {self.max_portfolio_heat:.1%}"
            )

        return True, "OK"

    def calculate_portfolio_heat(
        self,
        account_balance: float,
        open_positions: List[PositionRisk]
    ) -> float:
        """
        Calculate current portfolio heat (total risk as % of account)

        Args:
            account_balance: Current account balance
            open_positions: List of open positions

        Returns:
            Portfolio heat as decimal (e.g., 0.05 = 5%)
        """
        if not open_positions:
            return 0.0

        total_risk = sum(pos.risk_amount for pos in open_positions)
        heat = total_risk / account_balance if account_balance > 0 else 0.0

        return heat


class DrawdownProtection:
    """Protect against excessive drawdowns"""

    def __init__(
        self,
        warning_threshold: float = 0.10,
        critical_threshold: float = 0.15,
        recovery_days: int = 3
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.recovery_days = recovery_days

    def check_drawdown_status(
        self,
        starting_balance: float,
        current_balance: float,
        recent_daily_returns: List[float]
    ) -> Dict:
        """
        Check drawdown status and determine action

        Args:
            starting_balance: Starting account balance
            current_balance: Current account balance
            recent_daily_returns: List of recent daily returns (as decimals)

        Returns:
            Dictionary with status and action
        """
        # Calculate peak balance
        peak_balance = max(starting_balance, current_balance)

        # Calculate current drawdown
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0

        status = {
            'drawdown': drawdown,
            'drawdown_pct': drawdown * 100,
            'status': 'NORMAL',
            'action': 'CONTINUE',
            'position_size_multiplier': 1.0,
            'message': 'Trading normally'
        }

        # Check critical level
        if drawdown >= self.critical_threshold:
            status['status'] = 'CRITICAL'
            status['action'] = 'STOP_TRADING'
            status['position_size_multiplier'] = 0.0
            status['message'] = (
                f'CRITICAL DRAWDOWN: {drawdown:.1%}. '
                f'All trading stopped until recovery.'
            )

        # Check warning level
        elif drawdown >= self.warning_threshold:
            status['status'] = 'WARNING'
            status['action'] = 'REDUCE_SIZE'
            status['position_size_multiplier'] = 0.5
            status['message'] = (
                f'WARNING: Drawdown at {drawdown:.1%}. '
                f'Position sizes reduced by 50%.'
            )

        # Check for recovery
        if len(recent_daily_returns) >= self.recovery_days:
            consecutive_positive = all(
                r > 0 for r in recent_daily_returns[-self.recovery_days:]
            )

            if consecutive_positive and drawdown < self.warning_threshold:
                status['status'] = 'RECOVERED'
                status['action'] = 'RESUME_NORMAL'
                status['position_size_multiplier'] = 1.0
                status['message'] = (
                    f'RECOVERY: {self.recovery_days} consecutive positive days. '
                    f'Resuming normal trading.'
                )

        return status


class VolatilityRegimeDetector:
    """Detect market volatility regime"""

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period

    def detect_regime(
        self,
        current_atr: float,
        historical_atr: pd.Series
    ) -> str:
        """
        Detect volatility regime

        Args:
            current_atr: Current ATR value
            historical_atr: Historical ATR series

        Returns:
            'LOW', 'NORMAL', or 'HIGH'
        """
        if len(historical_atr) < self.lookback_period:
            return 'NORMAL'

        # Calculate average ATR
        avg_atr = historical_atr.tail(self.lookback_period).mean()

        if avg_atr == 0:
            return 'NORMAL'

        # Calculate ratio
        ratio = current_atr / avg_atr

        if ratio < 0.5:
            return 'LOW'
        elif ratio > 1.5:
            return 'HIGH'
        else:
            return 'NORMAL'

    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period

        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = true_range.rolling(window=period).mean()

        return atr


class RiskManager:
    """Main risk manager integrating all components"""

    def __init__(self, params: RiskParameters = None):
        if params is None:
            params = RiskParameters()

        self.params = params

        # Initialize components
        self.position_sizer = VolatilityPositionSizer(
            risk_per_trade=params.risk_per_trade,
            atr_multiplier=params.atr_multiplier
        )

        self.stop_loss_calculator = DynamicStopLoss()

        self.heat_manager = PortfolioHeatManager(
            max_portfolio_heat=params.max_portfolio_heat,
            max_positions=params.max_positions
        )

        self.drawdown_protector = DrawdownProtection(
            warning_threshold=params.warning_drawdown,
            critical_threshold=params.critical_drawdown,
            recovery_days=params.recovery_days
        )

        self.volatility_detector = VolatilityRegimeDetector()

        logger.info("🛡️ Adaptive Risk Manager initialized")
        logger.info(f"   Risk per trade: {params.risk_per_trade:.1%}")
        logger.info(f"   Max portfolio heat: {params.max_portfolio_heat:.1%}")
        logger.info(f"   Max positions: {params.max_positions}")

    def evaluate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        df: pd.DataFrame,
        account_balance: float,
        starting_balance: float,
        open_positions: List[PositionRisk],
        recent_daily_returns: List[float]
    ) -> Dict:
        """
        Comprehensive trade evaluation

        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Proposed entry price
            df: Price DataFrame with OHLC data
            account_balance: Current account balance
            starting_balance: Starting balance (for drawdown)
            open_positions: Currently open positions
            recent_daily_returns: Recent daily returns

        Returns:
            Dictionary with trade approval and parameters
        """
        result = {
            'approved': False,
            'reason': '',
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_amount': 0.0,
            'risk_percentage': 0.0,
            'volatility_regime': 'NORMAL'
        }

        # 1. Check drawdown status
        drawdown_status = self.drawdown_protector.check_drawdown_status(
            starting_balance,
            account_balance,
            recent_daily_returns
        )

        if drawdown_status['action'] == 'STOP_TRADING':
            result['reason'] = drawdown_status['message']
            return result

        # 2. Calculate ATR
        atr_series = self.volatility_detector.calculate_atr(df)
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.0

        if current_atr <= 0:
            result['reason'] = "Invalid ATR value"
            return result

        # 3. Detect volatility regime
        volatility_regime = self.volatility_detector.detect_regime(
            current_atr,
            atr_series
        )
        result['volatility_regime'] = volatility_regime

        # 4. Calculate position size
        position_multiplier = drawdown_status['position_size_multiplier']
        position_size = self.position_sizer.calculate_position_size(
            account_balance,
            entry_price,
            current_atr,
            drawdown_multiplier=position_multiplier
        )

        # 5. Calculate stop-loss and take-profit
        stop_loss = self.stop_loss_calculator.calculate_stop_loss(
            entry_price,
            direction,
            current_atr,
            volatility_regime
        )

        take_profit = self.stop_loss_calculator.calculate_take_profit(
            entry_price,
            stop_loss,
            direction,
            risk_reward_ratio=2.0
        )

        # 6. Calculate risk amount
        risk_per_unit = abs(entry_price - stop_loss)
        position_units = position_size / entry_price
        risk_amount = risk_per_unit * position_units
        risk_percentage = risk_amount / account_balance if account_balance > 0 else 0.0

        # 7. Check portfolio heat
        can_open, heat_reason = self.heat_manager.can_open_position(
            account_balance,
            open_positions,
            risk_amount
        )

        if not can_open:
            result['reason'] = heat_reason
            return result

        # 8. All checks passed
        result.update({
            'approved': True,
            'reason': 'Trade approved',
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'drawdown_status': drawdown_status,
            'atr': current_atr
        })

        return result

    def get_portfolio_status(
        self,
        account_balance: float,
        starting_balance: float,
        open_positions: List[PositionRisk],
        recent_daily_returns: List[float]
    ) -> PortfolioRiskStatus:
        """Get current portfolio risk status"""

        # Calculate heat
        total_heat = self.heat_manager.calculate_portfolio_heat(
            account_balance,
            open_positions
        )

        available_heat = self.params.max_portfolio_heat - total_heat

        # Check drawdown
        drawdown_status = self.drawdown_protector.check_drawdown_status(
            starting_balance,
            account_balance,
            recent_daily_returns
        )

        return PortfolioRiskStatus(
            total_heat=total_heat,
            open_positions=len(open_positions),
            available_heat=max(0, available_heat),
            drawdown=drawdown_status['drawdown'],
            status=drawdown_status['status'],
            action=drawdown_status['action'],
            position_size_multiplier=drawdown_status['position_size_multiplier']
        )


# Preset risk profiles
RISK_PROFILES = {
    'conservative': RiskParameters(
        risk_per_trade=0.01,
        max_portfolio_heat=0.05,
        max_positions=3,
        atr_multiplier=2.5,
        warning_drawdown=0.08,
        critical_drawdown=0.12
    ),

    'moderate': RiskParameters(
        risk_per_trade=0.02,
        max_portfolio_heat=0.06,
        max_positions=5,
        atr_multiplier=2.0,
        warning_drawdown=0.10,
        critical_drawdown=0.15
    ),

    'aggressive': RiskParameters(
        risk_per_trade=0.03,
        max_portfolio_heat=0.10,
        max_positions=8,
        atr_multiplier=1.5,
        warning_drawdown=0.15,
        critical_drawdown=0.20
    )
}


def create_risk_manager(profile: str = 'moderate') -> RiskManager:
    """
    Create risk manager with preset profile

    Args:
        profile: 'conservative', 'moderate', or 'aggressive'

    Returns:
        RiskManager instance
    """
    params = RISK_PROFILES.get(profile, RISK_PROFILES['moderate'])
    return RiskManager(params)
