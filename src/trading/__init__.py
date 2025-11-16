"""
🤖💰 Autonomous Trading System Module
نظام التداول المستقل الذكي
"""

from .live_market_data_collector import (
    LiveMarketDataCollector,
    MarketCandle,
    TechnicalIndicators
)

from .trading_model_trainer import (
    TradingModelTrainer,
    TrainingConfig,
    TrainingResult,
    LSTMTradingModel,
    TransformerTradingModel
)

from .trading_predictor import (
    TradingPredictor,
    TradingSignal,
    PredictionResult
)

from .multi_symbol_trainer import (
    MultiSymbolTrainer,
    BatchTrainingResult
)

from .crypto_symbols import (
    TOP_100_CRYPTO_SYMBOLS,
    TOP_20_CRYPTO_SYMBOLS,
    get_top_n_symbols
)

from .autonomous_trading_system import (
    AutonomousTradingSystem,
    TradingCycleResult
)

from .binance_connector import (
    BinanceConnector,
    BinanceConfig
)

from .binance_data_collector import (
    BinanceDataCollector
)

from .risk_manager import (
    RiskManager,
    RiskParameters,
    PositionRisk,
    PortfolioRiskStatus,
    VolatilityPositionSizer,
    DynamicStopLoss,
    PortfolioHeatManager,
    DrawdownProtection,
    VolatilityRegimeDetector,
    create_risk_manager,
    RISK_PROFILES
)

from .trade_execution_engine import (
    TradeExecutionEngine,
    ExecutedTrade,
    OrderStatus
)

from .trade_logger import (
    TradeLogger
)

# Phase 4 - Analytics & Intelligence
from .trade_analyzer import (
    TradeAnalyzer,
    PerformanceMetrics,
    SymbolPerformance
)

from .portfolio_allocator import (
    PortfolioAllocator,
    AllocationStrategy,
    SymbolAllocation
)

from .backtesting_engine import (
    BacktestingEngine,
    BacktestResult
)

from .adaptive_learning import (
    AdaptiveLearning,
    RetrainingTrigger
)

__all__ = [
    # Data Collection
    'LiveMarketDataCollector',
    'MarketCandle',
    'TechnicalIndicators',

    # Binance Integration
    'BinanceConnector',
    'BinanceConfig',
    'BinanceDataCollector',

    # Risk Management & Execution
    'RiskManager',
    'RiskParameters',
    'PositionRisk',
    'PortfolioRiskStatus',
    'VolatilityPositionSizer',
    'DynamicStopLoss',
    'PortfolioHeatManager',
    'DrawdownProtection',
    'VolatilityRegimeDetector',
    'create_risk_manager',
    'RISK_PROFILES',
    'TradeExecutionEngine',
    'ExecutedTrade',
    'OrderStatus',
    'TradeLogger',

    # Training
    'TradingModelTrainer',
    'TrainingConfig',
    'TrainingResult',
    'LSTMTradingModel',
    'TransformerTradingModel',

    # Multi-Symbol Training
    'MultiSymbolTrainer',
    'BatchTrainingResult',

    # Symbols
    'TOP_100_CRYPTO_SYMBOLS',
    'TOP_20_CRYPTO_SYMBOLS',
    'get_top_n_symbols',

    # Prediction
    'TradingPredictor',
    'TradingSignal',
    'PredictionResult',

    # Autonomous System
    'AutonomousTradingSystem',
    'TradingCycleResult',

    # Phase 4 - Analytics & Intelligence
    'TradeAnalyzer',
    'PerformanceMetrics',
    'SymbolPerformance',
    'PortfolioAllocator',
    'AllocationStrategy',
    'SymbolAllocation',
    'BacktestingEngine',
    'BacktestResult',
    'AdaptiveLearning',
    'RetrainingTrigger',
]
