#!/usr/bin/env python3
"""
🤖💰 Autonomous Trading System - نظام التداول المستقل
يدمج جمع البيانات + التدريب + التنبؤ في نظام مستقل كامل
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.live_market_data_collector import LiveMarketDataCollector, MarketCandle
from src.trading.trading_model_trainer import TradingModelTrainer, TrainingConfig, TrainingResult
from src.trading.trading_predictor import TradingPredictor, TradingSignal, PredictionResult
from src.trading.multi_symbol_trainer import MultiSymbolTrainer, BatchTrainingResult
from src.trading.crypto_symbols import TOP_100_CRYPTO_SYMBOLS, get_top_n_symbols
from src.government.president import President

logger = logging.getLogger(__name__)


@dataclass
class TradingCycleResult:
    """نتيجة دورة تداول كاملة"""
    timestamp: str
    symbols_processed: List[str]
    signals_generated: List[TradingSignal]
    models_trained: int
    predictions_made: int
    total_time: float
    status: str


class AutonomousTradingSystem:
    """
    🤖💰 نظام التداول المستقل

    دورة كاملة:
    1. جمع بيانات السوق الحية
    2. تدريب/تحديث النماذج (كل N أيام)
    3. التنبؤ بالصفقات
    4. استشارة الرئيس (اختياري)
    5. إنتاج إشارات التداول
    """

    def __init__(
        self,
        work_dir: str = "/home/noogh/projects/noogh_unified_system",
        symbols: List[str] = None,
        retrain_interval_days: int = 7,
        president: President = None,
        use_100_symbols: bool = False,
        use_multi_trainer: bool = False
    ):
        self.work_dir = Path(work_dir)

        # Symbol selection: support for 100 coins
        if symbols is not None:
            self.symbols = symbols
        elif use_100_symbols:
            self.symbols = TOP_100_CRYPTO_SYMBOLS.copy()
            logger.info("📊 Using TOP 100 cryptocurrency symbols!")
        else:
            # Default: top 20 for reasonable training time
            self.symbols = get_top_n_symbols(20)
            logger.info("📊 Using top 20 cryptocurrency symbols")

        self.retrain_interval_days = retrain_interval_days
        self.president = president
        self.use_multi_trainer = use_multi_trainer or len(self.symbols) > 10

        # Components
        self.data_collector = LiveMarketDataCollector(
            work_dir=str(self.work_dir),
            symbols=self.symbols
        )

        # Use multi-symbol trainer for large symbol lists
        if self.use_multi_trainer:
            self.multi_trainer = MultiSymbolTrainer(
                work_dir=str(self.work_dir),
                batch_size=5,
                max_workers=3
            )
            logger.info("🚀 Using MultiSymbolTrainer for batch training")
        else:
            self.trainer = TradingModelTrainer(
                work_dir=str(self.work_dir)
            )

        self.predictor = TradingPredictor(
            work_dir=str(self.work_dir)
        )

        # State
        self.last_training_date: Dict[str, datetime] = {}
        self.cycles_completed = 0

        # Stats
        self.total_signals_generated = 0
        self.total_models_trained = 0

        logger.info(f"🤖💰 AutonomousTradingSystem initialized")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   Multi-trainer: {self.use_multi_trainer}")
        logger.info(f"   Retrain interval: {self.retrain_interval_days} days")
        logger.info(f"   President connected: {self.president is not None}")

    def _should_retrain(self, symbol: str) -> bool:
        """هل يجب إعادة تدريب النموذج؟"""

        if symbol not in self.last_training_date:
            return True

        days_since_training = (datetime.now() - self.last_training_date[symbol]).days

        return days_since_training >= self.retrain_interval_days

    async def _consult_president(self, request: str, context: Dict) -> Optional[Dict]:
        """استشارة الرئيس"""
        if not self.president:
            return None
        try:
            response = await self.president.process_request(request, context)
            return response
        except Exception as e:
            logger.warning(f"   ⚠️ President consultation failed: {e}")
            return None

    async def run_trading_cycle(self) -> TradingCycleResult:
        """
        تشغيل دورة تداول كاملة

        Returns:
            TradingCycleResult
        """

        start_time = datetime.now()

        logger.info(f"\n{'='*70}")
        logger.info(f"🤖💰 AUTONOMOUS TRADING CYCLE #{self.cycles_completed + 1}")
        logger.info(f"{'='*70}\n")

        signals_generated = []
        models_trained = 0

        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 1: جمع بيانات السوق
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            logger.info(f"📊 Step 1: Collecting market data...")

            datasets = await self.data_collector.collect_and_prepare_all_symbols(
                days=30,
                sequence_length=60
            )

            logger.info(f"   ✅ Collected data for {len(datasets)} symbols")

            # استشارة الرئيس
            if self.president:
                analysis_advice = await self._consult_president(
                    "Analyze market data",
                    {
                        "summary": f"{len(datasets)} symbols, {self.data_collector.candles_collected} candles",
                        "symbols": self.symbols
                    }
                )
                if analysis_advice:
                    logger.info(f"   🏛️ President advice received")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 2: تدريب/تحديث النماذج
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            logger.info(f"\n🏋️ Step 2: Training/Updating models...")

            for dataset in datasets:
                symbol = dataset['symbol']

                # تحقق: هل يجب إعادة التدريب؟
                if self._should_retrain(symbol):
                    logger.info(f"\n   📈 Training model for {symbol}...")

                    # استشارة الرئيس
                    if self.president:
                        training_advice = await self._consult_president(
                            f"Should I retrain the model for {symbol}?",
                            {
                                "symbol": symbol,
                                "samples": dataset['num_samples']
                            }
                        )
                        if training_advice:
                            logger.info(f"      🏛️ President advice received")

                    # تدريب النموذج
                    training_result = await self.trainer.train(dataset)

                    if training_result.status == "success":
                        logger.info(f"      ✅ Model trained: Accuracy {training_result.accuracy:.2%}")
                        self.last_training_date[symbol] = datetime.now()
                        models_trained += 1
                        self.total_models_trained += 1
                    else:
                        logger.warning(f"      ⚠️ Training failed for {symbol}")
                else:
                    logger.info(f"   ⏭️ Skipping {symbol} (trained recently)")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 3: التنبؤ بالصفقات
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            logger.info(f"\n🔮 Step 3: Generating trading signals...")

            # تحضير بيانات للتنبؤ
            market_data = {}
            for dataset in datasets:
                symbol = dataset['symbol']

                # Load dataset
                loaded = self.data_collector.load_dataset(symbol)
                if not loaded:
                    continue

                # جمع آخر بيانات
                candles = await self.data_collector.collect_historical_data(symbol, days=3)
                if len(candles) < 60:
                    continue

                indicators = self.data_collector.calculate_technical_indicators(candles)

                # تحويل لقواميس
                candles_dict = [asdict(c) for c in candles]
                indicators_dict = [asdict(i) for i in indicators]

                market_data[symbol] = {
                    'candles': candles_dict,
                    'indicators': indicators_dict
                }

            # التنبؤ لكل الرموز
            predictions = await self.predictor.predict_multiple(
                symbols=self.symbols,
                market_data=market_data
            )

            logger.info(f"   ✅ Generated {len(predictions)} predictions")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Step 4: استشارة الرئيس لكل إشارة
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            logger.info(f"\n🏛️💰 Step 4: Consulting President...")

            for pred in predictions:
                if pred.status == "success" and pred.signal:
                    signal = pred.signal

                    # استشارة الرئيس
                    if self.president and signal.action != "hold":
                        finance_advice = await self._consult_president(
                            f"Review trading signal for {signal.symbol}",
                            {
                                "symbol": signal.symbol,
                                "signal": signal.action,
                                "confidence": signal.confidence,
                                "reasoning": signal.reasoning
                            }
                        )

                        if finance_advice:
                            logger.info(f"   🏛️ President reviewed {signal.symbol}: {signal.action}")

                    signals_generated.append(signal)

            self.total_signals_generated += len(signals_generated)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Summary
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            summary = self.predictor.get_signals_summary(predictions)

            logger.info(f"\n{'='*70}")
            logger.info(f"✅ TRADING CYCLE COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"   Models trained: {models_trained}")
            logger.info(f"   Predictions made: {len(predictions)}")
            logger.info(f"   Buy signals: {summary['buy_signals']}")
            logger.info(f"   Sell signals: {summary['sell_signals']}")
            logger.info(f"   Hold signals: {summary['hold_signals']}")
            logger.info(f"   Duration: {(datetime.now() - start_time).total_seconds():.1f}s")
            logger.info(f"{'='*70}\n")

            self.cycles_completed += 1

            return TradingCycleResult(
                timestamp=datetime.now().isoformat(),
                symbols_processed=self.symbols,
                signals_generated=signals_generated,
                models_trained=models_trained,
                predictions_made=len(predictions),
                total_time=(datetime.now() - start_time).total_seconds(),
                status="success"
            )

        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}", exc_info=True)

            return TradingCycleResult(
                timestamp=datetime.now().isoformat(),
                symbols_processed=[],
                signals_generated=[],
                models_trained=0,
                predictions_made=0,
                total_time=0.0,
                status="failed"
            )

    def get_trading_stats(self) -> Dict:
        """إحصائيات نظام التداول"""

        return {
            'cycles_completed': self.cycles_completed,
            'total_signals_generated': self.total_signals_generated,
            'total_models_trained': self.total_models_trained,
            'symbols': self.symbols,
            'data_collector': self.data_collector.get_stats(),
            'trainer': self.trainer.get_stats(),
            'predictor': self.predictor.get_stats()
        }

    async def run_continuous(self, interval_hours: float = 1.0):
        """
        تشغيل مستمر

        Args:
            interval_hours: الفاصل بين الدورات (بالساعات)
        """

        logger.info(f"🤖💰 Starting continuous trading...")
        logger.info(f"   Interval: {interval_hours} hours")
        logger.info(f"   Symbols: {self.symbols}")

        cycle = 0

        while True:
            cycle += 1

            logger.info(f"\n{'━'*70}")
            logger.info(f"🔄 Cycle #{cycle} - {datetime.now()}")
            logger.info(f"{'━'*70}\n")

            # تشغيل دورة
            result = await self.run_trading_cycle()

            if result.status == "success":
                logger.info(f"✅ Cycle #{cycle} successful")
            else:
                logger.warning(f"⚠️ Cycle #{cycle} failed")

            # انتظار
            wait_seconds = interval_hours * 3600
            logger.info(f"\n⏳ Waiting {interval_hours} hours until next cycle...")
            await asyncio.sleep(wait_seconds)


async def main():
    """اختبار النظام المستقل"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("🤖💰 Testing Autonomous Trading System")
    print("="*70 + "\n")

    # Create system
    president = President()
    trading_system = AutonomousTradingSystem(
        symbols=["BTC/USDT", "ETH/USDT"],
        president=president
    )

    # Run one cycle
    result = await trading_system.run_trading_cycle()

    # Display stats
    stats = trading_system.get_trading_stats()

    print("\n" + "="*70)
    print("📊 Trading Stats:")
    print("="*70)
    print(f"Cycles: {stats['cycles_completed']}")
    print(f"Signals: {stats['total_signals_generated']}")
    print(f"Models trained: {stats['total_models_trained']}")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
