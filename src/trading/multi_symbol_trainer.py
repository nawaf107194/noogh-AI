#!/usr/bin/env python3
"""
🚀 Multi-Symbol Parallel Trainer - مدرب متعدد العملات المتوازي
يدرب نماذج متعددة بشكل متوازي على GPU لـ 100 عملة
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import time

from .live_market_data_collector import LiveMarketDataCollector
from .trading_model_trainer import TradingModelTrainer, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class BatchTrainingResult:
    """نتيجة تدريب دفعة من النماذج"""
    total_symbols: int
    successful_trainings: int
    failed_trainings: int
    training_time: float
    results: Dict[str, TrainingResult]
    failed_symbols: List[str]


class MultiSymbolTrainer:
    """
    🚀 مدرب متعدد العملات المتوازي

    القدرات:
    - تدريب نماذج متعددة بشكل متوازي
    - إدارة ذاكرة GPU بكفاءة
    - معالجة الأخطاء التلقائية
    - تقارير تقدم مفصلة
    """

    def __init__(
        self,
        work_dir: str = "/home/noogh/projects/noogh_unified_system",
        batch_size: int = 5,  # Number of models to train in parallel
        max_workers: int = 3,  # Max concurrent trainings
        config: TrainingConfig = None
    ):
        self.work_dir = Path(work_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Default training config (faster for 100 models)
        self.config = config or TrainingConfig(
            model_type="lstm",
            hidden_size=64,  # Smaller for speed
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=20,  # Fewer epochs for 100 models
            early_stopping_patience=5
        )

        # Data collector
        self.data_collector = None

        # Stats
        self.total_trained = 0
        self.total_failed = 0

        logger.info(f"🚀 MultiSymbolTrainer initialized")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Config: {self.config.model_type}, {self.config.epochs} epochs")

    async def train_single_symbol(
        self,
        symbol: str,
        trainer: TradingModelTrainer,
        candles_data: Dict,
        indicators_data: Dict
    ) -> Optional[TrainingResult]:
        """
        تدريب نموذج واحد لرمز معين

        Args:
            symbol: الرمز
            trainer: المدرب
            candles_data: بيانات الشموع
            indicators_data: بيانات المؤشرات

        Returns:
            TrainingResult أو None
        """

        try:
            logger.info(f"🎯 Training model for {symbol}...")

            # Prepare dataset
            dataset = self.data_collector.prepare_training_dataset(
                candles=candles_data[symbol],
                indicators=indicators_data[symbol],
                symbol=symbol
            )

            if not dataset or dataset['X'].shape[0] < 100:
                logger.warning(f"   ⚠️ Not enough data for {symbol}")
                return None

            # Train model (symbol is already in the dataset)
            result = await trainer.train(
                dataset=dataset
            )

            if result and result.status == "success":
                logger.info(f"   ✅ {symbol} trained successfully!")
                self.total_trained += 1
                return result
            else:
                logger.warning(f"   ⚠️ {symbol} training failed")
                self.total_failed += 1
                return None

        except Exception as e:
            logger.error(f"   ❌ Error training {symbol}: {e}")
            self.total_failed += 1
            return None

    async def train_batch(
        self,
        symbols: List[str],
        candles_data: Dict,
        indicators_data: Dict
    ) -> Dict[str, TrainingResult]:
        """
        تدريب دفعة من الرموز

        Args:
            symbols: قائمة الرموز
            candles_data: بيانات الشموع لكل رمز
            indicators_data: بيانات المؤشرات لكل رمز

        Returns:
            Dict من النتائج
        """

        results = {}

        # Create a trainer for this batch
        trainer = TradingModelTrainer(
            work_dir=str(self.work_dir),
            config=self.config
        )

        # Train symbols sequentially in this batch
        # (GPU memory limits parallel GPU usage)
        for symbol in symbols:
            if symbol not in candles_data or symbol not in indicators_data:
                logger.warning(f"⚠️ No data for {symbol}, skipping")
                continue

            result = await self.train_single_symbol(
                symbol=symbol,
                trainer=trainer,
                candles_data=candles_data,
                indicators_data=indicators_data
            )

            if result:
                results[symbol] = result

        return results

    async def collect_all_data(
        self,
        symbols: List[str],
        days: int = 30
    ) -> Tuple[Dict, Dict]:
        """
        جمع البيانات لجميع الرموز

        Args:
            symbols: قائمة الرموز
            days: عدد الأيام

        Returns:
            (candles_data, indicators_data)
        """

        logger.info(f"📥 Collecting data for {len(symbols)} symbols...")

        candles_data = {}
        indicators_data = {}

        start_time = time.time()

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"   [{i}/{len(symbols)}] Collecting {symbol}...")

                # Collect historical data
                candles = await self.data_collector.collect_historical_data(
                    symbol=symbol,
                    days=days
                )

                if len(candles) < 100:
                    logger.warning(f"      ⚠️ Not enough data for {symbol} ({len(candles)} candles)")
                    continue

                # Calculate indicators
                indicators = self.data_collector.calculate_technical_indicators(candles)

                candles_data[symbol] = candles
                indicators_data[symbol] = indicators

                logger.info(f"      ✅ {len(candles)} candles collected")

            except Exception as e:
                logger.error(f"      ❌ Failed to collect {symbol}: {e}")
                continue

        collection_time = time.time() - start_time

        logger.info(f"\n📊 Data collection complete!")
        logger.info(f"   Symbols with data: {len(candles_data)}/{len(symbols)}")
        logger.info(f"   Collection time: {collection_time:.1f}s")

        return candles_data, indicators_data

    async def train_all_symbols(
        self,
        symbols: List[str],
        days: int = 30
    ) -> BatchTrainingResult:
        """
        تدريب جميع الرموز

        Args:
            symbols: قائمة الرموز (100 عملة)
            days: عدد الأيام من البيانات

        Returns:
            BatchTrainingResult
        """

        start_time = time.time()

        logger.info("\n" + "="*70)
        logger.info(f"🚀 Starting multi-symbol training for {len(symbols)} symbols")
        logger.info("="*70 + "\n")

        # Initialize data collector
        self.data_collector = LiveMarketDataCollector(
            work_dir=str(self.work_dir),
            symbols=symbols
        )

        # Step 1: Collect all data
        logger.info("📊 Step 1: Collecting market data...")
        candles_data, indicators_data = await self.collect_all_data(symbols, days)

        if not candles_data:
            logger.error("❌ No data collected! Aborting training.")
            return BatchTrainingResult(
                total_symbols=len(symbols),
                successful_trainings=0,
                failed_trainings=len(symbols),
                training_time=0,
                results={},
                failed_symbols=symbols
            )

        # Step 2: Split into batches
        symbols_with_data = list(candles_data.keys())
        batches = [
            symbols_with_data[i:i + self.batch_size]
            for i in range(0, len(symbols_with_data), self.batch_size)
        ]

        logger.info(f"\n🔄 Step 2: Training {len(symbols_with_data)} models in {len(batches)} batches...")

        # Step 3: Train batches
        all_results = {}

        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"\n📦 Batch {batch_idx}/{len(batches)}: {len(batch)} symbols")
            logger.info(f"   Symbols: {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")

            batch_results = await self.train_batch(
                symbols=batch,
                candles_data=candles_data,
                indicators_data=indicators_data
            )

            all_results.update(batch_results)

            logger.info(f"   ✅ Batch complete: {len(batch_results)}/{len(batch)} successful")
            logger.info(f"   📊 Overall progress: {len(all_results)}/{len(symbols_with_data)} models trained")

        # Calculate results
        training_time = time.time() - start_time
        failed_symbols = [s for s in symbols if s not in all_results]

        logger.info("\n" + "="*70)
        logger.info("✅ Multi-symbol training complete!")
        logger.info("="*70)
        logger.info(f"Total symbols: {len(symbols)}")
        logger.info(f"Successful: {len(all_results)}")
        logger.info(f"Failed: {len(failed_symbols)}")
        logger.info(f"Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"Avg time per model: {training_time/max(len(all_results), 1):.1f}s")
        logger.info("="*70 + "\n")

        return BatchTrainingResult(
            total_symbols=len(symbols),
            successful_trainings=len(all_results),
            failed_trainings=len(failed_symbols),
            training_time=training_time,
            results=all_results,
            failed_symbols=failed_symbols
        )

    def get_training_summary(self, result: BatchTrainingResult) -> str:
        """ملخص التدريب"""

        lines = []
        lines.append("\n" + "="*70)
        lines.append("📊 Multi-Symbol Training Summary")
        lines.append("="*70)

        lines.append(f"\n🎯 Overall Results:")
        lines.append(f"   Total symbols: {result.total_symbols}")
        lines.append(f"   ✅ Successful: {result.successful_trainings}")
        lines.append(f"   ❌ Failed: {result.failed_trainings}")
        lines.append(f"   Success rate: {result.successful_trainings/result.total_symbols*100:.1f}%")

        lines.append(f"\n⏱️ Performance:")
        lines.append(f"   Total time: {result.training_time:.1f}s ({result.training_time/60:.1f}m)")
        if result.successful_trainings > 0:
            lines.append(f"   Avg per model: {result.training_time/result.successful_trainings:.1f}s")

        # Top performing models
        if result.results:
            sorted_results = sorted(
                result.results.items(),
                key=lambda x: x[1].accuracy if x[1].accuracy else 0,
                reverse=True
            )[:5]

            lines.append(f"\n🏆 Top 5 Models:")
            for symbol, res in sorted_results:
                lines.append(f"   {symbol}: {res.accuracy:.2%} accuracy")

        if result.failed_symbols:
            lines.append(f"\n⚠️ Failed symbols ({len(result.failed_symbols)}):")
            lines.append(f"   {', '.join(result.failed_symbols[:10])}{'...' if len(result.failed_symbols) > 10 else ''}")

        lines.append("\n" + "="*70)

        return "\n".join(lines)


async def main():
    """اختبار المدرب المتعدد"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    from .crypto_symbols import get_top_n_symbols

    print("\n" + "="*70)
    print("🚀 Testing Multi-Symbol Trainer")
    print("="*70 + "\n")

    # Test with top 10 symbols first
    symbols = get_top_n_symbols(10)

    print(f"Testing with {len(symbols)} symbols:")
    for symbol in symbols:
        print(f"  - {symbol}")

    # Create trainer
    trainer = MultiSymbolTrainer(
        batch_size=3,
        max_workers=2,
        config=TrainingConfig(
            epochs=10,  # Quick test
            hidden_size=64
        )
    )

    # Train all symbols
    result = await trainer.train_all_symbols(symbols, days=30)

    # Print summary
    print(trainer.get_training_summary(result))

    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
