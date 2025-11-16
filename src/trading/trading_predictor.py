#!/usr/bin/env python3
"""
🔮 Trading Predictor - المتنبئ بالصفقات
يستخدم النماذج المدربة للتنبؤ بالصفقات في الوقت الفعلي
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PyTorch not available - trading predictor will not work")

from .live_market_data_collector import MarketCandle, TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """إشارة تداول"""
    symbol: str
    action: str  # buy, sell, hold
    confidence: float  # 0-1
    predicted_probability: float  # احتمالية الارتفاع
    timestamp: str
    features: Dict
    reasoning: str


@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    symbol: str
    signal: TradingSignal
    model_used: str
    prediction_time: float
    status: str


class TradingPredictor:
    """
    🔮 المتنبئ بالصفقات

    القدرات:
    - تحميل النماذج المدربة
    - التنبؤ بالصفقات من البيانات الحية
    - حساب الثقة
    - إنتاج إشارات تداول
    """

    def __init__(
        self,
        work_dir: str = "/home/noogh/projects/noogh_unified_system",
        confidence_threshold: float = 0.6
    ):
        self.work_dir = Path(work_dir)
        self.models_dir = self.work_dir / "models" / "trading"
        self.confidence_threshold = confidence_threshold

        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"  # Auto-detect GPU

        # Loaded models cache
        self.loaded_models: Dict[str, nn.Module] = {}

        # Stats
        self.predictions_made = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0

        logger.info(f"🔮 TradingPredictor initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Confidence threshold: {self.confidence_threshold}")
        logger.info(f"   Models dir: {self.models_dir}")

        if not TORCH_AVAILABLE:
            logger.error("❌ PyTorch not available!")
            raise ImportError("PyTorch is required for TradingPredictor")

    def load_model(self, symbol: str) -> Optional[nn.Module]:
        """
        تحميل النموذج لرمز معين

        Args:
            symbol: BTC/USDT, ETH/USDT, etc.

        Returns:
            النموذج المحمل أو None
        """
        # Check cache
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        # Find latest model file
        symbol_clean = symbol.replace("/", "_")
        model_files = list(self.models_dir.glob(f"{symbol_clean}_model_*.pth"))

        if not model_files:
            logger.warning(f"⚠️ No model found for {symbol}")
            return None

        # Get latest
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        try:
            # Load checkpoint
            checkpoint = torch.load(latest_model, map_location=self.device)

            # Recreate model from config
            config = checkpoint['config']
            model_type = config['model_type']

            # Import model classes
            from .trading_model_trainer import LSTMTradingModel, TransformerTradingModel

            # Assume input size (will be in checkpoint if we saved it)
            # For now, use default: 14 features
            input_size = 14

            if model_type == "lstm":
                model = LSTMTradingModel(
                    input_size=input_size,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                )
            elif model_type == "transformer":
                model = TransformerTradingModel(
                    input_size=input_size,
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Cache
            self.loaded_models[symbol] = model

            logger.info(f"   ✅ Loaded model for {symbol} from {latest_model.name}")

            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model for {symbol}: {e}")
            return None

    def preprocess_input(
        self,
        candles: List[Dict],
        indicators: List[Dict],
        sequence_length: int = 60
    ) -> Optional[np.ndarray]:
        """
        معالجة البيانات المدخلة

        Args:
            candles: آخر N شمعة
            indicators: المؤشرات الفنية
            sequence_length: طول التسلسل المطلوب

        Returns:
            Array جاهز للتنبؤ (1, seq_len, features)
        """

        if len(candles) < sequence_length or len(indicators) < sequence_length:
            logger.warning(f"⚠️ Not enough data: {len(candles)} candles, need {sequence_length}")
            return None

        # Get last sequence_length items
        candles = candles[-sequence_length:]
        indicators = indicators[-sequence_length:]

        # Combine features
        features_list = []
        for candle, ind in zip(candles, indicators):
            features = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
                'rsi': ind['rsi'],
                'macd': ind['macd'],
                'macd_signal': ind['macd_signal'],
                'bb_upper': ind['bb_upper'],
                'bb_middle': ind['bb_middle'],
                'bb_lower': ind['bb_lower'],
                'ema_9': ind['ema_9'],
                'ema_21': ind['ema_21'],
                'ema_50': ind['ema_50']
            }
            features_list.append(features)

        df = pd.DataFrame(features_list)

        # Normalize (simple z-score) with epsilon to avoid division by zero
        mean = df.mean()
        std = df.std()
        std = std.replace(0, 1e-8)  # Replace 0 with small value
        df_normalized = (df - mean) / std

        # Convert to numpy
        X = df_normalized.values
        X = np.expand_dims(X, axis=0)  # Add batch dimension

        return X.astype(np.float32)

    async def predict(
        self,
        symbol: str,
        candles: List[Dict],
        indicators: List[Dict],
        sequence_length: int = 60
    ) -> Optional[PredictionResult]:
        """
        التنبؤ بالصفقة

        Args:
            symbol: الرمز
            candles: الشموع الأخيرة
            indicators: المؤشرات الفنية
            sequence_length: طول التسلسل

        Returns:
            PredictionResult
        """

        start_time = datetime.now()

        logger.info(f"🔮 Predicting for {symbol}...")

        try:
            # 1. Load model
            model = self.load_model(symbol)
            if model is None:
                logger.warning(f"   ⚠️ No model available for {symbol}")
                return None

            # 2. Preprocess input
            X = self.preprocess_input(candles, indicators, sequence_length)
            if X is None:
                logger.warning(f"   ⚠️ Failed to preprocess data")
                return None

            # 3. Predict
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            # 4. Create signal
            action = "buy" if predicted_class == 1 else "sell"

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                action = "hold"
                self.hold_signals += 1
            elif action == "buy":
                self.buy_signals += 1
            else:
                self.sell_signals += 1

            # Reasoning
            buy_prob = probabilities[0, 1].item()
            reasoning = self._generate_reasoning(
                action,
                confidence,
                buy_prob,
                candles[-1],
                indicators[-1]
            )

            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                predicted_probability=buy_prob,
                timestamp=datetime.now().isoformat(),
                features={
                    'last_close': candles[-1]['close'],
                    'last_rsi': indicators[-1]['rsi'],
                    'last_macd': indicators[-1]['macd']
                },
                reasoning=reasoning
            )

            # Prediction time
            prediction_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"   ✅ Prediction: {action.upper()}")
            logger.info(f"      Confidence: {confidence:.2%}")
            logger.info(f"      Buy probability: {buy_prob:.2%}")
            logger.info(f"      Time: {prediction_time*1000:.1f}ms")

            self.predictions_made += 1

            return PredictionResult(
                symbol=symbol,
                signal=signal,
                model_used=model.__class__.__name__,
                prediction_time=prediction_time,
                status="success"
            )

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            return PredictionResult(
                symbol=symbol,
                signal=None,
                model_used="",
                prediction_time=0.0,
                status="failed"
            )

    def _generate_reasoning(
        self,
        action: str,
        confidence: float,
        buy_prob: float,
        last_candle: Dict,
        last_indicator: Dict
    ) -> str:
        """توليد تفسير للقرار"""

        reasoning_parts = []

        # Action
        if action == "buy":
            reasoning_parts.append(f"📈 إشارة شراء (احتمال ارتفاع: {buy_prob:.1%})")
        elif action == "sell":
            reasoning_parts.append(f"📉 إشارة بيع (احتمال انخفاض: {1-buy_prob:.1%})")
        else:
            reasoning_parts.append(f"⏸️ انتظار (ثقة منخفضة: {confidence:.1%})")

        # RSI
        rsi = last_indicator['rsi']
        if rsi > 70:
            reasoning_parts.append(f"RSI مرتفع ({rsi:.1f}) - منطقة تشبع شرائي")
        elif rsi < 30:
            reasoning_parts.append(f"RSI منخفض ({rsi:.1f}) - منطقة تشبع بيعي")

        # MACD
        macd = last_indicator['macd']
        macd_signal = last_indicator['macd_signal']
        if macd > macd_signal:
            reasoning_parts.append(f"MACD إيجابي (صاعد)")
        else:
            reasoning_parts.append(f"MACD سلبي (هابط)")

        # Price vs EMA
        price = last_candle['close']
        ema_21 = last_indicator['ema_21']
        if price > ema_21:
            reasoning_parts.append(f"السعر فوق EMA21 (اتجاه صاعد)")
        else:
            reasoning_parts.append(f"السعر تحت EMA21 (اتجاه هابط)")

        return " | ".join(reasoning_parts)

    async def predict_multiple(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict]
    ) -> List[PredictionResult]:
        """
        التنبؤ لعدة رموز

        Args:
            symbols: قائمة الرموز
            market_data: بيانات السوق لكل رمز
                {"BTC/USDT": {"candles": [...], "indicators": [...]}}

        Returns:
            قائمة PredictionResult
        """

        logger.info(f"🔮 Predicting for {len(symbols)} symbols...")

        results = []

        for symbol in symbols:
            if symbol not in market_data:
                logger.warning(f"   ⚠️ No data for {symbol}")
                continue

            data = market_data[symbol]
            result = await self.predict(
                symbol=symbol,
                candles=data['candles'],
                indicators=data['indicators']
            )

            if result:
                results.append(result)

        logger.info(f"   ✅ Predictions complete: {len(results)} signals")

        return results

    def get_signals_summary(self, results: List[PredictionResult]) -> Dict:
        """ملخص الإشارات"""

        buy_signals = [r for r in results if r.signal and r.signal.action == "buy"]
        sell_signals = [r for r in results if r.signal and r.signal.action == "sell"]
        hold_signals = [r for r in results if r.signal and r.signal.action == "hold"]

        return {
            'total_predictions': len(results),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'hold_signals': len(hold_signals),
            'buy_symbols': [r.symbol for r in buy_signals],
            'sell_symbols': [r.symbol for r in sell_signals],
            'top_buy_confidence': max([r.signal.confidence for r in buy_signals], default=0.0),
            'top_sell_confidence': max([r.signal.confidence for r in sell_signals], default=0.0)
        }

    def get_stats(self) -> Dict:
        """إحصائيات المتنبئ"""
        return {
            'predictions_made': self.predictions_made,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            'hold_signals': self.hold_signals,
            'loaded_models': len(self.loaded_models),
            'device': self.device
        }


async def main():
    """اختبار المتنبئ"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("🔮 Testing Trading Predictor")
    print("="*70 + "\n")

    # Create dummy data
    candles = []
    indicators = []

    base_price = 40000.0
    for i in range(60):
        price = base_price * (1 + np.random.randn() * 0.01)
        candles.append({
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': 1000000
        })
        indicators.append({
            'rsi': 50 + np.random.randn() * 10,
            'macd': np.random.randn() * 100,
            'macd_signal': np.random.randn() * 100,
            'bb_upper': price * 1.02,
            'bb_middle': price,
            'bb_lower': price * 0.98,
            'ema_9': price,
            'ema_21': price,
            'ema_50': price
        })

    # Create predictor
    predictor = TradingPredictor()

    # Note: This will fail if no model exists, which is expected
    # In real usage, you'd train a model first

    print("\n" + "="*70)
    print("📊 Predictor initialized")
    print("="*70)
    print(f"Device: {predictor.device}")
    print(f"Confidence threshold: {predictor.confidence_threshold}")
    print("\nNote: Actual prediction requires a trained model.")
    print("Run trading_model_trainer.py first to train a model.")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
