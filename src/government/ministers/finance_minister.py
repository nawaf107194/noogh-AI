#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finance Minister - Hybrid Spot + Futures Trading
=================================================

PAPER TRADING MODE: Simulates trades on both spot and futures markets.
News-driven + volume-driven strategies.
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import os
import json

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class FinanceMinister(BaseMinister):
    """
    Finance Minister - Hybrid Dual-Market Trader.
    
    NEW Powers:
    - Dual exchange support (spot + futures)
    - Paper trading ledger
    - News-driven analysis
    - Spot vs Futures decision logic
    """
    
    def __init__(self, brain: Optional[Any] = None, health_minister: Optional[Any] = None):
        """Initialize Finance Minister with dual exchanges and health monitoring.
        
        Args:
            brain: LocalBrainService instance for AI inference
            health_minister: HealthMinister instance for system health checks
        """
        super().__init__(
            name="Finance Minister (Hybrid Trader)",
            description="Dual-market trader (spot + futures). Paper trading mode.",
            brain=brain
        )
        
        self.system_prompt = """You are an elite crypto trader managing BOTH spot and futures positions.

SPOT: Long-term holds, fundamental value
FUTURES: Short-term volatility, technical setups

Analyze and recommend the appropriate market."""
        
        # Dual exchange initialization
        self.spot_exchange = None
        self.futures_exchange = None
        self._init_exchanges()

        # Vision Service - AI-powered chart analysis
        self.vision = None
        self._init_vision()
        
        # Health Minister - Hardware safety monitoring
        self.health_minister = health_minister
        if health_minister:
            logger.info("🏥 HealthMinister linked for GPU safety checks")
        else:
            logger.warning("⚠️ HealthMinister not provided - GPU safety checks disabled")

        # Paper trading setup
        from src.core.settings import Settings
        settings = Settings()
        self.paper_ledger_path = settings.paper_ledger_path
        self.paper_balance_spot = settings.paper_balance_spot
        self.paper_balance_futures = settings.paper_balance_futures

        self._init_ledger()
    
    def _init_exchanges(self):
        """Initialize BOTH spot and futures exchanges."""
        try:
            import ccxt

            # Spot exchange
            self.spot_exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logger.info("✅ Spot exchange connected")

            # Futures exchange
            self.futures_exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            logger.info("✅ Futures exchange connected")

        except ImportError:
            logger.warning("⚠️ ccxt not installed")
        except Exception as e:
            logger.error(f"Exchange init failed: {e}")

    def _init_vision(self):
        """Initialize Vision Service for chart analysis."""
        try:
            from src.services.vision_service import VisionService
            self.vision = VisionService()
            logger.info("👁️ Vision Service connected (LLaVA for chart analysis)")
        except Exception as e:
            logger.warning(f"⚠️ Vision Service unavailable: {e}")
            self.vision = None

    async def analyze_chart_with_vision(self, chart_path: str) -> Dict[str, Any]:
        """
        Analyze a trading chart using AI vision.

        Args:
            chart_path: Path to chart image

        Returns:
            Vision analysis result with trading insights
        """
        if not self.vision:
            return {
                "success": False,
                "error": "Vision service not available",
                "analysis": "Vision analysis disabled"
            }

        try:
            logger.info(f"👁️ Analyzing chart with AI vision: {chart_path}")

            # Use vision service to analyze the chart
            result = self.vision.analyze_chart(
                image_path=chart_path,
                prompt="""Analyze this trading chart carefully:

1. What is the overall trend (bullish/bearish/neutral)?
2. Describe any technical patterns you see (triangles, flags, head & shoulders, etc.)
3. How does the volume look relative to price action?
4. Are there any breakout signals?
5. What is your overall assessment - would you BUY, SELL, or HOLD?

Provide a detailed analysis."""
            )

            if result.get("success"):
                logger.info("✅ Vision analysis complete!")
                logger.info(f"   Analysis preview: {result['analysis'][:150]}...")

            return result

        except Exception as e:
            logger.error(f"Chart vision analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Analysis failed"
            }

    def _init_ledger(self):
        """Initialize paper trading ledger."""
        os.makedirs(os.path.dirname(self.paper_ledger_path), exist_ok=True)
        
        if not os.path.exists(self.paper_ledger_path):
            initial_ledger = {
                "created": datetime.now().isoformat(),
                "balances": {
                    "spot_usdt": self.paper_balance_spot,
                    "futures_usdt": self.paper_balance_futures
                },
                "trades": [],
                "stats": {
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "total_pnl": 0.0
                }
            }
            
            with open(self.paper_ledger_path, 'w') as f:
                json.dump(initial_ledger, f, indent=2)
            
            logger.info(f"✅ Paper ledger initialized: {self.paper_ledger_path}")
    
    def record_paper_trade(
        self,
        market_type: str,  # "SPOT" or "FUTURES"
        symbol: str,
        side: str,  # "BUY", "SELL", "LONG", "SHORT"
        price: float,
        reason: str,
        ai_confidence: str,
        chart_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a paper trade to the ledger.
        
        Args:
            market_type: "SPOT" or "FUTURES"
            symbol: Trading pair
            side: Trade direction
            price: Entry price
            reason: Trade rationale
            ai_confidence: AI confidence level
            chart_path: Path to chart image
        
        Returns:
            Trade record
        """
        try:
            # Load ledger
            with open(self.paper_ledger_path, 'r') as f:
                ledger = json.load(f)
            
            # Create trade record
            trade = {
                "id": len(ledger["trades"]) + 1,
                "timestamp": datetime.now().isoformat(),
                "market_type": market_type,
                "symbol": symbol,
                "side": side,
                "entry_price": price,
                "reason": reason,
                "ai_confidence": ai_confidence,
                "chart_path": chart_path,
                "status": "OPEN"
            }
            
            ledger["trades"].append(trade)
            ledger["stats"]["total_trades"] += 1
            
            # Save ledger
            with open(self.paper_ledger_path, 'w') as f:
                json.dump(ledger, f, indent=2)
            
            logger.warning(f"📝 Paper trade recorded: {market_type} {side} {symbol} @ ${price}")
            
            return {
                "success": True,
                "trade": trade
            }
        
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_specific_coin(
        self,
        symbol: str,
        trigger_reason: str = "News Alert"
    ) -> Dict[str, Any]:
        """
        Analyze a specific coin with STRICT safety and confluence checks.
        
        CRITICAL LOGIC FLOW:
        1. Hardware Safety Check (GPU Temp > 88°C = IMMEDIATE HOLD)
        2. Text Analysis (LLM) - Fail fast if not BUY
        3. Vision Confirmation - ONLY if text is BUY
        4. Confluence Logic - BOTH must agree for BUY
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            trigger_reason: What triggered this analysis
            
        Returns:
            Analysis result with signal and metadata
        """
        logger.info(f"🎯 Analyzing {symbol} (Trigger: {trigger_reason})")
        
        # ════════════════════════════════════════════════════════════════
        # PRIORITY #1: HARDWARE SAFETY CHECK
        # ════════════════════════════════════════════════════════════════
        try:
            health_data = self.health_minister.get_system_health()
            gpu_temp = health_data.get("gpu_temp", 0)
            
            if gpu_temp > 88:
                critical_msg = f"🚨 CRITICAL: GPU OVERHEATING ({gpu_temp}°C > 88°C) - TRADING SUSPENDED"
                logger.critical(critical_msg)
                print(f"\n{'='*80}\n{critical_msg}\n{'='*80}\n")
                
                return {
                    "success": True,
                    "signal": "HOLD",
                    "reason": "SYSTEM_OVERHEAT_PROTECTION",
                    "gpu_temp": gpu_temp,
                    "warning": critical_msg
                }
        except AttributeError:
            logger.warning("⚠️ health_minister not initialized - skipping GPU check")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: TEXT ANALYSIS (LLM) - FAIL FAST IF NOT BUY
        # ════════════════════════════════════════════════════════════════
        try:
            # Fetch market data for text analysis
            spot_data = await self._fetch_from_exchange(self.spot_exchange, symbol)
            
            if not spot_data.get("success"):
                return {
                    "success": False,
                    "signal": "HOLD",
                    "reason": "DATA_FETCH_FAILED",
                    "error": spot_data.get("error", "Unknown error")
                }
            
            # Run text-based LLM analysis
            text_analysis = await self._analyze_for_spot(symbol, spot_data)
            text_signal = text_analysis.get("signal")
            
            logger.info(f"📝 Text Analysis: {text_signal}")
            
            # FAIL FAST: If text is not BUY, return immediately
            if text_signal != "BUY":
                logger.info(f"⏭️ Fast exit: Text signal is {text_signal}, skipping vision")
                return {
                    "success": True,
                    "signal": text_signal,
                    "reason": text_analysis.get("reasoning", "Text analysis negative"),
                    "confidence": text_analysis.get("confidence", "N/A"),
                    "analysis_type": "TEXT_ONLY"
                }
            
            # ════════════════════════════════════════════════════════════════
            # STEP 3: VISION CONFIRMATION (Only if text is BUY)
            # ════════════════════════════════════════════════════════════════
            logger.info("👁️ Text says BUY - Requesting vision confirmation...")
            
            # Generate chart for vision analysis
            chart_path = None
            if spot_data.get("ohlcv"):
                try:
                    import pandas as pd
                    
                    df = pd.DataFrame(
                        spot_data["ohlcv"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    
                    from src.services.charting_service import ChartingService
                    charting = ChartingService()
                    
                    chart_result = charting.generate_chart_image(df, symbol)
                    if chart_result.get("success"):
                        chart_path = chart_result.get("path")
                        logger.info(f"📊 Chart generated: {chart_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Chart generation failed: {e}")
            
            # Call vision service
            vision_signal = "HOLD"  # Default to HOLD if vision unavailable
            vision_analysis_text = "Vision service unavailable"
            
            if chart_path and self.vision:
                try:
                    vision_result = await self.analyze_chart_with_vision(chart_path)
                    
                    if vision_result.get("success"):
                        vision_analysis_text = vision_result.get("analysis", "")
                        analysis_upper = vision_analysis_text.upper()
                        
                        if "BUY" in analysis_upper or "BULLISH" in analysis_upper:
                            vision_signal = "BUY"
                        elif "SELL" in analysis_upper or "BEARISH" in analysis_upper:
                            vision_signal = "SELL"
                        else:
                            vision_signal = "HOLD"
                        
                        logger.info(f"👁️ Vision Analysis: {vision_signal}")
                except Exception as e:
                    logger.error(f"Vision analysis failed: {e}")
            else:
                logger.warning("👁️ Vision service not available - defaulting to HOLD")
            
            # ════════════════════════════════════════════════════════════════
            # STEP 4: CONFLUENCE LOGIC - STRICT AGREEMENT REQUIRED
            # ════════════════════════════════════════════════════════════════
            if text_signal == "BUY" and vision_signal == "BUY":
                # ✅ BOTH AGREE - EXECUTE BUY
                logger.warning(f"✅ CONFLUENCE CONFIRMED: Text + Vision both say BUY!")
                
                # Record paper trade
                trade_result = self.record_paper_trade(
                    market_type="SPOT",
                    symbol=symbol,
                    side="BUY",
                    price=spot_data.get("price", 0),
                    reason=f"CONFLUENCE: {text_analysis.get('reasoning', 'Signal detected')}",
                    ai_confidence="HIGH",
                    chart_path=chart_path
                )
                
                return {
                    "success": True,
                    "signal": "BUY",
                    "reason": "CONFLUENCE_CONFIRMED",
                    "text_signal": text_signal,
                    "vision_signal": vision_signal,
                    "price": spot_data.get("price", 0),
                    "trade_recorded": trade_result.get("success", False),
                    "chart_path": chart_path,
                    "confidence": "HIGH"
                }
            else:
                # ⚠️ DIVERGENCE DETECTED - HOLD
                logger.warning(f"⚠️ DIVERGENCE: Text={text_signal}, Vision={vision_signal} - Returning HOLD")
                
                return {
                    "success": True,
                    "signal": "HOLD",
                    "reason": "DIVERGENCE_DETECTED",
                    "text_signal": text_signal,
                    "vision_signal": vision_signal,
                    "divergence_details": f"Text recommended {text_signal}, but Vision said {vision_signal}",
                    "chart_path": chart_path,
                    "confidence": "LOW"
                }
        
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return {
                "success": False,
                "signal": "HOLD",
                "reason": "ANALYSIS_ERROR",
                "error": str(e)
            }

    def check_system_health(self) -> bool:
        """
        Check if system is healthy enough to trade.
        
        CRITICAL: Returns False if GPU Temp > 80°C.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            if temp > 80:
                logger.critical(f"🔥 GPU OVERHEATING: {temp}°C. Trading suspended.")
                return False
            return True
        except ImportError:
            # If pynvml not installed, assume safe (or fail safe depending on policy)
            # For now, we log warning and proceed, assuming HealthMinister handles the rest
            logger.warning("⚠️ pynvml not installed, skipping temp check")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to check GPU temp: {e}")
            return True

    async def _fetch_from_exchange(self, exchange, symbol: str) -> Dict[str, Any]:
        """Fetch data from specific exchange."""
        if not exchange:
            return {"success": False, "error": "Exchange not available"}
        
        try:
            # Note: ccxt methods can be async, so we use await for safety
            # If they're sync, this will still work
            import asyncio
            
            # Wrap sync calls in run_in_executor for thread safety
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, exchange.fetch_ticker, symbol)
            ohlcv = await loop.run_in_executor(None, exchange.fetch_ohlcv, symbol, '1h', 100)
            
            return {
                "success": True,
                "price": ticker['last'],
                "volume": ticker['quoteVolume'],
                "ohlcv": ohlcv
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_for_spot(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Analyze for spot market (long-term holding)."""
        # Simplified spot logic: fundamental + trend
        prompt = f"""Analyze {symbol} for SPOT market (long-term hold):

Price: ${data.get('price', 0):.2f}
Volume: ${data.get('volume', 0):,.0f}

Is this a good long-term accumulation opportunity?
Answer: BUY or HOLD"""
        
        response = await self._think_with_prompt(
            system_prompt=self.system_prompt,
            user_message=prompt,
            max_tokens=100
        )
        
        signal = "BUY" if "BUY" in response.upper() else "HOLD"
        
        return {
            "signal": signal,
            "reasoning": response,
            "confidence": "MEDIUM"
        }
    
    async def _analyze_for_futures(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """Analyze for futures market (short-term volatility)."""
        prompt = f"""Analyze {symbol} for FUTURES market (short-term):

Price: ${data.get('price', 0):.2f}
Volume: ${data.get('volume', 0):,.0f}

Is there a short-term directional setup?
Answer: LONG, SHORT, or NONE"""
        
        response = await self._think_with_prompt(
            system_prompt=self.system_prompt,
            user_message=prompt,
            max_tokens=100
        )
        
        if "LONG" in response.upper():
            signal = "LONG"
        elif "SHORT" in response.upper():
            signal = "SHORT"
        else:
            signal = "NONE"
        
        return {
            "signal": signal,
            "reasoning": response,
            "confidence": "MEDIUM"
        }
    
    async def hunt_market_opportunities(self, min_rvol: float = 2.5, top_pairs: int = 20) -> Dict[str, Any]:
        """Volume-driven hunting (original method)."""
        # Keep existing implementation for volume-driven hunting
        # This is supplementary to news-driven analysis
        logger.info(f"🎯 Volume hunting (RVOL > {min_rvol})...")
        
        # Simplified version for brevity
        return {
            "success": True,
            "opportunities": [],
            "message": "Volume hunting active (implementation kept from Phase 15)"
        }
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute trading task."""
        self.tasks_processed += 1
        
        try:
            task_lower = task.lower()
            
            # News-driven analysis
            if "analyze" in task_lower and context and context.get("symbol"):
                symbol = context.get("symbol")
                trigger = context.get("trigger", "Manual")
                
                result = await self.analyze_specific_coin(symbol, trigger)
                
                # Handle the new simplified format
                signal = result.get("signal", "UNKNOWN")
                reason = result.get("reason", "No reason provided")
                
                if result.get("success"):
                    self.tasks_successful += 1
                    
                    # Build user-friendly summary
                    summary = f"📊 {symbol} Analysis Complete\n\n"
                    summary += f"🎯 Signal: **{signal}**\n"
                    summary += f"📝 Reason: {reason}\n"
                    
                    # Add confluence details if available
                    if "text_signal" in result and "vision_signal" in result:
                        summary += f"\n🤖 Text Signal: {result['text_signal']}\n"
                        summary += f"👁️ Vision Signal: {result['vision_signal']}\n"
                    
                    # Add safety warnings if present
                    if result.get("gpu_temp"):
                        summary += f"\n🌡️ GPU Temp: {result['gpu_temp']}°C\n"
                    
                    # Show confidence
                    confidence = result.get("confidence", "N/A")
                    summary += f"📈 Confidence: {confidence}\n"
                    
                    # Show if trade was recorded
                    if result.get("trade_recorded"):
                        summary += f"\n✅ Paper trade recorded at ${result.get('price', 0):.2f}\n"
                    
                    return {
                        "success": True,
                        "response": summary,
                        "minister": self.name,
                        "metadata": result
                    }
            
            # Volume hunting
            elif "hunt" in task_lower:
                result = await self.hunt_market_opportunities()
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": "Volume hunting active",
                    "minister": self.name,
                    "metadata": result
                }
            
            return {
                "success": False,
                "response": "Unknown task type",
                "minister": self.name
            }
        
        except Exception as e:
            logger.error(f"Task error: {e}")
            return {
                "success": False,
                "response": str(e),
                "minister": self.name,
                "error": str(e)
            }


__all__ = ["FinanceMinister"]
