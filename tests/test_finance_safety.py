import asyncio
import sys
import os
from unittest.mock import MagicMock, patch
import logging

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.government.ministers.finance_minister import FinanceMinister

# Mock matplotlib and mplfinance BEFORE importing ChartingService
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['mplfinance'] = MagicMock()
sys.modules['pynvml'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['pandas_ta'] = MagicMock()

# Explicitly import to ensure patch works
import src.services.charting_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_finance_safety():
    print("\n🧪 Testing Finance Minister Safety & Logic...\n")
    
    # Mock dependencies
    mock_brain = MagicMock()
    # Make think return an awaitable
    mock_brain.think.side_effect = lambda *args, **kwargs: asyncio.sleep(0.01, result="BUY because it looks good")
    
    # Initialize Minister
    minister = FinanceMinister(brain=mock_brain)
    
    # Mock Exchange Data
    minister._fetch_from_exchange = MagicMock(side_effect=lambda ex, sym: asyncio.Future())
    minister._fetch_from_exchange.side_effect = lambda ex, sym: asyncio.sleep(0.01, result={
        "success": True,
        "price": 50000.0,
        "volume": 1000000.0,
        "ohlcv": [
            [1600000000000, 49000, 51000, 48000, 50000, 100] for _ in range(10)
        ]
    })
    
    # Mock LLM Analysis (Spot)
    minister._analyze_for_spot = MagicMock(side_effect=lambda sym, data: asyncio.Future())
    minister._analyze_for_spot.side_effect = lambda sym, data: asyncio.sleep(0.01, result={
        "signal": "BUY",
        "reasoning": "Strong fundamentals",
        "confidence": "HIGH"
    })
    
    # Mock Vision Service
    minister.vision = MagicMock()
    minister.analyze_chart_with_vision = MagicMock(side_effect=lambda path: asyncio.Future())
    
    # Mock Charting Service
    with patch('src.services.charting_service.ChartingService') as MockCharting:
        mock_charting_instance = MockCharting.return_value
        mock_charting_instance.generate_chart_image.return_value = {
            "success": True,
            "path": "/tmp/fake_chart.png"
        }
        
        # TEST 1: High GPU Temp -> Should Block
        print("🔹 Test 1: High GPU Temperature (Safety Lock)")
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetHandleByIndex'), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=85): # 85°C > 80°C
            
            result = await minister.analyze_specific_coin("BTC/USDT")
            
            if result['success'] == False and "High GPU Temperature" in result['error']:
                print("✅ PASSED: Trading blocked due to high temp.")
            else:
                print(f"❌ FAILED: Trading NOT blocked. Result: {result}")

        # TEST 2: Normal Temp + Vision Agrees -> Should Trade
        print("\n🔹 Test 2: Normal Temp + Vision CONFIRMS Buy")
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetHandleByIndex'), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=60): # 60°C < 80°C
            
            # Vision says BUY
            minister.analyze_chart_with_vision.side_effect = lambda path: asyncio.sleep(0.01, result={
                "success": True,
                "analysis": "The chart looks BULLISH. I recommend a BUY."
            })
            
            result = await minister.analyze_specific_coin("BTC/USDT")
            
            trades = result.get('trades_recorded', [])
            # trades[0] is {"success": True, "trade": {...}}
            if len(trades) > 0 and trades[0]['trade']['side'] == 'BUY':
                print("✅ PASSED: Trade executed when Vision agrees.")
            else:
                print(f"❌ FAILED: Trade NOT executed. Result: {result}")

        # TEST 3: Normal Temp + Vision Disagrees -> Should Block
        print("\n🔹 Test 3: Normal Temp + Vision REJECTS Buy")
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetHandleByIndex'), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=60):
            
            # Vision says SELL
            minister.analyze_chart_with_vision.side_effect = lambda path: asyncio.sleep(0.01, result={
                "success": True,
                "analysis": "The chart looks BEARISH. I recommend a SELL."
            })
            
            result = await minister.analyze_specific_coin("BTC/USDT")
            
            trades = result.get('trades_recorded', [])
            # We expect NO spot buy trade because Vision disagreed with the Text model's BUY
            spot_trades = [t for t in trades if t['market_type'] == 'SPOT']
            
            if len(spot_trades) == 0:
                print("✅ PASSED: Trade aborted when Vision disagrees.")
            else:
                print(f"❌ FAILED: Trade executed despite Vision disagreement. Trades: {spot_trades}")

if __name__ == "__main__":
    asyncio.run(test_finance_safety())
