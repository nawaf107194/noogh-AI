#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finance Minister - Critical Logic Test
========================================

Tests the new security logic flow:
1. Hardware Safety Check (GPU temp > 88°C)
2. Text Analysis (fail-fast if not BUY)
3. Vision Confirmation
4. Strict Confluence Logic
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_hardware_safety_check():
    """Test #1: GPU Overheat Protection"""
    print("\n" + "="*80)
    print("TEST #1: Hardware Safety Check (GPU > 88°C)")
    print("="*80)
    
    # Import after setup
    from src.government.ministers.finance_minister import FinanceMinister
    
    # Create mock health minister that reports high temperature
    mock_health = Mock()
    mock_health.get_system_health = Mock(return_value={"gpu_temp": 92})
    
    # Create Finance Minister with mocked health minister
    finance = FinanceMinister(brain=None, health_minister=mock_health)
    
    # Test analysis - should return HOLD immediately
    result = await finance.analyze_specific_coin("BTC/USDT", "Test")
    
    print(f"\n📊 Result:")
    print(f"  Signal: {result.get('signal')}")
    print(f"  Reason: {result.get('reason')}")
    print(f"  GPU Temp: {result.get('gpu_temp')}°C")
    
    # Verify
    assert result.get('signal') == 'HOLD', "Should return HOLD on overheat"
    assert result.get('reason') == 'SYSTEM_OVERHEAT_PROTECTION', "Should have correct reason"
    assert result.get('gpu_temp') == 92, "Should report GPU temp"
    
    print("\n✅ Test #1 PASSED: Hardware safety working!")
    return True


async def test_fail_fast_text_analysis():
    """Test #2: Fail-fast when text analysis is not BUY"""
    print("\n" + "="*80)
    print("TEST #2: Fail-Fast Text Analysis (Signal != BUY)")
    print("="*80)
    
    from src.government.ministers.finance_minister import FinanceMinister
    
    # Mock health minister (safe temperature)
    mock_health = Mock()
    mock_health.get_system_health = Mock(return_value={"gpu_temp": 65})
    
    # Create Finance Minister
    finance = FinanceMinister(brain=None, health_minister=mock_health)
    
    # Mock exchange to return data
    mock_exchange_data = {
        "success": True,
        "price": 45000.0,
        "volume": 1000000,
        "ohlcv": []  # Empty for simplicity
    }
    
    # Mock _analyze_for_spot to return HOLD
    async def mock_analyze_spot(symbol, data):
        return {
            "signal": "HOLD",
            "reasoning": "Market conditions not favorable",
            "confidence": "MEDIUM"
        }
    
    # Patch methods
    with patch.object(finance, '_fetch_from_exchange', AsyncMock(return_value=mock_exchange_data)):
        with patch.object(finance, '_analyze_for_spot', mock_analyze_spot):
            result = await finance.analyze_specific_coin("BTC/USDT", "Test")
    
    print(f"\n📊 Result:")
    print(f"  Signal: {result.get('signal')}")
    print(f"  Reason: {result.get('reason')}")
    print(f"  Analysis Type: {result.get('analysis_type')}")
    
    # Verify - should return immediately without vision
    assert result.get('signal') == 'HOLD', "Should return HOLD"
    assert result.get('analysis_type') == 'TEXT_ONLY', "Should skip vision"
    assert 'vision_signal' not in result, "Should not call vision"
    
    print("\n✅ Test #2 PASSED: Fail-fast logic working!")
    return True


async def test_confluence_both_buy():
    """Test #3: Confluence - Both Text and Vision say BUY"""
    print("\n" + "="*80)
    print("TEST #3: Confluence Logic (Both BUY)")
    print("="*80)
    
    from src.government.ministers.finance_minister import FinanceMinister
    
    # Mock health minister (safe temperature)
    mock_health = Mock()
    mock_health.get_system_health = Mock(return_value={"gpu_temp": 65})
    
    # Create Finance Minister
    finance = FinanceMinister(brain=None, health_minister=mock_health)
    
    # Mock exchange data
    mock_exchange_data = {
        "success": True,
        "price": 45000.0,
        "volume": 1000000,
        "ohlcv": []
    }
    
    # Mock text analysis to return BUY
    async def mock_analyze_spot(symbol, data):
        return {
            "signal": "BUY",
            "reasoning": "Strong bullish momentum detected",
            "confidence": "HIGH"
        }
    
    # Mock vision analysis to return BUY
    async def mock_vision_analysis(chart_path):
        return {
            "success": True,
            "analysis": "Chart shows clear BUY signal with bullish breakout pattern"
        }
    
    # Patch methods
    finance.vision = Mock()  # Enable vision
    
    with patch.object(finance, '_fetch_from_exchange', AsyncMock(return_value=mock_exchange_data)):
        with patch.object(finance, '_analyze_for_spot', mock_analyze_spot):
            with patch.object(finance, 'analyze_chart_with_vision', mock_vision_analysis):
                with patch.object(finance, 'record_paper_trade', Mock(return_value={"success": True})):
                    result = await finance.analyze_specific_coin("BTC/USDT", "Test")
    
    print(f"\n📊 Result:")
    print(f"  Signal: {result.get('signal')}")
    print(f"  Reason: {result.get('reason')}")
    print(f"  Text Signal: {result.get('text_signal')}")
    print(f"  Vision Signal: {result.get('vision_signal')}")
    print(f"  Trade Recorded: {result.get('trade_recorded')}")
    
    # Verify - should return BUY and record trade
    assert result.get('signal') == 'BUY', "Should return BUY"
    assert result.get('reason') == 'CONFLUENCE_CONFIRMED', "Should confirm confluence"
    assert result.get('text_signal') == 'BUY', "Text should be BUY"
    assert result.get('vision_signal') == 'BUY', "Vision should be BUY"
    
    print("\n✅ Test #3 PASSED: Confluence confirmation working!")
    return True


async def test_divergence_detection():
    """Test #4: Divergence - Text BUY but Vision HOLD"""
    print("\n" + "="*80)
    print("TEST #4: Divergence Detection (Text=BUY, Vision=HOLD)")
    print("="*80)
    
    from src.government.ministers.finance_minister import FinanceMinister
    
    # Mock health minister (safe temperature)
    mock_health = Mock()
    mock_health.get_system_health = Mock(return_value={"gpu_temp": 65})
    
    # Create Finance Minister
    finance = FinanceMinister(brain=None, health_minister=mock_health)
    
    # Mock exchange data
    mock_exchange_data = {
        "success": True,
        "price": 45000.0,
        "volume": 1000000,
        "ohlcv": []
    }
    
    # Mock text analysis to return BUY
    async def mock_analyze_spot(symbol, data):
        return {
            "signal": "BUY",
            "reasoning": "News indicates positive sentiment",
            "confidence": "MEDIUM"
        }
    
    # Mock vision analysis to return HOLD/NEUTRAL
    async def mock_vision_analysis(chart_path):
        return {
            "success": True,
            "analysis": "Chart shows consolidation, no clear direction. Assessment: HOLD"
        }
    
    # Patch methods
    finance.vision = Mock()  # Enable vision
    
    with patch.object(finance, '_fetch_from_exchange', AsyncMock(return_value=mock_exchange_data)):
        with patch.object(finance, '_analyze_for_spot', mock_analyze_spot):
            with patch.object(finance, 'analyze_chart_with_vision', mock_vision_analysis):
                result = await finance.analyze_specific_coin("BTC/USDT", "Test")
    
    print(f"\n📊 Result:")
    print(f"  Signal: {result.get('signal')}")
    print(f"  Reason: {result.get('reason')}")
    print(f"  Text Signal: {result.get('text_signal')}")
    print(f"  Vision Signal: {result.get('vision_signal')}")
    print(f"  Divergence Details: {result.get('divergence_details')}")
    
    # Verify - should return HOLD due to divergence
    assert result.get('signal') == 'HOLD', "Should return HOLD on divergence"
    assert result.get('reason') == 'DIVERGENCE_DETECTED', "Should detect divergence"
    assert result.get('text_signal') == 'BUY', "Text should be BUY"
    assert result.get('vision_signal') == 'HOLD', "Vision should be HOLD"
    
    print("\n✅ Test #4 PASSED: Divergence detection working!")
    return True


async def main():
    """Run all tests"""
    print("\n" + "🔬"*40)
    print("FINANCE MINISTER - CRITICAL LOGIC TESTS")
    print("🔬"*40)
    
    try:
        # Run all tests
        await test_hardware_safety_check()
        await test_fail_fast_text_analysis()
        await test_confluence_both_buy()
        await test_divergence_detection()
        
        # Summary
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ Hardware Safety: WORKING")
        print("✅ Fail-Fast Logic: WORKING")
        print("✅ Confluence Detection: WORKING")
        print("✅ Divergence Detection: WORKING")
        print("\n🔐 Finance Minister security patch is OPERATIONAL!\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
