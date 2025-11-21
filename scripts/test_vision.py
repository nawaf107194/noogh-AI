#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Service Test Script
===========================

Tests the LLaVA vision model with a sample trading chart.

This script:
1. Downloads a sample trading chart image
2. Analyzes it using VisionService
3. Prints the AI's visual analysis

Usage:
    python scripts/test_vision.py
"""

import sys
import logging
from pathlib import Path
import urllib.request

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.vision_service import VisionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_sample_chart():
    """
    Download a sample trading chart for testing.

    Returns:
        Path to downloaded chart
    """
    # Create test data directory
    test_dir = project_root / "data" / "test_charts"
    test_dir.mkdir(parents=True, exist_ok=True)

    chart_path = test_dir / "btc_sample_chart.png"

    # If already exists, use it
    if chart_path.exists():
        logger.info(f"✅ Using existing chart: {chart_path}")
        return chart_path

    # Download a sample BTC chart from a public source
    # Using a simple technical analysis chart image
    logger.info("📥 Downloading sample trading chart...")

    try:
        # Sample chart URL (public domain trading chart for testing)
        # This is a generic candlestick chart image for demonstration
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Candlestick_chart_sample.png/800px-Candlestick_chart_sample.png"

        urllib.request.urlretrieve(url, chart_path)
        logger.info(f"✅ Chart downloaded: {chart_path}")
        return chart_path

    except Exception as e:
        logger.error(f"❌ Failed to download chart: {e}")
        logger.info("💡 Creating a simple test image instead...")

        # Fallback: Create a simple colored rectangle as test image
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create a simple chart-like image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)

            # Draw some simple chart elements
            draw.rectangle([50, 50, 750, 550], outline='black', width=2)
            draw.line([50, 300, 750, 300], fill='gray', width=1)
            draw.line([400, 50, 400, 550], fill='gray', width=1)

            # Draw some "candlesticks" (simple rectangles)
            colors = ['green', 'red', 'green', 'green', 'red', 'green']
            x = 100
            for color in colors:
                draw.rectangle([x, 200, x+80, 400], fill=color, outline='black')
                x += 120

            # Add title
            try:
                draw.text((300, 20), "BTC/USDT Test Chart", fill='black')
            except:
                pass  # Font may not be available

            img.save(chart_path)
            logger.info(f"✅ Created test chart: {chart_path}")
            return chart_path

        except Exception as e2:
            logger.error(f"❌ Failed to create test image: {e2}")
            raise


def test_vision_service():
    """
    Test the vision service with a sample chart.
    """
    print("=" * 80)
    print("👁️ NOOGH VISION SERVICE TEST")
    print("=" * 80)
    print()

    try:
        # Step 1: Download/prepare sample chart
        print("📊 Step 1: Preparing sample chart...")
        chart_path = download_sample_chart()
        print(f"   Chart ready: {chart_path}")
        print()

        # Step 2: Initialize Vision Service
        print("🧠 Step 2: Initializing LLaVA Vision Model...")
        print("   (This may take 30-60 seconds on first run)")
        print()

        vision = VisionService()

        # Step 3: Analyze the chart
        print("🔍 Step 3: Analyzing chart with AI vision...")
        print()

        result = vision.analyze_chart(
            image_path=str(chart_path),
            prompt="Analyze this trading chart. What patterns do you see? Is it bullish or bearish?"
        )

        # Step 4: Display results
        print("=" * 80)
        print("📊 VISION ANALYSIS RESULT")
        print("=" * 80)
        print()

        if result.get("success"):
            print("✅ Analysis Status: SUCCESS")
            print()
            print(f"🖼️  Image: {result.get('image_path')}")
            print(f"🤖 Model: {result.get('model')}")
            print(f"📊 Confidence: {result.get('confidence', 0) * 100:.1f}%")
            print(f"🎯 Real Vision: {not result.get('simulated', True)}")
            print()
            print("=" * 80)
            print("👁️ AI VISUAL ANALYSIS:")
            print("=" * 80)
            print()
            print(result.get('analysis', 'No analysis'))
            print()
            print("=" * 80)
            print("✅ VISION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            # Test the status method
            print()
            print("📊 Vision Service Status:")
            status = vision.get_status()
            for key, value in status.items():
                print(f"   {key}: {value}")

        else:
            print("❌ Analysis Status: FAILED")
            print(f"   Error: {result.get('error')}")
            print()
            print("=" * 80)
            print("💡 TROUBLESHOOTING:")
            print("=" * 80)
            print("1. Install dependencies: pip install transformers accelerate bitsandbytes")
            print("2. Ensure you have enough VRAM (~4GB for LLaVA)")
            print("3. Check logs above for specific errors")
            print("=" * 80)

    except Exception as e:
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("💡 TROUBLESHOOTING:")
        print("=" * 80)
        print("1. Install dependencies: pip install transformers accelerate bitsandbytes pillow")
        print("2. Ensure GPU is available (CUDA)")
        print("3. Check if you have ~4GB VRAM free")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    test_vision_service()
