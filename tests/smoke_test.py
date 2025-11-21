#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Smoke Tests - اختبارات سريعة للنظام
Verify critical systems work after refactoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_1_president_initialization():
    """Test 1: President can be initialized"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: President Initialization")
    logger.info("="*60)

    try:
        from src.government.president import create_president

        president = create_president(verbose=False)

        assert president is not None, "President should not be None"
        assert len(president.cabinet) > 0, "Cabinet should have ministers"

        logger.info(f"✅ PASSED: President created with {len(president.cabinet)} ministers")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


async def test_2_president_process_request():
    """Test 2: President can process requests"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: President Process Request")
    logger.info("="*60)

    try:
        from src.government.president import create_president

        president = create_president(verbose=False)
        result = await president.process_request("Hello, test message")

        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dict"

        logger.info(f"✅ PASSED: Request processed successfully")
        logger.info(f"   Result keys: {list(result.keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_3_knowledge_kernel_initialization():
    """Test 3: Knowledge Kernel can be initialized"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 3: Knowledge Kernel Initialization")
    logger.info("="*60)

    try:
        from src.knowledge_kernel_v4_1 import KnowledgeKernelV41

        kernel = KnowledgeKernelV41(
            enable_brain=False,
            enable_allam=False,
            enable_intent_routing=False,
            enable_web_search=False,
            enable_reflection=False
        )

        assert kernel is not None, "Kernel should not be None"

        logger.info(f"✅ PASSED: Knowledge Kernel initialized")
        logger.info(f"   Knowledge chunks: {len(kernel.knowledge_index):,}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_4_auto_data_collector_initialization():
    """Test 4: Auto Data Collector can be initialized"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 4: Auto Data Collector Initialization")
    logger.info("="*60)

    try:
        from src.data_collection.auto_data_collector import AutoDataCollector

        collector = AutoDataCollector()

        assert collector is not None, "Collector should not be None"

        stats = collector.get_stats()
        assert stats is not None, "Stats should not be None"

        logger.info(f"✅ PASSED: Auto Data Collector initialized")
        logger.info(f"   Data directory: {stats['data_directory']}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


async def test_5_auto_data_collector_collect():
    """Test 5: Auto Data Collector can collect data"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 5: Auto Data Collector - Data Collection")
    logger.info("="*60)

    try:
        from src.data_collection.auto_data_collector import AutoDataCollector

        collector = AutoDataCollector()

        # Collect small amount of synthetic data
        result = await collector.collect_training_data(
            target_samples=20,
            task_type="general",
            sources=["synthetic"]
        )

        assert result is not None, "Result should not be None"
        assert 'train' in result, "Should have train data"
        assert 'test' in result, "Should have test data"
        assert len(result['train']) > 0, "Should have training samples"

        logger.info(f"✅ PASSED: Data collection successful")
        logger.info(f"   Train samples: {len(result['train'])}")
        logger.info(f"   Test samples: {len(result['test'])}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_6_external_awareness():
    """Test 6: External Awareness (Web Search) initialization"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 6: External Awareness - Web Search")
    logger.info("="*60)

    try:
        from src.external_awareness import WebSearchProvider

        provider = WebSearchProvider(cache_ttl_hours=1)

        assert provider is not None, "Provider should not be None"
        assert provider._cache is not None, "Cache should be initialized"

        logger.info(f"✅ PASSED: WebSearchProvider initialized")
        logger.info(f"   Cache TTL: 1 hour")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_7_reflection_tracker():
    """Test 7: Reflection (Experience Tracker) initialization"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 7: Reflection - Experience Tracker")
    logger.info("="*60)

    try:
        from src.reflection import ExperienceTracker

        # Use in-memory database for testing
        tracker = ExperienceTracker(db_path=":memory:")

        assert tracker is not None, "Tracker should not be None"
        assert tracker.conn is not None, "Database connection should exist"

        # Track a test experience
        tracker.track(
            question="Test question",
            intent="test",
            answer="Test answer",
            sources=[],
            confidence=0.9,
            success=True,
            execution_time=0.1,
            handler="test_handler",
            used_web_search=False
        )

        # Get stats
        stats = tracker.get_session_stats()
        assert stats['queries_in_session'] == 1, "Should have 1 query"

        logger.info(f"✅ PASSED: ExperienceTracker initialized and working")
        logger.info(f"   Test experiences tracked: 1")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_8_vision_scene_understanding():
    """Test 8: Vision - Scene Understanding Engine"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 8: Vision - Scene Understanding")
    logger.info("="*60)

    try:
        from src.vision.scene_understanding import SceneUnderstandingEngine

        engine = SceneUnderstandingEngine()

        assert engine is not None, "Engine should not be None"

        stats = engine.get_statistics()
        assert stats is not None, "Stats should not be None"
        assert stats['total_analyzed'] == 0, "Should start with 0 analyzed"

        logger.info(f"✅ PASSED: SceneUnderstandingEngine initialized")
        logger.info(f"   ImageAnalyzer available: {engine.image_analyzer is not None}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


def test_9_vision_material_analyzer():
    """Test 9: Vision - Material Analyzer"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 9: Vision - Material Analyzer")
    logger.info("="*60)

    try:
        from src.vision.material_analyzer import MaterialAnalyzer

        analyzer = MaterialAnalyzer()

        assert analyzer is not None, "Analyzer should not be None"

        stats = analyzer.get_statistics()
        assert stats is not None, "Stats should not be None"
        assert stats['total_analyzed'] == 0, "Should start with 0 analyzed"

        logger.info(f"✅ PASSED: MaterialAnalyzer initialized")
        logger.info(f"   ImageAnalyzer available: {analyzer.image_analyzer is not None}")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        return False


async def run_all_tests():
    """Run all smoke tests"""
    logger.info("\n" + "🚀"*30)
    logger.info("🚀 NOOGH UNIFIED SYSTEM - SMOKE TESTS")
    logger.info("🚀"*30)

    results = []

    # Core System Tests (Synchronous)
    results.append(("President Initialization", test_1_president_initialization()))
    results.append(("Knowledge Kernel Initialization", test_3_knowledge_kernel_initialization()))
    results.append(("Auto Data Collector Initialization", test_4_auto_data_collector_initialization()))

    # Core System Tests (Async)
    results.append(("President Process Request", await test_2_president_process_request()))
    results.append(("Auto Data Collector Collection", await test_5_auto_data_collector_collect()))

    # Optional Modules Tests
    results.append(("External Awareness (Web Search)", test_6_external_awareness()))
    results.append(("Reflection (Experience Tracker)", test_7_reflection_tracker()))
    results.append(("Vision - Scene Understanding", test_8_vision_scene_understanding()))
    results.append(("Vision - Material Analyzer", test_9_vision_material_analyzer()))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")

    logger.info("="*60)
    logger.info(f"📊 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("="*60)

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - System is working correctly! 🎉")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
