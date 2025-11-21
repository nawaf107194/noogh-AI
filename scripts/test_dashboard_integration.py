#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Integration Test
===========================

Tests all dashboard backend connections WITHOUT opening a browser.
Simulates what the Streamlit dashboard does to verify system wiring.

This script verifies:
1. GovernmentService initialization (singleton)
2. Chat flow (President.process_request)
3. Health telemetry (HealthMinister.check_vital_signs)
4. Chart path accessibility
5. Voice service availability

Usage:
    python scripts/test_dashboard_integration.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DashboardIntegrationTester:
    """
    Simulates dashboard backend operations to verify connections.
    """

    def __init__(self):
        self.test_results = {}
        self.president = None

    def run_all_tests(self):
        """Run all integration tests."""
        print("=" * 80)
        print("🔍 DASHBOARD BACKEND INTEGRATION TEST")
        print("=" * 80)
        print()

        tests = [
            ("Initialize President (Singleton)", self.test_president_init),
            ("Chat Flow (Process Message)", self.test_chat_flow),
            ("Health Telemetry (GPU/CPU/RAM)", self.test_health_telemetry),
            ("Chart Path Accessibility", self.test_chart_path),
            ("Voice Service Availability", self.test_voice_service),
            ("Government Service (Alternative)", self.test_government_service),
        ]

        for test_name, test_func in tests:
            print(f"\n{'─' * 80}")
            print(f"🧪 TEST: {test_name}")
            print(f"{'─' * 80}")

            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"\n{status}: {result.get('message', '')}")

            except Exception as e:
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "message": f"Test crashed: {e}"
                }
                print(f"\n❌ FAIL: Test crashed - {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        self.print_summary()

    def test_president_init(self) -> Dict[str, Any]:
        """Test 1: Initialize President (like dashboard does)."""
        try:
            from src.government.president import President

            print("   Initializing President...")
            self.president = President(verbose=False)

            print(f"   ✓ President initialized")
            print(f"   ✓ Cabinet size: {len(self.president.cabinet)} ministers")

            # Verify key ministers exist
            required_ministers = ['health', 'finance', 'education', 'security']
            missing = [m for m in required_ministers if m not in self.president.cabinet]

            if missing:
                return {
                    "success": False,
                    "message": f"Missing ministers: {missing}"
                }

            return {
                "success": True,
                "message": f"President ready with {len(self.president.cabinet)} ministers",
                "data": {
                    "cabinet_size": len(self.president.cabinet),
                    "ministers": list(self.president.cabinet.keys())
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to initialize President: {e}"
            }

    def test_chat_flow(self) -> Dict[str, Any]:
        """Test 2: Process a message (simulate chat input)."""
        try:
            if not self.president:
                return {"success": False, "message": "President not initialized"}

            print("   Sending message: 'Hello, what is your status?'")

            response = self.president.process_request(
                user_input="Hello, what is your status?",
                context={},
                priority="medium"
            )

            print(f"   ✓ Received response")
            print(f"   ✓ Response type: {type(response)}")
            print(f"   ✓ Response preview: {str(response)[:150]}...")

            # Verify response structure
            if not isinstance(response, dict):
                return {
                    "success": False,
                    "message": f"Invalid response type: {type(response)}"
                }

            if 'response' not in response:
                return {
                    "success": False,
                    "message": "Response missing 'response' key"
                }

            return {
                "success": True,
                "message": "Chat flow working correctly",
                "data": {
                    "response_preview": str(response.get('response', ''))[:100],
                    "has_minister_info": 'minister' in response
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Chat flow failed: {e}"
            }

    def test_health_telemetry(self) -> Dict[str, Any]:
        """Test 3: Get health telemetry (simulate sidebar metrics)."""
        try:
            if not self.president:
                return {"success": False, "message": "President not initialized"}

            print("   Accessing Health Minister...")

            if 'health' not in self.president.cabinet:
                return {
                    "success": False,
                    "message": "Health Minister not found in cabinet"
                }

            health_minister = self.president.cabinet['health']

            print("   Calling check_vital_signs()...")
            vitals = health_minister.check_vital_signs()

            print(f"   ✓ Telemetry received")
            print(f"   ✓ Keys: {list(vitals.keys())}")

            # Verify telemetry structure
            required_keys = ['gpu', 'cpu', 'memory', 'disk']
            missing = [k for k in required_keys if k not in vitals]

            if missing:
                return {
                    "success": False,
                    "message": f"Missing telemetry keys: {missing}"
                }

            # Display sample data
            gpu_data = vitals.get('gpu', {})
            cpu_data = vitals.get('cpu', {})
            memory_data = vitals.get('memory', {})

            print(f"\n   📊 Sample Telemetry:")
            if gpu_data.get('status') != 'unavailable':
                print(f"      GPU: {gpu_data.get('temperature_c')}°C | {gpu_data.get('vram_percent', 0):.1f}% VRAM")
            else:
                print(f"      GPU: Unavailable (no NVIDIA GPU or pynvml not installed)")
            print(f"      CPU: {cpu_data.get('percent', 0):.1f}% | {cpu_data.get('cores', 0)} cores")
            print(f"      RAM: {memory_data.get('percent', 0):.1f}% | {memory_data.get('used_gb', 0):.1f}GB used")

            return {
                "success": True,
                "message": "Health telemetry working correctly",
                "data": {
                    "gpu_available": gpu_data.get('status') != 'unavailable',
                    "gpu_temp": gpu_data.get('temperature_c'),
                    "cpu_percent": cpu_data.get('percent'),
                    "memory_percent": memory_data.get('percent')
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Health telemetry failed: {e}"
            }

    def test_chart_path(self) -> Dict[str, Any]:
        """Test 4: Verify chart directory is accessible."""
        try:
            from src.core.settings import settings

            charts_dir = settings.data_dir / "charts"

            print(f"   Charts directory: {charts_dir}")
            print(f"   ✓ Path exists: {charts_dir.exists()}")

            if not charts_dir.exists():
                charts_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✓ Created charts directory")

            # Check for existing charts
            charts = list(charts_dir.glob("*.png"))
            print(f"   ✓ Charts found: {len(charts)}")

            if charts:
                print(f"   📊 Latest chart: {charts[-1].name}")

            return {
                "success": True,
                "message": f"Chart path accessible ({len(charts)} charts found)",
                "data": {
                    "path": str(charts_dir),
                    "exists": charts_dir.exists(),
                    "chart_count": len(charts)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Chart path test failed: {e}"
            }

    def test_voice_service(self) -> Dict[str, Any]:
        """Test 5: Check if VoiceService can be initialized."""
        try:
            from src.services.voice_service import VoiceService

            print("   Initializing VoiceService...")
            voice = VoiceService()

            print(f"   ✓ VoiceService initialized")
            print(f"   ✓ TTS type: {voice.tts_type if hasattr(voice, 'tts_type') else 'Unknown'}")

            # Check if STT is available
            has_stt = voice.stt_model is not None if hasattr(voice, 'stt_model') else False
            print(f"   ✓ STT available: {has_stt}")

            return {
                "success": True,
                "message": "Voice service available",
                "data": {
                    "tts_available": voice.tts_type is not None if hasattr(voice, 'tts_type') else False,
                    "stt_available": has_stt
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Voice service test failed (may be expected if dependencies not installed): {e}"
            }

    def test_government_service(self) -> Dict[str, Any]:
        """Test 6: Test GovernmentService (alternative to direct President access)."""
        try:
            from src.services.government_service import GovernmentService

            print("   Testing GovernmentService singleton...")

            # Get president via service
            president1 = GovernmentService.get_president()
            president2 = GovernmentService.get_president()

            # Verify singleton
            is_singleton = president1 is president2
            print(f"   ✓ Singleton pattern: {is_singleton}")

            if not is_singleton:
                return {
                    "success": False,
                    "message": "GovernmentService not maintaining singleton!"
                }

            return {
                "success": True,
                "message": "GovernmentService singleton working correctly",
                "data": {
                    "singleton": is_singleton
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"GovernmentService test failed: {e}"
            }

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.test_results.values() if r.get("success", False))
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅" if result.get("success") else "❌"
            print(f"{status} {test_name}")
            if not result.get("success"):
                print(f"   └─ {result.get('message', 'Unknown error')}")

        print("\n" + "─" * 80)
        print(f"Results: {passed}/{total} tests passed")
        print("─" * 80)

        if passed == total:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ DASHBOARD BACKEND IS WIRED CORRECTLY\n")
            return 0
        else:
            print(f"\n⚠️ {total - passed} TEST(S) FAILED")
            print("❌ DASHBOARD HAS INTEGRATION ISSUES\n")
            return 1


def main():
    """Run dashboard integration tests."""
    tester = DashboardIntegrationTester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
