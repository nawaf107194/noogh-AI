#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Tests for the Self-Audit Engine
"""
import asyncio
from typing import Dict, Any

# Add project root to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.government.president import President

def run_async(coro):
    """Helper to run an async function from a sync context."""
    return asyncio.run(coro)

def test_decision_confidence_measurement() -> Dict[str, Any]:
    """
    Automated test for Audit Question ID 2:
    "Does it have a way to measure decision 'confidence' numerically?"
    """
    try:
        # 1. Initialize the president
        # We create a new instance to ensure a clean state for the test
        president = President(verbose=False)

        # 2. Process a simple request
        request_input = "ما هي حالة النظام؟"
        # Running the async process_request method using the helper
        result = run_async(president.process_request(request_input))

        # 3. Check for the 'final_decision' and 'confidence' key
        final_decision = result.get('final_decision')
        if not final_decision:
            return {
                "passed": False,
                "score": 0.0,
                "evidence": "The 'final_decision' key was not found in the president's response."
            }

        confidence = final_decision.get('confidence')
        if confidence is None:
            return {
                "passed": False,
                "score": 0.0,
                "evidence": "The 'confidence' key was not found in the final_decision."
            }

        # 4. Verify the type and range of the confidence score
        if isinstance(confidence, (float, int)) and 0.0 <= confidence <= 1.0:
            return {
                "passed": True,
                "score": 1.0,
                "evidence": f"Confidence score found and is a valid value: {confidence:.2f}"
            }
        else:
            return {
                "passed": False,
                "score": 0.2,  # Partial credit for having the key but wrong type/value
                "evidence": f"Confidence score was found, but its value or type is invalid. Value: {confidence}"
            }

    except Exception as e:
        return {
            "passed": False,
            "score": 0.0,
            "evidence": f"An exception occurred during the test: {str(e)}"
        }

# A dictionary to map question IDs to their test functions
# This will be imported by the SelfAuditEngine
AUTOMATED_TEST_FUNCTIONS = {
    2: test_decision_confidence_measurement,
    # Other test functions will be added here
}

class AutomatedTestSuite:
    """A base class for automated test suites."""
    def get_all_tests(self) -> Dict[int, Any]:
        return AUTOMATED_TEST_FUNCTIONS
