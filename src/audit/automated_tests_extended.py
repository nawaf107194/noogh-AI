from typing import Dict, Any
from audit.automated_tests import AutomatedTestSuite, AUTOMATED_TEST_FUNCTIONS

class ExtendedAutomatedTests(AutomatedTestSuite):
    """
    Extended suite of automated tests.
    Restored to fix missing file error.
    """
    def get_all_extended_tests(self) -> Dict[int, Any]:
        """
        Returns all extended tests.
        For now, returns the base set of tests.
        """
        return AUTOMATED_TEST_FUNCTIONS
