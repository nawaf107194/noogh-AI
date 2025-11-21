import asyncio
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.government.president import President
from src.government.minister_types_universal import MinisterType
from src.government.education_minister import EducationMinister

@pytest.fixture
def president():
    """Fixture to create a President instance for testing."""
    # We disable verbose logging for cleaner test output
    return President(verbose=False)

@pytest.mark.asyncio
async def test_president_initializes_real_ministers(president):
    """
    Test that the president initializes real, functional ministers
    instead of placeholders.
    """
    assert president.cabinet, "Cabinet dictionary should not be empty"
    
    # Check if a specific, known minister (EducationMinister) is initialized
    education_minister = president.cabinet.get(MinisterType.EDUCATION.value)
    assert education_minister is not None, "Education Minister should be initialized"
    assert isinstance(education_minister, EducationMinister), "Should be a real instance of EducationMinister"
    
    # Check that it has the real execution method
    assert hasattr(education_minister, '_execute_specific_task'), "Minister should have the real execution method"

@pytest.mark.asyncio
async def test_consult_ministers_calls_real_minister(president):
    """
    Test that process_request calls the real execute_task method
    on a minister, instead of simulating a result.
    """
    # Get the real education minister to spy on it
    education_minister = president.cabinet.get(MinisterType.EDUCATION.value)
    assert education_minister is not None
    
    # Mock the real minister's execute_task method to act as a spy
    # We use patch.object to mock the method on the actual instance
    with patch.object(education_minister, 'execute_task', new_callable=MagicMock) as mock_execute_task:
        # Set up a mock return value that looks like a real MinisterReport
        mock_report = MagicMock()
        mock_report.minister_type = MinisterType.EDUCATION
        mock_report.to_dict.return_value = {
            "minister": "education",
            "status": "completed",
            "result": {"success": True, "message": "Mocked real execution"}
        }
        mock_execute_task.return_value = asyncio.Future()
        mock_execute_task.return_value.set_result(mock_report)

        # Define a request that should be routed to the Education Minister
        user_input = "شرح لي كيفية عمل لغة بايثون"
        
        # Call the method under test
        result = await president.process_request(
            user_input=user_input,
            priority="medium"
        )

        # --- Assertions ---
        # 1. Check that the minister's execute_task was actually called
        mock_execute_task.assert_called_once()
        
        # 2. Check that the result contains the data from our mock
        assert result.get("result", {}).get("message") == "Mocked real execution"
        
        print("\n✅ Test passed: President successfully called the real minister's execute_task method.")
