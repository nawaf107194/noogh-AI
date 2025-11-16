import asyncio
import pytest
import logging
from src.government.education_minister import EducationMinister, generate_task_id

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_education_minister_full_flow():
    """
    Test the full flow of the Education Minister.
    """
    logger.info("🎓 Education Minister v2.0 - Test")
    logger.info("=" * 70)
    logger.info("")

    # Initialize minister
    minister = EducationMinister(verbose=True)

    # Print status
    minister.print_status()

    # Test adding a resource
    logger.info("\n📌 Testing: Add learning resource...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="add_learning_resource",
        task_data={
            "url": "https://www.example.com/python-tutorial",
            "title": "Python Programming Tutorial",
            "tags": ["python", "programming", "tutorial"]
        }
    )
    logger.info(f"Result: {result.result}")
    assert result.status.value == "completed"
    assert result.result['success'] is True

    # Print statistics
    logger.info("\n📊 Education Statistics:")
    stats = minister.get_education_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n✅ Education Minister test complete!")
