import asyncio
import pytest
import logging
from src.government.communication_minister import create_communication_minister, generate_task_id

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_communication_minister_full_flow():
    """
    Test the full flow of the Communication Minister.
    """
    logger.info("🧪 Testing Communication Minister...\n")

    minister = create_communication_minister(verbose=True)

    # Test general communication
    logger.info(f"\n🧪 Testing general communication...")
    task_id_comm = generate_task_id()
    report_comm = await minister.execute_task(
        task_id=task_id_comm,
        task_type="communication",
        task_data={"input": "مرحباً، كيف يمكنني تحليل BTC؟"}
    )

    logger.info(f"\n📋 Communication Report Summary:")
    logger.info(f"   Status: {report_comm.status.value}")
    logger.info(f"   Confidence: {report_comm.confidence:.1%}")
    assert report_comm.status.value == "completed"
    assert 'response' in report_comm.result
    logger.info(f"   Response: {report_comm.result['response'][:80]}...")

    # Test translation
    logger.info(f"\n🧪 Testing translation...")
    task_id_trans = generate_task_id()
    report_trans = await minister.execute_task(
        task_id=task_id_trans,
        task_type="translation",
        task_data={
            "input": "Hello, how are you?",
            "target_language": "arabic"
        }
    )

    logger.info(f"\n📋 Translation Report Summary:")
    logger.info(f"   Status: {report_trans.status.value}")
    assert report_trans.status.value == "completed"
    assert 'translated_text' in report_trans.result
    logger.info(f"   Translation: {report_trans.result['translated_text']}")

    # Print final status
    minister.print_status()

    logger.info(f"\n✅ Communication Minister test completed!")
