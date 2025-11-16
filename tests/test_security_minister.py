import asyncio
import pytest
import logging
from src.government.security_minister import SecurityMinister, generate_task_id

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_security_minister_full_flow():
    """
    Test the full flow of the Security Minister.
    """
    logger.info("🔐 Security Minister v2.0 - Test")
    logger.info("=" * 70)
    logger.info("")

    # Initialize minister
    minister = SecurityMinister(verbose=True)

    # Print status
    minister.print_status()

    # Test 1: Threat detection
    logger.info("\n📌 Test 1: Threat Detection...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="detect_threat",
        task_data={
            "content": "SELECT * FROM users WHERE id=1 OR 1=1",
            "source_ip": "192.168.1.100"
        }
    )
    logger.info(f"Result: {result.result}")
    assert result.status.value == "completed"
    assert result.result['threats_detected'] > 0

    # Test 2: Access control
    logger.info("\n📌 Test 2: Access Control...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="check_access",
        task_data={
            "user_role": "developer",
            "resource": "/api/data",
            "action": "write"
        }
    )
    logger.info(f"Result: {result.result}")
    assert result.status.value == "completed"
    assert result.result['access_granted'] is True

    # Test 3: Rate limiting
    logger.info("\n📌 Test 3: Rate Limiting...")
    for i in range(3):
        task_id = generate_task_id()
        result = await minister.execute_task(
            task_id=task_id,
            task_type="check_rate_limit",
            task_data={
                "identifier": "192.168.1.101",
                "max_requests": 2,
                "time_window": 60
            }
        )
        logger.info(f"  Request {i+1}: {result.result}")
        if i == 2:
            assert result.result['rate_limit_exceeded'] is True

    # Print statistics
    logger.info("\n📊 Security Statistics:")
    stats = minister.get_security_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\n🎯 Security Score: {minister.get_security_score():.1f}/100")
    logger.info("\n✅ Security Minister test complete!")
