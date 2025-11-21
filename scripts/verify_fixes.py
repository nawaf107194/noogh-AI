import asyncio
import inspect
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audit.audit_scheduler import AuditScheduler
from autonomy.decision_loop import DecisionLoop, DecisionLoopConfig
from services.user_service import UserService
from core.settings import settings

async def verify_audit_scheduler():
    print("\n🔍 Verifying AuditScheduler...")
    scheduler = AuditScheduler()
    
    if not inspect.iscoroutinefunction(scheduler.start):
        print("❌ AuditScheduler.start is NOT async")
        return False
    
    if not inspect.iscoroutinefunction(scheduler._scheduler_loop):
        print("❌ AuditScheduler._scheduler_loop is NOT async")
        return False
        
    print("✅ AuditScheduler methods are async")
    return True

async def verify_decision_loop():
    print("\n🔍 Verifying DecisionLoop...")
    # Mock dependencies
    config = DecisionLoopConfig(interval_sec=1)
    loop = DecisionLoop(config, None, None)
    
    if not inspect.iscoroutinefunction(loop.start):
        print("❌ DecisionLoop.start is NOT async")
        return False
        
    if not inspect.iscoroutinefunction(loop._run):
        print("❌ DecisionLoop._run is NOT async")
        return False
        
    print("✅ DecisionLoop methods are async")
    return True

def verify_user_service_security():
    print("\n🔍 Verifying UserService Security...")
    file_path = Path(__file__).parent.parent / "src" / "services" / "user_service.py"
    content = file_path.read_text()
    
    if "SecurePass123!" in content:
        print("❌ Hardcoded password found in user_service.py")
        return False
        
    if "settings.default_user_password.get_secret_value()" not in content:
        print("❌ Settings usage not found in user_service.py")
        return False
        
    print("✅ No hardcoded password found")
    return True

async def main():
    print("🚀 Starting Verification...")
    
    checks = [
        await verify_audit_scheduler(),
        await verify_decision_loop(),
        verify_user_service_security()
    ]
    
    if all(checks):
        print("\n✨ ALL CHECKS PASSED! System is clean.")
        sys.exit(0)
    else:
        print("\n❌ SOME CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
