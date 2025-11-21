import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    print("🔄 Attempting to import app from src.api.main...")
    from src.api.main import app
    from fastapi import FastAPI
    
    if isinstance(app, FastAPI):
        print("✅ Successfully initialized FastAPI app")
        print(f"   Title: {app.title}")
        print(f"   Version: {app.version}")
        print(f"   Routes: {len(app.routes)}")
    else:
        print("❌ Imported object is not a FastAPI instance")
        sys.exit(1)

except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
