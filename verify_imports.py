import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath("src"))

try:
    print("Attempting to import api.main...")
    from api import main
    print("✅ Successfully imported api.main")
except ImportError as e:
    print(f"❌ Failed to import api.main: {e}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
