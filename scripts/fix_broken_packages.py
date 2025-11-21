import os
import sys
import importlib.util
from pathlib import Path

def fix_streamlit_mic_recorder():
    """
    Patches the streamlit_mic_recorder package by creating a missing source map file.
    Error fixed: FileNotFoundError: .../bootstrap.min.css.map
    """
    package_name = "streamlit_mic_recorder"
    print(f"🔍 Locating package: {package_name}...")

    # Dynamic package location
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.origin:
        print(f"❌ ERROR: Package '{package_name}' not found in environment.")
        return False

    package_dir = Path(spec.origin).parent
    target_dir = package_dir / "frontend" / "build"
    target_file = target_dir / "bootstrap.min.css.map"

    print(f"📂 Package path: {package_dir}")
    print(f"🎯 Target directory: {target_dir}")

    if not target_dir.exists():
        print(f"⚠️  Warning: Target directory does not exist. Attempting to create...")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ ERROR: Could not create directory: {e}")
            return False

    if target_file.exists():
        print(f"ℹ️  File already exists: {target_file}")
        print("✅ No action needed.")
        return True

    try:
        # Create empty map file
        target_file.touch()
        print(f"✅ PATCH APPLIED: Created missing map file at:")
        print(f"   {target_file}")
        print("🚀 Dashboard should now launch without crashing.")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 STARTING PACKAGE REPAIR SEQUENCE...")
    success = fix_streamlit_mic_recorder()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
