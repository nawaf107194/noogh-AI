#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 HARD RESET - System Cleanup Script
======================================

Clears all temporary files, corrupted charts, and emergency stops.
Use this to prepare the system for a fresh start.

Usage:
    python scripts/hard_reset.py
"""

import os
import shutil
from pathlib import Path
from typing import List


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def delete_files_in_directory(directory: Path, pattern: str = "*") -> int:
    """
    Delete all files matching pattern in directory.

    Args:
        directory: Directory path
        pattern: Glob pattern (default: all files)

    Returns:
        Number of files deleted
    """
    count = 0
    if not directory.exists():
        print(f"   ⚠️  Directory does not exist: {directory}")
        return 0

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {file_path.name}: {e}")

    return count


def delete_directory(directory: Path) -> bool:
    """
    Delete entire directory and its contents.

    Args:
        directory: Directory path

    Returns:
        True if successful, False otherwise
    """
    if not directory.exists():
        return False

    try:
        shutil.rmtree(directory)
        return True
    except Exception as e:
        print(f"   ❌ Failed to delete {directory}: {e}")
        return False


def main():
    """Main cleanup routine."""
    project_root = get_project_root()

    print("━" * 70)
    print("🔥 NOOGH HARD RESET - System Cleanup")
    print("━" * 70)
    print()

    # Track statistics
    total_deleted = 0
    directories_cleaned: List[str] = []

    # 1. Clear all chart images
    print("📊 Clearing chart images...")
    charts_dir = project_root / "data" / "charts"
    deleted = delete_files_in_directory(charts_dir, "*.png")
    total_deleted += deleted
    if deleted > 0:
        directories_cleaned.append(f"data/charts/ ({deleted} images)")
    print(f"   ✅ Deleted {deleted} chart files")
    print()

    # 2. Remove EMERGENCY_STOP file
    print("🛑 Removing emergency stop flag...")
    stop_file = project_root / "data" / "EMERGENCY_STOP"
    if stop_file.exists():
        try:
            stop_file.unlink()
            print("   ✅ Emergency stop removed")
            total_deleted += 1
        except Exception as e:
            print(f"   ❌ Failed to remove: {e}")
    else:
        print("   ℹ️  No emergency stop file found")
    print()

    # 3. Clear temporary files
    print("🗑️  Clearing temporary files...")
    temp_dir = project_root / "temp"
    deleted = delete_files_in_directory(temp_dir)
    total_deleted += deleted
    if deleted > 0:
        directories_cleaned.append(f"temp/ ({deleted} files)")
    print(f"   ✅ Deleted {deleted} temporary files")
    print()

    # 4. Clear Python cache
    print("🐍 Clearing Python cache...")
    pycache_count = 0
    for pycache_dir in project_root.rglob("__pycache__"):
        if delete_directory(pycache_dir):
            pycache_count += 1
    if pycache_count > 0:
        directories_cleaned.append(f"__pycache__/ ({pycache_count} dirs)")
    print(f"   ✅ Cleared {pycache_count} __pycache__ directories")
    print()

    # 5. Clear .pyc files
    print("🔧 Clearing compiled Python files...")
    pyc_count = 0
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_count += 1
        except Exception:
            pass
    total_deleted += pyc_count
    print(f"   ✅ Deleted {pyc_count} .pyc files")
    print()

    # 6. Clear old log files (optional - keep recent)
    print("📜 Cleaning old logs...")
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        # Keep api.log but clear .log.1, .log.2, etc.
        old_logs = list(logs_dir.glob("*.log.*"))
        for log_file in old_logs:
            try:
                log_file.unlink()
                total_deleted += 1
            except Exception:
                pass
        print(f"   ✅ Cleared {len(old_logs)} old log backups")
    else:
        print("   ℹ️  No logs directory found")
    print()

    # Summary
    print("━" * 70)
    print("✅ CLEANUP COMPLETE")
    print("━" * 70)
    print(f"   Total items deleted: {total_deleted}")
    if directories_cleaned:
        print(f"   Cleaned directories:")
        for dir_info in directories_cleaned:
            print(f"      • {dir_info}")
    print()
    print("🚀 System is now ready for a fresh start!")
    print("   Run: streamlit run src/interface/dashboard.py")
    print("━" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cleanup interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
