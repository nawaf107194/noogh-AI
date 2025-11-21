#!/usr/bin/env python3
"""
Script to automatically fix all sys.path.insert() hacks in the codebase.
Removes sys.path manipulation and ensures proper package structure.
"""

import re
from pathlib import Path
from typing import List, Tuple

def find_files_with_syspath(root_dir: Path) -> List[Path]:
    """Find all Python files that contain sys.path.insert()"""
    files_with_syspath = []
    
    for py_file in root_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            if 'sys.path.insert' in content or 'sys.path.append' in content:
                files_with_syspath.append(py_file)
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return files_with_syspath

def remove_syspath_manipulation(file_path: Path) -> Tuple[bool, str]:
    """
    Remove sys.path.insert/append and related imports from a file.
    Returns (changed, new_content)
    """
    content = file_path.read_text(encoding='utf-8')
    original = content
    lines = content.split('\n')
    new_lines = []
    
    skip_next_blank = False
    removed_sys_import = False
    
    for i, line in enumerate(lines):
        # Skip sys.path.insert/append lines
        if 'sys.path.insert' in line or 'sys.path.append' in line:
            skip_next_blank = True
            continue
        
        # Skip standalone "import sys" if it's only used for path manipulation
        if line.strip() == 'import sys':
            # Look ahead to see if sys is used for anything else
            has_other_sys_usage = False
            for future_line in lines[i+1:]:
                if 'sys.' in future_line and 'sys.path' not in future_line:
                    has_other_sys_usage = True
                    break
            
            if not has_other_sys_usage:
                removed_sys_import = True
                skip_next_blank = True
                continue
        
        # Skip "from pathlib import Path" if only used for sys.path
        if 'from pathlib import Path' in line and not removed_sys_import:
            # Check if Path is used elsewhere
            has_other_path_usage = any('Path(' in l and 'sys.path' not in l for l in lines)
            if not has_other_path_usage:
                skip_next_blank = True
                continue
        
        # Skip one blank line after removing sys.path stuff
        if skip_next_blank and line.strip() == '':
            skip_next_blank = False
            continue
        
        new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    changed = (new_content != original)
    
    return changed, new_content

def main():
    """Main function to fix all files"""
    project_root = Path('/home/noogh/projects/noogh_unified_system')
    src_dir = project_root / 'src'
    
    print("🔍 Finding files with sys.path manipulation...")
    files = find_files_with_syspath(src_dir)
    print(f"Found {len(files)} files to fix\n")
    
    fixed_count = 0
    for file_path in files:
        rel_path = file_path.relative_to(project_root)
        changed, new_content = remove_syspath_manipulation(file_path)
        
        if changed:
            file_path.write_text(new_content, encoding='utf-8')
            print(f"✅ Fixed: {rel_path}")
            fixed_count += 1
        else:
            print(f"⏭️  No changes needed: {rel_path}")
    
    print(f"\n🎉 Done! Fixed {fixed_count} files out of {len(files)} total")

if __name__ == "__main__":
    main()
