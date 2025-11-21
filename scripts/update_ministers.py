#!/usr/bin/env python3
"""
Script to update all minister __init__ signatures to accept brain_hub parameter
"""
import re
from pathlib import Path

# Find all minister files
ministers_dir = Path("src/government")
minister_files = [
    f for f in ministers_dir.glob("*_minister.py")
    if f.name != "base_minister.py" and f.name != "education_minister.py"
]

print(f"Found {len(minister_files)} minister files to update:")
for f in minister_files:
    print(f"  - {f.name}")

for minister_file in minister_files:
    content = minister_file.read_text()
    
    # Pattern to find __init__ method
    # Look for def __init__(self, and capture until the closing ):
    init_pattern = r'(def __init__\([\s\S]*?\n\s{4}\):)'
    
    match = re.search(init_pattern, content)
    if match:
        init_signature = match.group(1)
        
        # Check if brain_hub is already there
        if 'brain_hub' in init_signature:
            print(f"✓ {minister_file.name} already has brain_hub parameter")
            continue
        
        # Add brain_hub parameter before the closing ):
        # Find the last parameter line before ):
        lines = init_signature.split('\n')
        
        # Find the line with ): 
        closing_line_idx = None
        for i, line in enumerate(lines):
            if '):' in line:
                closing_line_idx = i
                break
        
        if closing_line_idx is not None:
            # Insert brain_hub parameter before closing
            # Get the indentation of the previous parameter
            prev_line = lines[closing_line_idx - 1]
            indent = len(prev_line) - len(prev_line.lstrip())
            
            # Add comma to previous line if it doesn't have one
            if not prev_line.rstrip().endswith(','):
                lines[closing_line_idx - 1] = prev_line.rstrip() + ','
            
            # Insert brain_hub line
            brain_hub_line = ' ' * indent + 'brain_hub: Any = None'
            lines.insert(closing_line_idx, brain_hub_line)
            
            new_init_signature = '\n'.join(lines)
            
            # Replace in content
            content = content.replace(init_signature, new_init_signature)
            
            # Now find the super().__init__ call and add brain_hub to it
            super_pattern = r'(super\(\).__init__\([\s\S]*?\n\s{8}\))'
            super_match = re.search(super_pattern, content)
            
            if super_match:
                super_call = super_match.group(1)
                
                # Check if brain_hub is already in super call
                if 'brain_hub' not in super_call:
                    # Add brain_hub parameter
                    # Find the last line before closing )
                    super_lines = super_call.split('\n')
                    
                    # Find closing )
                    super_closing_idx = None
                    for i, line in enumerate(super_lines):
                        if line.strip() == ')':
                            super_closing_idx = i
                            break
                    
                    if super_closing_idx is not None:
                        # Add comma to previous line if needed
                        prev_line = super_lines[super_closing_idx - 1]
                        if not prev_line.rstrip().endswith(','):
                            super_lines[super_closing_idx - 1] = prev_line.rstrip() + ','
                        
                        # Get indentation
                        indent = len(prev_line) - len(prev_line.lstrip())
                        
                        # Insert brain_hub
                        brain_hub_line = ' ' * indent + 'brain_hub=brain_hub'
                        super_lines.insert(super_closing_idx, brain_hub_line)
                        
                        new_super_call = '\n'.join(super_lines)
                        content = content.replace(super_call, new_super_call)
            
            # Write back
            minister_file.write_text(content)
            print(f"✓ Updated {minister_file.name}")
        else:
            print(f"✗ Could not find closing ): in {minister_file.name}")
    else:
        print(f"✗ Could not find __init__ in {minister_file.name}")

print("\n✅ All ministers updated!")
