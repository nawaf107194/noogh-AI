#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Development Minister - File Master & Code Generator
====================================================

AI-powered code generation AND file system indexing.
"""

from typing import Optional, Dict, Any, List
import logging
import os
import json
import time
from pathlib import Path

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class DevelopmentMinister(BaseMinister):
    """
    Minister of Development - Senior Python developer AI + File Master.
    
    NEW Powers:
    - File system indexing
    - Directory mapping
    - Storage recommendations
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Development Minister."""
        super().__init__(
            name="Development Minister (File Master)",
            description="Senior Python developer & file system indexer.",
            brain=brain
        )
        
        self.system_prompt = """You are a Senior Python Developer with 10+ years of experience.
Write clean, well-documented, production-quality Python code.

Follow these principles:
- PEP 8 style guide
- Type hints where appropriate
- Docstrings for all functions/classes
- Error handling with try/except
- Clear variable names
- Comments for complex logic

Provide code only, with minimal explanation unless asked."""
    
    def index_directory(
        self,
        path: str,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively index a directory with file metadata.
        
        Args:
            path: Directory path to index
            max_depth: Maximum recursion depth
            current_depth: Current depth (internal)
        
        Returns:
            Index data
        """
        if current_depth > max_depth:
            return {"files": [], "directories": [], "error": "Max depth reached"}
        
        files = []
        directories = []
        total_size = 0
        file_types = {}
        
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        # Skip hidden files/dirs
                        if entry.name.startswith('.'):
                            continue
                        
                        if entry.is_file(follow_symlinks=False):
                            stat = entry.stat()
                            size_mb = stat.st_size / (1024 ** 2)
                            
                            # Determine file type
                            ext = Path(entry.name).suffix.lower()
                            file_type = ext if ext else "no_extension"
                            
                            file_info = {
                                "filename": entry.name,
                                "path": entry.path,
                                "size_mb": round(size_mb, 2),
                                "type": file_type,
                                "modified": time.ctime(stat.st_mtime)
                            }
                            
                            files.append(file_info)
                            total_size += size_mb
                            
                            # Count file types
                            file_types[file_type] = file_types.get(file_type, 0) + 1
                        
                        elif entry.is_dir(follow_symlinks=False):
                            # Recursively index subdirectory
                            if current_depth < max_depth:
                                subdir_index = self.index_directory(
                                    entry.path,
                                    max_depth,
                                    current_depth + 1
                                )
                                
                                directories.append({
                                    "name": entry.name,
                                    "path": entry.path,
                                    "file_count": len(subdir_index.get("files", [])),
                                    "subdirs": len(subdir_index.get("directories", []))
                                })
                                
                                # Merge stats
                                files.extend(subdir_index.get("files", []))
                                total_size += subdir_index.get("total_size_mb", 0)
                                
                                for ftype, count in subdir_index.get("file_types", {}).items():
                                    file_types[ftype] = file_types.get(ftype, 0) + count
                    
                    except PermissionError:
                        logger.warning(f"Permission denied: {entry.path}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing {entry.path}: {e}")
                        continue
        
        except PermissionError:
            logger.error(f"Permission denied: {path}")
            return {"error": f"Permission denied: {path}"}
        except Exception as e:
            logger.error(f"Error indexing {path}: {e}")
            return {"error": str(e)}
        
        return {
            "path": path,
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_size_mb": round(total_size, 2),
            "file_types": file_types
        }
    
    def save_index(self, index_data: Dict[str, Any], output_path: str = "data/file_index.json"):
        """
        Save file index to JSON.
        
        Args:
            index_data: Index data to save
            output_path: Output file path
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"✅ Index saved to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute development task (code generation OR file indexing).
        
        Args:
            task: Task description
            context: Optional context
        
        Returns:
            Generated code or file index
        """
        self.tasks_processed += 1
        
        try:
            task_lower = task.lower()
            
            # File indexing
            if "index" in task_lower or "scan" in task_lower:
                # Extract path from context or use default
                index_path = context.get("path", ".") if context else "."
                max_depth = context.get("max_depth", 3) if context else 3
                
                logger.info(f"📂 Indexing directory: {index_path}")
                
                # Index directory
                index_data = self.index_directory(index_path, max_depth=max_depth)
                
                if "error" in index_data:
                    return {
                        "success": False,
                        "response": f"Indexing failed: {index_data['error']}",
                        "minister": self.name,
                        "error": index_data["error"]
                    }
                
                # Save index
                self.save_index(index_data)
                
                # Generate AI analysis
                total_files = index_data.get("total_files", 0)
                total_size = index_data.get("total_size_mb", 0)
                file_types = index_data.get("file_types", {})
                
                structure_summary = f"""Indexed {total_files} files ({total_size:.1f} MB)

File Types Distribution:
{chr(10).join([f"- {ftype}: {count} files" for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]])}

Analysis Request:
Based on this directory structure, suggest:
1. Where to store new AI models
2. Recommended folder organization
3. Files that could be archived/cleaned up"""
                
                analysis = await self._think_with_prompt(
                    system_prompt="You are a File System Architect. Analyze directory structures and provide optimization recommendations.",
                    user_message=structure_summary,
                    max_tokens=500
                )
                
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": analysis,
                    "minister": self.name,
                    "domain": "file_indexing",
                    "metadata": {
                        "index_data": {
                            "total_files": total_files,
                            "total_size_mb": total_size,
                            "file_types": file_types
                        }
                    }
                }
            
            # Code generation (original functionality)
            else:
                response = await self._think_with_prompt(
                    system_prompt=self.system_prompt,
                    user_message=f"Write Python code for: {task}",
                    max_tokens=800
                )
                
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": response,
                    "minister": self.name,
                    "domain": "development",
                    "metadata": {
                        "language": "python"
                    }
                }
        
        except Exception as e:
            logger.error(f"Development Minister error: {e}")
            return {
                "success": False,
                "response": f"Task failed: {str(e)}",
                "minister": self.name,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["DevelopmentMinister"]
