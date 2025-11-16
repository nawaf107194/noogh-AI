#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Indexer - Fallback implementation for knowledge indexing
"""

import sys
from pathlib import Path
from datetime import datetime
from enum import Enum

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    # Try to import from core if it exists
    from core.knowledge_indexer import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import from core.knowledge_indexer: {e}")
    
    # Provide fallback implementation
    class KnowledgeType(Enum):
        FACTUAL = "factual"
        PROCEDURAL = "procedural"
        CONCEPTUAL = "conceptual"
        METACOGNITIVE = "metacognitive"
    
    class KnowledgeEntry:
        def __init__(self, content, knowledge_type=KnowledgeType.FACTUAL, metadata=None):
            self.content = content
            self.knowledge_type = knowledge_type
            self.metadata = metadata or {}
            self.created_at = datetime.now()
            self.updated_at = datetime.now()
            self.id = hash(content + str(datetime.now()))
        
        def to_dict(self):
            return {
                "id": self.id,
                "content": self.content,
                "type": self.knowledge_type.value if isinstance(self.knowledge_type, KnowledgeType) else self.knowledge_type,
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }
    
    class KnowledgeIndexer:
        def __init__(self):
            self.entries = []
            self.index = {}
            self.metadata = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "fallback": True
            }
        
        def add_knowledge(self, content, knowledge_type=KnowledgeType.FACTUAL, metadata=None):
            """Add knowledge to the index"""
            entry = KnowledgeEntry(content, knowledge_type, metadata)
            self.entries.append(entry)
            
            # Simple indexing by keywords
            words = content.lower().split()
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(entry.id)
            
            return entry
        
        def search(self, query):
            """Search for knowledge"""
            words = query.lower().split()
            matching_ids = set()
            
            for word in words:
                if word in self.index:
                    matching_ids.update(self.index[word])
            
            # Return matching entries
            results = [e for e in self.entries if e.id in matching_ids]
            return [e.to_dict() for e in results]
        
        def get_by_type(self, knowledge_type):
            """Get all knowledge of a specific type"""
            results = [e for e in self.entries if e.knowledge_type == knowledge_type]
            return [e.to_dict() for e in results]
        
        def get_all(self):
            """Get all knowledge entries"""
            return [e.to_dict() for e in self.entries]
        
        def get_stats(self):
            """Get indexer statistics"""
            return {
                "total_entries": len(self.entries),
                "indexed_words": len(self.index),
                "types": {
                    kt.value: len([e for e in self.entries if e.knowledge_type == kt])
                    for kt in KnowledgeType
                },
                "fallback": True
            }
        
        def load_index(self, filepath=None):
            """Load index from file (fallback does nothing)"""
            return {"status": "loaded", "fallback": True, "entries": len(self.entries)}
        
        def save_index(self, filepath=None):
            """Save index to file (fallback does nothing)"""
            return {"status": "saved", "fallback": True, "entries": len(self.entries)}

# Singleton instance
_indexer = None

def get_knowledge_indexer():
    """Get or create knowledge indexer singleton"""
    global _indexer
    if _indexer is None:
        _indexer = KnowledgeIndexer()
    return _indexer

__all__ = ['KnowledgeType', 'KnowledgeEntry', 'KnowledgeIndexer', 'get_knowledge_indexer']
