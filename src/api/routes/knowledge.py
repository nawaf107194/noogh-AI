"""
🔍 Knowledge Search API
البحث الذكي في قاعدة المعرفة المحلية
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_indexer import KnowledgeIndexer

# Import cache manager for performance
try:
    from api.utils.cache_manager import get_cache
    cache = get_cache()
    cache_available = True
except ImportError:
    cache_available = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Global indexer instance
indexer = None


class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    path: str
    chunk: str
    score: float
    file_type: str
    chunk_id: int


@router.on_event("startup")
async def load_knowledge_index():
    """تحميل الفهرس عند بدء التشغيل"""
    global indexer
    try:
        indexer = KnowledgeIndexer()
        if indexer.load_index():
            logger.info(f"✅ Knowledge index loaded: {len(indexer.metadata)} chunks")
        else:
            logger.warning("⚠️ No knowledge index found. Run indexer first.")
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge index: {e}")


@router.get("/health")
async def health():
    """فحص صحة النظام"""
    if indexer is None:
        return {"status": "error", "message": "Indexer not initialized"}
    
    return {
        "status": "healthy",
        "total_chunks": len(indexer.metadata) if indexer.metadata else 0,
        "index_loaded": indexer.index is not None
    }


@router.post("/search", response_model=List[SearchResult])
async def search_knowledge(query: SearchQuery):
    """
    البحث في قاعدة المعرفة

    Example:
        POST /knowledge/search
        {"query": "كيف يعمل النظام؟", "top_k": 5}
    """
    if indexer is None:
        raise HTTPException(status_code=503, detail="Knowledge indexer not initialized")

    if not indexer.metadata:
        raise HTTPException(status_code=503, detail="Knowledge index is empty. Run indexer first.")

    # Check cache first
    cache_key = f"knowledge:search:{query.query}:{query.top_k}"
    if cache_available:
        cached_results = cache.get(cache_key)
        if cached_results:
            logger.debug(f"✅ Cache hit for query: {query.query[:50]}...")
            return cached_results

    try:
        results = indexer.search(query.query, top_k=query.top_k)

        search_results = [
            SearchResult(
                path=r['path'],
                chunk=r['full_chunk'] if 'full_chunk' in r else r['chunk'],
                score=r['score'],
                file_type=r.get('file_type', 'unknown'),
                chunk_id=r.get('chunk_id', 0)
            )
            for r in results
        ]

        # Cache the results for 10 minutes
        if cache_available:
            cache.set(cache_key, search_results, ttl=600)
            logger.debug(f"💾 Cached results for query: {query.query[:50]}...")

        return search_results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild")
async def rebuild_index():
    """
    إعادة بناء الفهرس
    
    ⚠️ هذه العملية تأخذ وقت!
    """
    global indexer
    
    if indexer is None:
        indexer = KnowledgeIndexer()
    
    try:
        # Reset index
        indexer.metadata = []
        indexer.index = __import__('faiss').IndexFlatIP(indexer.dimension)
        
        # Rebuild
        stats = indexer.index_directory()
        
        # Save
        indexer.save_index()
        
        return {
            "status": "success",
            "message": "Index rebuilt successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """إحصائيات قاعدة المعرفة"""
    if indexer is None or not indexer.metadata:
        return {
            "total_chunks": 0,
            "files": {},
            "types": {}
        }
    
    # Count files and types
    files = {}
    types = {}
    
    for meta in indexer.metadata:
        # Handle both dict and string metadata
        if isinstance(meta, dict):
            path = meta.get('path', 'unknown')
            file_type = meta.get('file_type', 'unknown')
        else:
            # If meta is a string, use it as path
            path = str(meta)
            file_type = 'unknown'
        
        files[path] = files.get(path, 0) + 1
        types[file_type] = types.get(file_type, 0) + 1
    
    return {
        "total_chunks": len(indexer.metadata),
        "total_files": len(files),
        "files": dict(sorted(files.items(), key=lambda x: x[1], reverse=True)[:20]),
        "types": types
    }
