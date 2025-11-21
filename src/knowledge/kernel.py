#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Noogh Knowledge Kernel v4.1 - Enhanced Cognitive Core
========================================================

Enhanced unified cognitive core that integrates:
- Neural Brain v4.0 (32K neurons)
- Knowledge Base (287K+ chunks)
- ALLaM Bridge
- Intent Understanding & Routing ✨ NEW
- External Awareness (Web Search) ✨ NEW
- Reflection & Learning System ✨ NEW
- Search Engine (keyword + semantic hybrid)

This is the evolved heart of Noogh v4.1 - now with awareness and learning!

Author: Noogh AI Team
Version: 4.1.0
Date: 2025-10-25
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import time
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeKernelV41:
    """
    🌐 النواة المعرفية الموحدة v4.1

    Enhanced unified system that combines:
    - Neural Brain v4.0 (32,000 neurons)
    - Knowledge Search (287K+ chunks)
    - ALLaM Integration
    - Intent Routing ✨ NEW
    - Web Search ✨ NEW
    - Reflection & Learning ✨ NEW
    """

    def __init__(
        self,
        knowledge_index_path: str = "/home/noogh/projects/noogh_unified_system/data/simple_index.json",
        brain_device: str = "cpu",
        enable_brain: bool = False,  # CPU expensive, disabled by default
        enable_allam: bool = True,
        enable_intent_routing: bool = True,  # ✨ NEW
        enable_web_search: bool = True,  # ✨ NEW
        enable_reflection: bool = True,  # ✨ NEW
        reflection_db_path: str = "data/reflection.db"
    ):
        """Initialize Enhanced Knowledge Kernel v4.1"""

        logger.info("=" * 80)
        logger.info("🌐 Initializing Noogh Knowledge Kernel v4.1...")
        logger.info("=" * 80)

        # Load knowledge base
        self.knowledge_index = []
        self.knowledge_loaded = False

        try:
            if Path(knowledge_index_path).exists():
                with open(knowledge_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'chunks' in data:
                        self.knowledge_index = data['chunks']
                    elif isinstance(data, list):
                        self.knowledge_index = data
                    else:
                        self.knowledge_index = []
                        logger.warning("⚠️ Unknown knowledge index format")
                
                self.knowledge_loaded = True
                logger.info(f"✅ Knowledge Base: {len(self.knowledge_index):,} chunks loaded")
            else:
                logger.warning(f"⚠️ Knowledge index not found at {knowledge_index_path}")
        except Exception as e:
            logger.error(f"❌ Error loading knowledge: {e}")

        # Neural Brain v4.0 (optional - CPU expensive)
        self.brain = None
        self.brain_enabled = enable_brain

        if enable_brain:
            try:
                # Try different import paths for brain_v4
                # Try different import paths for brain_v4
                try:
                    from src.brain.core import create_brain_v4
                except ImportError:
                    try:
                        from brain.core import create_brain_v4
                    except ImportError:
                        from src.brain_v4 import create_brain_v4 # Fallback just in case

                self.brain = create_brain_v4(device=brain_device)
                logger.info(f"✅ Neural Brain v4.0: {self.brain.total_neurons:,} neurons active")
            except Exception as e:
                logger.error(f"❌ Error loading Brain v4.0: {e}")
                self.brain_enabled = False
        else:
            logger.info("ℹ️ Neural Brain v4.0: Disabled (enable_brain=False)")

        # ALLaM Bridge (Lazy Loading)
        self.allam = None
        self.allam_enabled = enable_allam
        self._allam_loading = False  # Flag to prevent concurrent loading
        self._allam_loading_lock = threading.Lock()  # Thread-safe lock for lazy loading

        # Don't load ALLaM at initialization - use lazy loading
        if enable_allam:
            logger.info("ℹ️ ALLaM Bridge: Ready for lazy loading (saves ~4GB RAM)")
        else:
            logger.info("ℹ️ ALLaM Bridge: Disabled")

        # ✨ NEW: Intent Router
        self.intent_router = None
        self.intent_routing_enabled = enable_intent_routing

        if enable_intent_routing:
            try:
                # Try different import paths for IntentRouter
                # Try different import paths for IntentRouter
                try:
                    from src.nlp.intent import IntentRouter
                except ImportError:
                    try:
                        from nlp.intent import IntentRouter
                    except ImportError:
                        from src.intent import IntentRouter # Fallback

                self.intent_router = IntentRouter(
                    knowledge_kernel=None,  # Will set self-reference later
                    enable_web_search=enable_web_search
                )
                logger.info("✅ Intent Router: Active (11 intent types)")
            except Exception as e:
                logger.error(f"❌ Error loading Intent Router: {e}")
                self.intent_routing_enabled = False
        else:
            logger.info("ℹ️ Intent Router: Disabled")

        # ✨ NEW: Web Search
        self.web_search = None
        self.web_search_enabled = enable_web_search

        if enable_web_search:
            try:
                # Try different import paths for external_awareness
                try:
                    from src.external_awareness import WebSearchProvider
                except ImportError:
                    try:
                        from .external_awareness import WebSearchProvider
                    except ImportError:
                        from external_awareness import WebSearchProvider

                self.web_search = WebSearchProvider(cache_ttl_hours=24)
                logger.info("✅ Web Search: Active (DuckDuckGo)")
            except Exception as e:
                logger.error(f"❌ Error loading Web Search: {e}")
                self.web_search_enabled = False
        else:
            logger.info("ℹ️ Web Search: Disabled")

        # ✨ NEW: Reflection & Learning
        self.reflection_enabled = enable_reflection
        self.experience_tracker = None

        if enable_reflection:
            try:
                # Try different import paths for reflection
                try:
                    from src.reflection import ExperienceTracker
                except ImportError:
                    try:
                        from .reflection import ExperienceTracker
                    except ImportError:
                        from reflection import ExperienceTracker

                self.experience_tracker = ExperienceTracker(db_path=reflection_db_path)
                logger.info("✅ Reflection System: Active (tracking all experiences)")
            except Exception as e:
                logger.error(f"❌ Error loading Reflection System: {e}")
                self.reflection_enabled = False
        else:
            logger.info("ℹ️ Reflection System: Disabled")

        # Set self-reference for Intent Router
        if self.intent_router:
            self.intent_router.knowledge_kernel = self

        # Statistics
        self.query_count = 0
        self.search_count = 0
        self.learning_cycles = 0

        # Learning buffer (for backward compatibility)
        self.learning_buffer = []
        self.max_learning_buffer = 1000

        logger.info("=" * 80)
        logger.info("🎉 Knowledge Kernel v4.1 Ready!")
        logger.info(f"   Knowledge: {len(self.knowledge_index):,} chunks")
        logger.info(f"   Brain v4.0: {'Enabled' if self.brain_enabled else 'Disabled'}")
        logger.info(f"   ALLaM: {'Lazy Loading' if self.allam_enabled else 'Disabled'}")
        logger.info(f"   Intent Routing: {'Enabled' if self.intent_routing_enabled else 'Disabled'}")
        logger.info(f"   Web Search: {'Enabled' if self.web_search_enabled else 'Disabled'}")
        logger.info(f"   Reflection: {'Enabled' if self.reflection_enabled else 'Disabled'}")
        logger.info("=" * 80)

    # ... (existing code) ...

    def _load_embedding_model(self):
        """Lazy load the embedding model"""
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            try:
                logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
            except ImportError:
                logger.error("❌ sentence-transformers not installed. Vector memory disabled.")
                self.embedding_model = None
            except Exception as e:
                logger.error(f"❌ Error loading embedding model: {e}")
                self.embedding_model = None

    def learn(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        🧠 Learn a new fact (store in vector memory)
        
        Args:
            text: The information to learn
            metadata: Additional info (source, timestamp, etc.)
        """
        if not text:
            return False
            
        self._load_embedding_model()
        if not self.embedding_model:
            return False
            
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Create memory entry
            memory_entry = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to knowledge index (in-memory for now, persistence can be added)
            self.knowledge_index.append(memory_entry)
            self.learning_cycles += 1
            
            logger.info(f"🧠 Learned: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Learning failed: {e}")
            return False

    def recall(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        💭 Recall information based on semantic similarity
        
        Args:
            query: The query to recall info for
            top_k: Number of results
            threshold: Minimum similarity score (0-1)
        """
        self._load_embedding_model()
        if not self.embedding_model:
            return self.search_knowledge(query, top_k, method="keyword")
            
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 1. Embed query
            query_embedding = self.embedding_model.encode(query).reshape(1, -1)
            
            # 2. Get all memory embeddings
            memories = [m for m in self.knowledge_index if 'embedding' in m]
            if not memories:
                return []
                
            memory_embeddings = np.array([m['embedding'] for m in memories])
            
            # 3. Calculate similarity
            similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
            
            # 4. Filter and sort results
            results = []
            for i, score in enumerate(similarities):
                if score >= threshold:
                    memory = memories[i].copy()
                    memory['score'] = float(score)
                    del memory['embedding'] # Don't return the vector
                    results.append(memory)
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Recall failed: {e}")
            return self.search_knowledge(query, top_k, method="keyword")

    def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
        method: str = "keyword"
    ) -> List[Dict[str, Any]]:
        """
        🔍 بحث في قاعدة المعرفة

        Args:
            query: Search query
            top_k: Number of results
            method: "keyword" or "hybrid" (future)

        Returns:
            List of search results with scores
        """
        self.search_count += 1

        if not self.knowledge_loaded:
            logger.warning("⚠️ Knowledge base not loaded")
            return []

        if method == "keyword":
            return self._keyword_search(query, top_k)
        elif method == "hybrid":
            # TODO: Implement semantic + keyword hybrid
            logger.warning("⚠️ Hybrid search not implemented yet, falling back to keyword")
            return self._keyword_search(query, top_k)
        else:
            raise ValueError(f"Unknown search method: {method}")

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword-based search (fast)"""
        keywords = set(re.findall(r'\w+', query.lower()))

        results = []
        for chunk in self.knowledge_index:
            # Handle different chunk formats
            if isinstance(chunk, str):
                text = chunk.lower()
                chunk_data = {'text': chunk}
            elif isinstance(chunk, dict):
                # Check 'text', 'full_chunk', or 'chunk' keys
                text = chunk.get('text', chunk.get('full_chunk', chunk.get('chunk', ''))).lower()
                chunk_data = chunk.copy()
            else:
                continue

            matches = sum(1 for kw in keywords if kw in text)

            if matches > 0:
                chunk_data['score'] = matches
                chunk_data['method'] = 'keyword'
                results.append(chunk_data)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        use_brain: bool = False,
        use_allam: bool = True,
        use_intent_routing: bool = True,  # ✨ NEW
        track_experience: bool = True  # ✨ NEW
    ) -> Dict[str, Any]:
        """
        💡 اسأل نوغ سؤالاً (Enhanced with Intent Routing)

        Complete intelligent question answering using all systems

        Args:
            question: User question
            context: Optional context (session, user info, etc.)
            use_brain: Use Neural Brain v4.0 for processing
            use_allam: Use ALLaM for linguistic analysis
            use_intent_routing: Use Intent Router for intelligent response ✨ NEW
            track_experience: Track this query in reflection system ✨ NEW

        Returns:
            Dictionary with answer, sources, confidence, etc.
        """
        self.query_count += 1
        start_time = time.time()

        logger.info(f"💡 Query #{self.query_count}: {question[:100]}...")

        # ✨ NEW: Route through Intent System if enabled
        if use_intent_routing and self.intent_routing_enabled and self.intent_router:
            try:
                response = self.intent_router.route(question, context)

                # Convert RouterResponse to dict
                result = {
                    "answer": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "intent": response.intent.value,
                    "handler": response.handler_used,
                    "processing_time": response.processing_time,
                    "method": "intent_routing",
                    "metadata": response.metadata
                }

                # ✨ Track experience if enabled
                if track_experience and self.reflection_enabled and self.experience_tracker:
                    success = response.confidence >= 0.5 and len(response.sources) > 0
                    self.experience_tracker.track(
                        question=question,
                        intent=response.intent.value,
                        answer=response.answer,
                        sources=response.sources,
                        confidence=response.confidence,
                        success=success,
                        execution_time=response.processing_time,
                        handler=response.handler_used,
                        used_web_search=response.metadata.get('used_web_search', False)
                    )

                logger.info(f"✅ Answer generated via Intent Router in {response.processing_time:.3f}s")
                return result

            except Exception as e:
                logger.error(f"❌ Intent Router error: {e}, falling back to legacy method")
                # Fall through to legacy method

        # Legacy method (v4.0 compatible)
        return self._legacy_ask(question, context, use_brain, use_allam)

    def _legacy_ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        use_brain: bool,
        use_allam: bool
    ) -> Dict[str, Any]:
        """
        Legacy ask method (v4.0 compatible)
        Used as fallback when intent routing is disabled or fails
        """
        start_time = time.time()

        # 1. Search knowledge base
        search_results = self.search_knowledge(question, top_k=5)

        # 2. Extract context from search results
        knowledge_context = []
        sources = []

        for result in search_results[:3]:
            chunk_text = result.get('full_chunk', result.get('chunk', ''))
            knowledge_context.append(chunk_text)
            if result.get('path'):
                sources.append(result['path'])

        context_text = "\n\n".join(knowledge_context)

        # 3. ALLaM linguistic analysis (if enabled with lazy loading)
        linguistic_analysis = {}
        if use_allam and self.allam_enabled:
            # 🔄 Lazy load ALLaM only when needed
            if self._load_allam_if_needed() and self.allam is not None:
                try:
                    # Check if the expected method exists
                    if hasattr(self.allam, '_analyze_intent') and callable(self.allam._analyze_intent):
                        linguistic_analysis = self.allam._analyze_intent(question)
                    elif hasattr(self.allam, 'analyze') and callable(self.allam.analyze):
                        linguistic_analysis = self.allam.analyze(question)
                    elif hasattr(self.allam, 'analyze_intent') and callable(self.allam.analyze_intent):
                        linguistic_analysis = self.allam.analyze_intent(question)
                    else:
                        logger.warning("ALLaM doesn't have expected analysis method")
                except Exception as e:
                    logger.warning(f"ALLaM analysis failed: {e}")
            else:
                logger.warning("ALLaM not available, skipping linguistic analysis")

        # 4. Neural Brain processing (if enabled)
        brain_output = {}
        if use_brain and self.brain_enabled and self.brain:
            try:
                brain_output = self.brain.process(question, context_text)
            except Exception as e:
                logger.warning(f"Brain processing failed: {e}")

        # 5. Generate answer
        if context_text:
            answer = f"📚 من قاعدة المعرفة:\n{context_text[:1500]}"
            if len(context_text) > 1500:
                answer += "..."

            if sources:
                answer += f"\n\n📁 المصادر:\n"
                for i, source in enumerate(sources[:5], 1):
                    filename = source.split('/')[-1]
                    answer += f"   {i}. {filename}\n"

            confidence = min(0.7 + (len(search_results) * 0.05), 0.95)
        else:
            answer = "لم أجد معلومات كافية في قاعدة المعرفة الداخلية."
            confidence = 0.1

        processing_time = time.time() - start_time

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "linguistic_analysis": linguistic_analysis,
            "brain_output": brain_output,
            "search_results_count": len(search_results),
            "processing_time": processing_time,
            "method": "legacy"
        }

        # Store in learning buffer (v4.0 compatible)
        self._store_learning_experience({
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"✅ Answer generated in {processing_time:.3f}s ({len(sources)} sources)")

        return result

    def _store_learning_experience(self, experience: Dict[str, Any]):
        """Store experience in learning buffer (v4.0 compatible)"""
        self.learning_buffer.append(experience)

        # Keep buffer size under limit
        if len(self.learning_buffer) > self.max_learning_buffer:
            self.learning_buffer = self.learning_buffer[-self.max_learning_buffer:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "version": "4.1.0",
            "knowledge_base": {
                "chunks": len(self.knowledge_index),
                "loaded": self.knowledge_loaded
            },
            "brain_v4": {
                "enabled": self.brain_enabled,
                "neurons": self.brain.total_neurons if self.brain else 0
            },
            "allam": {
                "enabled": self.allam_enabled
            },
            "intent_routing": {
                "enabled": self.intent_routing_enabled
            },
            "web_search": {
                "enabled": self.web_search_enabled
            },
            "reflection": {
                "enabled": self.reflection_enabled
            },
            "usage": {
                "total_queries": self.query_count,
                "total_searches": self.search_count,
                "learning_cycles": self.learning_cycles,
                "buffer_size": len(self.learning_buffer)
            }
        }

        # Add session stats from experience tracker
        if self.reflection_enabled and self.experience_tracker:
            try:
                session_stats = self.experience_tracker.get_session_stats()
                stats["session"] = session_stats
            except:
                pass

        return stats


# Factory function
def create_knowledge_kernel(
    enable_brain: bool = False,
    enable_allam: bool = True,
    enable_intent_routing: bool = True,
    enable_web_search: bool = True,
    enable_reflection: bool = True,
    brain_device: str = "cpu"
) -> KnowledgeKernelV41:
    """
    Factory function to create Knowledge Kernel v4.1

    Args:
        enable_brain: Enable Neural Brain v4.0 (CPU expensive)
        enable_allam: Enable ALLaM linguistic analysis
        enable_intent_routing: Enable intent-based routing ✨ NEW
        enable_web_search: Enable web search fallback ✨ NEW
        enable_reflection: Enable experience tracking ✨ NEW
        brain_device: Device for brain ("cpu" or "cuda")

    Returns:
        KnowledgeKernelV41 instance
    """
    return KnowledgeKernelV41(
        enable_brain=enable_brain,
        enable_allam=enable_allam,
        enable_intent_routing=enable_intent_routing,
        enable_web_search=enable_web_search,
        enable_reflection=enable_reflection,
        brain_device=brain_device
    )


if __name__ == "__main__":
    # Test the enhanced kernel
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("🧪 Testing Knowledge Kernel v4.1")
    print("=" * 80 + "\n")

    # Create kernel with all v4.1 features
    kernel = create_knowledge_kernel(
        enable_brain=False,  # CPU expensive
        enable_allam=True,
        enable_intent_routing=True,  # ✨ NEW
        enable_web_search=True,  # ✨ NEW
        enable_reflection=True  # ✨ NEW
    )

    # Test queries
    test_queries = [
        "What is PyTorch?",
        "ما هو نوغ v4.1؟",
        "Compare Python and JavaScript"
    ]

    print("\n" + "-" * 80)
    print("Running test queries...")
    print("-" * 80 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        result = kernel.ask(query)

        print(f"\nIntent: {result.get('intent', 'N/A')}")
        print(f"Handler: {result.get('handler', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Sources: {len(result.get('sources', []))}")
        print(f"Processing time: {result.get('processing_time', 0):.3f}s")
        print(f"Method: {result.get('method', 'N/A')}")
        print(f"\nAnswer preview:")
        print(result.get('answer', '')[:200] + "...")

    # Statistics
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)

    stats = kernel.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n✅ All tests completed!")

    def ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        use_brain: bool = False,
        use_allam: bool = True,
        use_intent_routing: bool = True,  # ✨ NEW
        track_experience: bool = True  # ✨ NEW
    ) -> Dict[str, Any]:
        """
        💡 اسأل نوغ سؤالاً (Enhanced with Intent Routing)

        Complete intelligent question answering using all systems

        Args:
            question: User question
            context: Optional context (session, user info, etc.)
            use_brain: Use Neural Brain v4.0 for processing
            use_allam: Use ALLaM for linguistic analysis
            use_intent_routing: Use Intent Router for intelligent response ✨ NEW
            track_experience: Track this query in reflection system ✨ NEW

        Returns:
            Dictionary with answer, sources, confidence, etc.
        """
        self.query_count += 1
        start_time = time.time()

        logger.info(f"💡 Query #{self.query_count}: {question[:100]}...")

        # ✨ NEW: Route through Intent System if enabled
        if use_intent_routing and self.intent_routing_enabled and self.intent_router:
            try:
                response = self.intent_router.route(question, context)

                # Convert RouterResponse to dict
                result = {
                    "answer": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "intent": response.intent.value,
                    "handler": response.handler_used,
                    "processing_time": response.processing_time,
                    "method": "intent_routing",
                    "metadata": response.metadata
                }

                # ✨ Track experience if enabled
                if track_experience and self.reflection_enabled and self.experience_tracker:
                    success = response.confidence >= 0.5 and len(response.sources) > 0
                    self.experience_tracker.track(
                        question=question,
                        intent=response.intent.value,
                        answer=response.answer,
                        sources=response.sources,
                        confidence=response.confidence,
                        success=success,
                        execution_time=response.processing_time,
                        handler=response.handler_used,
                        used_web_search=response.metadata.get('used_web_search', False)
                    )

                logger.info(f"✅ Answer generated via Intent Router in {response.processing_time:.3f}s")
                return result

            except Exception as e:
                logger.error(f"❌ Intent Router error: {e}, falling back to legacy method")
                # Fall through to legacy method

        # Legacy method (v4.0 compatible)
        return self._legacy_ask(question, context, use_brain, use_allam)

    def _legacy_ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        use_brain: bool,
        use_allam: bool
    ) -> Dict[str, Any]:
        """
        Legacy ask method (v4.0 compatible)
        Used as fallback when intent routing is disabled or fails
        """
        start_time = time.time()

        # 1. Search knowledge base
        search_results = self.search_knowledge(question, top_k=5)

        # 2. Extract context from search results
        knowledge_context = []
        sources = []

        for result in search_results[:3]:
            chunk_text = result.get('full_chunk', result.get('chunk', ''))
            knowledge_context.append(chunk_text)
            if result.get('path'):
                sources.append(result['path'])

        context_text = "\n\n".join(knowledge_context)

        # 3. ALLaM linguistic analysis (if enabled with lazy loading)
        linguistic_analysis = {}
        if use_allam and self.allam_enabled:
            # 🔄 Lazy load ALLaM only when needed
            if self._load_allam_if_needed() and self.allam is not None:
                try:
                    # Check if the expected method exists
                    if hasattr(self.allam, '_analyze_intent') and callable(self.allam._analyze_intent):
                        linguistic_analysis = self.allam._analyze_intent(question)
                    elif hasattr(self.allam, 'analyze') and callable(self.allam.analyze):
                        linguistic_analysis = self.allam.analyze(question)
                    elif hasattr(self.allam, 'analyze_intent') and callable(self.allam.analyze_intent):
                        linguistic_analysis = self.allam.analyze_intent(question)
                    else:
                        logger.warning("ALLaM doesn't have expected analysis method")
                except Exception as e:
                    logger.warning(f"ALLaM analysis failed: {e}")
            else:
                logger.warning("ALLaM not available, skipping linguistic analysis")

        # 4. Neural Brain processing (if enabled)
        brain_output = {}
        if use_brain and self.brain_enabled and self.brain:
            try:
                brain_output = self.brain.process(question, context_text)
            except Exception as e:
                logger.warning(f"Brain processing failed: {e}")

        # 5. Generate answer
        if context_text:
            answer = f"📚 من قاعدة المعرفة:\n{context_text[:1500]}"
            if len(context_text) > 1500:
                answer += "..."

            if sources:
                answer += f"\n\n📁 المصادر:\n"
                for i, source in enumerate(sources[:5], 1):
                    filename = source.split('/')[-1]
                    answer += f"   {i}. {filename}\n"

            confidence = min(0.7 + (len(search_results) * 0.05), 0.95)
        else:
            answer = "لم أجد معلومات كافية في قاعدة المعرفة الداخلية."
            confidence = 0.1

        processing_time = time.time() - start_time

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "linguistic_analysis": linguistic_analysis,
            "brain_output": brain_output,
            "search_results_count": len(search_results),
            "processing_time": processing_time,
            "method": "legacy"
        }

        # Store in learning buffer (v4.0 compatible)
        self._store_learning_experience({
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"✅ Answer generated in {processing_time:.3f}s ({len(sources)} sources)")

        return result

    def _store_learning_experience(self, experience: Dict[str, Any]):
        """Store experience in learning buffer (v4.0 compatible)"""
        self.learning_buffer.append(experience)

        # Keep buffer size under limit
        if len(self.learning_buffer) > self.max_learning_buffer:
            self.learning_buffer = self.learning_buffer[-self.max_learning_buffer:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "version": "4.1.0",
            "knowledge_base": {
                "chunks": len(self.knowledge_index),
                "loaded": self.knowledge_loaded
            },
            "brain_v4": {
                "enabled": self.brain_enabled,
                "neurons": self.brain.total_neurons if self.brain else 0
            },
            "allam": {
                "enabled": self.allam_enabled
            },
            "intent_routing": {
                "enabled": self.intent_routing_enabled
            },
            "web_search": {
                "enabled": self.web_search_enabled
            },
            "reflection": {
                "enabled": self.reflection_enabled
            },
            "usage": {
                "total_queries": self.query_count,
                "total_searches": self.search_count,
                "learning_cycles": self.learning_cycles,
                "buffer_size": len(self.learning_buffer)
            }
        }

        # Add session stats from experience tracker
        if self.reflection_enabled and self.experience_tracker:
            try:
                session_stats = self.experience_tracker.get_session_stats()
                stats["session"] = session_stats
            except:
                pass

        return stats


# Factory function
def create_knowledge_kernel(
    enable_brain: bool = False,
    enable_allam: bool = True,
    enable_intent_routing: bool = True,
    enable_web_search: bool = True,
    enable_reflection: bool = True,
    brain_device: str = "cpu"
) -> KnowledgeKernelV41:
    """
    Factory function to create Knowledge Kernel v4.1

    Args:
        enable_brain: Enable Neural Brain v4.0 (CPU expensive)
        enable_allam: Enable ALLaM linguistic analysis
        enable_intent_routing: Enable intent-based routing ✨ NEW
        enable_web_search: Enable web search fallback ✨ NEW
        enable_reflection: Enable experience tracking ✨ NEW
        brain_device: Device for brain ("cpu" or "cuda")

    Returns:
        KnowledgeKernelV41 instance
    """
    return KnowledgeKernelV41(
        enable_brain=enable_brain,
        enable_allam=enable_allam,
        enable_intent_routing=enable_intent_routing,
        enable_web_search=enable_web_search,
        enable_reflection=enable_reflection,
        brain_device=brain_device
    )


if __name__ == "__main__":
    # Test the enhanced kernel
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("🧪 Testing Knowledge Kernel v4.1")
    print("=" * 80 + "\n")

    # Create kernel with all v4.1 features
    kernel = create_knowledge_kernel(
        enable_brain=False,  # CPU expensive
        enable_allam=True,
        enable_intent_routing=True,  # ✨ NEW
        enable_web_search=True,  # ✨ NEW
        enable_reflection=True  # ✨ NEW
    )

    # Test queries
    test_queries = [
        "What is PyTorch?",
        "ما هو نوغ v4.1؟",
        "Compare Python and JavaScript"
    ]

    print("\n" + "-" * 80)
    print("Running test queries...")
    print("-" * 80 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        result = kernel.ask(query)

        print(f"\nIntent: {result.get('intent', 'N/A')}")
        print(f"Handler: {result.get('handler', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Sources: {len(result.get('sources', []))}")
        print(f"Processing time: {result.get('processing_time', 0):.3f}s")
        print(f"Method: {result.get('method', 'N/A')}")
        print(f"\nAnswer preview:")
        print(result.get('answer', '')[:200] + "...")

    # Statistics
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)

    stats = kernel.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n✅ All tests completed!")