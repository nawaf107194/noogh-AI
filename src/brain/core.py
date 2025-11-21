#!/usr/bin/env python3
"""
🧠 Brain v4.0 - Enhanced Contextual Thinking Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced reasoning with session memory, contextual awareness, and pattern detection
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
from pathlib import Path

from src.core.database import SessionLocal
from src.core.models import Memory

logger = logging.getLogger(__name__)


class SessionMemory:
    """Session memory to store recent interactions"""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.memories: deque = deque(maxlen=max_size)
        self.session_start = datetime.now()

    def add(self, query: str, insights: Dict[str, Any]):
        """Add interaction to memory"""
        self.memories.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "insights": insights
        })

    def search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search memory for similar past interactions"""
        results = []
        for memory in self.memories:
            query_lower = memory["query"].lower()
            if any(kw.lower() in query_lower for kw in keywords):
                results.append(memory)
        return results

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get N most recent memories"""
        return list(self.memories)[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        uptime = (datetime.now() - self.session_start).total_seconds()
        return {
            "session_start": self.session_start.isoformat(),
            "uptime_seconds": uptime,
            "total_interactions": len(self.memories),
            "memory_usage": f"{len(self.memories)}/{self.max_size}"
        }


class RuleBasedBrain:
    """
    Enhanced Brain v4.0 with contextual reasoning and memory
    
    Features:
    - Keyword extraction and analysis
    - Sentiment analysis
    - Pattern detection (comparison, troubleshooting, etc.)
    - Session memory
    - Context-aware reasoning
    - Confidence scoring
    """
    def __init__(self, device: str = "cpu", memory_size: int = 100):
        self.device = device
        self.total_neurons = 0  # Not a real neural network
        self.memory = SessionMemory(max_size=memory_size)
        
        # Load persisted memories from DB
        self._load_memories()

        logger.info(f"🧠 Brain v4.0: Initialized on device '{self.device}' with memory")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts common words, excluding stopwords."""
        stopwords = {"a", "an", "the", "is", "in", "on", "of", "for", "to"}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stopwords and len(word) > 2]

    def _analyze_sentiment(self, text: str) -> str:
        """Performs very basic sentiment analysis."""
        positive_words = {"good", "great", "excellent", "success", "positive"}
        negative_words = {"bad", "terrible", "error", "fail", "negative"}
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        if neg_count > pos_count:
            return "negative"
        return "neutral"

    def _load_memories(self):
        """Load persisted memories from DB"""
        session = SessionLocal()
        try:
            # Load last 100 memories
            db_memories = session.query(Memory).order_by(Memory.timestamp.desc()).limit(100).all()
            for mem in reversed(db_memories): # Reverse to keep chronological order in deque
                self.memory.memories.append({
                    "timestamp": mem.timestamp.isoformat(),
                    "query": mem.content, # Mapping content to query
                    "insights": mem.context # Mapping context to insights
                })
            logger.info(f"Loaded {len(db_memories)} memories from DB")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
        finally:
            session.close()

    def _save_memory(self, query: str, insights: Dict[str, Any]):
        """Persist memory to DB"""
        session = SessionLocal()
        try:
            memory = Memory(
                timestamp=datetime.now(),
                memory_type="interaction",
                content=query,
                context=insights,
                importance=0.5 # Default importance
            )
            session.add(memory)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")
            session.rollback()
        finally:
            session.close()

    def _calculate_confidence(self, common_keywords: List[str], context_text: str, similar_memories: List[Dict]) -> float:
        """Calculate confidence score for the reasoning"""
        confidence = 0.5  # Base confidence

        # Boost for common keywords
        if len(common_keywords) > 0:
            confidence += min(0.2, len(common_keywords) * 0.05)

        # Boost for context availability
        if len(context_text) > 100:
            confidence += 0.1

        # Boost for similar past queries
        if len(similar_memories) > 0:
            confidence += min(0.2, len(similar_memories) * 0.05)

        return min(1.0, confidence)

    def process(self, question: str, context_text: str) -> Dict[str, Any]:
        """
        Processes the question and context to generate insights.

        Args:
            question: The user's question.
            context_text: The context retrieved from the knowledge search.

        Returns:
            A dictionary of insights with enhanced reasoning.
        """
        logger.info("Brain v4.0: Processing input with contextual reasoning")

        # Extract keywords
        question_keywords = self._extract_keywords(question)
        context_keywords = self._extract_keywords(context_text)
        common_keywords = list(set(question_keywords) & set(context_keywords))

        # Search for similar past queries
        similar_memories = self.memory.search(question_keywords[:3])

        # Calculate confidence
        confidence = self._calculate_confidence(common_keywords, context_text, similar_memories)

        # Build insights
        insights = {
            "question_keywords": question_keywords,
            "context_keywords": context_keywords,
            "common_keywords": common_keywords,
            "context_sentiment": self._analyze_sentiment(context_text),
            "confidence": confidence,
            "similar_past_queries": len(similar_memories),
            "reasoning_trace": [
                "Analyzed question and context.",
                f"Found {len(common_keywords)} common keywords.",
                f"Searched memory and found {len(similar_memories)} similar past queries.",
                "Performed sentiment analysis on context.",
                f"Calculated confidence score: {confidence:.2f}"
            ]
        }

        # Enhanced pattern detection
        detected_patterns = []

        # Comparison pattern
        if "compare" in question.lower() or "vs" in question.lower() or "versus" in question.lower():
            detected_patterns.append("comparison")
            insights["reasoning_trace"].append("Detected: Comparison query")

        # Troubleshooting pattern
        if any(kw in question.lower() for kw in ["error", "fix", "debug", "issue", "problem", "not working"]):
            detected_patterns.append("troubleshooting")
            insights["reasoning_trace"].append("Detected: Troubleshooting query")

        # How-to pattern
        if question.lower().startswith("how") or "how to" in question.lower():
            detected_patterns.append("how-to")
            insights["reasoning_trace"].append("Detected: How-to query")

        # Conceptual/explanation pattern
        if any(kw in question.lower() for kw in ["what is", "explain", "define", "meaning"]):
            detected_patterns.append("conceptual")
            insights["reasoning_trace"].append("Detected: Conceptual query")

        insights["detected_patterns"] = detected_patterns

        # Add session context
        insights["session_context"] = {
            "recent_queries": len(self.memory.get_recent(5)),
            "session_stats": self.memory.get_stats()
        }

        # Store in memory
        self.memory.add(question, insights)

        # Persist to DB
        self._save_memory(question, insights)

        return insights

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.memory.get_stats()

    def search_memories(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search past memories"""
        return self.memory.search(keywords)

def create_brain_v4(device: str = "cpu", memory_size: int = 100, persist_path: Optional[str] = None) -> RuleBasedBrain:
    """
    Factory function to create the RuleBasedBrain with enhanced features.

    Args:
        device: Device to run on (cpu/cuda)
        memory_size: Maximum number of memories to keep in session
        persist_path: Deprecated, ignored.

    Returns:
        RuleBasedBrain instance with full v4.0 capabilities
    """
    logger.info(f"create_brain_v4: Creating enhanced Brain v4.0 on device '{device}'")

    return RuleBasedBrain(device=device, memory_size=memory_size)

if __name__ == '__main__':
    # Example Usage
    brain = create_brain_v4()
    
    test_question = "Compare Python vs JavaScript for web development"
    test_context = """
    Python is a popular language for web development, especially with frameworks like Django and Flask.
    JavaScript is essential for front-end development and can also be used on the back-end with Node.js.
    Both have large ecosystems. Python is often praised for its readability, while JavaScript is praised for its speed on the client-side.
    """
    
    brain_output = brain.process(test_question, test_context)
    
    import json
    print("\n--- Brain Output ---")
    print(json.dumps(brain_output, indent=2))