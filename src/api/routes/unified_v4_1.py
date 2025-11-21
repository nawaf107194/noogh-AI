"""
🧠 Unified Answer System v4.1 - Enhanced with Intent Routing & Learning
نظام الإجابة الموحد v4.1 - محسّن مع توجيه النوايا والتعلم

New Features:
- ✨ Intent Classification (11 types)
- ✨ Specialized Handlers
- ✨ Web Search Integration
- ✨ Reflection & Learning Database
- ✨ Automatic Experience Tracking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.kernel import create_knowledge_kernel

# Import validators for input sanitization
try:
    from api.utils.validators import validate_text_input
    validators_available = True
except ImportError:
    validators_available = False

# Import cache manager for performance
try:
    from api.utils.cache_manager import get_cache
    cache = get_cache()
    cache_available = True
except ImportError:
    cache_available = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Global kernel instance
knowledge_kernel = None


class QuestionRequest(BaseModel):
    question: str
    use_intent_routing: Optional[bool] = True
    use_web_search: Optional[bool] = True
    track_experience: Optional[bool] = True
    context: Optional[Dict[str, Any]] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    intent: str
    handler: str
    confidence: float
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


class StatisticsResponse(BaseModel):
    version: str
    knowledge_base: Dict[str, Any]
    usage: Dict[str, Any]
    session: Optional[Dict[str, Any]] = None
    reflection: Optional[Dict[str, Any]] = None


@router.on_event("startup")
async def initialize():
    """Initialize Knowledge Kernel v4.1"""
    global knowledge_kernel

    try:
        logger.info("🚀 Initializing Knowledge Kernel v4.1...")

        knowledge_kernel = create_knowledge_kernel(
            enable_brain=False,  # CPU expensive
            enable_allam=True,
            enable_intent_routing=True,
            enable_web_search=True,
            enable_reflection=True
        )

        logger.info("✅ Knowledge Kernel v4.1 ready!")

    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """
    نظام الإجابة الذكي v4.1

    Features:
    1. Intent classification (LEARN, SEARCH, ANALYZE, etc.)
    2. Specialized handlers for each intent
    3. Web search fallback when needed
    4. Automatic experience tracking
    5. Learning from interactions

    Example:
        POST /v4.1/ask
        {
            "question": "What is machine learning?",
            "use_intent_routing": true,
            "use_web_search": true,
            "track_experience": true
        }
    """

    if knowledge_kernel is None:
        raise HTTPException(status_code=503, detail="Knowledge Kernel not initialized")

    # Validate input
    if validators_available:
        validated_question = validate_text_input(req.question, max_length=2000)
        req.question = validated_question

    # Check cache for non-learning queries
    cache_key = None
    if cache_available and not req.track_experience:
        cache_key = f"unified:ask:{req.question}:{req.use_intent_routing}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.debug(f"✅ Cache hit for question: {req.question[:50]}...")
            return cached_result

    try:
        # Ask the kernel
        result = knowledge_kernel.ask(
            question=req.question,
            context=req.context,
            use_intent_routing=req.use_intent_routing,
            track_experience=req.track_experience
        )

        # Build response
        response = AnswerResponse(
            answer=result.get('answer', ''),
            sources=result.get('sources', []),
            intent=result.get('intent', 'unknown'),
            handler=result.get('handler', 'default'),
            confidence=result.get('confidence', 0.0),
            processing_time=result.get('processing_time', 0.0),
            metadata=result.get('metadata', {})
        )

        # Cache result if not learning-based
        if cache_available and cache_key and not req.track_experience:
            cache.set(cache_key, response, ttl=600)
            logger.debug(f"💾 Cached result for: {req.question[:50]}...")

        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get system statistics

    Returns:
    - Version info
    - Knowledge base stats
    - Usage statistics
    - Session stats
    - Reflection database stats
    """

    if knowledge_kernel is None:
        raise HTTPException(status_code=503, detail="Knowledge Kernel not initialized")

    try:
        stats = knowledge_kernel.get_statistics()

        # Extract reflection stats
        reflection_stats = None
        if hasattr(knowledge_kernel, 'experience_tracker') and knowledge_kernel.experience_tracker:
            try:
                overall = knowledge_kernel.experience_tracker.get_overall_stats()
                reflection_stats = {
                    "total_experiences": overall.get('total_experiences', 0),
                    "overall_success_rate": overall.get('overall_success_rate', 0.0),
                    "open_gaps": overall.get('open_gaps', 0),
                    "learning_cycles": overall.get('total_cycles', 0)
                }
            except Exception as e:
                logger.warning(f"Could not get reflection stats: {e}")
                pass

        return StatisticsResponse(
            version=stats.get('version', '4.1.0'),
            knowledge_base=stats.get('knowledge_base', {}),
            usage=stats.get('usage', {}),
            session=stats.get('session'),
            reflection=reflection_stats
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""

    return {
        "status": "healthy" if knowledge_kernel else "not_ready",
        "version": "4.1.0",
        "features": {
            "intent_routing": knowledge_kernel.intent_routing_enabled if knowledge_kernel else False,
            "web_search": knowledge_kernel.web_search_enabled if knowledge_kernel else False,
            "reflection": knowledge_kernel.reflection_enabled if knowledge_kernel else False,
            "allam": knowledge_kernel.allam_enabled if knowledge_kernel else False
        }
    }


@router.get("/reflection/recent")
async def get_recent_experiences(limit: int = 50):
    """Get recent user experiences"""

    if knowledge_kernel is None or not knowledge_kernel.experience_tracker:
        raise HTTPException(status_code=503, detail="Reflection system not available")

    try:
        experiences = knowledge_kernel.experience_tracker.db.get_recent_experiences(limit=limit)
        return {
            "experiences": experiences,
            "count": len(experiences)
        }
    except Exception as e:
        logger.error(f"Error getting experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflection/gaps")
async def get_knowledge_gaps(status: str = "open", limit: int = 20):
    """Get knowledge gaps"""

    if knowledge_kernel is None or not knowledge_kernel.experience_tracker:
        raise HTTPException(status_code=503, detail="Reflection system not available")

    try:
        gaps = knowledge_kernel.experience_tracker.db.get_knowledge_gaps(status=status, limit=limit)
        return {
            "gaps": gaps,
            "count": len(gaps)
        }
    except Exception as e:
        logger.error(f"Error getting knowledge gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflection/analyze")
async def analyze_patterns(days: int = 7):
    """Analyze learning patterns"""

    if knowledge_kernel is None or not knowledge_kernel.experience_tracker:
        raise HTTPException(status_code=503, detail="Reflection system not available")

    try:
        from reflection import PatternAnalyzer

        analyzer = PatternAnalyzer(db_path="data/reflection.db")
        analysis = analyzer.analyze_recent_patterns(days=days, min_experiences=1)

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflection/priorities")
async def get_learning_priorities(top_n: int = 10):
    """Get top learning priorities"""

    if knowledge_kernel is None or not knowledge_kernel.experience_tracker:
        raise HTTPException(status_code=503, detail="Reflection system not available")

    try:
        from reflection import PatternAnalyzer

        analyzer = PatternAnalyzer(db_path="data/reflection.db")
        priorities = analyzer.identify_learning_priorities(top_n=top_n)

        return {
            "priorities": priorities,
            "count": len(priorities)
        }

    except Exception as e:
        logger.error(f"Error getting priorities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
