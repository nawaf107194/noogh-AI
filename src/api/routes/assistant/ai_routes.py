"""
مسارات API للمساعد الذكي
AI Assistant API Routes
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
import os

# إضافة المسار للمشروع
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ai_assistant.intelligent_assistant import IntelligentAssistant
from ai_assistant.memory_system import MemorySystem
from ai_assistant.sentiment_analyzer import SentimentAnalyzer

router = APIRouter(prefix="/ai-assistant", tags=["AI Assistant"])

# تهيئة المكونات
assistant = IntelligentAssistant(name="نوغ AI", language="ar")
memory = MemorySystem(db_path="data/memory.db")
sentiment = SentimentAnalyzer()


# النماذج
class ChatRequest(BaseModel):
    message: str = Field(..., description="رسالة المستخدم")
    context: Optional[Dict[str, Any]] = Field(None, description="سياق إضافي")


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    metadata: Optional[Dict] = None


class SentimentRequest(BaseModel):
    text: str = Field(..., description="النص المراد تحليله")


class MemoryRequest(BaseModel):
    content: str = Field(..., description="المحتوى")
    memory_type: str = Field("conversation", description="نوع الذاكرة")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="الأهمية")


class CreativeTextRequest(BaseModel):
    prompt: str = Field(..., description="الموضوع أو البداية")
    style: str = Field("general", description="النمط: story, poem, article")


# المسارات

@router.get("/health")
async def health_check():
    """فحص صحة المساعد"""
    return {
        "status": "healthy",
        "assistant": assistant.name,
        "language": assistant.language
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    محادثة مع المساعد الذكي

    Example:
    ```json
    {
        "message": "مرحباً، كيف حالك؟",
        "context": {"user_id": "123"}
    }
    ```
    """
    try:
        response = await assistant.chat(request.message, request.context)

        # حفظ في الذاكرة
        await memory.store_memory(
            content=f"User: {request.message}\nAssistant: {response}",
            memory_type="conversation",
            importance=0.5
        )

        from datetime import datetime
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat(),
            metadata={"message_length": len(response)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """
    تحليل المشاعر في النص

    Example:
    ```json
    {
        "text": "أنا سعيد جداً اليوم!"
    }
    ```
    """
    try:
        result = sentiment.analyze(request.text)
        return {
            "sentiment": result.overall_sentiment,
            "score": result.score,
            "confidence": result.confidence,
            "emotions": result.emotions,
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/summary")
async def get_sentiment_summary(request: SentimentRequest):
    """الحصول على ملخص العواطف"""
    try:
        summary = sentiment.get_emotion_summary(request.text)
        return {
            "summary": summary,
            "text_length": len(request.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/store")
async def store_memory(request: MemoryRequest):
    """تخزين ذاكرة جديدة"""
    try:
        memory_id = await memory.store_memory(
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance
        )
        return {
            "memory_id": memory_id,
            "status": "stored"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/recall")
async def recall_memories(
    query: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 10
):
    """
    استرجاع الذكريات

    Parameters:
    - query: نص البحث (اختياري)
    - memory_type: نوع الذاكرة (اختياري)
    - limit: عدد النتائج (افتراضي 10)
    """
    try:
        memories = await memory.recall_memory(query, memory_type, limit)
        return {
            "count": len(memories),
            "memories": [
                {
                    "id": m.memory_id,
                    "type": m.memory_type,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    "importance": m.importance,
                    "created_at": m.created_at
                }
                for m in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def get_memory_stats():
    """إحصائيات الذاكرة"""
    try:
        stats = await memory.get_memory_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/clear")
async def clear_memories():
    """مسح جميع الذكريات"""
    try:
        await memory.clear_all_memories()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/creative")
async def generate_creative_text(request: CreativeTextRequest):
    """
    توليد نص إبداعي

    Example:
    ```json
    {
        "prompt": "الذكاء الاصطناعي",
        "style": "poem"
    }
    ```

    Styles: story, poem, article
    """
    try:
        text = await assistant.generate_creative_text(request.prompt, request.style)
        return {
            "generated_text": text,
            "style": request.style,
            "prompt": request.prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_conversation_history(limit: int = 10):
    """
    الحصول على سجل المحادثة

    Parameters:
    - limit: عدد الرسائل (افتراضي 10)
    """
    try:
        history = assistant.get_conversation_history(limit)
        return {
            "count": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/clear")
async def clear_conversation_history():
    """مسح سجل المحادثة"""
    try:
        assistant.clear_history()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_assistant_stats():
    """إحصائيات المساعد"""
    try:
        stats = assistant.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question")
async def ask_question(question: str = Body(..., embed=True)):
    """
    طرح سؤال على المساعد

    Example:
    ```json
    {
        "question": "ما هو الذكاء الاصطناعي؟"
    }
    ```
    """
    try:
        response = await assistant.chat(question)
        return {
            "question": question,
            "answer": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/help")
async def get_help():
    """الحصول على مساعدة حول استخدام API"""
    return {
        "endpoints": {
            "POST /chat": "محادثة مع المساعد",
            "POST /sentiment/analyze": "تحليل المشاعر",
            "POST /memory/store": "تخزين ذاكرة",
            "GET /memory/recall": "استرجاع الذكريات",
            "POST /generate/creative": "توليد نص إبداعي",
            "GET /history": "سجل المحادثة",
            "GET /stats": "إحصائيات المساعد"
        },
        "documentation": "زر /docs للحصول على التوثيق الكامل"
    }
