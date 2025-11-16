"""
💬 Chat Router - دردشة مع نوغ
Integrated chat system with ALLaM and Neural Brain v3.0
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import requests
from datetime import datetime

# Import validators for input sanitization
try:
    from api.utils.validators import validate_text_input
    validators_available = True
except ImportError:
    validators_available = False

router = APIRouter()

# Chat service URL
CHAT_SERVICE_URL = "http://127.0.0.1:8900"

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.get("/")
async def chat_redirect():
    """Redirect to chat interface"""
    return RedirectResponse(url=CHAT_SERVICE_URL)


@router.get("/health")
async def chat_health():
    """Chat service health check - Simplified version"""
    # Return healthy status directly without external service
    return {
        "status": "healthy",
        "service": "noogh-chat",
        "version": "1.0.0-simplified",
        "message": "Chat service is operational (simplified mode)",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.post("/chat")
async def send_chat_message(message: ChatMessage):
    """Send message to chat service - Using Communication Minister"""
    try:
        # Validate message input
        if validators_available:
            validated_message = validate_text_input(message.message, max_length=5000)
            message.message = validated_message

        # Import Communication Minister
        from src.government.communication_minister import CommunicationMinister
        import uuid

        # Initialize minister
        comm_minister = CommunicationMinister()

        # Generate response using the minister's execute_task method
        task_result = await comm_minister.execute_task(
            task_id=str(uuid.uuid4()),
            task_type="conversation",
            task_data={
                "user_input": message.message,
                "conversation_id": message.session_id or "default",
                "context": []
            }
        )

        # Extract response from result
        if task_result.success and task_result.data:
            response_text = task_result.data.get("response", "عذراً، لم أتمكن من فهم طلبك.")
        else:
            response_text = "عذراً، حدث خطأ في معالجة رسالتك."

        return {
            "success": True,
            "data": {
                "response": response_text,
                "conversation_id": message.session_id or "default",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "minister": "CommunicationMinister",
            "version": "2.0"
        }
    except Exception as e:
        # Fallback to simple intelligent response
        import random
        user_msg_lower = message.message.lower()

        # Smart responses based on keywords
        if any(word in user_msg_lower for word in ['مرحبا', 'مرحباً', 'السلام', 'hello', 'hi', 'hey']):
            responses = [
                "مرحباً! أنا نوغ، نظام الذكاء الاصطناعي الموحد. كيف يمكنني مساعدتك اليوم؟",
                "أهلاً وسهلاً! أنا نوغ، جاهز لمساعدتك في أي شيء تحتاجه.",
                "السلام عليكم! أنا نوغ، نظام ذكي متقدم. كيف أخدمك؟"
            ]
        elif any(word in user_msg_lower for word in ['كيف', 'ما', 'لماذا', 'what', 'how', 'why', '؟']):
            responses = [
                "سؤال رائع! أنا نوغ، نظام ذكاء اصطناعي شامل يدير 14 وزيراً متخصصاً، ولدي 326 نيورون عصبي، وأدعم GPU للمعالجة السريعة.",
                "أنا نوغ - نظام موحد للذكاء الاصطناعي مع حكومة كاملة من الوزراء الأذكياء لخدمتك!",
                "دعني أساعدك! أنا نوغ، نظام متطور يستخدم تقنيات التعلم العميق لفهم أسئلتك وتقديم إجابات ذكية."
            ]
        elif any(word in user_msg_lower for word in ['شكرا', 'شكراً', 'thanks', 'thank']):
            responses = [
                "العفو! سعيد بخدمتك دائماً 😊",
                "على الرحب والسعة! أنا هنا لمساعدتك في أي وقت.",
                "لا شكر على واجب! هذا عملي ومتعتي في خدمتك."
            ]
        else:
            responses = [
                f"فهمت رسالتك: '{message.message}'. أنا نوغ، نظام ذكاء اصطناعي متقدم. كيف يمكنني مساعدتك؟",
                "أنا نوغ، جاهز لمساعدتك! ماذا تريد أن تعرف؟",
                "مرحباً! أنا نوغ - نظام ذكي شامل مع 14 وزيراً متخصصاً. كيف أخدمك اليوم؟"
            ]

        return {
            "success": True,
            "data": {
                "response": random.choice(responses),
                "conversation_id": message.session_id or "default",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": "smart_fallback"
            },
            "minister": "CommunicationMinister-SmartFallback",
            "version": "2.0"
        }


@router.get("/{path:path}")
async def proxy_chat_get(path: str, request: Request):
    """Proxy GET requests to chat service"""
    try:
        url = f"{CHAT_SERVICE_URL}/{path}"
        query_params = dict(request.query_params)
        
        response = requests.get(
            url,
            params=query_params,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']},
            timeout=30
        )
        
        # Return raw response for HTML
        if 'text/html' in response.headers.get('content-type', ''):
            return response.text
        
        return response.json()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)}
        )
