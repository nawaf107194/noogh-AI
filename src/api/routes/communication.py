#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📡 Communication Minister API Routes
API endpoints for Communication Minister v2.0

Features:
- General Communication
- Translation Services
- Summarization
- Conversation Management
- Report Generation
- Message Queue
- External API Integration
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio

router = APIRouter()

# Singleton instance
_communication_minister = None


def get_communication_minister():
    """Get or create Communication Minister instance"""
    global _communication_minister
    if _communication_minister is None:
        from government.communication_minister import CommunicationMinister
        _communication_minister = CommunicationMinister(verbose=True)
    return _communication_minister


def generate_task_id() -> str:
    """Generate unique task ID"""
    return f"comm_{uuid.uuid4().hex[:12]}"


# ==================== Request Models ====================

class CommunicationRequest(BaseModel):
    """Request model for general communication"""
    input: str = Field(..., description="User input text")
    language: Optional[str] = Field(None, description="Preferred language (arabic/english)")


class TranslationRequest(BaseModel):
    """Request model for translation"""
    input: str = Field(..., description="Text to translate")
    source_language: str = Field("auto", description="Source language")
    target_language: str = Field("arabic", description="Target language")


class SummarizationRequest(BaseModel):
    """Request model for summarization"""
    input: str = Field(..., description="Text to summarize")
    max_length: int = Field(100, description="Maximum summary length")


class ConversationRequest(BaseModel):
    """Request model for conversation"""
    input: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")


class ExplanationRequest(BaseModel):
    """Request model for explanation"""
    input: str = Field(..., description="Concept to explain")
    detail_level: str = Field("medium", description="Detail level: basic, medium, advanced")
    language: str = Field("arabic", description="Response language")


class ReportRequest(BaseModel):
    """Request model for report generation"""
    data: Dict[str, Any] = Field(..., description="Data for report")
    report_type: str = Field("summary", description="Report type: trading, performance, summary")


class MessageQueueRequest(BaseModel):
    """Request model for message queue"""
    recipient: str = Field(..., description="Recipient minister")
    message: str = Field(..., description="Message content")
    priority: str = Field("medium", description="Priority: low, medium, high, urgent")


class ExternalAPIRequest(BaseModel):
    """Request model for external API calls"""
    url: str = Field(..., description="API endpoint URL")
    method: str = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, Any]] = None


# ==================== API Endpoints ====================

@router.post("/communicate")
async def communicate(request: CommunicationRequest):
    """
    تواصل عام - General communication

    Process user input and generate appropriate response.
    Supports:
    - Greetings
    - Questions
    - Requests
    - General conversation
    """
    minister = get_communication_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="communication",
        task_data={
            "input": request.input,
            "language": request.language
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/translate")
async def translate(request: TranslationRequest):
    """
    ترجمة - Translation service

    Translate text between Arabic and English.
    """
    minister = get_communication_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="translation",
        task_data={
            "input": request.input,
            "source_language": request.source_language,
            "target_language": request.target_language
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/summarize")
async def summarize(request: SummarizationRequest):
    """
    تلخيص - Summarization service

    Summarize long text into concise summary.
    """
    minister = get_communication_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="summarization",
        task_data={
            "input": request.input,
            "max_length": request.max_length
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/conversation")
async def conversation(request: ConversationRequest):
    """
    محادثة - Conversation management

    Manage multi-turn conversations with context.
    """
    minister = get_communication_minister()

    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="conversation",
        task_data={
            "input": request.input,
            "conversation_id": conversation_id
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "conversation_id": conversation_id,
        "result": report.result
    }


@router.post("/explain")
async def explain(request: ExplanationRequest):
    """
    شرح - Explanation service

    Explain concepts in simple terms.
    """
    minister = get_communication_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="explanation",
        task_data={
            "input": request.input,
            "detail_level": request.detail_level,
            "language": request.language
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/report/generate")
async def generate_report(request: ReportRequest):
    """
    توليد تقرير - Report generation

    Generate formatted reports from data.
    Types: trading, performance, summary
    """
    minister = get_communication_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="report_writing",
        task_data={
            "data": request.data,
            "report_type": request.report_type
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/conversation/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    سجل المحادثة - Get conversation history

    Retrieve conversation history for a specific conversation ID.
    """
    minister = get_communication_minister()

    history = minister._get_conversation_context(conversation_id)

    return {
        "success": True,
        "conversation_id": conversation_id,
        "message_count": len(history),
        "history": history
    }


@router.get("/languages")
async def get_supported_languages():
    """
    اللغات المدعومة - Get supported languages

    Returns list of supported languages.
    """
    minister = get_communication_minister()

    return {
        "success": True,
        "languages": minister.resources.get('supported_languages', []),
        "styles": minister.resources.get('communication_styles', [])
    }


@router.get("/templates")
async def get_response_templates():
    """
    قوالب الردود - Get response templates

    Returns available response templates.
    """
    minister = get_communication_minister()

    templates = minister.resources.get('response_templates', {})

    return {
        "success": True,
        "total_templates": len(templates),
        "templates": list(templates.keys())
    }


@router.post("/message/send")
async def send_message(request: MessageQueueRequest):
    """
    إرسال رسالة - Send message to another minister

    Send a message to another minister via message queue.
    """
    minister = get_communication_minister()

    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    # Store message in queue (would be implemented in full system)
    message = {
        "message_id": message_id,
        "sender": "communication",
        "recipient": request.recipient,
        "message": request.message,
        "priority": request.priority,
        "timestamp": datetime.now().isoformat(),
        "status": "sent"
    }

    return {
        "success": True,
        "message_id": message_id,
        "message": message
    }


@router.post("/api/call")
async def call_external_api(request: ExternalAPIRequest):
    """
    استدعاء API خارجي - Call external API

    Make HTTP requests to external APIs.
    """
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            if request.method.upper() == "GET":
                response = await client.get(
                    request.url,
                    headers=request.headers or {}
                )
            elif request.method.upper() == "POST":
                response = await client.post(
                    request.url,
                    headers=request.headers or {},
                    json=request.body or {}
                )
            elif request.method.upper() == "PUT":
                response = await client.put(
                    request.url,
                    headers=request.headers or {},
                    json=request.body or {}
                )
            elif request.method.upper() == "DELETE":
                response = await client.delete(
                    request.url,
                    headers=request.headers or {}
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")

            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics():
    """
    إحصائيات - Get communication statistics

    Returns communication minister statistics.
    """
    minister = get_communication_minister()

    return {
        "success": True,
        "statistics": {
            "total_conversations": len(minister.resources.get('conversation_history', [])),
            "supported_languages": len(minister.resources.get('supported_languages', [])),
            "response_templates": len(minister.resources.get('response_templates', {})),
            "communication_styles": len(minister.resources.get('communication_styles', []))
        },
        "authorities": minister.authorities,
        "minister_type": minister.minister_type.value
    }


@router.get("/report")
async def get_minister_report():
    """
    تقرير الوزير - Get minister report

    Full report for President.
    """
    minister = get_communication_minister()
    report = await minister.report_to_president()

    return {
        "success": True,
        "report": report
    }


@router.get("/health")
async def health_check():
    """
    فحص الصحة - Health check

    Check if Communication Minister is operational.
    """
    minister = get_communication_minister()

    return {
        "status": "operational",
        "minister": minister.name,
        "type": minister.minister_type.value,
        "authorities": minister.authorities,
        "timestamp": datetime.now().isoformat()
    }


# ==================== WebSocket Endpoints ====================

class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket Chat - Real-time communication

    Real-time chat endpoint using WebSocket.
    """
    await manager.connect(websocket)
    minister = get_communication_minister()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Process message with Communication Minister
            task_id = generate_task_id()
            report = await minister.execute_task(
                task_id=task_id,
                task_type="communication",
                task_data={"input": data}
            )

            # Send response back to client
            response = {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": data,
                "response": report.result.get('response', 'No response generated')
            }

            await manager.send_personal_message(str(response), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
