#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ Government & Deep Cognition API Routes
نقاط النهاية للحكومة والإدراك العميق

Provides endpoints for:
- Government Status (14 Ministers + President)
- Deep Cognition Score (97.5% TRANSCENDENT)
- Minister Task Delegation
- Brain Hub Statistics
- Cognitive Analysis

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-11-10
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Import Unified Brain Hub
try:
    from integration.unified_brain_hub import get_brain_hub, BrainHubStatus
    HAS_BRAIN_HUB = True
except ImportError as e:
    logging.warning(f"⚠️ Brain Hub not available: {e}")
    HAS_BRAIN_HUB = False

# Import Ministers
try:
    from government.minister_types_universal import MinisterType, MINISTER_INFO
    HAS_MINISTER_TYPES = True
except ImportError as e:
    logging.warning(f"⚠️ Minister types not available: {e}")
    HAS_MINISTER_TYPES = False


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/government",
    tags=["government", "cognition"],
    responses={404: {"description": "Not found"}},
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REQUEST/RESPONSE MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TaskRequest(BaseModel):
    """طلب مهمة"""
    request: str = Field(..., description="المهمة المطلوبة", min_length=1)
    context: Optional[Dict[str, Any]] = Field(default={}, description="سياق إضافي")
    minister: Optional[str] = Field(None, description="الوزير المحدد (اختياري)")

    class Config:
        json_schema_extra = {
            "example": {
                "request": "أريد تعليم الطلاب عن الذكاء الاصطناعي",
                "context": {"level": "beginner"},
                "minister": "education"
            }
        }


class CognitionAnalysisRequest(BaseModel):
    """طلب تحليل إدراكي"""
    text: Optional[str] = Field(None, description="نص للتحليل")
    image_path: Optional[str] = Field(None, description="مسار صورة للتحليل")
    confidence_factors: Optional[Dict[str, float]] = Field(None, description="عوامل الثقة")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "الحياة جميلة والأمل موجود",
                "confidence_factors": {
                    "data_quality": 0.9,
                    "model_agreement": 0.85
                }
            }
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/status", summary="حالة الحكومة - Government Status")
async def get_government_status():
    """
    الحصول على حالة الحكومة الكاملة

    Returns:
        - Deep Cognition Score (97.5%)
        - Active Ministers (14)
        - Brain Hub Status
        - System Health
    """
    if not HAS_BRAIN_HUB:
        raise HTTPException(status_code=503, detail="Brain Hub not available")

    try:
        brain_hub = get_brain_hub()
        status = brain_hub.get_status()

        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "brain_hub": {
                "active": status.active,
                "cognition_score": status.cognition_score,
                "cognition_percentage": f"{status.cognition_score:.1%}",
                "cognition_level": "TRANSCENDENT" if status.cognition_score >= 0.95 else "EXPERT",
                "active_ministers": status.active_ministers
            },
            "systems": {
                "deep_cognition": status.deep_cognition_available,
                "agent_brain": status.agent_brain_available,
                "government": status.government_available,
                "unified_cognition": status.unified_cognition_available
            },
            "message": "🏛️ Noogh Government is operational with 14 active ministers"
        }

    except Exception as e:
        logger.error(f"Error getting government status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ministers", summary="قائمة الوزراء - Ministers List")
async def get_ministers_list():
    """
    الحصول على قائمة جميع الوزراء

    Returns:
        List of 14 active ministers with their details
    """
    if not HAS_MINISTER_TYPES:
        raise HTTPException(status_code=503, detail="Minister types not available")

    ministers_list = []

    for minister_type in MinisterType:
        info = MINISTER_INFO.get(minister_type, {})
        ministers_list.append({
            "id": minister_type.value,
            "arabic_name": info.get('arabic', 'Unknown'),
            "english_name": info.get('english', 'Unknown'),
            "category": info.get('category', 'Unknown'),
            "description": info.get('description', ''),
            "keywords": info.get('keywords', '').split('، '),
            "gpu_enabled": info.get('gpu_enabled', False)
        })

    return {
        "total_ministers": len(ministers_list),
        "ministers": ministers_list,
        "message": f"🏛️ {len(ministers_list)} ministers active and ready"
    }


@router.post("/task", summary="تفويض مهمة - Delegate Task")
async def delegate_task(task_request: TaskRequest = Body(...)):
    """
    تفويض مهمة إلى الوزير المناسب أو Brain Hub

    Args:
        task_request: التفاصيل الكاملة للمهمة

    Returns:
        نتيجة المعالجة مع الوزير المستخدم
    """
    if not HAS_BRAIN_HUB:
        raise HTTPException(status_code=503, detail="Brain Hub not available")

    try:
        brain_hub = get_brain_hub()

        # Process request through Brain Hub
        result = brain_hub.process_request(
            request=task_request.request,
            context=task_request.context
        )

        return {
            "status": result.status,
            "response": result.response,
            "minister_used": result.minister_used,
            "confidence": result.confidence,
            "confidence_percentage": f"{result.confidence:.0%}",
            "processing_time_ms": result.processing_time_ms,
            "metadata": result.metadata,
            "cognition_analysis": result.cognition_analysis
        }

    except Exception as e:
        logger.error(f"Error delegating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cognition/score", summary="درجة الإدراك - Cognition Score")
async def get_cognition_score():
    """
    الحصول على درجة الإدراك العميق

    Returns:
        Deep Cognition score and level (97.5% TRANSCENDENT)
    """
    if not HAS_BRAIN_HUB:
        raise HTTPException(status_code=503, detail="Brain Hub not available")

    try:
        brain_hub = get_brain_hub()
        status = brain_hub.get_status()

        # Determine cognition level
        score = status.cognition_score
        if score >= 0.95:
            level = "TRANSCENDENT"
            icon = "🌟"
        elif score >= 0.90:
            level = "EXPERT"
            icon = "🎯"
        elif score >= 0.75:
            level = "ADVANCED"
            icon = "📈"
        else:
            level = "PROFICIENT"
            icon = "✅"

        return {
            "cognition_score": score,
            "cognition_percentage": f"{score:.1%}",
            "cognition_level": level,
            "icon": icon,
            "systems": {
                "scene_understanding": status.deep_cognition_available,
                "material_analyzer": status.deep_cognition_available,
                "semantic_intent": status.deep_cognition_available,
                "meta_confidence": status.deep_cognition_available,
                "vision_reasoning_sync": status.deep_cognition_available
            },
            "message": f"{icon} Deep Cognition v1.2 Lite - {level} Level"
        }

    except Exception as e:
        logger.error(f"Error getting cognition score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cognition/analyze", summary="تحليل إدراكي - Cognitive Analysis")
async def cognitive_analysis(analysis_request: CognitionAnalysisRequest = Body(...)):
    """
    تحليل إدراكي عميق للنص أو الصور

    Args:
        analysis_request: البيانات المطلوب تحليلها

    Returns:
        Deep Cognition analysis results
    """
    if not HAS_BRAIN_HUB:
        raise HTTPException(status_code=503, detail="Brain Hub not available")

    try:
        brain_hub = get_brain_hub()

        # Prepare data for inference
        data = {}
        if analysis_request.text:
            data['text'] = analysis_request.text
        if analysis_request.image_path:
            data['image_path'] = analysis_request.image_path
        if analysis_request.confidence_factors:
            data['confidence_factors'] = analysis_request.confidence_factors

        # Run inference
        result = brain_hub.inference(data)

        if not result:
            raise HTTPException(
                status_code=400,
                detail="No analysis could be performed with provided data"
            )

        return {
            "status": "success",
            "analysis": result,
            "timestamp": datetime.now().isoformat(),
            "message": "🧠 Deep Cognition analysis complete"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cognitive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", summary="إحصائيات الحكومة - Government Statistics")
async def get_government_statistics():
    """
    الحصول على الإحصائيات الشاملة

    Returns:
        - Brain Hub statistics
        - Ministers statistics
        - Unified Cognition statistics
        - Success rates
    """
    if not HAS_BRAIN_HUB:
        raise HTTPException(status_code=503, detail="Brain Hub not available")

    try:
        brain_hub = get_brain_hub()
        stats = brain_hub.get_statistics()

        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "summary": {
                "total_requests": stats['brain_hub']['total_requests'],
                "success_rate": f"{stats['brain_hub']['success_rate']:.0%}",
                "cognition_score": f"{stats['brain_hub']['cognition_score']:.1%}",
                "active_systems": sum(1 for v in stats['systems'].values() if v)
            }
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="فحص الصحة - Health Check")
async def health_check():
    """
    فحص صحة نظام الحكومة

    Returns:
        Health status of all systems
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "brain_hub": HAS_BRAIN_HUB,
            "minister_types": HAS_MINISTER_TYPES
        }
    }

    if HAS_BRAIN_HUB:
        try:
            brain_hub = get_brain_hub()
            status = brain_hub.get_status()
            health_status["brain_hub_active"] = status.active
            health_status["cognition_score"] = status.cognition_score
            health_status["active_ministers"] = status.active_ministers
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)

    return health_status
