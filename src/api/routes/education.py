#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Education Minister Dashboard API Routes
======================================

REST API endpoints for managing learning resources, lessons, and curricula

Features:
- ✅ Add/manage learning resources (YouTube, websites, documents, images)
- ✅ Extract content from resources
- ✅ Generate lessons automatically
- ✅ Create and manage curricula
- ✅ Track education statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Education Minister
try:
    from src.government.education_minister import (
        EducationMinister,
        ContentType,
        LessonDifficulty,
        generate_task_id
    )
    EDUCATION_MINISTER_AVAILABLE = True
except ImportError as e:
    EDUCATION_MINISTER_AVAILABLE = False
    logging.warning(f"Education Minister not available: {e}")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/education", tags=["education"])

# Global Education Minister instance
education_minister: Optional[EducationMinister] = None


def get_education_minister() -> EducationMinister:
    """Get or create Education Minister instance"""
    global education_minister
    if education_minister is None:
        if not EDUCATION_MINISTER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Education Minister not available"
            )
        education_minister = EducationMinister(verbose=True)
    return education_minister


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request/Response Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AddResourceRequest(BaseModel):
    """Request to add a learning resource"""
    url: str = Field(..., description="URL of the learning resource (YouTube, website, PDF, image)")
    title: Optional[str] = Field(None, description="Title of the resource")
    description: Optional[str] = Field(None, description="Description of the resource")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class AddResourceResponse(BaseModel):
    """Response after adding a resource"""
    success: bool
    resource_id: str
    content_type: str
    message: str


class ExtractContentRequest(BaseModel):
    """Request to extract content from a resource"""
    resource_id: Optional[str] = Field(None, description="ID of existing resource")
    url: Optional[str] = Field(None, description="Direct URL (if not using resource_id)")


class ExtractContentResponse(BaseModel):
    """Response with extracted content"""
    success: bool
    content_type: str
    extracted_content: Dict[str, Any]


class GenerateLessonRequest(BaseModel):
    """Request to generate a lesson"""
    content: str = Field(..., description="Content for the lesson")
    title: str = Field(..., description="Title of the lesson")
    difficulty: Optional[str] = Field("intermediate", description="Difficulty level: beginner, intermediate, advanced, expert")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class GenerateLessonResponse(BaseModel):
    """Response after generating a lesson"""
    success: bool
    lesson_id: str
    title: str
    word_count: int
    estimated_duration: int
    key_points_count: int


class CreateCurriculumRequest(BaseModel):
    """Request to create a curriculum"""
    title: str = Field(..., description="Title of the curriculum")
    lesson_ids: List[str] = Field(..., description="List of lesson IDs to include")
    description: Optional[str] = Field(None, description="Description of the curriculum")


class CreateCurriculumResponse(BaseModel):
    """Response after creating a curriculum"""
    success: bool
    curriculum_id: str
    title: str
    total_lessons: int
    total_duration_minutes: int


class AnalyzeImageRequest(BaseModel):
    """Request to analyze an image"""
    image_path: str = Field(..., description="Path to the image file")


class ExtractTextRequest(BaseModel):
    """Request to extract text from an image"""
    image_path: str = Field(..., description="Path to the image file")
    language: Optional[str] = Field("auto", description="Language: auto, ar, en")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Learning Resources Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/resources/add", response_model=AddResourceResponse)
async def add_learning_resource(request: AddResourceRequest):
    """
    إضافة مصدر تعليمي جديد
    Add a new learning resource (YouTube video, website, PDF, image)

    Example:
    ```json
    {
        "url": "https://www.youtube.com/watch?v=xxx",
        "title": "Python Tutorial for Beginners",
        "description": "Complete Python course",
        "tags": ["python", "programming", "tutorial"]
    }
    ```
    """
    try:
        minister = get_education_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="add_learning_resource",
            task_data={
                "url": request.url,
                "title": request.title,
                "description": request.description,
                "tags": request.tags
            }
        )

        if report.status.value == "completed":
            result = report.result
            return AddResourceResponse(
                success=result["success"],
                resource_id=result["resource_id"],
                content_type=result["content_type"],
                message=result["message"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add resource: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error adding resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/list")
async def list_learning_resources(limit: int = 100):
    """
    قائمة جميع الموارد التعليمية
    List all learning resources
    """
    try:
        minister = get_education_minister()
        resources = minister.get_learning_resources(limit=limit)

        return {
            "success": True,
            "total": len(resources),
            "resources": resources
        }

    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/extract", response_model=ExtractContentResponse)
async def extract_content(request: ExtractContentRequest, background_tasks: BackgroundTasks):
    """
    استخراج محتوى من مصدر تعليمي
    Extract content from a learning resource

    Supports:
    - YouTube videos (transcript extraction)
    - Websites (scraping paragraphs, headings, code blocks)
    - Images (vision analysis)
    - PDF documents (OCR text extraction)

    Example:
    ```json
    {
        "url": "https://www.youtube.com/watch?v=xxx"
    }
    ```
    """
    try:
        minister = get_education_minister()

        if not request.resource_id and not request.url:
            raise HTTPException(
                status_code=400,
                detail="Either resource_id or url must be provided"
            )

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="extract_content",
            task_data={
                "resource_id": request.resource_id,
                "url": request.url
            }
        )

        if report.status.value == "completed":
            result = report.result
            return ExtractContentResponse(
                success=result["success"],
                content_type=result["content_type"],
                extracted_content=result["extracted_content"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Content extraction failed: {report.result.get('error', 'Unknown error')}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Computer Vision Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/vision/analyze-image")
async def analyze_image(request: AnalyzeImageRequest):
    """
    تحليل صورة باستخدام الرؤية الحاسوبية
    Analyze an image using computer vision (scene understanding)

    Returns:
    - Scene type (indoor, outdoor, workspace, etc.)
    - Lighting conditions
    - Complexity score
    - Contextual clues
    """
    try:
        minister = get_education_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="analyze_image",
            task_data={"image_path": request.image_path}
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "analysis": report.result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision/extract-text")
async def extract_text_from_image(request: ExtractTextRequest):
    """
    استخراج نص من صورة باستخدام OCR
    Extract text from an image using OCR

    Supports:
    - Arabic text
    - English text
    - Automatic language detection
    """
    try:
        minister = get_education_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="extract_text",
            task_data={
                "image_path": request.image_path,
                "language": request.language
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "ocr_result": report.result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Text extraction failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lesson Generation Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/lessons/generate", response_model=GenerateLessonResponse)
async def generate_lesson(request: GenerateLessonRequest):
    """
    توليد درس تلقائياً من محتوى
    Generate a lesson automatically from content

    Takes content and creates a structured lesson with:
    - Key points extraction
    - Duration estimation
    - Difficulty level

    Example:
    ```json
    {
        "content": "Python is a high-level programming language...",
        "title": "Introduction to Python",
        "difficulty": "beginner",
        "tags": ["python", "programming"]
    }
    ```
    """
    try:
        minister = get_education_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="generate_lesson",
            task_data={
                "content": request.content,
                "title": request.title,
                "difficulty": request.difficulty,
                "tags": request.tags
            }
        )

        if report.status.value == "completed":
            result = report.result
            return GenerateLessonResponse(
                success=result["success"],
                lesson_id=result["lesson_id"],
                title=result["title"],
                word_count=result["word_count"],
                estimated_duration=result["estimated_duration"],
                key_points_count=result["key_points_count"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Lesson generation failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error generating lesson: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lessons/list")
async def list_lessons(limit: int = 100):
    """
    قائمة جميع الدروس المولدة
    List all generated lessons
    """
    try:
        minister = get_education_minister()
        lessons = minister.get_generated_lessons(limit=limit)

        return {
            "success": True,
            "total": len(lessons),
            "lessons": lessons
        }

    except Exception as e:
        logger.error(f"Error listing lessons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Curriculum Management Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/curriculum/create", response_model=CreateCurriculumResponse)
async def create_curriculum(request: CreateCurriculumRequest):
    """
    إنشاء منهج تدريبي
    Create a training curriculum from multiple lessons

    Example:
    ```json
    {
        "title": "Python Programming Complete Course",
        "lesson_ids": ["lesson_abc123", "lesson_def456"],
        "description": "Full Python course from beginner to advanced"
    }
    ```
    """
    try:
        minister = get_education_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="create_curriculum",
            task_data={
                "title": request.title,
                "lesson_ids": request.lesson_ids,
                "description": request.description
            }
        )

        if report.status.value == "completed":
            result = report.result
            return CreateCurriculumResponse(
                success=result["success"],
                curriculum_id=result["curriculum_id"],
                title=result["title"],
                total_lessons=result["total_lessons"],
                total_duration_minutes=result["total_duration_minutes"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Curriculum creation failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error creating curriculum: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/curriculum/list")
async def list_curricula():
    """
    قائمة جميع المناهج التدريبية
    List all training curricula
    """
    try:
        minister = get_education_minister()
        curricula = minister.get_curricula()

        return {
            "success": True,
            "total": len(curricula),
            "curricula": list(curricula.values())
        }

    except Exception as e:
        logger.error(f"Error listing curricula: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/curriculum/{curriculum_id}")
async def get_curriculum(curriculum_id: str):
    """
    الحصول على تفاصيل منهج معين
    Get details of a specific curriculum
    """
    try:
        minister = get_education_minister()
        curricula = minister.get_curricula()

        if curriculum_id not in curricula:
            raise HTTPException(status_code=404, detail="Curriculum not found")

        return {
            "success": True,
            "curriculum": curricula[curriculum_id]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting curriculum: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistics & Status Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/statistics")
async def get_education_statistics():
    """
    إحصائيات وزير التعليم
    Get Education Minister statistics

    Returns:
    - Total resources, lessons, curricula
    - Vision capabilities status
    - Web scraping status
    """
    try:
        minister = get_education_minister()
        stats = minister.get_education_statistics()

        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_education_minister_status():
    """
    حالة وزير التعليم
    Get Education Minister status report
    """
    try:
        minister = get_education_minister()
        status = minister.get_status_report()

        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Combined Workflow Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/workflow/resource-to-lesson")
async def resource_to_lesson_workflow(
    url: str,
    title: str,
    extract_content: bool = True,
    generate_lesson: bool = True,
    difficulty: str = "intermediate",
    tags: List[str] = []
):
    """
    سير عمل كامل: من مصدر إلى درس
    Complete workflow: From resource to lesson

    Steps:
    1. Add resource
    2. Extract content
    3. Generate lesson

    Example:
    ```json
    {
        "url": "https://www.example.com/python-tutorial",
        "title": "Python Tutorial",
        "extract_content": true,
        "generate_lesson": true,
        "difficulty": "beginner",
        "tags": ["python", "programming"]
    }
    ```
    """
    try:
        minister = get_education_minister()
        workflow_results = {}

        # Step 1: Add resource
        logger.info(f"Step 1: Adding resource - {url}")
        add_task_id = generate_task_id()
        add_report = await minister.execute_task(
            task_id=add_task_id,
            task_type="add_learning_resource",
            task_data={"url": url, "title": title, "tags": tags}
        )
        workflow_results["resource_added"] = add_report.result

        if not extract_content:
            return {
                "success": True,
                "workflow": "resource_only",
                "results": workflow_results
            }

        # Step 2: Extract content
        logger.info("Step 2: Extracting content")
        resource_id = add_report.result.get("resource_id")
        extract_task_id = generate_task_id()
        extract_report = await minister.execute_task(
            task_id=extract_task_id,
            task_type="extract_content",
            task_data={"resource_id": resource_id}
        )
        workflow_results["content_extracted"] = extract_report.result

        if not generate_lesson:
            return {
                "success": True,
                "workflow": "resource_and_extract",
                "results": workflow_results
            }

        # Step 3: Generate lesson
        logger.info("Step 3: Generating lesson")
        extracted = extract_report.result.get("extracted_content", {})

        # Combine content from different types
        content = ""
        if "full_transcript" in extracted:
            content = extracted["full_transcript"]
        elif "paragraphs" in extracted:
            content = " ".join(extracted["paragraphs"])
        elif "full_text" in extracted:
            content = extracted["full_text"]
        else:
            content = str(extracted)

        if content:
            lesson_task_id = generate_task_id()
            lesson_report = await minister.execute_task(
                task_id=lesson_task_id,
                task_type="generate_lesson",
                task_data={
                    "content": content,
                    "title": title,
                    "difficulty": difficulty,
                    "tags": tags
                }
            )
            workflow_results["lesson_generated"] = lesson_report.result

        return {
            "success": True,
            "workflow": "complete",
            "results": workflow_results
        }

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🎓 Education API Routes initialized")
    print(f"   Available: {EDUCATION_MINISTER_AVAILABLE}")
