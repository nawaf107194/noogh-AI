#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - Education Minister v2.0
نظام الحكومة الداخلية لنوغ - وزير التعليم

Version: 2.0.0
Features:
- ✅ Computer vision integration (OCR, scene understanding)
- ✅ Content extraction (YouTube videos, websites)
- ✅ Automatic lesson generation
- ✅ Curriculum management
- ✅ Learning resource tracking
- ✅ Training data preparation
"""

from pathlib import Path
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json
import re

# Base Minister Framework
from .base_minister import (
    BaseMinister,
    MinisterType,
    MinisterReport,
    MinisterResponse,
    Priority,
    TaskStatus,
    generate_task_id
)

# Vision modules
try:
    from vision.ocr_engine import OCREngine, OCRLanguage
    from vision.scene_understanding import SceneUnderstandingEngine
    from vision.image_analyzer import ImageAnalyzer
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    logging.warning(f"Vision modules not available: {e}")

# Content extraction
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """أنواع المحتوى التعليمي"""
    YOUTUBE_VIDEO = "youtube_video"
    WEBSITE = "website"
    PDF_DOCUMENT = "pdf_document"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


class LessonDifficulty(Enum):
    """مستوى صعوبة الدرس"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EducationMinister(BaseMinister):
    """
    وزير التعليم - Minister of Education

    المسؤوليات:
    1. إدارة الموارد التعليمية (YouTube, websites, documents)
    2. استخراج المحتوى باستخدام الرؤية الحاسوبية
    3. توليد دروس تلقائياً من المحتوى
    4. إدارة المناهج التدريبية
    5. تتبع التقدم التعليمي
    """

    def __init__(
        self,
        verbose: bool = True,
        enable_vision: bool = True,
        enable_web_scraping: bool = True,
        brain_hub: Any = None
    ):
        """
        Args:
            verbose: عرض التفاصيل
            enable_vision: تفعيل الرؤية الحاسوبية
            enable_web_scraping: تفعيل استخراج محتوى المواقع
            brain_hub: Reference to UnifiedBrainHub
        """
        # Authorities - الصلاحيات
        authorities = [
            "manage_learning_resources",
            "extract_content",
            "generate_lessons",
            "manage_curriculum",
            "access_training_data",
            "analyze_visual_content",
            "scrape_websites",
            "process_youtube_videos",
            "track_learning_progress"
        ]

        # Resources - الموارد
        resources = {
            "vision_enabled": VISION_AVAILABLE and enable_vision,
            "web_scraping_enabled": WEB_SCRAPING_AVAILABLE and enable_web_scraping,
            "learning_resources": [],
            "generated_lessons": [],
            "curricula": []
        }

        super().__init__(
            minister_type=MinisterType.EDUCATION,
            name="Education Minister",
            authorities=authorities,
            resources=resources,
            verbose=verbose,
            specialty="Educational Content Management & Curriculum Development",
            description="Manages learning resources, extracts content, and generates training curricula",
            expertise_level=0.90,
            brain_hub=brain_hub
        )

        # Initialize vision modules
        self.ocr_engine = None
        self.scene_engine = None
        self.image_analyzer = None

        if VISION_AVAILABLE and enable_vision:
            try:
                self.ocr_engine = OCREngine(default_language=OCRLanguage.AUTO)
                self.scene_engine = SceneUnderstandingEngine()
                self.image_analyzer = ImageAnalyzer()
                logger.info("✅ Vision modules initialized")
            except Exception as e:
                logger.warning(f"⚠️  Vision initialization failed: {e}")

        # Learning resources database
        self.learning_resources: List[Dict[str, Any]] = []
        self.generated_lessons: List[Dict[str, Any]] = []
        self.curricula: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_resources_added = 0
        self.total_lessons_generated = 0
        self.total_content_extracted = 0

        if self.verbose:
            logger.info(f"\n🎓 {self.get_arabic_title()} initialized")
            logger.info(f"   Vision: {'✅ Enabled' if self.ocr_engine else '❌ Disabled'}")
            logger.info(f"   Web Scraping: {'✅ Enabled' if WEB_SCRAPING_AVAILABLE else '❌ Disabled'}")

    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بوزير التعليم"""
        # The President sends a general "educational" task type.
        if task_type == "educational":
            return True

        # Also handle specific internal tasks
        education_tasks = [
            "add_learning_resource",
            "extract_content",
            "generate_lesson",
            "create_curriculum",
            "analyze_image",
            "extract_text",
            "scrape_website",
            "process_youtube"
        ]

        return task_type in education_tasks

    async def _execute_specific_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """تنفيذ المهام الخاصة بوزير التعليم"""

        # If the president sends a general request, we try to generate a lesson.
        if task_type == "educational":
            user_input = task_data.get("user_input", "")
            return await self._generate_lesson({"content": user_input, "title": f"Lesson on {user_input[:20]}"})

        if task_type == "add_learning_resource":
            return await self._add_learning_resource(task_data)

        elif task_type == "extract_content":
            return await self._extract_content(task_data)

        elif task_type == "generate_lesson":
            return await self._generate_lesson(task_data)

        elif task_type == "create_curriculum":
            return await self._create_curriculum(task_data)

        elif task_type == "analyze_image":
            return await self._analyze_image(task_data)

        elif task_type == "extract_text":
            return await self._extract_text_from_image(task_data)

        elif task_type == "scrape_website":
            return await self._scrape_website(task_data)

        elif task_type == "process_youtube":
            return await self._process_youtube_video(task_data)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Learning Resource Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _add_learning_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        إضافة مصدر تعليمي جديد

        Args:
            data: {
                "url": str,
                "title": str (optional),
                "description": str (optional),
                "tags": List[str] (optional)
            }
        """
        url = data.get("url")
        if not url:
            raise ValueError("URL is required")

        # Determine content type
        content_type = self._detect_content_type(url)

        resource = {
            "id": generate_task_id(),
            "url": url,
            "title": data.get("title", "Untitled Resource"),
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "content_type": content_type.value,
            "added_at": datetime.now().isoformat(),
            "processed": False,
            "lessons_generated": 0
        }

        self.learning_resources.append(resource)
        self.total_resources_added += 1

        return {
            "success": True,
            "resource_id": resource["id"],
            "content_type": content_type.value,
            "message": f"Resource added: {resource['title']}"
        }

    def _detect_content_type(self, url: str) -> ContentType:
        """كشف نوع المحتوى من URL"""
        url_lower = url.lower()

        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return ContentType.YOUTUBE_VIDEO
        elif url_lower.endswith(".pdf"):
            return ContentType.PDF_DOCUMENT
        elif url_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            return ContentType.IMAGE
        elif url_lower.startswith("http"):
            return ContentType.WEBSITE
        else:
            return ContentType.UNKNOWN

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Content Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _extract_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        استخراج المحتوى من مصدر تعليمي

        Args:
            data: {"resource_id": str} or {"url": str}
        """
        # Get resource
        resource_id = data.get("resource_id")
        url = data.get("url")

        if resource_id:
            resource = next((r for r in self.learning_resources if r["id"] == resource_id), None)
            if not resource:
                raise ValueError(f"Resource not found: {resource_id}")
            url = resource["url"]
            content_type = ContentType(resource["content_type"])
        else:
            content_type = self._detect_content_type(url)

        # Extract based on type
        if content_type == ContentType.YOUTUBE_VIDEO:
            result = await self._process_youtube_video({"url": url})
        elif content_type == ContentType.WEBSITE:
            result = await self._scrape_website({"url": url})
        elif content_type == ContentType.IMAGE:
            result = await self._analyze_image({"image_path": url})
        elif content_type == ContentType.PDF_DOCUMENT:
            result = await self._extract_text_from_pdf({"pdf_path": url})
        else:
            raise ValueError(f"Unsupported content type: {content_type.value}")

        self.total_content_extracted += 1

        return {
            "success": True,
            "content_type": content_type.value,
            "extracted_content": result
        }

    async def _scrape_website(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج محتوى من موقع ويب"""
        if not WEB_SCRAPING_AVAILABLE:
            return {
                "error": "Web scraping not available (missing dependencies)",
                "required": "pip install requests beautifulsoup4"
            }

        url = data.get("url")
        if not url:
            raise ValueError("URL is required")

        try:
            # Fetch the page
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Educational Bot)'
            })
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"

            # Extract main content
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('body')

            # Extract paragraphs
            paragraphs = [p.get_text().strip() for p in main_content.find_all('p') if p.get_text().strip()]

            # Extract headings
            headings = []
            for i in range(1, 7):
                headings.extend([h.get_text().strip() for h in main_content.find_all(f'h{i}')])

            # Extract code blocks (for programming tutorials)
            code_blocks = [code.get_text().strip() for code in main_content.find_all('code')]

            return {
                "url": url,
                "title": title_text,
                "headings": headings,
                "paragraphs": paragraphs,
                "code_blocks": code_blocks,
                "total_text_length": sum(len(p) for p in paragraphs),
                "extraction_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Website scraping failed: {e}")
            return {
                "error": str(e),
                "url": url
            }

    async def _process_youtube_video(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        معالجة فيديو YouTube

        ملاحظة: يتطلب youtube-transcript-api للحصول على النصوص
        """
        url = data.get("url")
        if not url:
            raise ValueError("URL is required")

        try:
            # Extract video ID
            video_id = self._extract_youtube_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")

            # Try to get transcript
            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                # Get transcript (try Arabic first, then English)
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ar'])
                    language = 'ar'
                except Exception:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                    language = 'en'

                # Combine transcript segments
                full_text = " ".join([entry['text'] for entry in transcript])
                timestamps = [{"time": entry['start'], "text": entry['text']} for entry in transcript]

                return {
                    "video_id": video_id,
                    "url": url,
                    "transcript_available": True,
                    "language": language,
                    "full_transcript": full_text,
                    "timestamped_segments": timestamps,
                    "duration_seconds": transcript[-1]['start'] + transcript[-1]['duration'] if transcript else 0,
                    "total_segments": len(transcript)
                }

            except ImportError:
                return {
                    "video_id": video_id,
                    "url": url,
                    "transcript_available": False,
                    "error": "youtube-transcript-api not installed",
                    "suggestion": "pip install youtube-transcript-api"
                }
            except Exception as e:
                return {
                    "video_id": video_id,
                    "url": url,
                    "transcript_available": False,
                    "error": str(e)
                }

        except Exception as e:
            logger.error(f"YouTube processing failed: {e}")
            return {
                "error": str(e),
                "url": url
            }

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """استخراج معرف فيديو YouTube من URL"""
        # Pattern for youtube.com/watch?v=...
        pattern1 = r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})'
        # Pattern for youtu.be/...
        pattern2 = r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})'

        match = re.search(pattern1, url) or re.search(pattern2, url)
        return match.group(1) if match else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Computer Vision Integration
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _analyze_image(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل صورة باستخدام الرؤية الحاسوبية"""
        if not VISION_AVAILABLE or not self.scene_engine:
            return {
                "error": "Vision modules not available",
                "vision_enabled": False
            }

        image_path = data.get("image_path")
        if not image_path:
            raise ValueError("image_path is required")

        try:
            # Scene understanding
            scene_analysis = self.scene_engine.analyze_scene(image_path)

            return {
                "image_path": image_path,
                "scene_type": scene_analysis.scene_context.scene_type.value,
                "lighting": scene_analysis.scene_context.lighting_condition.value,
                "complexity": scene_analysis.complexity_score,
                "interpretability": scene_analysis.interpretability,
                "contextual_clues": scene_analysis.scene_context.contextual_clues,
                "confidence": scene_analysis.scene_context.confidence
            }

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "error": str(e),
                "image_path": image_path
            }

    async def _extract_text_from_image(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج نص من صورة باستخدام OCR"""
        if not VISION_AVAILABLE or not self.ocr_engine:
            return {
                "error": "OCR engine not available",
                "vision_enabled": False
            }

        image_path = data.get("image_path")
        language = data.get("language", "auto")

        if not image_path:
            raise ValueError("image_path is required")

        try:
            # Convert language string to enum
            if language == "auto":
                lang_enum = OCRLanguage.AUTO
            elif language == "ar":
                lang_enum = OCRLanguage.ARABIC
            elif language == "en":
                lang_enum = OCRLanguage.ENGLISH
            else:
                lang_enum = OCRLanguage.AUTO

            # Extract text
            ocr_result = self.ocr_engine.extract_text(image_path, language=lang_enum)

            return {
                "image_path": image_path,
                "full_text": ocr_result.full_text,
                "detected_languages": ocr_result.detected_languages,
                "confidence": ocr_result.average_confidence,
                "confidence_level": ocr_result.confidence_level.value,
                "regions_count": len(ocr_result.regions)
            }

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "error": str(e),
                "image_path": image_path
            }

    async def _extract_text_from_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج نص من PDF"""
        if not VISION_AVAILABLE or not self.ocr_engine:
            return {
                "error": "OCR engine not available",
                "vision_enabled": False
            }

        pdf_path = data.get("pdf_path")
        page_number = data.get("page_number", 0)

        try:
            ocr_result = self.ocr_engine.extract_text_from_pdf(pdf_path, page_number)

            return {
                "pdf_path": pdf_path,
                "page_number": page_number,
                "extracted_text": ocr_result.full_text,
                "confidence": ocr_result.average_confidence
            }

        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {
                "error": str(e),
                "pdf_path": pdf_path
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Lesson Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _generate_lesson(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        توليد درس من محتوى مستخرج

        Args:
            data: {
                "content": str,
                "title": str,
                "difficulty": str (optional),
                "tags": List[str] (optional)
            }
        """
        content = data.get("content")
        title = data.get("title", "Untitled Lesson")
        difficulty = data.get("difficulty", "intermediate")
        tags = data.get("tags", [])

        if not content:
            raise ValueError("Content is required")

        # Analyze content
        word_count = len(content.split())
        estimated_duration_minutes = max(5, word_count // 150)  # Assuming 150 words per minute

        # Extract key points (simple implementation)
        key_points = self._extract_key_points(content)

        # Generate lesson structure
        lesson = {
            "id": generate_task_id(),
            "title": title,
            "difficulty": difficulty,
            "tags": tags,
            "content": content,
            "key_points": key_points,
            "word_count": word_count,
            "estimated_duration_minutes": estimated_duration_minutes,
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }

        self.generated_lessons.append(lesson)
        self.total_lessons_generated += 1

        return {
            "success": True,
            "lesson_id": lesson["id"],
            "title": title,
            "word_count": word_count,
            "estimated_duration": estimated_duration_minutes,
            "key_points_count": len(key_points)
        }

    def _extract_key_points(self, content: str, max_points: int = 5) -> List[str]:
        """استخراج النقاط الرئيسية من المحتوى (تنفيذ بسيط)"""
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        # Take first few sentences as key points (simple heuristic)
        # In production, use NLP/ML for better extraction
        key_points = sentences[:max_points]

        return key_points

    async def _create_curriculum(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        إنشاء منهج تدريبي من دروس متعددة

        Args:
            data: {
                "title": str,
                "lesson_ids": List[str],
                "description": str (optional)
            }
        """
        title = data.get("title")
        lesson_ids = data.get("lesson_ids", [])
        description = data.get("description", "")

        if not title:
            raise ValueError("Title is required")
        if not lesson_ids:
            raise ValueError("At least one lesson is required")

        # Validate lessons exist
        lessons = [l for l in self.generated_lessons if l["id"] in lesson_ids]
        if len(lessons) != len(lesson_ids):
            raise ValueError("Some lesson IDs not found")

        curriculum = {
            "id": generate_task_id(),
            "title": title,
            "description": description,
            "lessons": lessons,
            "total_lessons": len(lessons),
            "total_duration_minutes": sum(l["estimated_duration_minutes"] for l in lessons),
            "created_at": datetime.now().isoformat()
        }

        curriculum_id = curriculum["id"]
        self.curricula[curriculum_id] = curriculum

        return {
            "success": True,
            "curriculum_id": curriculum_id,
            "title": title,
            "total_lessons": len(lessons),
            "total_duration_minutes": curriculum["total_duration_minutes"]
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_learning_resources(self, limit: int = 100) -> List[Dict[str, Any]]:
        """الحصول على قائمة الموارد التعليمية"""
        return self.learning_resources[:limit]

    def get_generated_lessons(self, limit: int = 100) -> List[Dict[str, Any]]:
        """الحصول على قائمة الدروس المولدة"""
        return self.generated_lessons[:limit]

    def get_curricula(self) -> Dict[str, Dict[str, Any]]:
        """الحصول على المناهج التدريبية"""
        return self.curricula

    def get_education_statistics(self) -> Dict[str, Any]:
        """إحصائيات وزير التعليم"""
        return {
            "total_resources": len(self.learning_resources),
            "total_lessons": len(self.generated_lessons),
            "total_curricula": len(self.curricula),
            "resources_added": self.total_resources_added,
            "lessons_generated": self.total_lessons_generated,
            "content_extracted": self.total_content_extracted,
            "vision_enabled": VISION_AVAILABLE and self.ocr_engine is not None,
            "web_scraping_enabled": WEB_SCRAPING_AVAILABLE
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    """Test Education Minister"""
    print("🎓 Education Minister v2.0 - Test")
    print("=" * 70)
    print()

    # Initialize minister
    minister = EducationMinister(verbose=True)

    # Print status
    minister.print_status()

    # Test adding a resource
    print("\n📌 Testing: Add learning resource...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="add_learning_resource",
        task_data={
            "url": "https://www.example.com/python-tutorial",
            "title": "Python Programming Tutorial",
            "tags": ["python", "programming", "tutorial"]
        }
    )
    print(f"Result: {result.result}")

    # Print statistics
    print("\n📊 Education Statistics:")
    stats = minister.get_education_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Education Minister test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
