#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 GPU-Accelerated Tools Router
موجّه أدوات GPU المتسارعة

أقوى أدوات الذكاء الاصطناعي المعتمدة على GPU
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import sys

try:
    from src.core.plugin.gpu_accelerated_tools import (
        StableDiffusionGenerator,
        WhisperTranscriber,
        CLIPImageAnalyzer,
        EmbeddingsEngine,
        CodeGenerator,
        NeuralTranslator,
        ObjectDetector,
        get_gpu_info,
        benchmark_gpu,
        USE_GPU,
        DEVICE,
        DEVICE_STR
    )
    GPU_TOOLS_AVAILABLE = True
except ImportError as e:
    GPU_TOOLS_AVAILABLE = False
    import logging
    logging.warning(f"⚠️  GPU Tools not available: {e}")

router = APIRouter()

# Initialize tools
sd_gen = None
whisper = None
clip_analyzer = None
embeddings = None
code_gen = None
translator = None
detector = None

if GPU_TOOLS_AVAILABLE:
    sd_gen = StableDiffusionGenerator()
    whisper = WhisperTranscriber()
    clip_analyzer = CLIPImageAnalyzer()
    embeddings = EmbeddingsEngine()
    code_gen = CodeGenerator()
    translator = NeuralTranslator()
    detector = ObjectDetector()


# ==============================================================================
# Request Models
# ==============================================================================

class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_images: int = 1
    height: int = 512
    width: int = 512
    steps: int = 50
    guidance_scale: float = 7.5


class TranscribeRequest(BaseModel):
    audio_path: str
    language: str = "ar"
    task: str = "transcribe"


class ImageAnalysisRequest(BaseModel):
    image_path: str
    labels: Optional[List[str]] = None


class EmbeddingsRequest(BaseModel):
    texts: List[str]


class SimilarityRequest(BaseModel):
    query: str
    corpus: List[str]
    top_k: int = 5


class CodeGenRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "ara_Arab"
    target_lang: str = "eng_Latn"


class ObjectDetectionRequest(BaseModel):
    image_path: str
    confidence: float = 0.25


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get("/status")
async def gpu_status():
    """
    🔍 حالة أدوات GPU
    GPU tools status
    """
    if not GPU_TOOLS_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "أدوات GPU غير متاحة - تحتاج تثبيت المكتبات المطلوبة",
            "gpu_available": False,
            "required_packages": [
                "torch",
                "diffusers",
                "transformers",
                "whisper",
                "sentence-transformers",
                "ultralytics",
                "Pillow"
            ]
        }

    gpu_info = get_gpu_info()

    return {
        "status": "available",
        "message": "✅ أدوات GPU جاهزة",
        "version": "1.0.0",
        "gpu_info": gpu_info,
        "device": DEVICE_STR,
        "tools": {
            "image_generation": True,
            "speech_to_text": True,
            "image_analysis": True,
            "embeddings": True,
            "code_generation": True,
            "translation": True,
            "object_detection": True
        }
    }


@router.get("/info")
async def gpu_info_endpoint():
    """
    📋 معلومات أدوات GPU
    GPU tools information
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPU tools غير متاحة")

    return {
        "name": "Noogh GPU-Accelerated Tools",
        "description": "أقوى أدوات الذكاء الاصطناعي المعتمدة على GPU",
        "version": "1.0.0",
        "device": DEVICE_STR,
        "gpu_available": USE_GPU,
        "capabilities": [
            "🎨 توليد صور (Stable Diffusion)",
            "🎙️ تحويل كلام لنص (Whisper)",
            "🔍 تحليل صور (CLIP)",
            "🧠 Embeddings والبحث الدلالي",
            "💻 توليد كود (CodeLlama)",
            "🌍 ترجمة (200+ لغة)",
            "🎯 كشف أجسام (YOLO)"
        ],
        "endpoints": {
            "status": "/gpu/status",
            "info": "/gpu/info",
            "benchmark": "/gpu/benchmark",
            "generate_image": "/gpu/generate-image",
            "transcribe": "/gpu/transcribe",
            "analyze_image": "/gpu/analyze-image",
            "embeddings": "/gpu/embeddings",
            "similarity": "/gpu/similarity",
            "generate_code": "/gpu/generate-code",
            "translate": "/gpu/translate",
            "detect_objects": "/gpu/detect-objects"
        }
    }


@router.get("/benchmark")
async def run_benchmark():
    """
    ⚡ قياس أداء GPU
    Benchmark GPU performance
    """
    if not GPU_TOOLS_AVAILABLE or not USE_GPU:
        raise HTTPException(status_code=503, detail="GPU غير متاح")

    result = benchmark_gpu()
    return {"success": True, "benchmark": result}


@router.post("/generate-image")
async def generate_image(request: ImageGenRequest):
    """
    🎨 توليد صور باستخدام Stable Diffusion
    Generate images using Stable Diffusion

    Example:
    ```json
    {
        "prompt": "a beautiful sunset over mountains, oil painting style",
        "negative_prompt": "ugly, blurry",
        "num_images": 2,
        "height": 512,
        "width": 512,
        "steps": 50
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stable Diffusion غير متاح")

    try:
        result = await sd_gen.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images=request.num_images,
            height=request.height,
            width=request.width,
            steps=request.steps,
            guidance_scale=request.guidance_scale
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التوليد: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """
    🎙️ تحويل الكلام إلى نص
    Speech-to-Text using Whisper

    Example:
    ```json
    {
        "audio_path": "/home/noogh/audio.mp3",
        "language": "ar",
        "task": "transcribe"
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Whisper غير متاح")

    try:
        result = await whisper.transcribe(
            audio_path=request.audio_path,
            language=request.language,
            task=request.task
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التحويل: {str(e)}")


@router.post("/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """
    🔍 تحليل صورة باستخدام CLIP
    Analyze image using CLIP

    Example:
    ```json
    {
        "image_path": "/home/noogh/image.jpg",
        "labels": ["person", "car", "tree"]
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="CLIP غير متاح")

    try:
        result = await clip_analyzer.analyze_image(
            image_path=request.image_path,
            candidate_labels=request.labels
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التحليل: {str(e)}")


@router.post("/embeddings")
async def compute_embeddings(request: EmbeddingsRequest):
    """
    🧠 حساب Embeddings للنصوص
    Compute embeddings for texts

    Example:
    ```json
    {
        "texts": ["مرحباً", "Hello", "السلام عليكم"]
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings غير متاح")

    try:
        result = await embeddings.compute_embeddings(request.texts)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل الحساب: {str(e)}")


@router.post("/similarity")
async def find_similarity(request: SimilarityRequest):
    """
    🔎 البحث عن النصوص المتشابهة
    Find similar texts

    Example:
    ```json
    {
        "query": "ذكاء اصطناعي",
        "corpus": ["AI", "machine learning", "deep learning", "cooking"],
        "top_k": 3
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings غير متاح")

    try:
        result = await embeddings.find_similar(
            query=request.query,
            corpus=request.corpus,
            top_k=request.top_k
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل البحث: {str(e)}")


@router.post("/generate-code")
async def generate_code(request: CodeGenRequest):
    """
    💻 توليد كود
    Generate code

    Example:
    ```json
    {
        "prompt": "def calculate_fibonacci(n):",
        "max_length": 256,
        "temperature": 0.7
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Code Generator غير متاح")

    try:
        result = await code_gen.generate_code(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التوليد: {str(e)}")


@router.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    🌍 ترجمة نص (200+ لغة)
    Translate text (200+ languages)

    Language codes:
    - ara_Arab: Arabic
    - eng_Latn: English
    - fra_Latn: French
    - deu_Latn: German
    - spa_Latn: Spanish

    Example:
    ```json
    {
        "text": "مرحباً بالعالم",
        "source_lang": "ara_Arab",
        "target_lang": "eng_Latn"
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Translator غير متاح")

    try:
        result = await translator.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشلت الترجمة: {str(e)}")


@router.post("/detect-objects")
async def detect_objects(request: ObjectDetectionRequest):
    """
    🎯 كشف الأجسام في الصور
    Detect objects in images using YOLO

    Example:
    ```json
    {
        "image_path": "/home/noogh/photo.jpg",
        "confidence": 0.25
    }
    ```
    """
    if not GPU_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Object Detector غير متاح")

    try:
        result = await detector.detect_objects(
            image_path=request.image_path,
            confidence=request.confidence
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل الكشف: {str(e)}")
