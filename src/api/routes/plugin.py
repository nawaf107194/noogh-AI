#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin Router - GPT-based File Analysis & Code Generation
نظام البلق إن - تحليل الملفات وتوليد الكود بواسطة GPT

🎉 الإصدار الموسّع الكامل
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
# Add plugin paths
# Try to import extended plugin core
try:
    from plugin_gpu_core_extended import (
        analyze_file_with_gpt,
        analyze_directory,
        write_file_via_gpt,
        batch_analyze_files,
        compare_files,
        refactor_code,
        generate_documentation
    )
    PLUGIN_AVAILABLE = True
except ImportError as e:
    PLUGIN_AVAILABLE = False
    import logging
    logging.warning(f"⚠️  Plugin core not available: {e}")

router = APIRouter()


class AnalyzeFileRequest(BaseModel):
    """Request model for file analysis"""
    path: str


class AnalyzeDirRequest(BaseModel):
    """Request model for directory analysis"""
    root: str
    patterns: List[str] = ["*.py", "*.md", "*.txt", "*.json"]


class WriteFileRequest(BaseModel):
    """Request model for file writing"""
    target_path: str
    instruction: str
    base_text: Optional[str] = None
    overwrite: bool = False


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch file analysis"""
    file_paths: List[str]


class CompareFilesRequest(BaseModel):
    """Request model for file comparison"""
    file1: str
    file2: str


class RefactorRequest(BaseModel):
    """Request model for code refactoring"""
    file_path: str
    instructions: str


class DocumentRequest(BaseModel):
    """Request model for documentation generation"""
    file_path: str


@router.get("/status")
async def plugin_status():
    """
    🔍 فحص حالة نظام البلاجن
    Check plugin system status
    """
    if PLUGIN_AVAILABLE:
        return {
            "status": "available",
            "message": "✅ نظام البلاجن يعمل بشكل كامل - الإصدار الموسّع",
            "version": "2.0.0-extended",
            "features": {
                "file_analysis": True,
                "directory_analysis": True,
                "code_generation": True,
                "batch_analysis": True,
                "file_comparison": True,
                "code_refactoring": True,
                "documentation_generation": True
            },
            "models": {
                "default": "gpt-4.1-mini",
                "analyze": "gpt-4.1",
                "write": "gpt-4.1"
            },
            "plugin_path": "/home/noogh/noogh_ai/plugin",
            "extended_path": "/home/noogh/noogh_unified_system/core/plugin"
        }
    else:
        return {
            "status": "unavailable",
            "message": "❌ نظام البلاجن غير متاح - تحتاج تثبيت OpenAI SDK وضبط API Key",
            "features": {
                "file_analysis": False,
                "directory_analysis": False,
                "code_generation": False,
                "batch_analysis": False,
                "file_comparison": False,
                "code_refactoring": False,
                "documentation_generation": False
            },
            "models": {},
            "required": ["pip install openai>=1.0", "export OPENAI_API_KEY=sk-..."]
        }


@router.get("/info")
async def plugin_info():
    """
    📋 معلومات نظام البلاجن
    Plugin system information
    """
    return {
        "name": "Noogh GPT Plugin System - Extended Edition",
        "description": "نظام تحليل الملفات وتوليد الكود بواسطة GPT-4 - الإصدار الموسّع",
        "version": "2.0.0",
        "status": "available" if PLUGIN_AVAILABLE else "unavailable",
        "capabilities": [
            "📄 تحليل ملف واحد بواسطة GPT-4",
            "📁 تحليل مجلد كامل (حتى 50 ملف)",
            "✏️ كتابة وتوليد ملفات جديدة",
            "📚 تحليل دفعات من الملفات",
            "🔄 مقارنة ملفين",
            "🔧 إعادة هيكلة الكود",
            "📖 توليد توثيق تلقائي",
            "💾 معالجة متوازية (6 ملفات في نفس الوقت)",
            "🚀 تسريع GPU (إذا كان متاحاً)"
        ],
        "endpoints": {
            "status": "/plugin/status",
            "info": "/plugin/info",
            "analyze_file": "/plugin/analyze-file",
            "analyze_directory": "/plugin/analyze-directory",
            "write_file": "/plugin/write-file",
            "batch_analyze": "/plugin/batch-analyze",
            "compare": "/plugin/compare",
            "refactor": "/plugin/refactor",
            "document": "/plugin/document"
        },
        "paths": {
            "original": "/home/noogh/noogh_ai/plugin",
            "extended": "/home/noogh/noogh_unified_system/core/plugin"
        }
    }


@router.post("/analyze-file")
async def analyze_file(request: AnalyzeFileRequest):
    """
    📄 تحليل ملف بواسطة GPT-4
    Analyze a single file using GPT-4

    Example:
    ```json
    {
        "path": "/home/noogh/test.py"
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="نظام البلاجن غير متاح. قم بتثبيت: pip install openai>=1.0 وضبط OPENAI_API_KEY"
        )

    try:
        result = await analyze_file_with_gpt(request.path)
        return {
            "success": True,
            "path": request.path,
            "analysis": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التحليل: {str(e)}")


@router.post("/analyze-directory")
async def analyze_dir(request: AnalyzeDirRequest):
    """
    📁 تحليل مجلد كامل بواسطة GPT-4
    Analyze an entire directory (up to 50 files)

    Example:
    ```json
    {
        "root": "/home/noogh/project",
        "patterns": ["*.py", "*.md"]
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="نظام البلاجن غير متاح"
        )

    try:
        result = await analyze_directory(request.root, tuple(request.patterns))
        return {
            "success": True,
            "root": request.root,
            "patterns": request.patterns,
            "analysis": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التحليل: {str(e)}")


@router.post("/write-file")
async def write_file(request: WriteFileRequest):
    """
    ✏️ كتابة أو تعديل ملف بواسطة GPT-4
    Generate or modify a file using GPT-4

    Example:
    ```json
    {
        "target_path": "/home/noogh/new_file.py",
        "instruction": "اكتب دالة Python لحساب فيبوناتشي",
        "base_text": null,
        "overwrite": false
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="نظام البلاجن غير متاح"
        )

    try:
        result = await write_file_via_gpt(
            request.target_path,
            request.instruction,
            request.base_text,
            request.overwrite
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشلت الكتابة: {str(e)}")


@router.post("/batch-analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    📚 تحليل دفعة من الملفات
    Analyze multiple files in batch

    Example:
    ```json
    {
        "file_paths": [
            "/home/noogh/file1.py",
            "/home/noogh/file2.py"
        ]
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="نظام البلاجن غير متاح")

    try:
        result = await batch_analyze_files(request.file_paths)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التحليل: {str(e)}")


@router.post("/compare")
async def compare(request: CompareFilesRequest):
    """
    🔄 مقارنة ملفين باستخدام GPT-4
    Compare two files

    Example:
    ```json
    {
        "file1": "/home/noogh/old.py",
        "file2": "/home/noogh/new.py"
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="نظام البلاجن غير متاح")

    try:
        result = await compare_files(request.file1, request.file2)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشلت المقارنة: {str(e)}")


@router.post("/refactor")
async def refactor(request: RefactorRequest):
    """
    🔧 إعادة هيكلة الكود
    Refactor existing code

    Example:
    ```json
    {
        "file_path": "/home/noogh/old_code.py",
        "instructions": "حسّن الأداء وأضف type hints"
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="نظام البلاجن غير متاح")

    try:
        result = await refactor_code(request.file_path, request.instructions)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشلت إعادة الهيكلة: {str(e)}")


@router.post("/document")
async def document(request: DocumentRequest):
    """
    📖 توليد توثيق تلقائي
    Generate documentation automatically

    Example:
    ```json
    {
        "file_path": "/home/noogh/module.py"
    }
    ```
    """
    if not PLUGIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="نظام البلاجن غير متاح")

    try:
        result = await generate_documentation(request.file_path)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل التوليد: {str(e)}")
