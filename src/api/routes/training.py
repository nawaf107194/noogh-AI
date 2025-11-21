#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Router - AI Training Data Management
نظام إدارة بيانات التدريب للذكاء الاصطناعي
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
# Add core to path
from training.data_loader import get_training_data_loader, TrainingDataLoader
from api.auth import get_current_user, User

router = APIRouter()


class TrainingStats(BaseModel):
    """Training data statistics"""
    total_files: int
    total_categories: int
    categories: List[str]
    total_size_mb: float
    files_by_extension: Dict[str, int]


class TrainingFile(BaseModel):
    """Training file model"""
    path: str
    name: str
    category: str
    size: int
    extension: str
    content: Optional[str] = None


@router.get("/stats", response_model=TrainingStats)
async def get_training_stats(user: User = Depends(get_current_user)):
    """
    إحصائيات بيانات التدريب
    Get training data statistics
    """
    try:
        loader = get_training_data_loader()
        stats = loader.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.get("/categories")
async def get_categories(user: User = Depends(get_current_user)):
    """
    قائمة جميع الفئات
    List all training categories
    """
    try:
        loader = get_training_data_loader()
        categories = loader.get_categories()
        return {
            "total": len(categories),
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")


@router.get("/files")
async def list_files(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: Optional[int] = Query(None, description="Limit number of files"),
    include_content: bool = Query(False, description="Include file content"),
    user: User = Depends(get_current_user)
):
    """
    قائمة جميع ملفات التدريب
    List all training files
    """
    try:
        loader = get_training_data_loader()

        if category:
            files = loader.load_by_category(category)
        else:
            files = loader.load_all(limit=limit)

        # Remove content if not requested to reduce response size
        if not include_content:
            for file in files:
                file.pop('content', None)

        return {
            "total": len(files),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@router.get("/file")
async def get_file(
    path: str = Query(..., description="File path"),
    user: User = Depends(get_current_user)
):
    """
    الحصول على ملف تدريب محدد
    Get specific training file
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        loader = get_training_data_loader()
        file_data = loader.load_file(file_path)

        if not file_data:
            raise HTTPException(status_code=500, detail="Error loading file")

        return file_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/prepare-dataset")
async def prepare_dataset(
    category: Optional[str] = None,
    max_files: Optional[int] = None,
    user: User = Depends(get_current_user)
):
    """
    تجهيز مجموعة بيانات للتدريب
    Prepare dataset for training
    """
    try:
        loader = get_training_data_loader()

        if category:
            files = loader.load_by_category(category)
        else:
            files = loader.load_all(limit=max_files)

        # Prepare training data
        dataset = []
        for file in files:
            dataset.append({
                'text': file['content'],
                'metadata': {
                    'name': file['name'],
                    'category': file['category'],
                    'path': file['path']
                }
            })

        return {
            "success": True,
            "dataset_size": len(dataset),
            "total_characters": sum(len(d['text']) for d in dataset),
            "categories": list(set(d['metadata']['category'] for d in dataset)),
            "message": "Dataset prepared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing dataset: {str(e)}")


@router.get("/info")
async def training_info():
    """معلومات نظام التدريب"""
    return {
        "name": "Noogh AI Training System",
        "description": "نظام تدريب الذكاء الاصطناعي من بيانات system-prompts-and-models",
        "version": "1.0.0",
        "data_source": "/home/noogh/system-prompts-and-models-of-ai-tools-main",
        "supported_formats": [".txt", ".md", ".json"],
        "features": [
            "Load training data from multiple categories",
            "Prepare datasets for AI training",
            "Statistics and analytics",
            "Category-based filtering",
            "Support for 233+ training files"
        ],
        "endpoints": {
            "stats": "/training/stats",
            "categories": "/training/categories",
            "files": "/training/files",
            "prepare": "/training/prepare-dataset"
        }
    }
