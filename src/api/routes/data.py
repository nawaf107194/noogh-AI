#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقاط نهاية إدارة البيانات
Data Management Endpoints

يحتوي على endpoints إدارة البيانات ومصادرها
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import pandas as pd
import csv
from datetime import datetime, timezone

# استخدام BASE_DIR من config بدلاً من hardcoded path
from config import BASE_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ..models import APIResponse, DataSource, DataQuery
from ..auth import get_current_user, require_permission, Permission, User
from api.utils.logger import get_logger, LogCategory
from api.utils.error_handler_factory import get_error_handler
from api.utils.error_handler import NooghError, ErrorCategory, ErrorSeverity

router = APIRouter()
logger = get_logger("data_api")
error_handler = get_error_handler()

@router.get("/sources", response_model=APIResponse)
async def list_data_sources(current_user: User = Depends(require_permission(Permission.DATA_READ))):
    """قائمة مصادر البيانات المتاحة"""
    
    try:
        sources = []
        data_dir = BASE_DIR / "data"
        
        if data_dir.exists():
            # فحص ملفات البيانات المختلفة
            data_files = {
                "market_data.csv": {"type": "market", "description": "بيانات السوق المالي"},
                "live_signals.csv": {"type": "signals", "description": "إشارات التداول المباشرة"},
                "training_data.csv": {"type": "training", "description": "بيانات التدريب"},
                "recommendations.csv": {"type": "recommendations", "description": "التوصيات"}
            }
            
            for filename, info in data_files.items():
                file_path = data_dir / filename
                if file_path.exists():
                    stat = file_path.stat()
                    
                    source = DataSource(
                        source_id=filename.replace('.csv', ''),
                        name=info["description"],
                        type=info["type"],
                        status="active",
                        last_update=datetime.fromtimestamp(stat.st_mtime).isoformat()
                    )
                    sources.append(source)
        
        logger.info(f"تم جلب مصادر البيانات بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم جلب مصادر البيانات بنجاح",
            data={
                "sources": [source.dict() for source in sources],
                "total": len(sources)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'list_data_sources',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب مصادر البيانات")

@router.post("/query", response_model=APIResponse)
async def query_data(
    query: DataQuery,
    current_user: User = Depends(require_permission(Permission.DATA_READ))
):
    """استعلام البيانات من مصدر معين"""
    
    try:
        data_file = BASE_DIR / "data" / f"{query.source}.csv"
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="مصدر البيانات غير موجود")
        
        # قراءة البيانات
        df = pd.read_csv(data_file)
        
        # تطبيق المرشحات
        for key, value in query.filters.items():
            if key in df.columns:
                df = df[df[key] == value]
        
        # ترتيب البيانات
        if query.sort_by and query.sort_by in df.columns:
            ascending = query.sort_order.lower() == 'asc'
            df = df.sort_values(by=query.sort_by, ascending=ascending)
        
        # تطبيق الحدود
        total_rows = len(df)
        df = df.iloc[query.offset:query.offset + query.limit]
        
        # تحويل إلى قاموس
        result_data = df.to_dict('records')
        
        logger.info(f"تم استعلام البيانات من {query.source} بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم استعلام البيانات بنجاح",
            data={
                "source": query.source,
                "data": result_data,
                "total_rows": total_rows,
                "returned_rows": len(result_data),
                "filters": query.filters
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'query_data',
            'user': current_user.username,
            'source': query.source
        })
        raise HTTPException(status_code=500, detail="فشل في استعلام البيانات")

@router.post("/upload", response_model=APIResponse)
async def upload_data_file(
    file: UploadFile = File(...),
    source_name: Optional[str] = None,
    overwrite: bool = False,
    current_user: User = Depends(require_permission(Permission.DATA_WRITE))
):
    """رفع ملف بيانات جديد"""
    
    try:
        # التحقق من نوع الملف
        allowed_extensions = ['.csv', '.json', '.xlsx']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"نوع الملف غير مدعوم. الأنواع المدعومة: {allowed_extensions}"
            )
        
        # تحديد اسم الملف
        if not source_name:
            source_name = Path(file.filename).stem
        
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / f"{source_name}{file_extension}"
        
        # فحص وجود الملف
        if file_path.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail="الملف موجود بالفعل. استخدم overwrite=true للكتابة فوقه"
            )
        
        # حفظ الملف
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # تحليل البيانات للحصول على معلومات إضافية
        file_info = {
            "source_id": source_name,
            "original_filename": file.filename,
            "file_size": len(content),
            "uploaded_by": current_user.username,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        
        # إذا كان ملف CSV، احصل على معلومات إضافية
        if file_extension == '.csv':
            try:
                df = pd.read_csv(file_path)
                file_info.update({
                    "rows_count": len(df),
                    "columns_count": len(df.columns),
                    "columns": list(df.columns)
                })
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Data validation error: {e}")
                pass
        
        logger.info(f"تم رفع ملف البيانات {source_name} بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم رفع ملف البيانات بنجاح",
            data=file_info,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'upload_data_file',
            'user': current_user.username,
            'filename': file.filename
        })
        raise HTTPException(status_code=500, detail="فشل في رفع ملف البيانات")

@router.get("/{source_id}/info", response_model=APIResponse)
async def get_data_source_info(
    source_id: str,
    current_user: User = Depends(require_permission(Permission.DATA_READ))
):
    """الحصول على معلومات مصدر بيانات معين"""
    
    try:
        data_file = BASE_DIR / "data" / f"{source_id}.csv"
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="مصدر البيانات غير موجود")
        
        # معلومات الملف
        stat = data_file.stat()
        file_info = {
            "source_id": source_id,
            "file_path": str(data_file),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        
        # تحليل محتوى الملف
        try:
            df = pd.read_csv(data_file)
            
            file_info.update({
                "rows_count": len(df),
                "columns_count": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head(5).to_dict('records'),
                "null_counts": df.isnull().sum().to_dict()
            })
            
            # إحصائيات أساسية للأعمدة الرقمية
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                file_info["statistics"] = df[numeric_columns].describe().to_dict()
        
        except Exception as e:
            file_info["analysis_error"] = str(e)
        
        logger.info(f"تم جلب معلومات مصدر البيانات {source_id} بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم جلب معلومات مصدر البيانات بنجاح",
            data=file_info,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_data_source_info',
            'user': current_user.username,
            'source_id': source_id
        })
        raise HTTPException(status_code=500, detail="فشل في جلب معلومات مصدر البيانات")

@router.delete("/{source_id}", response_model=APIResponse)
async def delete_data_source(
    source_id: str,
    current_user: User = Depends(require_permission(Permission.DATA_DELETE))
):
    """حذف مصدر بيانات"""
    
    try:
        deleted_files = []
        
        # البحث عن جميع الملفات المرتبطة بالمصدر
        data_dir = BASE_DIR / "data"
        
        for extension in ['.csv', '.json', '.xlsx']:
            file_path = data_dir / f"{source_id}{extension}"
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail="مصدر البيانات غير موجود")
        
        logger.warning(f"تم حذف مصدر البيانات {source_id} بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم حذف مصدر البيانات بنجاح",
            data={
                "source_id": source_id,
                "deleted_files": deleted_files
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'delete_data_source',
            'user': current_user.username,
            'source_id': source_id
        })
        raise HTTPException(status_code=500, detail="فشل في حذف مصدر البيانات")

@router.post("/clean", response_model=APIResponse)
async def clean_data(
    source_id: str,
    background_tasks: BackgroundTasks,
    remove_duplicates: bool = True,
    fill_missing: bool = True,
    current_user: User = Depends(require_permission(Permission.DATA_WRITE))
):
    """تنظيف البيانات"""
    
    try:
        data_file = BASE_DIR / "data" / f"{source_id}.csv"
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="مصدر البيانات غير موجود")
        
        cleaning_id = f"clean_{int(datetime.now().timestamp())}"
        
        logger.info(f"بدء تنظيف البيانات {source_id} بواسطة {current_user.username}", LogCategory.DATA)
        
        # إضافة مهمة التنظيف في الخلفية
        background_tasks.add_task(
            _clean_data_task, 
            source_id, 
            cleaning_id, 
            remove_duplicates, 
            fill_missing
        )
        
        return APIResponse(
            success=True,
            message="تم بدء عملية تنظيف البيانات",
            data={
                "cleaning_id": cleaning_id,
                "source_id": source_id,
                "status": "in_progress"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'clean_data',
            'user': current_user.username,
            'source_id': source_id
        })
        raise HTTPException(status_code=500, detail="فشل في بدء تنظيف البيانات")

async def _clean_data_task(source_id: str, cleaning_id: str, remove_duplicates: bool, fill_missing: bool):
    """مهمة تنظيف البيانات"""
    
    try:
        from api.utils.data_cleaner import DataCleaner
        
        data_file = BASE_DIR / "data" / f"{source_id}.csv"
        
        # قراءة البيانات
        df = pd.read_csv(data_file)
        original_rows = len(df)
        
        cleaner = DataCleaner()
        
        # تطبيق التنظيف
        if remove_duplicates:
            df = cleaner.remove_duplicates(df)
        
        if fill_missing:
            df = cleaner.handle_missing_values(df)
        
        # حفظ البيانات المنظفة
        cleaned_file = BASE_DIR / "data" / f"{source_id}_cleaned.csv"
        df.to_csv(cleaned_file, index=False)
        
        cleaned_rows = len(df)
        
        logger.info(
            f"تم تنظيف البيانات {source_id}: {original_rows} -> {cleaned_rows} صف",
            LogCategory.DATA
        )
        
    except Exception as e:
        logger.error(f"فشل في تنظيف البيانات {cleaning_id}: {e}", LogCategory.DATA)

@router.post("/export", response_model=APIResponse)
async def export_data(
    source_id: str,
    format: str = "csv",
    filters: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(require_permission(Permission.DATA_READ))
):
    """تصدير البيانات بتنسيق معين"""
    
    try:
        if format not in ['csv', 'json', 'excel']:
            raise HTTPException(status_code=400, detail="تنسيق التصدير غير مدعوم")
        
        data_file = BASE_DIR / "data" / f"{source_id}.csv"
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="مصدر البيانات غير موجود")
        
        # قراءة البيانات
        df = pd.read_csv(data_file)
        
        # تطبيق المرشحات إذا وجدت
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        # إنشاء مجلد التصدير
        export_dir = BASE_DIR / "data" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # تحديد اسم الملف المصدر
        timestamp = int(datetime.now().timestamp())
        export_filename = f"{source_id}_export_{timestamp}"
        
        # تصدير حسب التنسيق
        if format == 'csv':
            export_path = export_dir / f"{export_filename}.csv"
            df.to_csv(export_path, index=False)
        elif format == 'json':
            export_path = export_dir / f"{export_filename}.json"
            df.to_json(export_path, orient='records', indent=2)
        elif format == 'excel':
            export_path = export_dir / f"{export_filename}.xlsx"
            df.to_excel(export_path, index=False)
        
        logger.info(f"تم تصدير البيانات {source_id} بتنسيق {format} بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم تصدير البيانات بنجاح",
            data={
                "export_path": str(export_path),
                "format": format,
                "rows_exported": len(df),
                "file_size": export_path.stat().st_size
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'export_data',
            'user': current_user.username,
            'source_id': source_id,
            'format': format
        })
        raise HTTPException(status_code=500, detail="فشل في تصدير البيانات")

@router.get("/stats/summary", response_model=APIResponse)
async def get_data_stats(current_user: User = Depends(require_permission(Permission.DATA_READ))):
    """الحصول على إحصائيات البيانات العامة"""
    
    try:
        stats = {
            "total_sources": 0,
            "total_size_mb": 0,
            "total_rows": 0,
            "sources_by_type": {},
            "largest_source": None,
            "most_recent_update": None
        }
        
        data_dir = BASE_DIR / "data"
        
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            stats["total_sources"] = len(csv_files)
            
            largest_size = 0
            most_recent_time = 0
            
            for csv_file in csv_files:
                file_stat = csv_file.stat()
                file_size = file_stat.st_size / (1024 * 1024)  # MB
                stats["total_size_mb"] += file_size
                
                # أكبر ملف
                if file_size > largest_size:
                    largest_size = file_size
                    stats["largest_source"] = {
                        "name": csv_file.stem,
                        "size_mb": file_size
                    }
                
                # أحدث تحديث
                if file_stat.st_mtime > most_recent_time:
                    most_recent_time = file_stat.st_mtime
                    stats["most_recent_update"] = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                
                # عدد الصفوف
                try:
                    df = pd.read_csv(csv_file)
                    stats["total_rows"] += len(df)
                except (ValueError, KeyError, TypeError) as e:
                    logger.error(f"Data processing error: {e}")
                    pass
        
        logger.info(f"تم جلب إحصائيات البيانات بواسطة {current_user.username}", LogCategory.DATA)
        
        return APIResponse(
            success=True,
            message="تم جلب إحصائيات البيانات بنجاح",
            data=stats,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_data_stats',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب إحصائيات البيانات")
