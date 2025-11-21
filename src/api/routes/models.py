#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقاط نهاية إدارة النماذج
Models Management Endpoints

يحتوي على endpoints إدارة النماذج والتجميع
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import os
from datetime import datetime, timezone

# استخدام BASE_DIR من config بدلاً من hardcoded path
from src.core.config import BASE_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ..models import APIResponse, ModelInfo, ModelMetrics
from ..auth import get_current_user, require_permission, Permission, User
from api.utils.logger import get_logger, LogCategory
from api.utils.error_handler_factory import get_error_handler
from api.utils.error_handler import NooghError, ErrorCategory, ErrorSeverity

router = APIRouter()
logger = get_logger("models_api")
error_handler = get_error_handler()

@router.get("/", response_model=APIResponse)
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(require_permission(Permission.MODEL_READ))
):
    """قائمة النماذج المتاحة"""
    
    try:
        models = []
        
        # البحث في مجلد النماذج
        models_dir = BASE_DIR / "models"
        
        if models_dir.exists():
            # نماذج التجميع
            ensemble_dir = models_dir / "ensemble"
            if ensemble_dir.exists():
                for model_file in ensemble_dir.glob("*.json"):
                    try:
                        with open(model_file, 'r', encoding='utf-8') as f:
                            model_data = json.load(f)
                        
                        model_info = ModelInfo(
                            model_id=model_data.get('id', model_file.stem),
                            name=model_data.get('id', model_file.stem),
                            type="ensemble",
                            version=str(model_data.get('version', '1.0')),
                            size_mb=model_file.stat().st_size / (1024*1024),
                            accuracy=model_data.get('best_loss'),
                            created_at=model_data.get('timestamp', datetime.now().isoformat()),
                            status="active"
                        )
                        models.append(model_info)
                    except (ValueError, IOError) as e:
                        logger.warning(f"Model load error: {e}")
                        continue
            
            # نماذج حديقة النماذج
            zoo_dir = models_dir / "zoo"
            if zoo_dir.exists():
                for model_file in zoo_dir.glob("*.json"):
                    try:
                        with open(model_file, 'r', encoding='utf-8') as f:
                            model_data = json.load(f)
                        
                        model_info = ModelInfo(
                            model_id=model_data.get('id', model_file.stem),
                            name=model_data.get('name', model_file.stem),
                            type=model_data.get('type', 'unknown'),
                            version=str(model_data.get('version', '1.0')),
                            size_mb=model_file.stat().st_size / (1024*1024),
                            accuracy=model_data.get('performance', {}).get('accuracy'),
                            created_at=model_data.get('created_at', datetime.now().isoformat()),
                            status=model_data.get('status', 'inactive')
                        )
                        models.append(model_info)
                    except Exception:
                        continue
        
        # تطبيق المرشحات
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        # ترتيب وتطبيق الحدود
        models.sort(key=lambda x: x.created_at, reverse=True)
        total_models = len(models)
        models = models[offset:offset + limit]
        
        logger.info(f"تم جلب قائمة النماذج بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم جلب قائمة النماذج بنجاح",
            data={
                "models": [model.dict() for model in models],
                "total": total_models,
                "limit": limit,
                "offset": offset
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'list_models',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب قائمة النماذج")

@router.get("/{model_id}", response_model=APIResponse)
async def get_model_details(
    model_id: str,
    current_user: User = Depends(require_permission(Permission.MODEL_READ))
):
    """الحصول على تفاصيل نموذج معين"""
    
    try:
        model_data = None
        model_file_path = None
        
        # البحث في مجلدات النماذج
        search_dirs = [
            BASE_DIR / "models" / "ensemble",
            BASE_DIR / "models" / "zoo",
            BASE_DIR / "models" / "current"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                model_file = search_dir / f"{model_id}.json"
                if model_file.exists():
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                    model_file_path = str(model_file)
                    break
        
        if not model_data:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")
        
        logger.info(f"تم جلب تفاصيل النموذج {model_id} بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم جلب تفاصيل النموذج بنجاح",
            data={
                "model_data": model_data,
                "file_path": model_file_path,
                "file_size_mb": Path(model_file_path).stat().st_size / (1024*1024) if model_file_path else 0
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_model_details',
            'user': current_user.username,
            'model_id': model_id
        })
        raise HTTPException(status_code=500, detail="فشل في جلب تفاصيل النموذج")

@router.post("/ensemble", response_model=APIResponse)
async def create_ensemble(
    members: List[Dict[str, Any]],
    weights: List[float],
    task: str = "classification",
    current_user: User = Depends(require_permission(Permission.MODEL_WRITE))
):
    """إنشاء نموذج تجميع جديد"""
    
    try:
        from ...py.ensemble_manager_v2 import build_ensemble
        
        # التحقق من صحة البيانات
        if len(members) != len(weights):
            raise HTTPException(status_code=400, detail="عدد الأعضاء يجب أن يساوي عدد الأوزان")
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="مجموع الأوزان يجب أن يساوي 1.0")
        
        # حساب أفضل خسارة (متوسط خسائر الأعضاء)
        best_loss = sum(member.get('performance', {}).get('loss', 0.5) * weight 
                       for member, weight in zip(members, weights))
        
        # إنشاء التجميع
        ensemble_meta, ensemble_path = build_ensemble(members, weights, task, best_loss)
        
        logger.info(f"تم إنشاء نموذج تجميع بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم إنشاء نموذج التجميع بنجاح",
            data={
                "ensemble_id": ensemble_meta['id'],
                "ensemble_path": ensemble_path,
                "members_count": len(members),
                "task": task,
                "best_loss": best_loss
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'create_ensemble',
            'user': current_user.username,
            'members_count': len(members)
        })
        raise HTTPException(status_code=500, detail="فشل في إنشاء نموذج التجميع")

@router.post("/upload", response_model=APIResponse)
async def upload_model(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    model_type: str = "custom",
    current_user: User = Depends(require_permission(Permission.MODEL_WRITE))
):
    """رفع نموذج جديد"""
    
    try:
        # التحقق من نوع الملف
        allowed_extensions = ['.json', '.pkl', '.joblib', '.h5', '.pt', '.pth']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"نوع الملف غير مدعوم. الأنواع المدعومة: {allowed_extensions}"
            )
        
        # إنشاء مجلد الرفع
        upload_dir = BASE_DIR / "models" / "uploaded"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # تحديد اسم الملف
        if not model_name:
            model_name = Path(file.filename).stem
        
        file_path = upload_dir / f"{model_name}{file_extension}"
        
        # حفظ الملف
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # إنشاء ملف معلومات النموذج
        model_info = {
            "id": model_name,
            "name": model_name,
            "type": model_type,
            "version": "1.0",
            "uploaded_by": current_user.username,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "file_path": str(file_path),
            "file_size": len(content),
            "status": "uploaded"
        }
        
        info_file = upload_dir / f"{model_name}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم رفع النموذج {model_name} بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم رفع النموذج بنجاح",
            data={
                "model_id": model_name,
                "file_path": str(file_path),
                "file_size_mb": len(content) / (1024*1024),
                "model_type": model_type
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'upload_model',
            'user': current_user.username,
            'filename': file.filename
        })
        raise HTTPException(status_code=500, detail="فشل في رفع النموذج")

@router.delete("/{model_id}", response_model=APIResponse)
async def delete_model(
    model_id: str,
    current_user: User = Depends(require_permission(Permission.MODEL_DELETE))
):
    """حذف نموذج"""
    
    try:
        deleted_files = []
        
        # البحث وحذف النموذج من جميع المجلدات
        search_dirs = [
            BASE_DIR / "models" / "ensemble",
            BASE_DIR / "models" / "zoo",
            BASE_DIR / "models" / "uploaded"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                model_file = search_dir / f"{model_id}.json"
                if model_file.exists():
                    model_file.unlink()
                    deleted_files.append(str(model_file))
                
                # حذف ملفات النموذج الأخرى
                for ext in ['.pkl', '.joblib', '.h5', '.pt', '.pth']:
                    model_data_file = search_dir / f"{model_id}{ext}"
                    if model_data_file.exists():
                        model_data_file.unlink()
                        deleted_files.append(str(model_data_file))
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")
        
        logger.warning(f"تم حذف النموذج {model_id} بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم حذف النموذج بنجاح",
            data={
                "model_id": model_id,
                "deleted_files": deleted_files
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'delete_model',
            'user': current_user.username,
            'model_id': model_id
        })
        raise HTTPException(status_code=500, detail="فشل في حذف النموذج")

@router.post("/{model_id}/deploy", response_model=APIResponse)
async def deploy_model(
    model_id: str,
    current_user: User = Depends(require_permission(Permission.MODEL_DEPLOY))
):
    """نشر نموذج كنموذج حالي"""
    
    try:
        # البحث عن النموذج
        model_data = None
        source_file = None
        
        search_dirs = [
            BASE_DIR / "models" / "ensemble",
            BASE_DIR / "models" / "zoo",
            BASE_DIR / "models" / "uploaded"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                model_file = search_dir / f"{model_id}.json"
                if model_file.exists():
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                    source_file = model_file
                    break
        
        if not model_data:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")
        
        # إنشاء مجلد النماذج الحالية
        current_dir = BASE_DIR / "models" / "current"
        current_dir.mkdir(parents=True, exist_ok=True)
        
        # نسخ النموذج إلى المجلد الحالي
        import shutil
        current_file = current_dir / f"{model_id}.json"
        shutil.copy2(source_file, current_file)
        
        # تحديث ملف النموذج الحالي
        model_data['deployed_at'] = datetime.now(timezone.utc).isoformat()
        model_data['deployed_by'] = current_user.username
        model_data['status'] = 'deployed'
        
        with open(current_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        # تحديث رابط النموذج الحالي
        latest_link = current_dir / "latest.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        try:
            latest_link.symlink_to(current_file.name)
        except OSError:
            # إذا فشل الرابط الرمزي، انسخ الملف
            shutil.copy2(current_file, latest_link)
        
        logger.info(f"تم نشر النموذج {model_id} بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم نشر النموذج بنجاح",
            data={
                "model_id": model_id,
                "deployed_at": model_data['deployed_at'],
                "status": "deployed"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'deploy_model',
            'user': current_user.username,
            'model_id': model_id
        })
        raise HTTPException(status_code=500, detail="فشل في نشر النموذج")

@router.get("/{model_id}/metrics", response_model=APIResponse)
async def get_model_metrics(
    model_id: str,
    current_user: User = Depends(require_permission(Permission.MODEL_READ))
):
    """الحصول على مقاييس أداء النموذج"""
    
    try:
        # البحث عن النموذج
        model_data = None
        
        search_dirs = [
            BASE_DIR / "models" / "ensemble",
            BASE_DIR / "models" / "zoo",
            BASE_DIR / "models" / "current"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                model_file = search_dir / f"{model_id}.json"
                if model_file.exists():
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                    break
        
        if not model_data:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")
        
        # استخراج المقاييس
        performance = model_data.get('performance', {})
        
        metrics = ModelMetrics(
            accuracy=performance.get('accuracy', 0.0),
            precision=performance.get('precision', 0.0),
            recall=performance.get('recall', 0.0),
            f1_score=performance.get('f1_score', 0.0),
            loss=performance.get('loss', 0.0),
            training_time=performance.get('training_time', 0.0)
        )
        
        logger.info(f"تم جلب مقاييس النموذج {model_id} بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم جلب مقاييس النموذج بنجاح",
            data=metrics.dict(),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_model_metrics',
            'user': current_user.username,
            'model_id': model_id
        })
        raise HTTPException(status_code=500, detail="فشل في جلب مقاييس النموذج")

@router.get("/current/active", response_model=APIResponse)
async def get_current_model(current_user: User = Depends(require_permission(Permission.MODEL_READ))):
    """الحصول على النموذج الحالي النشط"""
    
    try:
        current_dir = BASE_DIR / "models" / "current"
        latest_file = current_dir / "latest.json"
        
        if not latest_file.exists():
            return APIResponse(
                success=True,
                message="لا يوجد نموذج نشط حالياً",
                data={"current_model": None},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # قراءة النموذج الحالي
        with open(latest_file, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        logger.info(f"تم جلب النموذج الحالي بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم جلب النموذج الحالي بنجاح",
            data={"current_model": model_data},
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_current_model',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب النموذج الحالي")

@router.get("/stats/summary", response_model=APIResponse)
async def get_models_stats(current_user: User = Depends(require_permission(Permission.MODEL_READ))):
    """الحصول على إحصائيات النماذج"""
    
    try:
        stats = {
            "total_models": 0,
            "by_type": {},
            "by_status": {},
            "total_size_mb": 0,
            "ensemble_models": 0,
            "zoo_models": 0,
            "uploaded_models": 0
        }
        
        # إحصاء النماذج في كل مجلد
        models_dir = BASE_DIR / "models"
        
        if models_dir.exists():
            for subdir in ["ensemble", "zoo", "uploaded"]:
                subdir_path = models_dir / subdir
                if subdir_path.exists():
                    model_files = list(subdir_path.glob("*.json"))
                    count = len(model_files)
                    stats[f"{subdir}_models"] = count
                    stats["total_models"] += count
                    
                    # حساب الحجم الإجمالي
                    for model_file in model_files:
                        stats["total_size_mb"] += model_file.stat().st_size / (1024*1024)
        
        logger.info(f"تم جلب إحصائيات النماذج بواسطة {current_user.username}", LogCategory.MODEL)
        
        return APIResponse(
            success=True,
            message="تم جلب إحصائيات النماذج بنجاح",
            data=stats,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_models_stats',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب إحصائيات النماذج")
