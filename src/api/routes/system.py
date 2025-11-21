#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقاط نهاية النظام
System Endpoints

يحتوي على endpoints إدارة النظام والتكوين
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import os
import sys
from pathlib import Path
import json
import psutil
from datetime import datetime, timezone

# استخدام BASE_DIR من config
from src.core.config import BASE_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from ..models import (
    APIResponse, SystemStatus, SystemConfig, ConfigUpdate,
    BackupRequest, RestoreRequest, SystemStats
)
from ..auth import get_current_user, require_permission, Permission, User
from api.utils.logger import get_logger, LogCategory
from api.utils.error_handler_factory import get_error_handler
from api.utils.error_handler import NooghError, ErrorCategory, ErrorSeverity
from api.utils.performance_monitor_factory import get_performance_monitor

router = APIRouter()
logger = get_logger("system_api")
error_handler = get_error_handler()
performance_monitor = get_performance_monitor()

@router.get("/status", response_model=APIResponse)
async def get_system_status(current_user: User = Depends(require_permission(Permission.SYSTEM_READ))):
    """الحصول على حالة النظام"""
    
    try:
        # جمع معلومات النظام
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # معلومات العمليات
        process_count = len(psutil.pids())
        
        # معلومات الشبكة
        network = psutil.net_io_counters()
        
        # حساب وقت التشغيل
        uptime = performance_monitor.get_performance_summary().get('uptime_seconds', 0)
        
        system_status = SystemStatus(
            status="healthy" if cpu_percent < 80 and memory.percent < 85 else "warning",
            version="1.0.0",  # يمكن جلبها من ملف الإصدار
            uptime=uptime,
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            active_connections=process_count
        )
        
        logger.info(f"تم جلب حالة النظام بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم جلب حالة النظام بنجاح",
            data=system_status.dict(),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_system_status',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب حالة النظام")

@router.get("/config", response_model=APIResponse)
async def get_system_config(current_user: User = Depends(require_permission(Permission.SYSTEM_READ))):
    """الحصول على تكوين النظام"""
    
    try:
        # قراءة ملف التكوين
        config_file = BASE_DIR / "config_default.yaml"
        
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}
        
        logger.info(f"تم جلب تكوين النظام بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم جلب التكوين بنجاح",
            data=config_data,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_system_config',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب التكوين")

@router.put("/config", response_model=APIResponse)
async def update_system_config(
    config_update: ConfigUpdate,
    current_user: User = Depends(require_permission(Permission.SYSTEM_WRITE))
):
    """تحديث تكوين النظام"""
    
    try:
        config_file = BASE_DIR / "config_default.yaml"
        
        # قراءة التكوين الحالي
        if config_file.exists():
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}
        
        # تحديث القيمة
        if config_update.section not in config_data:
            config_data[config_update.section] = {}
        
        config_data[config_update.section][config_update.key] = config_update.value
        
        # حفظ التكوين المحدث
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(
            f"تم تحديث التكوين بواسطة {current_user.username}: {config_update.section}.{config_update.key}",
            category=LogCategory.SYSTEM
        )
        
        return APIResponse(
            success=True,
            message="تم تحديث التكوين بنجاح",
            data={
                "section": config_update.section,
                "key": config_update.key,
                "new_value": config_update.value
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'update_system_config',
            'user': current_user.username,
            'config_update': config_update.dict()
        })
        raise HTTPException(status_code=500, detail="فشل في تحديث التكوين")

@router.post("/restart", response_model=APIResponse)
async def restart_system(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """إعادة تشغيل النظام"""
    
    try:
        logger.warning(f"طلب إعادة تشغيل النظام من {current_user.username}", category=LogCategory.SYSTEM)
        
        # إضافة مهمة إعادة التشغيل في الخلفية
        background_tasks.add_task(_restart_system_task)
        
        return APIResponse(
            success=True,
            message="تم طلب إعادة تشغيل النظام",
            data={"restart_initiated": True},
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'restart_system',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في إعادة تشغيل النظام")

async def _restart_system_task():
    """مهمة إعادة تشغيل النظام"""
    
    try:
        # انتظار قليل للسماح بإرسال الاستجابة
        await asyncio.sleep(2)
        
        logger.info("بدء إعادة تشغيل النظام...", category=LogCategory.SYSTEM)
        
        # تنظيف الموارد
        performance_monitor.cleanup()
        
        # إعادة تشغيل العملية
        os.execv(sys.executable, ['python'] + sys.argv)
        
    except Exception as e:
        logger.error(f"فشل في إعادة تشغيل النظام: {e}", category=LogCategory.SYSTEM)

@router.post("/shutdown", response_model=APIResponse)
async def shutdown_system(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """إيقاف النظام"""
    
    try:
        logger.warning(f"طلب إيقاف النظام من {current_user.username}", category=LogCategory.SYSTEM)
        
        # إضافة مهمة الإيقاف في الخلفية
        background_tasks.add_task(_shutdown_system_task)
        
        return APIResponse(
            success=True,
            message="تم طلب إيقاف النظام",
            data={"shutdown_initiated": True},
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'shutdown_system',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في إيقاف النظام")

async def _shutdown_system_task():
    """مهمة إيقاف النظام"""
    
    try:
        # انتظار قليل للسماح بإرسال الاستجابة
        await asyncio.sleep(2)
        
        logger.info("بدء إيقاف النظام...", category=LogCategory.SYSTEM)
        
        # تنظيف الموارد
        performance_monitor.cleanup()
        
        # إيقاف النظام
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"فشل في إيقاف النظام: {e}", category=LogCategory.SYSTEM)

@router.get("/logs", response_model=APIResponse)
async def get_system_logs(
    lines: int = 100,
    level: Optional[str] = None,
    current_user: User = Depends(require_permission(Permission.SYSTEM_READ))
):
    """الحصول على سجلات النظام"""
    
    try:
        logs_dir = BASE_DIR / "logs"
        log_file = logs_dir / "noogh_ai.log"
        
        if not log_file.exists():
            return APIResponse(
                success=True,
                message="لا توجد سجلات متاحة",
                data={"logs": []},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # قراءة آخر عدد من الأسطر
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        # تصفية حسب المستوى إذا تم تحديده
        if level:
            filtered_lines = [line for line in recent_lines if level.upper() in line]
            recent_lines = filtered_lines
        
        logger.info(f"تم جلب السجلات بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم جلب السجلات بنجاح",
            data={
                "logs": [line.strip() for line in recent_lines],
                "total_lines": len(recent_lines),
                "filter_level": level
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_system_logs',
            'user': current_user.username,
            'lines': lines,
            'level': level
        })
        raise HTTPException(status_code=500, detail="فشل في جلب السجلات")

@router.get("/stats", response_model=APIResponse)
async def get_system_stats(current_user: User = Depends(require_permission(Permission.SYSTEM_READ))):
    """الحصول على إحصائيات النظام"""
    
    try:
        # إحصائيات الأداء
        perf_stats = performance_monitor.get_performance_summary()
        
        # إحصائيات الأخطاء
        error_stats = error_handler.get_error_stats()
        
        # إحصائيات النظام
        system_stats = SystemStats(
            total_requests=perf_stats.get('total_requests', 0),
            successful_requests=perf_stats.get('successful_requests', 0),
            failed_requests=error_stats.get('total_errors', 0),
            average_response_time=perf_stats.get('average_response_time', 0),
            active_users=1,  # يمكن تحسينها لاحقاً
            system_load=psutil.cpu_percent(),
            memory_usage_percent=psutil.virtual_memory().percent,
            disk_usage_percent=(psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        )
        
        logger.info(f"تم جلب الإحصائيات بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم جلب الإحصائيات بنجاح",
            data={
                "system_stats": system_stats.dict(),
                "performance_stats": perf_stats,
                "error_stats": error_stats
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_system_stats',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب الإحصائيات")

@router.post("/backup", response_model=APIResponse)
async def create_backup(
    backup_request: BackupRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """إنشاء نسخة احتياطية"""
    
    try:
        backup_id = f"backup_{int(datetime.now().timestamp())}"
        
        logger.info(f"بدء إنشاء نسخة احتياطية بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        # إضافة مهمة النسخ الاحتياطي في الخلفية
        background_tasks.add_task(_create_backup_task, backup_request, backup_id)
        
        return APIResponse(
            success=True,
            message="تم بدء عملية النسخ الاحتياطي",
            data={
                "backup_id": backup_id,
                "backup_type": backup_request.backup_type,
                "status": "in_progress"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'create_backup',
            'user': current_user.username,
            'backup_request': backup_request.dict()
        })
        raise HTTPException(status_code=500, detail="فشل في بدء النسخ الاحتياطي")

async def _create_backup_task(backup_request: BackupRequest, backup_id: str):
    """مهمة إنشاء النسخة الاحتياطية"""
    
    try:
        import shutil
        import tarfile
        
        backup_dir = BASE_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"{backup_id}.tar.gz"
        
        with tarfile.open(backup_file, "w:gz") as tar:
            if backup_request.include_config:
                config_file = BASE_DIR / "config_default.yaml"
                if config_file.exists():
                    tar.add(config_file, arcname="config.yaml")
            
            if backup_request.include_data:
                data_dir = BASE_DIR / "data"
                if data_dir.exists():
                    tar.add(data_dir, arcname="data")
            
            if backup_request.include_models:
                models_dir = BASE_DIR / "models"
                if models_dir.exists():
                    tar.add(models_dir, arcname="models")
        
        logger.info(f"تم إنشاء النسخة الاحتياطية: {backup_file}", category=LogCategory.SYSTEM)
        
    except Exception as e:
        logger.error(f"فشل في إنشاء النسخة الاحتياطية: {e}", category=LogCategory.SYSTEM)

@router.post("/restore", response_model=APIResponse)
async def restore_backup(
    restore_request: RestoreRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """استعادة نسخة احتياطية"""
    
    try:
        backup_file = Path(restore_request.backup_file)
        
        if not backup_file.exists():
            raise HTTPException(status_code=404, detail="ملف النسخة الاحتياطية غير موجود")
        
        logger.warning(f"بدء استعادة النسخة الاحتياطية بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        # إضافة مهمة الاستعادة في الخلفية
        background_tasks.add_task(_restore_backup_task, restore_request)
        
        return APIResponse(
            success=True,
            message="تم بدء عملية الاستعادة",
            data={
                "backup_file": str(backup_file),
                "status": "in_progress"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'restore_backup',
            'user': current_user.username,
            'restore_request': restore_request.dict()
        })
        raise HTTPException(status_code=500, detail="فشل في بدء الاستعادة")

async def _restore_backup_task(restore_request: RestoreRequest):
    """مهمة استعادة النسخة الاحتياطية"""
    
    try:
        import tarfile
        
        backup_file = Path(restore_request.backup_file)
        
        with tarfile.open(backup_file, "r:gz") as tar:
            if restore_request.restore_config:
                try:
                    tar.extract("config.yaml", BASE_DIR)
                    logger.info("تم استعادة ملف التكوين", category=LogCategory.SYSTEM)
                except (OSError, RuntimeError) as e:
                    logger.error(f"System operation error: {e}")
                    pass
            
            if restore_request.restore_data:
                try:
                    tar.extractall(BASE_DIR, members=[m for m in tar.getmembers() if m.name.startswith("data/")])
                    logger.info("تم استعادة البيانات", category=LogCategory.SYSTEM)
                except (OSError, RuntimeError) as e:
                    logger.error(f"System operation error: {e}")
                    pass
            
            if restore_request.restore_models:
                try:
                    tar.extractall(BASE_DIR, members=[m for m in tar.getmembers() if m.name.startswith("models/")])
                    logger.info("تم استعادة النماذج", category=LogCategory.SYSTEM)
                except (OSError, RuntimeError) as e:
                    logger.error(f"System operation error: {e}")
                    pass
        
        logger.info("تم إنجاز عملية الاستعادة بنجاح", category=LogCategory.SYSTEM)
        
    except Exception as e:
        logger.error(f"فشل في استعادة النسخة الاحتياطية: {e}", category=LogCategory.SYSTEM)

@router.get("/processes", response_model=APIResponse)
async def get_system_processes(current_user: User = Depends(require_permission(Permission.SYSTEM_READ))):
    """الحصول على العمليات الجارية"""
    
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                proc_info = proc.info
                if proc_info['name'] and 'python' in proc_info['name'].lower():
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # ترتيب حسب استهلاك المعالج
        processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        
        logger.info(f"تم جلب العمليات بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم جلب العمليات بنجاح",
            data={
                "processes": processes[:20],  # أول 20 عملية
                "total_processes": len(processes)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_system_processes',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب العمليات")

@router.delete("/cache", response_model=APIResponse)
async def clear_system_cache(current_user: User = Depends(require_permission(Permission.SYSTEM_WRITE))):
    """مسح ذاكرة التخزين المؤقت"""
    
    try:
        import gc
        
        # تشغيل جامع القمامة
        collected = gc.collect()
        
        # مسح ملفات التخزين المؤقت
        cache_dirs = [
            BASE_DIR / "__pycache__",
            BASE_DIR / ".pytest_cache",
            BASE_DIR / "logs" / "temp"
        ]
        
        cleared_files = 0
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    cleared_files += 1
                except (OSError, RuntimeError) as e:
                    logger.error(f"System operation error: {e}")
                    pass
        
        logger.info(f"تم مسح التخزين المؤقت بواسطة {current_user.username}", category=LogCategory.SYSTEM)
        
        return APIResponse(
            success=True,
            message="تم مسح التخزين المؤقت بنجاح",
            data={
                "garbage_collected": collected,
                "cache_dirs_cleared": cleared_files
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'clear_system_cache',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في مسح التخزين المؤقت")
