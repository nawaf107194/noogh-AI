#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Minister Dashboard API Routes
======================================

REST API endpoints for security management, threat detection, and access control

Features:
- ✅ Threat detection and scanning
- ✅ Access control (RBAC)
- ✅ IP blocking management
- ✅ Rate limiting
- ✅ Security incident tracking
- ✅ Security statistics and monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Security Minister
try:
    from src.government.security_minister import (
        SecurityMinister,
        ThreatLevel,
        IncidentType,
        UserRole,
        generate_task_id
    )
    SECURITY_MINISTER_AVAILABLE = True
except ImportError as e:
    SECURITY_MINISTER_AVAILABLE = False
    logging.warning(f"Security Minister not available: {e}")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/security", tags=["security"])

# Global Security Minister instance
security_minister: Optional[SecurityMinister] = None


def get_security_minister() -> SecurityMinister:
    """Get or create Security Minister instance"""
    global security_minister
    if security_minister is None:
        if not SECURITY_MINISTER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Security Minister not available"
            )
        security_minister = SecurityMinister(
            verbose=True,
            enable_auto_blocking=True,
            enable_rate_limiting=True
        )
    return security_minister


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request/Response Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreatDetectionRequest(BaseModel):
    """Request to detect threats"""
    content: str = Field(..., description="Content to scan for threats")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    endpoint: Optional[str] = Field(None, description="API endpoint being accessed")


class AccessCheckRequest(BaseModel):
    """Request to check access permissions"""
    user_role: str = Field(..., description="User role: admin, developer, analyst, user, guest")
    resource: str = Field(..., description="Resource path (e.g., /api/data)")
    action: str = Field(..., description="Action: read, write, delete, execute, admin")


class RateLimitCheckRequest(BaseModel):
    """Request to check rate limit"""
    identifier: str = Field(..., description="IP address or user ID")
    max_requests: Optional[int] = Field(None, description="Maximum requests allowed")
    time_window: Optional[int] = Field(None, description="Time window in seconds")


class BlockIPRequest(BaseModel):
    """Request to block an IP"""
    ip: str = Field(..., description="IP address to block")
    reason: Optional[str] = Field(None, description="Reason for blocking")


class CreateIncidentRequest(BaseModel):
    """Request to create a security incident"""
    incident_type: str = Field(..., description="Type: sql_injection, xss_attack, brute_force, etc.")
    threat_level: str = Field(..., description="Level: critical, high, medium, low, info")
    source_ip: Optional[str] = Field(None, description="Source IP")
    user_id: Optional[str] = Field(None, description="User ID")
    endpoint: Optional[str] = Field(None, description="Endpoint")
    description: str = Field(..., description="Incident description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ValidateInputRequest(BaseModel):
    """Request to validate input"""
    input: str = Field(..., description="Input to validate")
    max_length: Optional[int] = Field(5000, description="Maximum length")
    allowed_chars: Optional[str] = Field(None, description="Allowed characters")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Threat Detection Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/threats/detect")
async def detect_threats(request: ThreatDetectionRequest):
    """
    كشف التهديدات في المحتوى
    Detect threats in content (SQL injection, XSS, malware patterns)

    Example:
    ```json
    {
        "content": "SELECT * FROM users WHERE id=1 OR 1=1",
        "source_ip": "192.168.1.100"
    }
    ```

    Returns threat level and details of detected threats.
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="detect_threat",
            task_data={
                "content": request.content,
                "source_ip": request.source_ip,
                "endpoint": request.endpoint
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "result": report.result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Threat detection failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error detecting threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threats/scan")
async def scan_for_threats(target: str, scan_type: str = "content"):
    """
    فحص شامل للتهديدات
    Comprehensive threat scanning

    Args:
        target: Content or file to scan
        scan_type: "content", "file", or "system"
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="scan_for_threats",
            task_data={
                "target": target,
                "scan_type": scan_type
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "scan_result": report.result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Scan failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error scanning for threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Access Control Endpoints (RBAC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/access/check")
async def check_access(request: AccessCheckRequest):
    """
    فحص صلاحيات الوصول
    Check access permissions (RBAC)

    Roles:
    - admin: Full access to everything
    - developer: Development access (/api/*)
    - analyst: Read access to statistics
    - user: Limited public access
    - guest: Minimal access
    - blocked: No access

    Example:
    ```json
    {
        "user_role": "developer",
        "resource": "/api/data",
        "action": "write"
    }
    ```
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="check_access",
            task_data={
                "user_role": request.user_role,
                "resource": request.resource,
                "action": request.action
            }
        )

        if report.status.value == "completed":
            result = report.result
            return {
                "success": True,
                "access_granted": result.get("access_granted", False),
                "user_role": result.get("user_role"),
                "resource": result.get("resource"),
                "action": result.get("action"),
                "reason": result.get("reason"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Access check failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error checking access: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rate Limiting Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/ratelimit/check")
async def check_rate_limit(request: RateLimitCheckRequest):
    """
    فحص تجاوز معدل الطلبات
    Check rate limit (DDoS protection)

    Default: 100 requests per 60 seconds

    Example:
    ```json
    {
        "identifier": "192.168.1.100",
        "max_requests": 10,
        "time_window": 60
    }
    ```

    Returns whether limit is exceeded and remaining requests.
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="check_rate_limit",
            task_data={
                "identifier": request.identifier,
                "max_requests": request.max_requests,
                "time_window": request.time_window
            }
        )

        if report.status.value == "completed":
            result = report.result
            return {
                "success": True,
                "rate_limit_exceeded": result.get("rate_limit_exceeded", False),
                "current_count": result.get("current_count"),
                "max_requests": result.get("max_requests"),
                "remaining_requests": result.get("remaining_requests"),
                "retry_after_seconds": result.get("retry_after_seconds"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Rate limit check failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IP Blocking Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/ips/block")
async def block_ip(request: BlockIPRequest):
    """
    حظر عنوان IP
    Block an IP address

    Example:
    ```json
    {
        "ip": "192.168.1.100",
        "reason": "Suspicious activity detected"
    }
    ```
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="block_ip",
            task_data={
                "ip": request.ip,
                "reason": request.reason
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "result": report.result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"IP blocking failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error blocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ips/unblock/{ip}")
async def unblock_ip(ip: str):
    """
    إلغاء حظر عنوان IP
    Unblock an IP address
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="unblock_ip",
            task_data={"ip": ip}
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "result": report.result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"IP unblocking failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error unblocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ips/blocked")
async def list_blocked_ips():
    """
    قائمة عناوين IP المحظورة
    List all blocked IP addresses
    """
    try:
        minister = get_security_minister()
        blocked_ips = minister.get_blocked_ips()

        return {
            "success": True,
            "total": len(blocked_ips),
            "blocked_ips": blocked_ips,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing blocked IPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ips/check/{ip}")
async def check_if_ip_blocked(ip: str):
    """
    فحص إذا كان IP محظور
    Check if an IP is blocked
    """
    try:
        minister = get_security_minister()
        is_blocked = minister.is_ip_blocked(ip)

        return {
            "success": True,
            "ip": ip,
            "is_blocked": is_blocked,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Security Incidents Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/incidents/create")
async def create_incident(request: CreateIncidentRequest):
    """
    إنشاء حادثة أمنية
    Create a security incident

    Example:
    ```json
    {
        "incident_type": "sql_injection",
        "threat_level": "critical",
        "source_ip": "192.168.1.100",
        "description": "SQL injection attempt detected"
    }
    ```
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="create_incident",
            task_data={
                "incident_type": request.incident_type,
                "threat_level": request.threat_level,
                "source_ip": request.source_ip,
                "user_id": request.user_id,
                "endpoint": request.endpoint,
                "description": request.description,
                "details": request.details
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "incident": report.result.get("incident"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Incident creation failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str):
    """
    حل حادثة أمنية
    Resolve a security incident
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="resolve_incident",
            task_data={"incident_id": incident_id}
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "message": report.result.get("message"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404 if "not found" in report.result.get("message", "").lower() else 500,
                detail=report.result.get("message", "Unknown error")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incidents/list")
async def list_incidents(
    limit: int = 100,
    threat_level: Optional[str] = None,
    incident_type: Optional[str] = None,
    resolved_only: bool = False
):
    """
    قائمة الحوادث الأمنية
    List security incidents

    Query parameters:
    - limit: Maximum number of incidents (default: 100)
    - threat_level: Filter by level (critical, high, medium, low, info)
    - incident_type: Filter by type (sql_injection, xss_attack, etc.)
    - resolved_only: Show only resolved incidents
    """
    try:
        minister = get_security_minister()

        # Convert string enums
        threat_level_enum = None
        if threat_level:
            try:
                threat_level_enum = ThreatLevel(threat_level.lower())
            except ValueError:
                pass

        incident_type_enum = None
        if incident_type:
            try:
                incident_type_enum = IncidentType(incident_type.lower())
            except ValueError:
                pass

        incidents = minister.get_incidents(
            limit=limit,
            threat_level=threat_level_enum,
            incident_type=incident_type_enum,
            resolved_only=resolved_only
        )

        return {
            "success": True,
            "total": len(incidents),
            "incidents": incidents,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input Validation Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/validate/input")
async def validate_input(request: ValidateInputRequest):
    """
    التحقق من صحة المدخلات
    Validate user input (detect threats, check length, sanitize)

    Example:
    ```json
    {
        "input": "Hello, world!",
        "max_length": 1000
    }
    ```

    Returns validation result and sanitized input if valid.
    """
    try:
        minister = get_security_minister()

        task_id = generate_task_id()
        report = await minister.execute_task(
            task_id=task_id,
            task_type="validate_input",
            task_data={
                "input": request.input,
                "max_length": request.max_length,
                "allowed_chars": request.allowed_chars
            }
        )

        if report.status.value == "completed":
            return {
                "success": True,
                "validation_result": report.result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {report.result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Error validating input: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistics & Monitoring Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/statistics")
async def get_security_statistics():
    """
    إحصائيات الأمان الشاملة
    Get comprehensive security statistics

    Returns:
    - Total incidents
    - Threats detected and blocked
    - Blocked IPs count
    - Incidents by type
    - Security score
    """
    try:
        minister = get_security_minister()
        stats = minister.get_security_statistics()
        security_score = minister.get_security_score()

        return {
            "success": True,
            "statistics": stats,
            "security_score": security_score,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_security_minister_status():
    """
    حالة وزير الأمن
    Get Security Minister status report
    """
    try:
        minister = get_security_minister()
        status = minister.get_status_report()

        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score")
async def get_security_score():
    """
    درجة الأمان (0-100)
    Get security score (0-100)

    Based on:
    - Unresolved incidents
    - Critical incidents count
    - Resolution rate
    """
    try:
        minister = get_security_minister()
        score = minister.get_security_score()

        # Determine grade
        if score >= 90:
            grade = "A"
            status = "Excellent"
        elif score >= 80:
            grade = "B"
            status = "Good"
        elif score >= 70:
            grade = "C"
            status = "Fair"
        elif score >= 60:
            grade = "D"
            status = "Poor"
        else:
            grade = "F"
            status = "Critical"

        return {
            "success": True,
            "security_score": score,
            "grade": grade,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting security score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Middleware Helper (for integration with FastAPI app)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def security_middleware(request: Request):
    """
    Security middleware helper
    Can be used to automatically check threats and rate limits
    """
    minister = get_security_minister()

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Check if IP is blocked
    if minister.is_ip_blocked(client_ip):
        raise HTTPException(status_code=403, detail="IP address is blocked")

    # Check rate limit
    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="check_rate_limit",
        task_data={"identifier": client_ip}
    )

    if report.result.get("rate_limit_exceeded"):
        retry_after = report.result.get("retry_after_seconds", 60)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)}
        )

    return True


if __name__ == "__main__":
    print("🔐 Security API Routes initialized")
    print(f"   Available: {SECURITY_MINISTER_AVAILABLE}")
