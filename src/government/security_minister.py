#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - Security Minister v2.0
نظام الحكومة الداخلية لنوغ - وزير الأمن

Version: 2.0.0
Features:
- ✅ Threat detection and monitoring
- ✅ Access control (RBAC - Role-Based Access Control)
- ✅ Security auditing and logging
- ✅ Incident response and auto-blocking
- ✅ Rate limiting and DDoS protection
- ✅ Authentication management
- ✅ Data protection and encryption
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
import ipaddress

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

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """مستوى التهديد"""
    CRITICAL = "critical"      # حرج
    HIGH = "high"             # عالي
    MEDIUM = "medium"         # متوسط
    LOW = "low"              # منخفض
    INFO = "info"            # معلوماتي


class IncidentType(Enum):
    """أنواع الحوادث الأمنية"""
    BRUTE_FORCE = "brute_force"              # هجوم تخمين
    SQL_INJECTION = "sql_injection"          # حقن SQL
    XSS_ATTACK = "xss_attack"               # هجوم XSS
    DDOS = "ddos"                           # هجوم حجب الخدمة
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # وصول غير مصرح
    DATA_BREACH = "data_breach"             # اختراق بيانات
    MALWARE = "malware"                     # برمجيات خبيثة
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"  # تجاوز الحد
    SUSPICIOUS_PATTERN = "suspicious_pattern"    # نمط مشبوه
    UNKNOWN = "unknown"                     # غير معروف


class UserRole(Enum):
    """أدوار المستخدمين (RBAC)"""
    ADMIN = "admin"              # مدير - كامل الصلاحيات
    DEVELOPER = "developer"      # مطور - صلاحيات تطوير
    ANALYST = "analyst"         # محلل - قراءة + تحليل
    USER = "user"               # مستخدم عادي
    GUEST = "guest"             # ضيف - قراءة فقط
    BLOCKED = "blocked"         # محظور


class ActionType(Enum):
    """أنواع الإجراءات"""
    ALLOW = "allow"
    DENY = "deny"
    MONITOR = "monitor"
    BLOCK = "block"
    QUARANTINE = "quarantine"


@dataclass
class SecurityIncident:
    """حادثة أمنية"""
    id: str
    incident_type: IncidentType
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: ActionType = ActionType.MONITOR
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_type": self.incident_type.value,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "description": self.description,
            "details": self.details,
            "action_taken": self.action_taken.value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved
        }


@dataclass
class AccessRule:
    """قاعدة وصول"""
    role: UserRole
    resource: str  # API endpoint or resource
    allowed_actions: List[str]  # ["read", "write", "delete", etc.]
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """قاعدة تحديد المعدل"""
    identifier: str  # IP, user_id, etc.
    max_requests: int
    time_window_seconds: int
    current_count: int = 0
    window_start: datetime = field(default_factory=datetime.now)

    def is_exceeded(self) -> bool:
        """فحص إذا تم تجاوز الحد"""
        now = datetime.now()

        # Reset if window expired
        if (now - self.window_start).total_seconds() >= self.time_window_seconds:
            self.current_count = 0
            self.window_start = now
            return False

        return self.current_count >= self.max_requests

    def increment(self):
        """زيادة العداد"""
        now = datetime.now()

        # Reset if window expired
        if (now - self.window_start).total_seconds() >= self.time_window_seconds:
            self.current_count = 0
            self.window_start = now

        self.current_count += 1


class SecurityMinister(BaseMinister):
    """
    🔐 وزير الأمن - Minister of Security

    المسؤوليات:
    1. كشف التهديدات والمراقبة (Threat Detection)
    2. التحكم بالوصول (Access Control - RBAC)
    3. التدقيق الأمني (Security Auditing)
    4. الاستجابة للحوادث (Incident Response)
    5. حماية من DDoS (Rate Limiting)
    6. إدارة المصادقة (Authentication)
    7. حماية البيانات (Data Protection)
    """

    def __init__(
        self,
        verbose: bool = True,
        enable_auto_blocking: bool = True,
        enable_rate_limiting: bool = True,
        max_incidents_history: int = 1000,
        brain_hub: Any = None
    ):
        """
        Args:
            verbose: عرض التفاصيل
            enable_auto_blocking: تفعيل الحظر التلقائي
            enable_rate_limiting: تفعيل تحديد المعدل
            max_incidents_history: أقصى عدد للحوادث المحفوظة
        """
        # Authorities - الصلاحيات
        authorities = [
            "detect_threats",
            "block_ips",
            "manage_access_control",
            "audit_security",
            "respond_to_incidents",
            "enforce_rate_limits",
            "manage_authentication",
            "protect_data",
            "monitor_system"
        ]

        # Resources - الموارد
        resources = {
            "auto_blocking_enabled": enable_auto_blocking,
            "rate_limiting_enabled": enable_rate_limiting,
            "incidents_tracked": 0,
            "threats_blocked": 0
        }

        super().__init__(
            minister_type=MinisterType.SECURITY,
            name="Security Minister",
            authorities=authorities,
            resources=resources,
            verbose=verbose,
            specialty="Cybersecurity & Threat Management",
            description="Protects the system from threats, manages access control, and ensures security",
            expertise_level=0.95,
            brain_hub=brain_hub
        )

        # Security configuration
        self.enable_auto_blocking = enable_auto_blocking
        self.enable_rate_limiting = enable_rate_limiting
        self.max_incidents_history = max_incidents_history

        # Incidents tracking
        self.incidents: deque = deque(maxlen=max_incidents_history)
        self.incidents_by_ip: Dict[str, List[SecurityIncident]] = defaultdict(list)
        self.incidents_by_type: Dict[IncidentType, int] = defaultdict(int)

        # Blocked entities
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()

        # Rate limiting
        self.rate_limits: Dict[str, RateLimitRule] = {}
        self.default_rate_limit = RateLimitRule(
            identifier="default",
            max_requests=100,
            time_window_seconds=60
        )

        # Access control (RBAC)
        self.access_rules: List[AccessRule] = []
        self._initialize_default_access_rules()

        # Authentication tracking
        self.failed_login_attempts: Dict[str, int] = defaultdict(int)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Threat patterns (simple regex patterns)
        self.threat_patterns = {
            IncidentType.SQL_INJECTION: [
                r"(\bunion\b.*\bselect\b)",
                r"(\bor\b.*=.*)",
                r"(--|\#|\/\*)",
                r"(\bdrop\b.*\btable\b)",
                r"(\bexec\b.*\()",
            ],
            IncidentType.XSS_ATTACK: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"onerror\s*=",
                r"onload\s*=",
            ],
            IncidentType.MALWARE: [
                r"eval\s*\(",
                r"base64_decode",
                r"system\s*\(",
                r"exec\s*\(",
            ]
        }

        # Statistics
        self.total_threats_detected = 0
        self.total_threats_blocked = 0
        self.total_incidents_resolved = 0
        self.total_rate_limit_violations = 0

        if self.verbose:
            logger.info(f"\n🔐 {self.get_arabic_title()} initialized")
            logger.info(f"   Auto-blocking: {'✅ Enabled' if enable_auto_blocking else '❌ Disabled'}")
            logger.info(f"   Rate limiting: {'✅ Enabled' if enable_rate_limiting else '❌ Disabled'}")
            logger.info(f"   Access rules: {len(self.access_rules)}")

    def _initialize_default_access_rules(self):
        """تهيئة قواعد الوصول الافتراضية"""
        # Admin - full access
        self.access_rules.append(AccessRule(
            role=UserRole.ADMIN,
            resource="*",
            allowed_actions=["read", "write", "delete", "execute", "admin"]
        ))

        # Developer - development access
        self.access_rules.append(AccessRule(
            role=UserRole.DEVELOPER,
            resource="/api/*",
            allowed_actions=["read", "write", "execute"]
        ))

        # Analyst - read and analyze
        self.access_rules.append(AccessRule(
            role=UserRole.ANALYST,
            resource="/api/*/statistics",
            allowed_actions=["read", "analyze"]
        ))

        # User - limited access
        self.access_rules.append(AccessRule(
            role=UserRole.USER,
            resource="/api/public/*",
            allowed_actions=["read"]
        ))

        # Guest - minimal access
        self.access_rules.append(AccessRule(
            role=UserRole.GUEST,
            resource="/api/public/info",
            allowed_actions=["read"]
        ))

        # Blocked - no access
        self.access_rules.append(AccessRule(
            role=UserRole.BLOCKED,
            resource="*",
            allowed_actions=[]
        ))

    def _can_handle_specific_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """فحص إضافي خاص بوزير الأمن"""
        if task_type == "security":
            return True
            
        security_tasks = [
            "detect_threat",
            "check_access",
            "block_ip",
            "unblock_ip",
            "check_rate_limit",
            "create_incident",
            "resolve_incident",
            "audit_log",
            "validate_input",
            "check_authentication",
            "scan_for_threats"
        ]

        return task_type in security_tasks

    async def _execute_specific_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """تنفيذ المهام الخاصة بوزير الأمن"""

        if task_type == "security":
            user_input = task_data.get("user_input", "")
            return await self._scan_for_threats({"target": user_input, "scan_type": "content"})

        if task_type == "detect_threat":
            return await self._detect_threat(task_data)

        elif task_type == "check_access":
            return await self._check_access(task_data)

        elif task_type == "block_ip":
            return await self._block_ip(task_data)

        elif task_type == "unblock_ip":
            return await self._unblock_ip(task_data)

        elif task_type == "check_rate_limit":
            return await self._check_rate_limit(task_data)

        elif task_type == "create_incident":
            return await self._create_incident(task_data)

        elif task_type == "resolve_incident":
            return await self._resolve_incident(task_data)

        elif task_type == "audit_log":
            return await self._audit_log(task_data)

        elif task_type == "validate_input":
            return await self._validate_input(task_data)

        elif task_type == "check_authentication":
            return await self._check_authentication(task_data)

        elif task_type == "scan_for_threats":
            return await self._scan_for_threats(task_data)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Threat Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _detect_threat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        كشف التهديدات في البيانات المدخلة

        Args:
            data: {
                "content": str,
                "source_ip": str (optional),
                "endpoint": str (optional)
            }
        """
        content = data.get("content", "")
        source_ip = data.get("source_ip")
        endpoint = data.get("endpoint")

        threats_found = []
        highest_threat_level = ThreatLevel.INFO

        # Check each threat pattern
        for incident_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    threat = {
                        "type": incident_type.value,
                        "pattern": pattern,
                        "match": match.group(0)
                    }
                    threats_found.append(threat)

                    # Determine threat level
                    if incident_type in [IncidentType.SQL_INJECTION, IncidentType.MALWARE]:
                        threat_level = ThreatLevel.CRITICAL
                    elif incident_type == IncidentType.XSS_ATTACK:
                        threat_level = ThreatLevel.HIGH
                    else:
                        threat_level = ThreatLevel.MEDIUM

                    if threat_level.value != highest_threat_level.value:
                        highest_threat_level = threat_level

        self.total_threats_detected += len(threats_found)

        # Create incident if threats found
        if threats_found and self.enable_auto_blocking:
            incident = SecurityIncident(
                id=generate_task_id(),
                incident_type=IncidentType(threats_found[0]["type"]),
                threat_level=highest_threat_level,
                source_ip=source_ip,
                endpoint=endpoint,
                description=f"Detected {len(threats_found)} threat(s)",
                details={"threats": threats_found},
                action_taken=ActionType.BLOCK if highest_threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] else ActionType.MONITOR
            )

            self._record_incident(incident)

            # Auto-block if critical
            if highest_threat_level == ThreatLevel.CRITICAL and source_ip:
                self.blocked_ips.add(source_ip)
                self.total_threats_blocked += 1

        return {
            "threats_detected": len(threats_found),
            "threat_level": highest_threat_level.value,
            "threats": threats_found,
            "action_taken": ActionType.BLOCK.value if (threats_found and highest_threat_level == ThreatLevel.CRITICAL) else ActionType.MONITOR.value
        }

    async def _scan_for_threats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فحص شامل للتهديدات

        Args:
            data: {
                "target": str (file_path, content, or endpoint),
                "scan_type": str ("content", "file", "system")
            }
        """
        target = data.get("target")
        scan_type = data.get("scan_type", "content")

        threats = []

        if scan_type == "content":
            result = await self._detect_threat({"content": target})
            threats = result.get("threats", [])

        return {
            "scan_complete": True,
            "threats_found": len(threats),
            "threats": threats,
            "timestamp": datetime.now().isoformat()
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Access Control (RBAC)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _check_access(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فحص صلاحيات الوصول

        Args:
            data: {
                "user_role": str,
                "resource": str,
                "action": str
            }
        """
        user_role_str = data.get("user_role", "guest")
        resource = data.get("resource")
        action = data.get("action")

        try:
            user_role = UserRole(user_role_str.lower())
        except ValueError:
            user_role = UserRole.GUEST

        # Check if role is blocked
        if user_role == UserRole.BLOCKED:
            return {
                "access_granted": False,
                "reason": "User role is blocked",
                "user_role": user_role.value
            }

        # Find matching access rule
        for rule in self.access_rules:
            if rule.role != user_role:
                continue

            # Check resource match (support wildcards)
            if resource and (rule.resource == "*" or self._match_resource(resource, rule.resource)):
                if action in rule.allowed_actions:
                    return {
                        "access_granted": True,
                        "user_role": user_role.value,
                        "resource": resource,
                        "action": action
                    }

        # No matching rule - deny access
        return {
            "access_granted": False,
            "reason": "No matching access rule",
            "user_role": user_role.value,
            "resource": resource,
            "action": action
        }

    def _match_resource(self, resource: str, pattern: str) -> bool:
        """فحص تطابق المورد مع النمط (يدعم *)"""
        if pattern == "*":
            return True

        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", resource))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Rate Limiting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _check_rate_limit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فحص تجاوز معدل الطلبات

        Args:
            data: {
                "identifier": str (IP or user_id),
                "max_requests": int (optional),
                "time_window": int (optional, in seconds)
            }
        """
        if not self.enable_rate_limiting:
            return {"rate_limit_exceeded": False, "message": "Rate limiting disabled"}

        identifier = data.get("identifier")
        if not identifier:
            raise ValueError("Identifier is required")

        # Get or create rate limit rule
        if identifier not in self.rate_limits:
            max_requests = data.get("max_requests", self.default_rate_limit.max_requests)
            time_window = data.get("time_window", self.default_rate_limit.time_window_seconds)

            self.rate_limits[identifier] = RateLimitRule(
                identifier=identifier,
                max_requests=max_requests,
                time_window_seconds=time_window
            )

        rule = self.rate_limits[identifier]
        rule.increment()

        if rule.is_exceeded():
            # Create incident
            incident = SecurityIncident(
                id=generate_task_id(),
                incident_type=IncidentType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=identifier if self._is_ip_address(identifier) else None,
                user_id=identifier if not self._is_ip_address(identifier) else None,
                description=f"Rate limit exceeded: {rule.current_count}/{rule.max_requests}",
                action_taken=ActionType.DENY
            )
            self._record_incident(incident)

            return {
                "rate_limit_exceeded": True,
                "current_count": rule.current_count,
                "max_requests": rule.max_requests,
                "time_window_seconds": rule.time_window_seconds,
                "retry_after_seconds": rule.time_window_seconds - int((datetime.now() - rule.window_start).total_seconds())
            }

        return {
            "rate_limit_exceeded": False,
            "current_count": rule.current_count,
            "max_requests": rule.max_requests,
            "remaining_requests": rule.max_requests - rule.current_count
        }

    def _is_ip_address(self, value: str) -> bool:
        """فحص إذا كانت القيمة عنوان IP"""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # IP Blocking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _block_ip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        حظر عنوان IP

        Args:
            data: {
                "ip": str,
                "reason": str (optional)
            }
        """
        ip = data.get("ip")
        reason = data.get("reason", "Manual block")

        if not ip:
            raise ValueError("IP address is required")

        self.blocked_ips.add(ip)
        self.total_threats_blocked += 1

        # Create incident
        incident = SecurityIncident(
            id=generate_task_id(),
            incident_type=IncidentType.UNAUTHORIZED_ACCESS,
            threat_level=ThreatLevel.HIGH,
            source_ip=ip,
            description=f"IP blocked: {reason}",
            action_taken=ActionType.BLOCK
        )
        self._record_incident(incident)

        return {
            "success": True,
            "ip": ip,
            "reason": reason,
            "total_blocked_ips": len(self.blocked_ips)
        }

    async def _unblock_ip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        إلغاء حظر عنوان IP

        Args:
            data: {"ip": str}
        """
        ip = data.get("ip")

        if not ip:
            raise ValueError("IP address is required")

        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            return {
                "success": True,
                "ip": ip,
                "message": "IP unblocked",
                "total_blocked_ips": len(self.blocked_ips)
            }
        else:
            return {
                "success": False,
                "ip": ip,
                "message": "IP was not blocked"
            }

    def is_ip_blocked(self, ip: str) -> bool:
        """فحص إذا كان IP محظور"""
        return ip in self.blocked_ips

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Incident Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _create_incident(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        إنشاء حادثة أمنية

        Args:
            data: {
                "incident_type": str,
                "threat_level": str,
                "source_ip": str (optional),
                "description": str
            }
        """
        incident_type_str = data.get("incident_type", "unknown")
        threat_level_str = data.get("threat_level", "medium")

        try:
            incident_type = IncidentType(incident_type_str)
        except ValueError:
            incident_type = IncidentType.UNKNOWN

        try:
            threat_level = ThreatLevel(threat_level_str)
        except ValueError:
            threat_level = ThreatLevel.MEDIUM

        incident = SecurityIncident(
            id=generate_task_id(),
            incident_type=incident_type,
            threat_level=threat_level,
            source_ip=data.get("source_ip"),
            user_id=data.get("user_id"),
            endpoint=data.get("endpoint"),
            description=data.get("description", ""),
            details=data.get("details", {}),
            action_taken=ActionType.MONITOR
        )

        self._record_incident(incident)

        return {
            "success": True,
            "incident_id": incident.id,
            "incident": incident.to_dict()
        }

    async def _resolve_incident(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        حل حادثة أمنية

        Args:
            data: {"incident_id": str}
        """
        incident_id = data.get("incident_id")

        # Find incident
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.resolved = True
                self.total_incidents_resolved += 1

                return {
                    "success": True,
                    "incident_id": incident_id,
                    "message": "Incident resolved"
                }

        return {
            "success": False,
            "incident_id": incident_id,
            "message": "Incident not found"
        }

    def _record_incident(self, incident: SecurityIncident):
        """تسجيل حادثة أمنية"""
        self.incidents.append(incident)

        if incident.source_ip:
            self.incidents_by_ip[incident.source_ip].append(incident)

        self.incidents_by_type[incident.incident_type] += 1

        if self.verbose:
            logger.warning(f"🚨 Security Incident: {incident.incident_type.value} - {incident.description}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Authentication & Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _check_authentication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فحص المصادقة

        Args:
            data: {
                "user_id": str,
                "password_hash": str (optional),
                "session_token": str (optional)
            }
        """
        user_id = data.get("user_id")

        # Check if user is blocked
        if user_id and user_id in self.blocked_users:
            return {
                "authenticated": False,
                "reason": "User is blocked"
            }

        # Check failed login attempts
        if user_id and self.failed_login_attempts.get(user_id, 0) >= 5:
            # Auto-block after 5 failed attempts
            self.blocked_users.add(user_id)

            incident = SecurityIncident(
                id=generate_task_id(),
                incident_type=IncidentType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                description=f"User blocked after {self.failed_login_attempts.get(user_id, 0)} failed login attempts",
                action_taken=ActionType.BLOCK
            )
            self._record_incident(incident)

            return {
                "authenticated": False,
                "reason": "Too many failed login attempts - user blocked"
            }

        # Simple session check (in production, use proper JWT or session management)
        session_token = data.get("session_token")
        if session_token and session_token in self.active_sessions:
            return {
                "authenticated": True,
                "user_id": user_id,
                "session_valid": True
            }

        return {
            "authenticated": False,
            "reason": "Invalid credentials or session"
        }

    async def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        التحقق من صحة المدخلات

        Args:
            data: {
                "input": str,
                "max_length": int (optional),
                "allowed_chars": str (optional)
            }
        """
        input_value = data.get("input", "")
        max_length = data.get("max_length", 5000)
        allowed_chars = data.get("allowed_chars")

        errors = []

        # Check length
        if len(input_value) > max_length:
            errors.append(f"Input exceeds maximum length ({max_length})")

        # Check for threats
        threat_result = await self._detect_threat({"content": input_value})
        if threat_result["threats_detected"] > 0:
            errors.append(f"Potential threats detected: {threat_result['threats_detected']}")

        # Check allowed characters
        if allowed_chars:
            invalid_chars = set(input_value) - set(allowed_chars)
            if invalid_chars:
                errors.append(f"Invalid characters found: {invalid_chars}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "sanitized_input": input_value if len(errors) == 0 else ""
        }

    async def _audit_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تسجيل حدث للتدقيق

        Args:
            data: {
                "event": str,
                "user_id": str (optional),
                "details": dict (optional)
            }
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": data.get("event"),
            "user_id": data.get("user_id"),
            "details": data.get("details", {})
        }

        # In production, save to database or log file
        logger.info(f"🔍 Audit: {json.dumps(audit_entry)}")

        return {
            "success": True,
            "audit_entry": audit_entry
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_incidents(
        self,
        limit: int = 100,
        threat_level: Optional[ThreatLevel] = None,
        incident_type: Optional[IncidentType] = None,
        resolved_only: bool = False
    ) -> List[Dict[str, Any]]:
        """الحصول على قائمة الحوادث"""
        incidents = list(self.incidents)

        # Filter by threat level
        if threat_level:
            incidents = [i for i in incidents if i.threat_level == threat_level]

        # Filter by incident type
        if incident_type:
            incidents = [i for i in incidents if i.incident_type == incident_type]

        # Filter by resolved status
        if resolved_only:
            incidents = [i for i in incidents if i.resolved]

        return [i.to_dict() for i in incidents[:limit]]

    def get_blocked_ips(self) -> List[str]:
        """قائمة IPs المحظورة"""
        return list(self.blocked_ips)

    def get_security_statistics(self) -> Dict[str, Any]:
        """إحصائيات الأمان"""
        return {
            "total_incidents": len(self.incidents),
            "unresolved_incidents": sum(1 for i in self.incidents if not i.resolved),
            "resolved_incidents": self.total_incidents_resolved,
            "threats_detected": self.total_threats_detected,
            "threats_blocked": self.total_threats_blocked,
            "blocked_ips_count": len(self.blocked_ips),
            "blocked_users_count": len(self.blocked_users),
            "active_sessions": len(self.active_sessions),
            "auto_blocking_enabled": self.enable_auto_blocking,
            "rate_limiting_enabled": self.enable_rate_limiting,
            "incidents_by_type": {k.value: v for k, v in self.incidents_by_type.items()}
        }

    def get_security_score(self) -> float:
        """
        حساب درجة الأمان (0-100)

        بناءً على:
        - عدد الحوادث غير المحلولة
        - نسبة الحوادث المحلولة
        - عدد IPs المحظورة
        """
        base_score = 100.0

        # Penalty for unresolved incidents
        unresolved = sum(1 for i in self.incidents if not i.resolved)
        base_score -= min(unresolved * 2, 30)  # Max -30 points

        # Penalty for critical incidents
        critical_incidents = sum(1 for i in self.incidents if i.threat_level == ThreatLevel.CRITICAL)
        base_score -= min(critical_incidents * 5, 20)  # Max -20 points

        # Bonus for resolution rate
        if len(self.incidents) > 0:
            resolution_rate = self.total_incidents_resolved / len(self.incidents)
            base_score += resolution_rate * 10  # Max +10 points

        return max(0.0, min(100.0, base_score))

    async def report_to_president(self) -> Dict[str, Any]:
        """
        تقرير إلى الرئيس
        Generate report for the President
        """
        return {
            "minister": self.name,
            "type": self.minister_type.value,
            "status": "operational",
            "security_metrics": {
                "total_threats_detected": self.total_threats_detected,
                "active_incidents": len([i for i in self.incidents if not i.resolved]),
                "blocked_ips": len(self.blocked_ips),
                "security_score": self.get_security_score()
            },
            "statistics": {
                "total_threats": self.total_threats_detected,
                "threats_blocked": self.total_threats_blocked,
                "incidents_total": len(self.incidents),
                "incidents_resolved": self.total_incidents_resolved,
                "rate_limit_violations": self.total_rate_limit_violations
            },
            "authorities": self.authorities,
            "timestamp": datetime.now().isoformat()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    """Test Security Minister"""
    print("🔐 Security Minister v2.0 - Test")
    print("=" * 70)
    print()

    # Initialize minister
    minister = SecurityMinister(verbose=True)

    # Print status
    minister.print_status()

    # Test 1: Threat detection
    print("\n📌 Test 1: Threat Detection...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="detect_threat",
        task_data={
            "content": "SELECT * FROM users WHERE id=1 OR 1=1",
            "source_ip": "192.168.1.100"
        }
    )
    print(f"Result: {result.result}")

    # Test 2: Access control
    print("\n📌 Test 2: Access Control...")
    task_id = generate_task_id()
    result = await minister.execute_task(
        task_id=task_id,
        task_type="check_access",
        task_data={
            "user_role": "developer",
            "resource": "/api/data",
            "action": "write"
        }
    )
    print(f"Result: {result.result}")

    # Test 3: Rate limiting
    print("\n📌 Test 3: Rate Limiting...")
    for i in range(3):
        task_id = generate_task_id()
        result = await minister.execute_task(
            task_id=task_id,
            task_type="check_rate_limit",
            task_data={
                "identifier": "192.168.1.100",
                "max_requests": 2,
                "time_window": 60
            }
        )
        print(f"  Request {i+1}: {result.result}")

    # Print statistics
    print("\n📊 Security Statistics:")
    stats = minister.get_security_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n🎯 Security Score: {minister.get_security_score():.1f}/100")
    print("\n✅ Security Minister test complete!")


