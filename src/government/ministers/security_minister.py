#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Minister - Guardian & Autonomous Vulnerability Scanner
================================================================

Scans logs for threats, detects attack patterns, and uses AI to generate
patches and mitigation strategies.
"""

from typing import Optional, Dict, Any, List
import logging
import re
from datetime import datetime

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class SecurityMinister(BaseMinister):
    """
    Minister of Security - Autonomous threat detection and mitigation.
    
    Capabilities:
    - Log scanning for attack patterns
    - SQL injection detection
    - XSS pattern recognition
    - AI-powered patch generation
    - Automated threat response
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Security Minister."""
        super().__init__(
            name="Security Minister (Guardian)",
            description="Autonomous security scanner and threat mitigator. Protects the system.",
            brain=brain
        )
        
        self.system_prompt = """You are an elite cybersecurity expert and threat analyst.
When analyzing security threats:
1. Identify the attack vector precisely
2. Assess the severity (Critical/High/Medium/Low)
3. Provide specific mitigation steps
4. Generate Python code to patch vulnerabilities
5. Recommend monitoring/prevention strategies

Be thorough, precise, and security-focused."""
        
        # Threat patterns (regex)
        self.threat_patterns = {
            "sql_injection": [
                r"(\bUNION\b.*\bSELECT\b|\bSELECT\b.*\bFROM\b.*\bWHERE\b.*=.*)",
                r"(\bDROP\b.*\bTABLE\b|\bDELETE\b.*\bFROM\b)",
                r"(--|\#|\/\*|\*\/|;).*",  # SQL comments
                r"(\bOR\b.*=.*\bOR\b|\'\s*OR\s*\'1\'\s*=\s*\'1)",
            ],
            "xss": [
                r"<script[\s\S]*?>[\s\S]*?</script>",
                r"javascript:",
                r"onerror\s*=",
                r"onload\s*=",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
            ],
            "command_injection": [
                r";\s*(ls|cat|wget|curl|rm|chmod|shutdown)",
                r"\|\s*(ls|cat|wget|curl|rm|chmod)",
            ]
        }
    
    async def scan_logs(
        self,
        logs: List[str],
        scan_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Scan logs for security threats.
        
        Args:
            logs: List of log entries to scan
            scan_type: Type of scan (all, sql_injection, xss, etc.)
        
        Returns:
            Scan results with detected threats
        """
        threats_found = []
        
        # Determine which patterns to check
        patterns_to_check = {}
        if scan_type == "all":
            patterns_to_check = self.threat_patterns
        elif scan_type in self.threat_patterns:
            patterns_to_check = {scan_type: self.threat_patterns[scan_type]}
        
        # Scan each log entry
        for i, log_entry in enumerate(logs):
            for threat_type, patterns in patterns_to_check.items():
                for pattern in patterns:
                    if re.search(pattern, log_entry, re.IGNORECASE):
                        threats_found.append({
                            "log_index": i,
                            "threat_type": threat_type,
                            "pattern": pattern,
                            "log_entry": log_entry,
                            "severity": self._assess_severity(threat_type)
                        })
                        break  # One threat per log entry
        
        return {
            "success": True,
            "threats_found": threats_found,
            "total_logs_scanned": len(logs),
            "threat_count": len(threats_found),
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_severity(self, threat_type: str) -> str:
        """Assess threat severity."""
        severity_map = {
            "sql_injection": "CRITICAL",
            "command_injection": "CRITICAL",
            "xss": "HIGH",
            "path_traversal": "HIGH"
        }
        return severity_map.get(threat_type, "MEDIUM")
    
    async def mitigate_threat(
        self,
        threat: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use AI to generate mitigation strategy and patch.
        
        Args:
            threat: Threat details from scan
        
        Returns:
            Mitigation strategy and patch code
        """
        prompt = f"""Security Threat Detected:

Threat Type: {threat['threat_type']}
Severity: {threat['severity']}
Attack Pattern: {threat['pattern']}
Log Entry: {threat['log_entry']}

As a cybersecurity expert:
1. Explain this attack vector
2. Assess the potential impact
3. Write Python code to validate and sanitize inputs to prevent this attack
4. Provide WAF (Web Application Firewall) rules if applicable
5. Recommend monitoring alerts to detect future attempts

Be specific and provide production-ready code."""
        
        mitigation = await self._think_with_prompt(
            system_prompt=self.system_prompt,
            user_message=prompt,
            max_tokens=800
        )
        
        return {
            "success": True,
            "threat": threat,
            "mitigation": mitigation
        }
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute security analysis task.
        
        Args:
            task: Task description or input to analyze
            context: Optional context with logs to scan
        
        Returns:
            Security analysis results
        """
        self.tasks_processed += 1
        
        try:
            # Check if context has logs to scan
            if context and "logs" in context:
                logs = context["logs"]
                
                # Scan logs
                scan_result = await self.scan_logs(logs)
                
                if scan_result["threat_count"] > 0:
                    # Mitigate first threat
                    first_threat = scan_result["threats_found"][0]
                    mitigation_result = await self.mitigate_threat(first_threat)
                    
                    self.tasks_successful += 1
                    
                    return {
                        "success": True,
                        "response": mitigation_result['mitigation'],
                        "minister": self.name,
                        "domain": "security",
                        "metadata": {
                            "scan_result": scan_result,
                            "threat_mitigated": first_threat
                        }
                    }
                else:
                    self.tasks_successful += 1
                    
                    return {
                        "success": True,
                        "response": "✅ All logs scanned. No threats detected.",
                        "minister": self.name,
                        "domain": "security",
                        "metadata": {
                            "scan_result": scan_result
                        }
                    }
            
            # Fallback: Analyze input directly
            scan_result = await self.scan_logs([task])
            
            if scan_result["threat_count"] > 0:
                first_threat = scan_result["threats_found"][0]
                mitigation_result = await self.mitigate_threat(first_threat)
                
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": f"⚠️ THREAT DETECTED: {first_threat['threat_type']}\n\n{mitigation_result['mitigation']}",
                    "minister": self.name,
                    "domain": "security",
                    "metadata": {
                        "threat": first_threat
                    }
                }
            else:
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": "✅ Input analyzed. No threats detected.",
                    "minister": self.name,
                    "domain": "security"
                }
        
        except Exception as e:
            logger.error(f"Security Minister error: {e}")
            return {
                "success": False,
                "response": f"Security analysis failed: {str(e)}",
                "minister": self.name,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["SecurityMinister"]
