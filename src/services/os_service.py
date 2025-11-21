#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OS Service - Safe Operating System Interface
=============================================

Centralized, controlled access to system commands and OS operations.
Implements safety locks and whitelisting.
"""

import subprocess
import logging
import hashlib
from typing import Optional, List, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)


class OSService:
    """
    Singleton service for safe OS command execution.
    
    Features:
    - Command whitelisting
    - Safety locks
    - Audit logging
    - Override mechanism for trusted operations
    """
    
    _instance: Optional['OSService'] = None
    _lock: Lock = Lock()
    _initialized: bool = False
    
    # Whitelist of safe commands
    SAFE_COMMANDS = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'df', 'du',
        'ps', 'top', 'free', 'uname', 'hostname', 'date', 'whoami',
        'lsblk', 'lsusb', 'lspci', 'nvidia-smi'
    }
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize OS Service."""
        if not self._initialized:
            self.override_enabled = False
            self.command_history: List[Dict[str, Any]] = []
            self._initialized = True
            logger.info("✅ OS Service initialized (Safety Mode: ON)")
    
    def enable_override(self, password: str) -> bool:
        """
        Enable override mode (allows all commands).
        
        Args:
            password: Override password (will be hashed and compared)
        
        Returns:
            True if successful
        """
        try:
            from ..core.settings import settings
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if password_hash == settings.override_password_hash:
                self.override_enabled = True
                logger.warning("⚠️ OVERRIDE MODE ENABLED - All commands allowed!")
                return True
            else:
                logger.warning("🚫 Invalid override password attempt")
                return False
        except Exception as e:
            logger.error(f"Error validating override password: {e}")
            return False
    
    def disable_override(self):
        """Disable override mode."""
        self.override_enabled = False
        logger.info("✅ Override mode disabled - Safety restored")
    
    def is_command_safe(self, command: str) -> bool:
        """
        Check if command is in whitelist.
        
        Args:
            command: Command to check
        
        Returns:
            True if safe
        """
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
        
        base_command = cmd_parts[0]
        
        # Check whitelist
        return base_command in self.SAFE_COMMANDS
    
    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        check_safety: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a system command safely.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            check_safety: Whether to check whitelist
        
        Returns:
            Dictionary with result
        """
        # Safety check
        if check_safety and not self.override_enabled:
            if not self.is_command_safe(command):
                logger.warning(f"🚫 Command blocked (not whitelisted): {command}")
                return {
                    "success": False,
                    "error": "Command not in whitelist. Enable override or use safe commands.",
                    "command": command
                }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Log execution
            self.command_history.append({
                "command": command,
                "success": result.returncode == 0,
                "return_code": result.returncode
            })
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": command
            }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {command}")
            return {
                "success": False,
                "error": "Command timed out",
                "command": command
            }
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return self.command_history[-limit:]
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False


# ============================================================================
# Exports
# ============================================================================

__all__ = ["OSService"]
