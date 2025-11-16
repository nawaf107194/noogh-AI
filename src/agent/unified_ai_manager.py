#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Unified AI Manager - مدير الذكاء الاصطناعي الموحد
=====================================================

يدير ويربط جميع نماذج الذكاء الاصطناعي في النظام:
- ALLaM (7B Arabic LLM) للمحادثات العربية
- ProjectManagementLLM (CodeLlama) لإدارة المشاريع
- UnifiedBrain (MegaBrain V5) للتعلم الآلي

Manages and connects all AI models in the system:
- ALLaM for Arabic conversations
- ProjectManagementLLM for project management
- UnifiedBrain for machine learning

Author: Noogh AI Team
Version: 1.0.0
Date: 2025-11-07
"""

import requests
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class UnifiedAIManager:
    """
    مدير موحد لجميع نماذج الذكاء الاصطناعي
    Unified manager for all AI models
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.allam_loaded = False
        self.pm_llm_loaded = False

        log.info("🤖 Initializing Unified AI Manager...")
        self._check_services()

    def _check_services(self):
        """فحص حالة الخدمات المتاحة"""
        try:
            # Check main server
            response = requests.get(f"{self.api_base_url}/health", timeout=2)
            if response.status_code == 200:
                log.info("✅ Main server is running")

            # Check ALLaM status
            allam_status = self.get_allam_status()
            self.allam_loaded = allam_status.get("loaded", False)

            # Check ProjectManager status
            pm_status = self.get_pm_status()
            self.pm_llm_loaded = pm_status.get("loaded", False)

            log.info(f"📊 ALLaM: {'✅ Loaded' if self.allam_loaded else '⚠️ Not loaded'}")
            log.info(f"📊 ProjectManager: {'✅ Loaded' if self.pm_llm_loaded else '⚠️ Not loaded'}")

        except Exception as e:
            log.warning(f"⚠️ Could not check services: {e}")

    # ========================
    # ALLaM Management
    # ========================

    def load_allam(self) -> Dict[str, Any]:
        """تحميل نموذج ALLaM"""
        log.info("📥 Loading ALLaM model...")
        try:
            response = requests.post(
                f"{self.api_base_url}/api/allam/load",
                timeout=180  # 3 minutes
            )
            result = response.json()

            if response.status_code == 200:
                self.allam_loaded = True
                log.info("✅ ALLaM loaded successfully")

            return result
        except Exception as e:
            log.error(f"❌ Failed to load ALLaM: {e}")
            return {"status": "error", "message": str(e)}

    def unload_allam(self) -> Dict[str, Any]:
        """تفريغ نموذج ALLaM"""
        log.info("🗑️ Unloading ALLaM model...")
        try:
            response = requests.post(
                f"{self.api_base_url}/api/allam/unload",
                timeout=30
            )
            result = response.json()

            if response.status_code == 200:
                self.allam_loaded = False
                log.info("✅ ALLaM unloaded successfully")

            return result
        except Exception as e:
            log.error(f"❌ Failed to unload ALLaM: {e}")
            return {"status": "error", "message": str(e)}

    def get_allam_status(self) -> Dict[str, Any]:
        """الحصول على حالة ALLaM"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/allam/status",
                timeout=5
            )
            return response.json()
        except Exception as e:
            return {"loaded": False, "error": str(e)}

    def chat_with_allam(
        self,
        message: str,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """محادثة مع ALLaM"""
        log.info(f"💬 Chatting with ALLaM: {message[:50]}...")

        if not self.allam_loaded:
            log.warning("⚠️ ALLaM not loaded, loading now...")
            load_result = self.load_allam()
            if load_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": "ALLaM is not loaded and failed to load"
                }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/allam/chat",
                json={
                    "message": message,
                    "max_length": max_length,
                    "temperature": temperature
                },
                timeout=60
            )
            return response.json()
        except Exception as e:
            log.error(f"❌ Failed to chat with ALLaM: {e}")
            return {"status": "error", "message": str(e)}

    # ========================
    # ProjectManager Management
    # ========================

    def load_project_manager(self, model_name: str = "codellama/CodeLlama-7b-Instruct-hf") -> Dict[str, Any]:
        """تحميل نموذج ProjectManager"""
        log.info(f"📥 Loading ProjectManager ({model_name})...")
        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/load",
                params={"model_name": model_name},
                timeout=180  # 3 minutes
            )
            result = response.json()

            if response.status_code == 200:
                self.pm_llm_loaded = True
                log.info("✅ ProjectManager loaded successfully")

            return result
        except Exception as e:
            log.error(f"❌ Failed to load ProjectManager: {e}")
            return {"status": "error", "message": str(e)}

    def unload_project_manager(self) -> Dict[str, Any]:
        """تفريغ نموذج ProjectManager"""
        log.info("🗑️ Unloading ProjectManager...")
        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/unload",
                timeout=30
            )
            result = response.json()

            if response.status_code == 200:
                self.pm_llm_loaded = False
                log.info("✅ ProjectManager unloaded successfully")

            return result
        except Exception as e:
            log.error(f"❌ Failed to unload ProjectManager: {e}")
            return {"status": "error", "message": str(e)}

    def get_pm_status(self) -> Dict[str, Any]:
        """الحصول على حالة ProjectManager"""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/project-manager/status",
                timeout=5
            )
            return response.json()
        except Exception as e:
            return {"loaded": False, "error": str(e)}

    def analyze_project(
        self,
        project_path: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """تحليل مشروع باستخدام ProjectManager"""
        log.info(f"🔍 Analyzing project: {project_path}")

        if not self.pm_llm_loaded:
            log.warning("⚠️ ProjectManager not loaded, loading now...")
            load_result = self.load_project_manager()
            if load_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": "ProjectManager is not loaded and failed to load"
                }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/analyze-project",
                json={
                    "project_path": project_path,
                    "focus_areas": focus_areas
                },
                timeout=120  # 2 minutes
            )
            return response.json()
        except Exception as e:
            log.error(f"❌ Failed to analyze project: {e}")
            return {"status": "error", "message": str(e)}

    def breakdown_task(
        self,
        feature_description: str,
        complexity: str = "medium"
    ) -> Dict[str, Any]:
        """تقسيم مهمة إلى خطوات"""
        log.info(f"📋 Breaking down task: {feature_description[:50]}...")

        if not self.pm_llm_loaded:
            log.warning("⚠️ ProjectManager not loaded, loading now...")
            load_result = self.load_project_manager()
            if load_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": "ProjectManager is not loaded and failed to load"
                }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/breakdown-task",
                json={
                    "feature_description": feature_description,
                    "complexity": complexity
                },
                timeout=90
            )
            return response.json()
        except Exception as e:
            log.error(f"❌ Failed to breakdown task: {e}")
            return {"status": "error", "message": str(e)}

    def review_code(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """مراجعة جودة الكود"""
        log.info(f"🔍 Reviewing {language} code...")

        if not self.pm_llm_loaded:
            log.warning("⚠️ ProjectManager not loaded, loading now...")
            load_result = self.load_project_manager()
            if load_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": "ProjectManager is not loaded and failed to load"
                }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/review-code",
                json={
                    "code": code,
                    "language": language
                },
                timeout=90
            )
            return response.json()
        except Exception as e:
            log.error(f"❌ Failed to review code: {e}")
            return {"status": "error", "message": str(e)}

    def generate_docs(
        self,
        code: str,
        doc_type: str = "docstring"
    ) -> Dict[str, Any]:
        """توليد توثيق للكود"""
        log.info(f"📝 Generating {doc_type} documentation...")

        if not self.pm_llm_loaded:
            log.warning("⚠️ ProjectManager not loaded, loading now...")
            load_result = self.load_project_manager()
            if load_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": "ProjectManager is not loaded and failed to load"
                }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/project-manager/generate-docs",
                json={
                    "code": code,
                    "doc_type": doc_type
                },
                timeout=90
            )
            return response.json()
        except Exception as e:
            log.error(f"❌ Failed to generate docs: {e}")
            return {"status": "error", "message": str(e)}

    # ========================
    # Smart Routing
    # ========================

    def route_request(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        توجيه ذكي للطلب إلى النموذج المناسب
        Smart routing to the appropriate model
        """
        input_lower = user_input.lower()

        # Project management keywords
        project_keywords = [
            "حلل المشروع", "analyze project", "تحليل المشروع",
            "فحص المشروع", "inspect project",
            "مراجعة الكود", "review code", "code review",
            "تقسيم المهمة", "break down", "breakdown task",
            "خطة المشروع", "project plan",
            "توليد توثيق", "generate docs", "documentation"
        ]

        # Conversation keywords (Arabic)
        conversation_keywords = [
            "مرحبا", "hello", "hi", "السلام",
            "كيف حالك", "how are you",
            "ما هو", "what is",
            "من أنت", "who are you",
            "شكرا", "thank"
        ]

        # Route to ProjectManager
        if any(keyword in input_lower for keyword in project_keywords):
            log.info("🎯 Routing to ProjectManager")

            # Analyze project
            if any(word in input_lower for word in ["حلل", "analyze", "فحص", "inspect"]):
                # Extract project path
                import re
                path_match = re.search(r'(/[\w/\-.]+)', user_input)
                if path_match:
                    project_path = path_match.group(1)
                else:
                    project_path = "."

                return {
                    "model": "ProjectManager",
                    "action": "analyze_project",
                    "result": self.analyze_project(project_path)
                }

            # Task breakdown
            elif any(word in input_lower for word in ["تقسيم", "break", "plan"]):
                return {
                    "model": "ProjectManager",
                    "action": "breakdown_task",
                    "result": self.breakdown_task(user_input)
                }

            # Code review
            elif any(word in input_lower for word in ["مراجعة", "review"]):
                # Extract code from context if available
                code = context.get("code", "") if context else ""
                if not code:
                    return {
                        "model": "ProjectManager",
                        "action": "review_code",
                        "result": {"status": "error", "message": "No code provided for review"}
                    }

                return {
                    "model": "ProjectManager",
                    "action": "review_code",
                    "result": self.review_code(code)
                }

        # Route to ALLaM for conversations
        elif any(keyword in input_lower for keyword in conversation_keywords):
            log.info("🎯 Routing to ALLaM")
            return {
                "model": "ALLaM",
                "action": "chat",
                "result": self.chat_with_allam(user_input)
            }

        # Default: Use ALLaM for general queries
        else:
            log.info("🎯 Routing to ALLaM (default)")
            return {
                "model": "ALLaM",
                "action": "chat",
                "result": self.chat_with_allam(user_input)
            }

    # ========================
    # System Management
    # ========================

    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام الكاملة"""
        return {
            "allam": {
                "loaded": self.allam_loaded,
                "status": self.get_allam_status()
            },
            "project_manager": {
                "loaded": self.pm_llm_loaded,
                "status": self.get_pm_status()
            },
            "api_base_url": self.api_base_url
        }

    def load_all_models(self) -> Dict[str, Any]:
        """تحميل جميع النماذج"""
        log.info("🚀 Loading all models...")

        results = {}

        # Load ALLaM
        if not self.allam_loaded:
            results["allam"] = self.load_allam()
        else:
            results["allam"] = {"status": "already_loaded"}

        # Load ProjectManager
        if not self.pm_llm_loaded:
            results["project_manager"] = self.load_project_manager()
        else:
            results["project_manager"] = {"status": "already_loaded"}

        return results

    def unload_all_models(self) -> Dict[str, Any]:
        """تفريغ جميع النماذج"""
        log.info("🗑️ Unloading all models...")

        results = {}

        # Unload ALLaM
        if self.allam_loaded:
            results["allam"] = self.unload_allam()
        else:
            results["allam"] = {"status": "not_loaded"}

        # Unload ProjectManager
        if self.pm_llm_loaded:
            results["project_manager"] = self.unload_project_manager()
        else:
            results["project_manager"] = {"status": "not_loaded"}

        return results


# ========================
# CLI Interface
# ========================

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("🤖 Unified AI Manager - مدير الذكاء الاصطناعي الموحد")
    print("=" * 80)
    print()

    manager = UnifiedAIManager()

    print("\n📊 System Status:")
    status = manager.get_system_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "load-all":
            print("\n🚀 Loading all models...")
            results = manager.load_all_models()
            print(json.dumps(results, indent=2, ensure_ascii=False))

        elif command == "unload-all":
            print("\n🗑️ Unloading all models...")
            results = manager.unload_all_models()
            print(json.dumps(results, indent=2, ensure_ascii=False))

        elif command == "test":
            print("\n🧪 Testing smart routing...")

            # Test 1: Arabic conversation
            print("\n--- Test 1: Arabic Conversation ---")
            result = manager.route_request("مرحبا، كيف حالك؟")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # Test 2: Project analysis
            print("\n--- Test 2: Project Analysis ---")
            result = manager.route_request("حلل المشروع /home/noogh/projects/noogh_unified_system")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: load-all, unload-all, test")

    print("\n✅ Done!")
