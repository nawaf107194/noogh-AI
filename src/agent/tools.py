#!/usr/bin/env python3
"""
Noogh Agent Tools - أدوات الوكيل الذكي
نظام أدوات متكامل لتنفيذ المهام بشكل مستقل
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ToolResult:
    """نتيجة تنفيذ الأداة"""
    def __init__(self, success: bool, output: Any, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error
        }


class BashTool:
    """أداة تنفيذ أوامر Bash"""

    @staticmethod
    def execute(command: str, timeout: int = 30, cwd: Optional[str] = None) -> ToolResult:
        """
        تنفيذ أمر bash

        Args:
            command: الأمر المراد تنفيذه
            timeout: الوقت الأقصى بالثواني
            cwd: مجلد العمل

        Returns:
            ToolResult مع النتيجة
        """
        try:
            log.info(f"🔧 Executing: {command}")

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output=result.stdout.strip()
                )
            else:
                return ToolResult(
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip()
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )


class FileTool:
    """أداة إدارة الملفات"""

    @staticmethod
    def read(file_path: str, max_lines: Optional[int] = None) -> ToolResult:
        """قراءة ملف"""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            with open(path, 'r', encoding='utf-8') as f:
                if max_lines:
                    lines = [f.readline() for _ in range(max_lines)]
                    content = ''.join(lines)
                else:
                    content = f.read()

            return ToolResult(success=True, output=content)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def write(file_path: str, content: str) -> ToolResult:
        """كتابة ملف"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"File written: {file_path}"
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def append(file_path: str, content: str) -> ToolResult:
        """إضافة محتوى لملف"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Content appended to: {file_path}"
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def list_dir(directory: str, pattern: Optional[str] = None) -> ToolResult:
        """عرض محتويات مجلد"""
        try:
            # Handle home directory references
            if directory == "~":
                directory = os.path.expanduser("~")
            elif directory.startswith("~/"):
                directory = os.path.expanduser(directory)
            # Note: "home" alone is NOT treated as home directory
            # Use full path /home/noogh or ~ instead

            path = Path(directory).resolve()

            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"❌ المجلد غير موجود: {directory}"
                )

            if not path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"❌ المسار ليس مجلداً: {directory}"
                )

            # Get files
            if pattern:
                files = list(path.glob(pattern))
            else:
                files = list(path.iterdir())

            # Sort: directories first, then files
            files.sort(key=lambda f: (not f.is_dir(), f.name.lower()))

            # Format output as readable text
            if not files:
                output_text = f"📁 المجلد '{directory}' فارغ"
            else:
                output_lines = [f"📁 محتويات '{directory}':", ""]

                for f in files:
                    try:
                        icon = "📁" if f.is_dir() else "📄"
                        size_str = ""
                        if f.is_file():
                            size = f.stat().st_size
                            if size < 1024:
                                size_str = f" ({size} B)"
                            elif size < 1024 * 1024:
                                size_str = f" ({size / 1024:.1f} KB)"
                            else:
                                size_str = f" ({size / (1024 * 1024):.1f} MB)"

                        output_lines.append(f"{icon} {f.name}{size_str}")
                    except Exception:
                        output_lines.append(f"❓ {f.name} (لا يمكن قراءة المعلومات)")

                output_lines.append(f"\n📊 الإجمالي: {len(files)} عنصر")
                output_text = "\n".join(output_lines)

            return ToolResult(success=True, output=output_text)

        except Exception as e:
            return ToolResult(success=False, output="", error=f"❌ خطأ: {str(e)}")


class SearchTool:
    """أداة البحث في الملفات"""

    @staticmethod
    def grep(pattern: str, path: str, recursive: bool = True) -> ToolResult:
        """البحث عن نص في ملفات"""
        try:
            if recursive:
                cmd = f"grep -r -n '{pattern}' {path}"
            else:
                cmd = f"grep -n '{pattern}' {path}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            # grep returns exit code 1 if no matches found
            if result.returncode == 0 or result.returncode == 1:
                matches = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return ToolResult(
                    success=True,
                    output={
                        "pattern": pattern,
                        "matches_count": len(matches),
                        "matches": matches[:100]  # First 100 matches
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=result.stderr.strip()
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def find_files(directory: str, name_pattern: str) -> ToolResult:
        """البحث عن ملفات بالاسم"""
        try:
            cmd = f"find {directory} -name '{name_pattern}' -type f"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return ToolResult(
                    success=True,
                    output={
                        "pattern": name_pattern,
                        "files_count": len(files),
                        "files": files
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=result.stderr.strip()
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class PythonTool:
    """أداة تنفيذ كود Python"""

    @staticmethod
    def execute(code: str, timeout: int = 30) -> ToolResult:
        """تنفيذ كود Python"""
        try:
            # Create temporary file
            temp_file = Path("/tmp/noogh_python_exec.py")
            temp_file.write_text(code, encoding='utf-8')

            # Execute
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Cleanup
            temp_file.unlink()

            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output=result.stdout.strip()
                )
            else:
                return ToolResult(
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip()
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Python execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GitTool:
    """أداة التحكم في Git"""

    @staticmethod
    def status(repo_path: str = ".") -> ToolResult:
        """حالة Git"""
        return BashTool.execute(f"git -C {repo_path} status", cwd=repo_path)

    @staticmethod
    def diff(repo_path: str = ".", file_path: Optional[str] = None) -> ToolResult:
        """عرض التغييرات"""
        if file_path:
            cmd = f"git -C {repo_path} diff {file_path}"
        else:
            cmd = f"git -C {repo_path} diff"

        return BashTool.execute(cmd, cwd=repo_path)

    @staticmethod
    def log(repo_path: str = ".", limit: int = 10) -> ToolResult:
        """عرض السجل"""
        cmd = f"git -C {repo_path} log --oneline -n {limit}"
        return BashTool.execute(cmd, cwd=repo_path)


class SystemTool:
    """أدوات النظام"""

    @staticmethod
    def check_process(process_name: str) -> ToolResult:
        """التحقق من عملية تعمل"""
        result = BashTool.execute(f"pgrep -f '{process_name}'")

        if result.success and result.output:
            pids = result.output.split('\n')
            return ToolResult(
                success=True,
                output={
                    "running": True,
                    "pids": pids,
                    "count": len(pids)
                }
            )
        else:
            return ToolResult(
                success=True,
                output={"running": False, "pids": [], "count": 0}
            )

    @staticmethod
    def get_gpu_info() -> ToolResult:
        """معلومات GPU"""
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        result = BashTool.execute(cmd)

        if result.success:
            try:
                values = result.output.split(',')
                gpu_info = {
                    "gpu_usage": int(values[0].strip()),
                    "memory_usage": int(values[1].strip()),
                    "memory_used_mb": int(values[2].strip()),
                    "memory_total_mb": int(values[3].strip()),
                    "temperature": int(values[4].strip())
                }
                return ToolResult(success=True, output=gpu_info)
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))
        else:
            return result

    @staticmethod
    def get_disk_usage(path: str = "/") -> ToolResult:
        """استخدام القرص"""
        cmd = f"df -h {path} | tail -1"
        result = BashTool.execute(cmd)

        if result.success:
            parts = result.output.split()
            return ToolResult(
                success=True,
                output={
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "usage_percent": parts[4]
                }
            )
        else:
            return result


class ProjectAnalysisTool:
    """أداة تحليل المشاريع البرمجية"""

    @staticmethod
    def analyze_project(project_path: str, detailed: bool = True) -> ToolResult:
        """تحليل مشروع برمجي بشكل شامل"""
        try:
            from agent.project_analyzer import project_analyzer

            log.info(f"🔍 Analyzing project: {project_path}")

            # تحليل المشروع
            analysis = project_analyzer.analyze_project(project_path)

            if "error" in analysis:
                return ToolResult(
                    success=False,
                    output="",
                    error=analysis["error"]
                )

            # توليد التقرير
            if detailed:
                report = project_analyzer.generate_report(analysis)
                return ToolResult(success=True, output=report)
            else:
                # تقرير مختصر
                summary = f"""📊 تحليل سريع للمشروع: {analysis['project_name']}

📁 {analysis['structure']['total_files']} ملف في {analysis['structure']['total_dirs']} مجلد
💻 اللغة الأساسية: {analysis['languages']['primary']}
📦 الحجم: {analysis['files_stats']['total_size_str']}
⚠️  مشاكل: {len(analysis['potential_issues'])}
💡 توصيات: {len(analysis['recommendations'])}"""

                return ToolResult(success=True, output=summary)

        except Exception as e:
            log.error(f"Project analysis error: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"خطأ في تحليل المشروع: {str(e)}"
            )


class ALLaMTool:
    """أداة المحادثة مع ALLaM"""

    @staticmethod
    def chat(message: str, system_prompt: Optional[str] = None,
             max_tokens: int = 300, temperature: float = 0.7) -> ToolResult:
        """
        محادثة مع نموذج ALLaM

        Args:
            message: الرسالة من المستخدم
            system_prompt: التعليمات الأولية للنموذج
            max_tokens: عدد أقصى من الكلمات
            temperature: درجة الإبداع (0.0 - 1.0)

        Returns:
            ToolResult مع رد ALLaM
        """
        try:
            import requests

            # استدعاء ALLaM API
            response = requests.post(
                "http://localhost:8000/api/allam/chat",
                json={
                    "message": message,
                    "system_prompt": system_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    allam_response = data.get("response", "")
                    return ToolResult(
                        success=True,
                        output=allam_response
                    )
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"ALLaM error: {data.get('detail', 'Unknown error')}"
                    )
            elif response.status_code == 400:
                # ALLaM not loaded
                return ToolResult(
                    success=False,
                    output="",
                    error="ALLaM model is not loaded. Please load it first via /api/allam/load"
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"ALLaM API returned status {response.status_code}"
                )

        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                output="",
                error="ALLaM request timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error calling ALLaM: {str(e)}"
            )


class ToolRegistry:
    """سجل الأدوات المتاحة"""

    def __init__(self):
        self.tools = {
            # Bash
            "bash": BashTool.execute,

            # Files
            "read_file": FileTool.read,
            "write_file": FileTool.write,
            "append_file": FileTool.append,
            "list_dir": FileTool.list_dir,

            # Search
            "grep": SearchTool.grep,
            "find_files": SearchTool.find_files,

            # Python
            "python": PythonTool.execute,

            # Git
            "git_status": GitTool.status,
            "git_diff": GitTool.diff,
            "git_log": GitTool.log,

            # System
            "check_process": SystemTool.check_process,
            "gpu_info": SystemTool.get_gpu_info,
            "disk_usage": SystemTool.get_disk_usage,

            # Project Analysis
            "analyze_project": ProjectAnalysisTool.analyze_project,

            # ALLaM Integration
            "allam_chat": ALLaMTool.chat,
        }

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """تنفيذ أداة"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool not found: {tool_name}"
            )

        try:
            tool_func = self.tools[tool_name]
            return tool_func(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution error: {str(e)}"
            )

    def has_tool(self, tool_name: str) -> bool:
        """Check if tool exists"""
        return tool_name in self.tools

    def list_tools(self) -> List[str]:
        """قائمة الأدوات المتاحة"""
        return list(self.tools.keys())


# Create global registry
tool_registry = ToolRegistry()


if __name__ == "__main__":
    # Test tools
    print("Testing Noogh Agent Tools...")

    # Test bash
    result = tool_registry.execute("bash", command="echo 'Hello from Noogh!'")
    print(f"Bash: {result.to_dict()}")

    # Test file read
    result = tool_registry.execute("read_file", file_path="/etc/hostname")
    print(f"Read file: {result.to_dict()}")

    # Test GPU info
    result = tool_registry.execute("gpu_info")
    print(f"GPU info: {result.to_dict()}")

    # List all tools
    print(f"\nAvailable tools: {tool_registry.list_tools()}")
