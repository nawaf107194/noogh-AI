#!/usr/bin/env python3
"""
Noogh Agent Brain - العقل المفكر
نظام التفكير والتخطيط المستقل
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
# Add parent to path
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not available - Agent will use rule-based planning")

from agent.tools import tool_registry, ToolResult

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Task:
    """مهمة واحدة"""
    def __init__(self, description: str, task_type: str = "general"):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.description = description
        self.task_type = task_type
        self.status = "pending"  # pending, in_progress, completed, failed
        self.steps: List[Dict] = []
        self.results: List[Dict] = []
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

    def add_step(self, step: str, tool: str, params: Dict):
        """إضافة خطوة للمهمة"""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "description": step,
            "tool": tool,
            "params": params,
            "status": "pending"
        })

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "status": self.status,
            "steps_count": len(self.steps),
            "steps": self.steps,
            "results": self.results,
            "created_at": str(self.created_at),
            "completed_at": str(self.completed_at) if self.completed_at else None
        }


class AgentBrain:
    """
    العقل المفكر للوكيل
    يحلل المهام، يخطط الخطوات، وينفذها
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tool_registry = tool_registry
        self.tasks: List[Task] = []
        self.current_task: Optional[Task] = None

        # Load model if available
        if HAS_TORCH and model_path and Path(model_path).exists():
            try:
                log.info(f"Loading Noogh model from {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                log.info("✅ Model loaded successfully")
            except Exception as e:
                log.warning(f"⚠️ Could not load model: {e}")
                log.info("Using rule-based planning instead")

    def analyze_task(self, user_input: str) -> Task:
        """
        تحليل مهمة من المستخدم

        Args:
            user_input: طلب المستخدم

        Returns:
            Task object مع التحليل والخطة
        """
        log.info(f"📋 Analyzing task: {user_input}")

        # Detect task type
        task_type = self._detect_task_type(user_input)
        task = Task(description=user_input, task_type=task_type)

        # Plan steps based on task type
        if task_type == "conversation":
            self._plan_conversation(task, user_input)
        elif task_type == "project_analysis":
            self._plan_project_analysis(task, user_input)
        elif task_type == "file_list":
            self._plan_file_list(task, user_input)
        elif task_type == "file_read":
            self._plan_file_read(task, user_input)
        elif task_type == "file_write":
            self._plan_file_write(task, user_input)
        elif task_type == "search":
            self._plan_search(task, user_input)
        elif task_type == "bash_command":
            self._plan_bash(task, user_input)
        elif task_type == "python_code":
            self._plan_python(task, user_input)
        elif task_type == "system_info":
            self._plan_system_info(task, user_input)
        elif task_type == "git_operation":
            self._plan_git(task, user_input)
        else:
            # General task - try to break it down
            self._plan_general(task, user_input)

        self.tasks.append(task)
        log.info(f"✅ Task planned with {len(task.steps)} steps")
        return task

    def _is_conversational(self, user_input: str) -> bool:
        """
        كاشف السياق الذكي - يحدد إذا كانت الرسالة تحاورية أم أمر
        ContextClassifier: detects if message is conversational or command
        """
        input_lower = user_input.lower().strip()

        # تحيات ومحادثات عامة
        greetings = ["مرحبا", "hello", "hi", "السلام", "صباح", "مساء", "كيف حالك", "how are you",
                     "شكرا", "thank", "ممتاز", "رائع", "good", "great"]

        # أسئلة عامة وفلسفية
        general_questions = ["ما هو", "what is", "من أنت", "who are you", "كيف تعمل", "how do",
                            "لماذا", "why", "هل تعتقد", "do you think", "ما رأيك", "your opinion"]

        # محادثة إذا بدأت بتحية أو سؤال عام
        if any(word in input_lower for word in greetings):
            return True
        if any(phrase in input_lower for phrase in general_questions):
            return True

        # إذا كانت قصيرة جداً ومجرد كلمة أو كلمتين بدون رموز برمجية
        words = input_lower.split()
        if len(words) <= 3 and not any(c in user_input for c in ["/", ".", "$", "ls", "cd", "git"]):
            return True

        return False

    def _detect_task_type(self, user_input: str) -> str:
        """تحديد نوع المهمة"""
        input_lower = user_input.lower()

        # أولاً: فحص إذا كانت محادثة عادية
        if self._is_conversational(user_input):
            return "conversation"

        # Project Analysis (check for combination of keywords)
        project_keywords = ["تحليل", "حلل", "analyze", "analysis", "فحص", "inspect"]
        project_targets = ["مشروع", "project", "كود", "code"]

        # Check if contains project analysis keywords
        has_analysis_word = any(word in input_lower for word in project_keywords)
        has_project_word = any(word in input_lower for word in project_targets)

        if has_analysis_word and has_project_word:
            return "project_analysis"

        # List directory (must come before file_read!)
        if any(word in input_lower for word in ["محتويات", "مجلد", "ls", "dir", "list", "شجرة", "tree"]):
            return "file_list"

        # File operations
        if any(word in input_lower for word in ["اقرأ", "read", "show", "عرض", "cat"]):
            return "file_read"
        if any(word in input_lower for word in ["اكتب", "write", "create file", "أنشئ ملف"]):
            return "file_write"

        # Search
        if any(word in input_lower for word in ["ابحث", "search", "find", "grep", "جد"]):
            return "search"

        # Bash
        if any(word in input_lower for word in ["شغل", "run", "execute", "نفذ", "command"]):
            return "bash_command"

        # Python
        if any(word in input_lower for word in ["python", "بايثون", "script", "كود"]):
            return "python_code"

        # System
        if any(word in input_lower for word in ["gpu", "disk", "memory", "process", "نظام"]):
            return "system_info"

        # Git
        if any(word in input_lower for word in ["git", "commit", "diff", "log"]):
            return "git_operation"

        return "general"

    def _plan_project_analysis(self, task: Task, user_input: str):
        """تخطيط تحليل مشروع"""
        # Extract project path from user input
        input_lower = user_input.lower()

        # Check for explicit path
        project_path = None

        # Try to find quoted paths first
        path_match = re.search(r'[\'"]([^\'"]+)[\'"]', user_input)
        if path_match:
            project_path = path_match.group(1)
        elif "/" in user_input or user_input.startswith("~") or user_input.startswith("."):
            # Extract path from words
            words = user_input.split()
            for word in words:
                if "/" in word or word.startswith("~") or word.startswith("."):
                    project_path = word
                    break

        # Default to current directory if no path found
        if not project_path:
            project_path = "."

        # Expand ~ to home directory
        if project_path.startswith("~"):
            project_path = os.path.expanduser(project_path)

        task.add_step(
            step=f"تحليل المشروع: {project_path}",
            tool="analyze_project",
            params={"project_path": project_path, "detailed": True}
        )

    def _plan_file_read(self, task: Task, user_input: str):
        """تخطيط قراءة ملف"""
        # Extract file path
        # Simple pattern matching - could be improved with NLU
        path_match = re.search(r'[\'"]([^\'"]+)[\'"]', user_input)
        if path_match:
            file_path = path_match.group(1)
        else:
            # Try to find path-like string
            words = user_input.split()
            file_path = None
            for word in words:
                if '/' in word or '.' in word:
                    file_path = word
                    break

            if not file_path:
                file_path = "/etc/hostname"  # default

        task.add_step(
            step=f"قراءة الملف: {file_path}",
            tool="read_file",
            params={"file_path": file_path}
        )

    def _plan_file_write(self, task: Task, user_input: str):
        """تخطيط كتابة ملف"""
        # This is complex - for now use simple approach
        task.add_step(
            step="كتابة ملف (يحتاج معلومات إضافية)",
            tool="write_file",
            params={"file_path": "/tmp/noogh_output.txt", "content": user_input}
        )

    def _plan_file_list(self, task: Task, user_input: str):
        """تخطيط عرض محتويات مجلد"""
        # Extract directory path
        input_lower = user_input.lower()
        directory = None

        # First, check for full paths (highest priority)
        if "/" in user_input:
            # Extract path from words
            words = user_input.split()
            for word in words:
                if "/" in word:
                    directory = word
                    break

        # If no full path found, check for common directory references
        if not directory:
            if "home" in input_lower and "/" not in user_input:
                # Only use home if no full path was specified
                directory = os.path.expanduser("~")
            else:
                directory = "."

        task.add_step(
            step=f"عرض محتويات: {directory}",
            tool="list_dir",
            params={"directory": directory}
        )

    def _plan_search(self, task: Task, user_input: str):
        """تخطيط البحث"""
        # Extract search pattern
        # Look for quoted strings first
        pattern_match = re.search(r'[\'"]([^\'"]+)[\'"]', user_input)
        if pattern_match:
            pattern = pattern_match.group(1)
        else:
            # Use last word as pattern
            words = [w for w in user_input.split() if len(w) > 2]
            pattern = words[-1] if words else "error"

        task.add_step(
            step=f"البحث عن: {pattern}",
            tool="grep",
            params={"pattern": pattern, "path": ".", "recursive": True}
        )

    def _plan_bash(self, task: Task, user_input: str):
        """تخطيط تنفيذ bash"""
        # Extract command
        # Look for command after keywords
        cmd_match = re.search(r'(?:run|execute|شغل|نفذ)\s+[\'"]?([^\'"]+)[\'"]?', user_input, re.IGNORECASE)
        if cmd_match:
            command = cmd_match.group(1)
        else:
            # Use everything after first word
            words = user_input.split(maxsplit=1)
            command = words[1] if len(words) > 1 else "echo 'No command specified'"

        task.add_step(
            step=f"تنفيذ: {command}",
            tool="bash",
            params={"command": command}
        )

    def _plan_python(self, task: Task, user_input: str):
        """تخطيط تنفيذ Python"""
        # Extract code
        code_match = re.search(r'```python\n(.*?)\n```', user_input, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = "print('No Python code found in input')"

        task.add_step(
            step="تنفيذ كود Python",
            tool="python",
            params={"code": code}
        )

    def _plan_system_info(self, task: Task, user_input: str):
        """تخطيط معلومات النظام"""
        input_lower = user_input.lower()

        if "gpu" in input_lower:
            task.add_step(
                step="جلب معلومات GPU",
                tool="gpu_info",
                params={}
            )

        if "disk" in input_lower:
            task.add_step(
                step="جلب معلومات القرص",
                tool="disk_usage",
                params={"path": "/"}
            )

        if "process" in input_lower:
            # Extract process name
            words = user_input.split()
            process_name = words[-1] if words else "python"
            task.add_step(
                step=f"التحقق من العملية: {process_name}",
                tool="check_process",
                params={"process_name": process_name}
            )

    def _plan_git(self, task: Task, user_input: str):
        """تخطيط عمليات Git"""
        input_lower = user_input.lower()

        if "status" in input_lower or "حالة" in input_lower:
            task.add_step(
                step="عرض حالة Git",
                tool="git_status",
                params={"repo_path": "."}
            )

        if "diff" in input_lower:
            task.add_step(
                step="عرض التغييرات",
                tool="git_diff",
                params={"repo_path": "."}
            )

        if "log" in input_lower or "سجل" in input_lower:
            task.add_step(
                step="عرض سجل Git",
                tool="git_log",
                params={"repo_path": ".", "limit": 10}
            )

    def _plan_general(self, task: Task, user_input: str):
        """تخطيط مهمة عامة"""
        # For general tasks, try to understand and break down
        # This is simplified - in real agent, would use LLM

        log.info("📝 Planning general task...")

        # Check if it's a complex task that needs breaking down
        if len(user_input.split()) > 10:
            # Multi-step task
            task.add_step(
                step="تحليل المهمة",
                tool="bash",
                params={"command": "echo 'Analyzing complex task...'"}
            )
        else:
            # Simple task - try bash
            task.add_step(
                step=f"تنفيذ: {user_input}",
                tool="bash",
                params={"command": user_input}
            )

    def _plan_conversation(self, task: Task, user_input: str):
        """
        تخطيط محادثة عادية - يتم إرسالها مباشرة لعلام (ALLaM)
        Plan conversational message - sends directly to ALLaM
        """
        log.info("💬 Detected conversational message, routing to ALLaM")

        # إضافة خطوة للمحادثة مع علام
        task.add_step(
            step=f"محادثة مع علام: {user_input[:50]}...",
            tool="allam_chat",
            params={
                "message": user_input,
                "system_prompt": "أنت مساعد ذكي اسمك نوغ (Noogh). تجيب بشكل احترافي وودود باللغة العربية أو الإنجليزية حسب لغة المستخدم.",
                "max_tokens": 300,
                "temperature": 0.7
            }
        )

    def execute_task(self, task: Task) -> Dict:
        """
        تنفيذ مهمة

        Args:
            task: المهمة المراد تنفيذها

        Returns:
            نتائج التنفيذ
        """
        log.info(f"▶️ Executing task: {task.description}")
        self.current_task = task
        task.status = "in_progress"

        results = []

        for step in task.steps:
            step_num = step["step_number"]
            log.info(f"⚙️ Step {step_num}/{len(task.steps)}: {step['description']}")

            step["status"] = "in_progress"

            try:
                # Execute tool
                result = self.tool_registry.execute(
                    step["tool"],
                    **step["params"]
                )

                step["status"] = "completed" if result.success else "failed"
                step["result"] = result.to_dict()

                results.append({
                    "step": step_num,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error
                })

                log.info(f"{'✅' if result.success else '❌'} Step {step_num}: {step['status']}")

                if not result.success:
                    log.error(f"Error: {result.error}")
                    # Continue to next step (could also break here)

            except Exception as e:
                log.error(f"❌ Step {step_num} failed: {e}")
                step["status"] = "failed"
                step["error"] = str(e)
                results.append({
                    "step": step_num,
                    "success": False,
                    "error": str(e)
                })

        # Mark task as completed
        task.status = "completed"
        task.completed_at = datetime.now()
        task.results = results

        log.info(f"✅ Task completed: {task.id}")

        return {
            "task_id": task.id,
            "status": task.status,
            "steps_completed": len([s for s in task.steps if s["status"] == "completed"]),
            "steps_total": len(task.steps),
            "results": results
        }

    def process_request(self, user_input: str) -> Dict:
        """
        معالجة طلب من المستخدم (تحليل + تنفيذ)

        Args:
            user_input: طلب المستخدم

        Returns:
            النتائج الكاملة
        """
        # Analyze and plan
        task = self.analyze_task(user_input)

        # Execute
        results = self.execute_task(task)

        return {
            "task": task.to_dict(),
            "execution": results
        }

    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """الحصول على سجل المهام"""
        return [t.to_dict() for t in self.tasks[-limit:]]


# Create global agent
agent_brain = None


def get_agent(model_path: Optional[str] = None) -> AgentBrain:
    """الحصول على Agent instance"""
    global agent_brain
    if agent_brain is None:
        agent_brain = AgentBrain(model_path=model_path)
    return agent_brain


if __name__ == "__main__":
    # Test agent
    print("Testing Noogh Agent Brain...")

    agent = get_agent()

    # Test various tasks
    test_cases = [
        "اقرأ ملف /etc/hostname",
        "ابحث عن كلمة 'error' في المجلد الحالي",
        "شغل أمر 'ls -lh /tmp'",
        "gpu معلومات",
    ]

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test}")
        print('='*60)

        result = agent.process_request(test)
        print(json.dumps(result, indent=2, ensure_ascii=False))
