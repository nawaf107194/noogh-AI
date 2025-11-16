#!/usr/bin/env python3
"""
Project Analyzer - محلل المشاريع البرمجية
يقوم بتحليل شامل لبنية ومحتوى المشاريع البرمجية
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

log = logging.getLogger(__name__)


class ProjectAnalyzer:
    """محلل المشاريع - يحلل بنية الكود والتبعيات والمشاكل"""

    def __init__(self):
        self.supported_languages = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React/JSX',
            '.tsx': 'React/TSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.sh': 'Shell',
            '.vue': 'Vue',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.xml': 'XML',
        }

        self.config_files = {
            'package.json': 'Node.js/NPM',
            'requirements.txt': 'Python/pip',
            'Pipfile': 'Python/pipenv',
            'pyproject.toml': 'Python/Poetry',
            'setup.py': 'Python/setuptools',
            'Cargo.toml': 'Rust/Cargo',
            'pom.xml': 'Java/Maven',
            'build.gradle': 'Java/Gradle',
            'go.mod': 'Go Modules',
            'composer.json': 'PHP/Composer',
            'Gemfile': 'Ruby/Bundler',
            'docker-compose.yml': 'Docker Compose',
            'Dockerfile': 'Docker',
            '.gitignore': 'Git',
            'README.md': 'Documentation',
            'tsconfig.json': 'TypeScript',
            'webpack.config.js': 'Webpack',
            'vite.config.js': 'Vite',
        }

        self.ignore_dirs = {
            'node_modules', '__pycache__', '.git', '.venv', 'venv',
            'dist', 'build', '.pytest_cache', '.mypy_cache',
            'target', 'bin', 'obj', '.next', '.nuxt', 'out',
            'coverage', '.coverage', 'htmlcov', '.tox'
        }

        self.ignore_files = {
            '.DS_Store', 'Thumbs.db', '.gitkeep', '.env', '.env.local'
        }

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        تحليل شامل للمشروع

        Returns:
            Dict with analysis results including:
            - structure: بنية المشروع
            - languages: اللغات المستخدمة
            - files_stats: إحصائيات الملفات
            - dependencies: التبعيات
            - potential_issues: المشاكل المحتملة
            - recommendations: توصيات التحسين
        """
        try:
            path = Path(project_path).resolve()

            if not path.exists():
                return {"error": f"❌ المسار غير موجود: {project_path}"}

            if not path.is_dir():
                return {"error": f"❌ المسار ليس مجلداً: {project_path}"}

            log.info(f"🔍 بدء تحليل المشروع: {project_path}")

            # جمع معلومات المشروع
            analysis = {
                "project_path": str(path),
                "project_name": path.name,
                "structure": self._analyze_structure(path),
                "languages": self._detect_languages(path),
                "files_stats": self._get_files_stats(path),
                "config_files": self._detect_config_files(path),
                "dependencies": self._analyze_dependencies(path),
                "code_metrics": self._analyze_code_metrics(path),
                "potential_issues": self._detect_issues(path),
                "recommendations": [],
            }

            # توليد التوصيات بناءً على التحليل
            analysis["recommendations"] = self._generate_recommendations(analysis)

            log.info(f"✅ اكتمل تحليل المشروع: {path.name}")
            return analysis

        except Exception as e:
            log.error(f"❌ خطأ في تحليل المشروع: {e}")
            return {"error": f"خطأ في التحليل: {str(e)}"}

    def _analyze_structure(self, path: Path) -> Dict[str, Any]:
        """تحليل البنية الشجرية للمشروع"""
        structure = {
            "total_dirs": 0,
            "total_files": 0,
            "max_depth": 0,
            "top_level_items": [],
            "main_directories": []
        }

        try:
            # العناصر في المستوى الأعلى
            top_items = []
            for item in path.iterdir():
                if item.name.startswith('.') and item.name not in ['.github', '.vscode']:
                    continue
                if item.name in self.ignore_dirs:
                    continue

                top_items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": self._get_size(item)
                })

            structure["top_level_items"] = sorted(top_items,
                                                 key=lambda x: (x["type"] != "dir", x["name"]))

            # حساب العمق والملفات
            for root, dirs, files in os.walk(path):
                # تجاهل المجلدات المستبعدة
                dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

                depth = len(Path(root).relative_to(path).parts)
                structure["max_depth"] = max(structure["max_depth"], depth)
                structure["total_dirs"] += len(dirs)
                structure["total_files"] += len([f for f in files
                                                if f not in self.ignore_files])

                # المجلدات الرئيسية (في المستوى الأول فقط)
                if depth == 1:
                    for d in dirs:
                        dir_path = Path(root) / d
                        structure["main_directories"].append({
                            "name": d,
                            "size": self._get_size(dir_path),
                            "files_count": sum(1 for _ in dir_path.rglob("*") if _.is_file())
                        })

            return structure

        except Exception as e:
            log.error(f"Error analyzing structure: {e}")
            return structure

    def _detect_languages(self, path: Path) -> Dict[str, Any]:
        """كشف اللغات البرمجية المستخدمة"""
        languages = defaultdict(lambda: {"count": 0, "lines": 0, "size": 0})

        try:
            for ext, lang in self.supported_languages.items():
                files = list(path.rglob(f"*{ext}"))
                # تصفية الملفات في المجلدات المستبعدة
                files = [f for f in files
                        if not any(ignored in f.parts for ignored in self.ignore_dirs)]

                if files:
                    total_lines = 0
                    total_size = 0

                    for file in files:
                        try:
                            total_size += file.stat().st_size
                            # عد الأسطر للملفات النصية
                            if ext not in ['.json', '.xml']:
                                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                                    total_lines += sum(1 for _ in f)
                        except:
                            pass

                    languages[lang]["count"] = len(files)
                    languages[lang]["lines"] = total_lines
                    languages[lang]["size"] = total_size

            # ترتيب حسب عدد الملفات
            sorted_langs = dict(sorted(languages.items(),
                                     key=lambda x: x[1]["count"],
                                     reverse=True))

            return {
                "detected": list(sorted_langs.keys()),
                "details": sorted_langs,
                "primary": list(sorted_langs.keys())[0] if sorted_langs else "Unknown"
            }

        except Exception as e:
            log.error(f"Error detecting languages: {e}")
            return {"detected": [], "details": {}, "primary": "Unknown"}

    def _get_files_stats(self, path: Path) -> Dict[str, Any]:
        """إحصائيات الملفات"""
        stats = {
            "total_size": 0,
            "total_files": 0,
            "largest_files": [],
            "file_types": defaultdict(int)
        }

        try:
            files_info = []

            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                if any(ignored in file.parts for ignored in self.ignore_dirs):
                    continue
                if file.name in self.ignore_files:
                    continue

                try:
                    size = file.stat().st_size
                    stats["total_size"] += size
                    stats["total_files"] += 1
                    stats["file_types"][file.suffix or "no_ext"] += 1

                    files_info.append({
                        "path": str(file.relative_to(path)),
                        "size": size,
                        "size_str": self._format_size(size)
                    })
                except:
                    pass

            # أكبر 10 ملفات
            stats["largest_files"] = sorted(files_info,
                                          key=lambda x: x["size"],
                                          reverse=True)[:10]

            stats["total_size_str"] = self._format_size(stats["total_size"])

            return stats

        except Exception as e:
            log.error(f"Error getting files stats: {e}")
            return stats

    def _detect_config_files(self, path: Path) -> List[Dict[str, str]]:
        """كشف ملفات التكوين الموجودة"""
        found_configs = []

        try:
            for config_name, config_type in self.config_files.items():
                config_path = path / config_name
                if config_path.exists():
                    found_configs.append({
                        "file": config_name,
                        "type": config_type,
                        "path": str(config_path.relative_to(path))
                    })

            return found_configs

        except Exception as e:
            log.error(f"Error detecting config files: {e}")
            return found_configs

    def _analyze_dependencies(self, path: Path) -> Dict[str, Any]:
        """تحليل التبعيات من ملفات التكوين"""
        dependencies = {
            "python": [],
            "node": [],
            "other": []
        }

        try:
            # Python - requirements.txt
            req_file = path / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r', encoding='utf-8') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    dependencies["python"] = deps[:20]  # أول 20 تبعية

            # Python - pyproject.toml
            pyproject_file = path / "pyproject.toml"
            if pyproject_file.exists():
                try:
                    with open(pyproject_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # استخراج بسيط للتبعيات
                        if '[tool.poetry.dependencies]' in content:
                            dependencies["python"].append("(Poetry dependencies found)")
                except:
                    pass

            # Node.js - package.json
            package_file = path / "package.json"
            if package_file.exists():
                try:
                    with open(package_file, 'r', encoding='utf-8') as f:
                        package_data = json.load(f)

                        deps = package_data.get("dependencies", {})
                        dev_deps = package_data.get("devDependencies", {})

                        dependencies["node"] = {
                            "dependencies": list(deps.keys())[:15],
                            "devDependencies": list(dev_deps.keys())[:15],
                            "total": len(deps) + len(dev_deps)
                        }
                except:
                    pass

            return dependencies

        except Exception as e:
            log.error(f"Error analyzing dependencies: {e}")
            return dependencies

    def _analyze_code_metrics(self, path: Path) -> Dict[str, Any]:
        """تحليل مقاييس الكود"""
        metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "avg_file_size": 0
        }

        try:
            code_files = []
            for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.go', '.rs']:
                code_files.extend(path.rglob(f"*{ext}"))

            # تصفية الملفات المستبعدة
            code_files = [f for f in code_files
                         if not any(ignored in f.parts for ignored in self.ignore_dirs)]

            if code_files:
                total_size = 0
                for file in code_files[:100]:  # تحليل أول 100 ملف فقط للسرعة
                    try:
                        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                metrics["total_lines"] += 1
                                stripped = line.strip()
                                if not stripped:
                                    metrics["blank_lines"] += 1
                                elif stripped.startswith('#') or stripped.startswith('//'):
                                    metrics["comment_lines"] += 1
                                else:
                                    metrics["code_lines"] += 1

                        total_size += file.stat().st_size
                    except:
                        pass

                metrics["avg_file_size"] = total_size // len(code_files) if code_files else 0
                metrics["files_analyzed"] = len(code_files[:100])
                metrics["total_code_files"] = len(code_files)

            return metrics

        except Exception as e:
            log.error(f"Error analyzing code metrics: {e}")
            return metrics

    def _detect_issues(self, path: Path) -> List[Dict[str, str]]:
        """كشف المشاكل المحتملة في المشروع"""
        issues = []

        try:
            # تحقق من وجود README
            if not (path / "README.md").exists() and not (path / "README").exists():
                issues.append({
                    "type": "documentation",
                    "severity": "medium",
                    "message": "لا يوجد ملف README - يفضل إضافة توثيق للمشروع"
                })

            # تحقق من .gitignore
            if not (path / ".gitignore").exists():
                issues.append({
                    "type": "version_control",
                    "severity": "low",
                    "message": "لا يوجد ملف .gitignore - قد يتم رفع ملفات غير مرغوبة"
                })

            # تحقق من وجود tests
            test_dirs = ['tests', 'test', '__tests__', 'spec']
            has_tests = any((path / test_dir).exists() for test_dir in test_dirs)
            if not has_tests:
                issues.append({
                    "type": "testing",
                    "severity": "medium",
                    "message": "لا توجد مجلدات للاختبارات - يفضل إضافة Unit Tests"
                })

            # تحقق من ملفات كبيرة جداً
            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                if any(ignored in file.parts for ignored in self.ignore_dirs):
                    continue

                try:
                    size = file.stat().st_size
                    # ملف أكبر من 5MB
                    if size > 5 * 1024 * 1024:
                        issues.append({
                            "type": "file_size",
                            "severity": "low",
                            "message": f"ملف كبير: {file.name} ({self._format_size(size)})"
                        })
                except:
                    pass

            return issues[:10]  # أول 10 مشاكل فقط

        except Exception as e:
            log.error(f"Error detecting issues: {e}")
            return issues

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """توليد توصيات بناءً على التحليل"""
        recommendations = []

        try:
            # توصيات بناءً على اللغات
            primary_lang = analysis["languages"].get("primary", "")

            if primary_lang == "Python":
                if not any(c["type"] == "Python/pip" for c in analysis["config_files"]):
                    recommendations.append("💡 يفضل إضافة ملف requirements.txt لتوثيق التبعيات")

                if not any(c["file"] == ".gitignore" for c in analysis["config_files"]):
                    recommendations.append("💡 أضف .gitignore لاستبعاد __pycache__ و .venv")

            elif primary_lang in ["JavaScript", "TypeScript"]:
                if not any(c["type"] == "Node.js/NPM" for c in analysis["config_files"]):
                    recommendations.append("💡 تأكد من وجود package.json بشكل صحيح")

            # توصيات بناءً على الحجم
            if analysis["files_stats"]["total_files"] > 1000:
                recommendations.append("📊 المشروع كبير - فكر في تقسيمه إلى modules منفصلة")

            # توصيات بناءً على المشاكل
            if len(analysis["potential_issues"]) > 5:
                recommendations.append("⚠️  يوجد عدة مشاكل محتملة - راجع قسم المشاكل")

            # توصيات عامة
            if analysis["structure"]["max_depth"] > 7:
                recommendations.append("🔄 البنية عميقة جداً - فكر في تسطيح Hierarchy")

            if not recommendations:
                recommendations.append("✅ المشروع يبدو منظماً بشكل جيد!")

            return recommendations

        except Exception as e:
            log.error(f"Error generating recommendations: {e}")
            return ["حدث خطأ في توليد التوصيات"]

    def _get_size(self, path: Path) -> int:
        """حساب الحجم الكلي لملف أو مجلد"""
        if path.is_file():
            return path.stat().st_size

        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except:
                        pass
        except:
            pass

        return total

    def _format_size(self, size: int) -> str:
        """تنسيق الحجم بشكل مقروء"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """توليد تقرير نصي شامل من نتائج التحليل"""
        if "error" in analysis:
            return f"❌ خطأ: {analysis['error']}"

        report_lines = []

        # العنوان
        report_lines.append("=" * 70)
        report_lines.append(f"📊 تقرير تحليل المشروع: {analysis['project_name']}")
        report_lines.append("=" * 70)
        report_lines.append("")

        # البنية
        struct = analysis["structure"]
        report_lines.append("📁 بنية المشروع:")
        report_lines.append(f"   • إجمالي المجلدات: {struct['total_dirs']}")
        report_lines.append(f"   • إجمالي الملفات: {struct['total_files']}")
        report_lines.append(f"   • أقصى عمق: {struct['max_depth']}")
        report_lines.append("")

        # اللغات
        langs = analysis["languages"]
        report_lines.append("💻 اللغات المكتشفة:")
        report_lines.append(f"   • اللغة الأساسية: {langs['primary']}")
        for lang, details in list(langs["details"].items())[:5]:
            report_lines.append(f"   • {lang}: {details['count']} ملف، {details['lines']:,} سطر")
        report_lines.append("")

        # الإحصائيات
        stats = analysis["files_stats"]
        report_lines.append("📈 الإحصائيات:")
        report_lines.append(f"   • الحجم الكلي: {stats['total_size_str']}")
        report_lines.append(f"   • عدد الملفات: {stats['total_files']}")
        report_lines.append("")

        # مقاييس الكود
        metrics = analysis["code_metrics"]
        if metrics["total_lines"] > 0:
            report_lines.append("📝 مقاييس الكود:")
            report_lines.append(f"   • إجمالي الأسطر: {metrics['total_lines']:,}")
            report_lines.append(f"   • أسطر الكود: {metrics['code_lines']:,}")
            report_lines.append(f"   • أسطر التعليقات: {metrics['comment_lines']:,}")
            report_lines.append(f"   • أسطر فارغة: {metrics['blank_lines']:,}")
            report_lines.append("")

        # التبعيات
        deps = analysis["dependencies"]
        if deps["python"] or deps["node"]:
            report_lines.append("📦 التبعيات:")
            if deps["python"]:
                report_lines.append(f"   • Python: {len(deps['python'])} تبعية")
            if deps["node"]:
                if isinstance(deps["node"], dict):
                    report_lines.append(f"   • Node.js: {deps['node'].get('total', 0)} تبعية")
            report_lines.append("")

        # المشاكل
        issues = analysis["potential_issues"]
        if issues:
            report_lines.append("⚠️  المشاكل المحتملة:")
            for issue in issues[:5]:
                report_lines.append(f"   • [{issue['severity']}] {issue['message']}")
            report_lines.append("")

        # التوصيات
        recommendations = analysis["recommendations"]
        if recommendations:
            report_lines.append("💡 التوصيات:")
            for rec in recommendations:
                report_lines.append(f"   {rec}")
            report_lines.append("")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)


# Instance عامة
project_analyzer = ProjectAnalyzer()


if __name__ == "__main__":
    # اختبار
    import sys

    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "."

    print(f"🔍 تحليل المشروع: {project_path}\n")

    analysis = project_analyzer.analyze_project(project_path)
    report = project_analyzer.generate_report(analysis)

    print(report)
