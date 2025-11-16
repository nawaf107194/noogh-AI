#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Development Minister API Routes
API endpoints for Development Minister v2.0

Features:
- Code Quality Scanning
- CI/CD Pipeline Management
- Code Generation
- Refactoring Tools
- Documentation Generation
- Dependency Management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

router = APIRouter()

# Singleton instance
_development_minister = None


def get_development_minister():
    """Get or create Development Minister instance"""
    global _development_minister
    if _development_minister is None:
        from government.development_minister import DevelopmentMinister
        _development_minister = DevelopmentMinister(verbose=True, auto_fix_enabled=False)
    return _development_minister


def generate_task_id() -> str:
    """Generate unique task ID"""
    return f"dev_{uuid.uuid4().hex[:12]}"


# ==================== Request Models ====================

class CodeScanRequest(BaseModel):
    """Request model for code quality scanning"""
    path: Optional[str] = Field(None, description="Path to scan (file or directory)")
    file_pattern: str = Field("*.py", description="File pattern to match")
    check_types: List[str] = Field(["all"], description="Check types: all, syntax, complexity, docs, security")


class PipelineRunRequest(BaseModel):
    """Request model for CI/CD pipeline"""
    project_name: str = Field(..., description="Project name")
    branch: str = Field("main", description="Git branch")
    stages: List[str] = Field(["build", "test", "deploy"], description="Pipeline stages")
    pipeline_id: Optional[str] = None


class CodeGenerationRequest(BaseModel):
    """Request model for code generation"""
    type: str = Field(..., description="Code type: function, class, test, api_endpoint")
    name: str = Field(..., description="Name of the code element")
    description: str = Field(..., description="Description of what the code should do")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    file_path: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class RefactorRequest(BaseModel):
    """Request model for code refactoring"""
    file_path: str = Field(..., description="File to refactor")
    type: str = Field("extract_function", description="Refactoring type")


class DocumentationRequest(BaseModel):
    """Request model for documentation generation"""
    path: str = Field(..., description="Path to document")
    format: str = Field("markdown", description="Output format: markdown, html, pdf")


class DependencyUpdateRequest(BaseModel):
    """Request model for dependency updates"""
    package: str = Field(..., description="Package name")
    version: str = Field("latest", description="Target version")


class PatchRequest(BaseModel):
    """Request model for applying patches"""
    type: str = Field("security", description="Patch type")
    file: str = Field(..., description="Target file")


class OptimizationRequest(BaseModel):
    """Request model for code optimization"""
    file_path: str = Field(..., description="File to optimize")
    type: str = Field("performance", description="Optimization type: performance, memory, speed")


# ==================== API Endpoints ====================

@router.post("/scan/code-quality")
async def scan_code_quality(request: CodeScanRequest):
    """
    فحص جودة الكود

    Scans code for:
    - Syntax errors
    - Complexity issues
    - Missing documentation
    - Security vulnerabilities

    Returns quality score (0-100) and detailed issues.
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="scan_code_quality",
        task_data={
            "path": request.path,
            "file_pattern": request.file_pattern,
            "check_types": request.check_types
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/quality/summary")
async def get_quality_summary():
    """
    ملخص جودة الكود

    Returns overall code quality metrics and statistics.
    """
    minister = get_development_minister()
    summary = minister.get_code_quality_summary()

    return {
        "success": True,
        "summary": summary
    }


@router.get("/quality/score")
async def get_quality_score():
    """
    درجة جودة الكود (0-100)

    Returns:
    - Quality score
    - Quality level (excellent, good, fair, poor, critical)
    - Grade (A, B, C, D, F)
    """
    minister = get_development_minister()
    score = minister.code_metrics.code_quality_score

    # Determine quality level
    from government.development_minister import CodeQualityLevel
    level = minister._get_quality_level(score)

    # Determine grade
    if score >= 90:
        grade, status = "A", "Excellent"
    elif score >= 75:
        grade, status = "B", "Good"
    elif score >= 60:
        grade, status = "C", "Fair"
    elif score >= 40:
        grade, status = "D", "Poor"
    else:
        grade, status = "F", "Critical"

    return {
        "quality_score": round(score, 2),
        "quality_level": level.value,
        "grade": grade,
        "status": status,
        "total_files": minister.code_metrics.total_files,
        "total_issues": len(minister.code_issues),
        "critical_issues": minister.code_metrics.critical_issues
    }


@router.post("/pipeline/run")
async def run_pipeline(request: PipelineRunRequest):
    """
    تشغيل خط أنابيب CI/CD

    Runs automated pipeline with stages:
    - build: Compile and build project
    - test: Run tests
    - package: Create deployment packages
    - deploy: Deploy to servers
    - verify: Verify deployment
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="run_pipeline",
        task_data={
            "pipeline_id": request.pipeline_id,
            "project_name": request.project_name,
            "branch": request.branch,
            "stages": request.stages
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/pipeline/active")
async def get_active_pipelines():
    """
    خطوط الأنابيب النشطة

    Returns list of all active/running pipelines.
    """
    minister = get_development_minister()
    pipelines = minister.get_active_pipelines()

    return {
        "success": True,
        "pipelines": pipelines
    }


@router.get("/pipeline/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """
    حالة خط أنابيب محدد

    Returns detailed pipeline status and logs.
    """
    minister = get_development_minister()

    if pipeline_id not in minister.active_pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    pipeline = minister.active_pipelines[pipeline_id]

    return {
        "success": True,
        "pipeline": pipeline.to_dict()
    }


@router.post("/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """
    توليد كود

    Generates code from templates:
    - function: Python function
    - class: Python class
    - test: Unit test
    - api_endpoint: FastAPI endpoint
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="generate_code",
        task_data={
            "type": request.type,
            "name": request.name,
            "description": request.description,
            "parameters": request.parameters,
            "file_path": request.file_path,
            "dependencies": request.dependencies
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/code/generated")
async def list_generated_code():
    """
    قائمة الكود المولد

    Returns list of all generated code.
    """
    minister = get_development_minister()

    generated = [code.to_dict() for code in minister.generated_code]

    return {
        "success": True,
        "total": len(generated),
        "generated_code": generated
    }


@router.post("/code/refactor")
async def refactor_code(request: RefactorRequest):
    """
    إعادة هيكلة الكود

    Refactoring operations:
    - extract_function: Extract code to new function
    - rename: Rename variables/functions
    - inline: Inline function calls
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="refactor_code",
        task_data={
            "file_path": request.file_path,
            "type": request.type
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/docs/generate")
async def generate_documentation(request: DocumentationRequest):
    """
    توليد التوثيق

    Generates documentation in multiple formats:
    - markdown
    - html
    - pdf
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="generate_docs",
        task_data={
            "path": request.path,
            "format": request.format
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/dependencies/update")
async def update_dependency(request: DependencyUpdateRequest):
    """
    تحديث التبعيات

    Updates project dependencies to specified versions.
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="update_dependencies",
        task_data={
            "package": request.package,
            "version": request.version
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/dependencies/list")
async def list_dependency_updates():
    """
    قائمة تحديثات التبعيات

    Returns all pending dependency updates.
    """
    minister = get_development_minister()

    return {
        "success": True,
        "total": len(minister.dependency_updates),
        "updates": minister.dependency_updates
    }


@router.post("/patch/apply")
async def apply_patch(request: PatchRequest):
    """
    تطبيق إصلاح

    Applies patches:
    - security: Security fixes
    - bugfix: Bug fixes
    - hotfix: Critical hotfixes
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="apply_patch",
        task_data={
            "type": request.type,
            "file": request.file
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.post("/optimize")
async def optimize_code(request: OptimizationRequest):
    """
    تحسين الكود

    Optimizations:
    - performance: Speed improvements
    - memory: Memory usage reduction
    - speed: Execution time reduction
    """
    minister = get_development_minister()

    task_id = generate_task_id()
    report = await minister.execute_task(
        task_id=task_id,
        task_type="optimize_code",
        task_data={
            "file_path": request.file_path,
            "type": request.type
        }
    )

    return {
        "success": True,
        "task_id": task_id,
        "result": report.result
    }


@router.get("/stats")
async def get_statistics():
    """
    إحصائيات وزير التطوير

    Returns comprehensive development statistics.
    """
    minister = get_development_minister()

    return {
        "success": True,
        "statistics": minister.stats,
        "code_quality": {
            "score": round(minister.code_metrics.code_quality_score, 2),
            "total_issues": len(minister.code_issues),
            "critical_issues": minister.code_metrics.critical_issues
        },
        "activity": {
            "active_pipelines": len(minister.active_pipelines),
            "code_generated": len(minister.generated_code),
            "dependency_updates": len(minister.dependency_updates)
        }
    }


@router.get("/report")
async def get_minister_report():
    """
    تقرير وزير التطوير للرئيس

    Full report for President.
    """
    minister = get_development_minister()
    report = await minister.report_to_president()

    return {
        "success": True,
        "report": report
    }


@router.get("/health")
async def health_check():
    """
    فحص صحة وزير التطوير

    Health status check.
    """
    minister = get_development_minister()

    return {
        "status": "operational",
        "minister": minister.name,
        "authorities": minister.authorities,
        "config": {
            "auto_fix_enabled": minister.auto_fix_enabled,
            "max_complexity": minister.max_complexity,
            "min_documentation_coverage": minister.min_documentation_coverage
        },
        "timestamp": datetime.now().isoformat()
    }
