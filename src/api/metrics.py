"""
Prometheus Metrics Exporter for Noogh Unified System
"""
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
import psutil
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# System Info
system_info = Info('noogh_system', 'Noogh Unified System Information')
system_info.info({
    'version': '1.0.0',
    'python_version': os.sys.version.split()[0],
    'environment': os.getenv('ENVIRONMENT', 'production')
})

# Request metrics
request_count = Counter(
    'noogh_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'noogh_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# API-specific metrics
api_errors = Counter(
    'noogh_api_errors_total',
    'Total API errors',
    ['error_type']
)

active_requests = Gauge(
    'noogh_active_requests',
    'Number of active requests'
)

# Minister metrics
minister_tasks = Counter(
    'noogh_minister_tasks_total',
    'Total tasks processed by ministers',
    ['minister', 'status']
)

minister_duration = Histogram(
    'noogh_minister_task_duration_seconds',
    'Minister task processing duration',
    ['minister']
)

# Knowledge Kernel metrics
knowledge_queries = Counter(
    'noogh_knowledge_queries_total',
    'Total knowledge queries',
    ['intent_type']
)

knowledge_chunks = Gauge(
    'noogh_knowledge_chunks',
    'Number of knowledge chunks loaded'
)

# Brain metrics
brain_active = Gauge(
    'noogh_brain_active',
    'Whether brain is active (1) or not (0)'
)

allam_queries = Counter(
    'noogh_allam_queries_total',
    'Total ALLaM queries'
)

# System resources
cpu_usage = Gauge(
    'noogh_cpu_usage_percent',
    'CPU usage percentage'
)

memory_usage = Gauge(
    'noogh_memory_usage_bytes',
    'Memory usage in bytes'
)

disk_usage = Gauge(
    'noogh_disk_usage_percent',
    'Disk usage percentage'
)

# Optional modules
module_status = Gauge(
    'noogh_module_status',
    'Module status (1=active, 0=inactive)',
    ['module']
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics Collection Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def update_system_metrics():
    """Update system resource metrics"""
    try:
        # CPU
        cpu_usage.set(psutil.cpu_percent(interval=1))

        # Memory
        memory = psutil.virtual_memory()
        memory_usage.set(memory.used)

        # Disk
        disk = psutil.disk_usage('/')
        disk_usage.set(disk.percent)

    except Exception as e:
        print(f"Error updating system metrics: {e}")


def track_request(method: str, endpoint: str, status: int, duration: float):
    """Track HTTP request"""
    request_count.labels(method=method, endpoint=endpoint, status=status).inc()
    request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def track_minister_task(minister: str, status: str, duration: float):
    """Track minister task"""
    minister_tasks.labels(minister=minister, status=status).inc()
    minister_duration.labels(minister=minister).observe(duration)


def track_knowledge_query(intent_type: str):
    """Track knowledge query"""
    knowledge_queries.labels(intent_type=intent_type).inc()


def update_module_status(module: str, active: bool):
    """Update module status"""
    module_status.labels(module=module).set(1 if active else 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics Endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_metrics() -> Response:
    """Generate Prometheus metrics"""
    update_system_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
