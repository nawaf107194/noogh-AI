#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ Noogh Unified AI System - Main API
نظام نوغ الموحد للذكاء الاصطناعي - الواجهة الرئيسية

Version: 1.0.0
Features:
- 🎩 Government System (14 Ministers + President)
- 💰 Trading & Crypto Prediction
- 📁 Smart File Management
- 🧠 Neural Brain (326 neurons)
- 📊 Analytics & Monitoring
- 🚀 GPU-Accelerated AI Tools
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # src/
SYSTEM_ROOT = PROJECT_ROOT.parent  # noogh_unified_system/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SYSTEM_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
from datetime import datetime
import os

# Import routers - load all available
routers_loaded = {}

router_imports = {
    'system': 'api.routes.system',
    'monitoring': 'api.routes.monitoring',
    'trading': 'api.routes.trading',
    'data': 'api.routes.data',
    'web': 'api.routes.web',
    'websocket': 'api.routes.websocket',
    'models': 'api.routes.models',
    'plugin': 'api.routes.plugin',
    'training': 'api.routes.training',
    'gpu': 'api.routes.gpu',
    'chat': 'api.routes.chat',
    'knowledge': 'api.routes.knowledge',
    'unified_v4_1': 'api.routes.unified_v4_1',
    'autonomous': 'api.routes.autonomy',  # Corrected path
    'government': 'api.routes.government',
    'cognition_stream': 'api.routes.cognition_stream',
    'training_auto': 'api.routes.training_auto',
    'feedback': 'api.routes.feedback',
    'finance_minister_dashboard': 'api.routes.finance_minister_dashboard',
    'education': 'api.routes.education',
    'security': 'api.routes.security',
    'development': 'api.routes.development',
    'communication': 'api.routes.communication',
    'brain': 'api.routes.brain',
    'audit': 'api.routes.audit_routes',
}

for name, module_path in router_imports.items():
    try:
        module = __import__(module_path, fromlist=['router'])
        routers_loaded[name] = module.router
        logging.info(f"✅ Loaded {name} router")
    except ImportError as e:
        logging.warning(f"⚠️  Could not import {name} router: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error loading {name} router: {e}", exc_info=True)

# Load unified settings
try:
    from config.settings import get_settings
    settings = get_settings()
    logger_configured = False
except ImportError:
    settings = None
    logger_configured = False

# Configure advanced logging
try:
    from api.utils.advanced_logger import setup_global_logging
    logger = setup_global_logging()
    logger_configured = True
except ImportError:
    # Fallback to basic logging
    log_level = settings.log_level if settings else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("noogh_unified")
    logger_configured = True

if settings:
    logger.info(f"✅ Loaded settings (Environment: {settings.environment})")

# Create FastAPI app
app = FastAPI(
    title="Noogh Unified AI System",
    description="""
    🏛️ نظام نوغ الموحد للذكاء الاصطناعي

    ## Features:
    - 🎩 Government System with 14 Ministers
    - 💰 Trading & Crypto Analysis
    - 📁 Smart File Operations
    - 🧠 Advanced Neural Brain
    - 📊 Real-time Monitoring
    - 🚀 GPU-Accelerated AI Tools
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register centralized error handlers
try:
    from api.utils.error_handlers import register_error_handlers
    register_error_handlers(app)
    logger.info("✅ Registered centralized error handlers")
except ImportError as e:
    logger.warning(f"⚠️  Error handlers not available: {e}")

# CORS Middleware - Secure configuration
# Get allowed origins from settings or environment
import os
if settings and settings.cors_origins:
    allowed_origins = [origin.strip() for origin in settings.cors_origins.split(',')]
else:
    allowed_origins_str = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080,https://nooogh.com')
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Register available routers
router_prefixes = {
    'system': ('/system', ['System']),
    'monitoring': ('', ['📊 Monitoring', 'Health Checks', 'Self-Monitor', 'Alerts', 'System Status']),
    'trading': ('/trading', ['Trading']),
    'data': ('/data', ['Data']),
    'web': ('/web', ['Web']),
    'websocket': ('/ws', ['WebSocket']),
    'models': ('/models', ['Models']),
    'plugin': ('/plugin', ['Plugin', 'GPT Analysis']),
    'training': ('/training', ['Training', 'AI Training']),
    'gpu': ('/gpu', ['GPU Tools', 'AI Acceleration']),
    'allam': ('/api/allam', ['🔥 ALLaM', 'Arabic LLM', 'علّام']),
    'chat': ('/chat', ['💬 Chat', 'ALLaM', 'Neural Brain v3.0']),
    'brain': ('/brain', ['🧠 Brain', 'Neural Network', 'Cognitive Core']),
    'knowledge': ('/knowledge', ['🔍 Knowledge', 'Search', 'Retrieval']),
    'unified_v4_1': ('/v4.1', ['✨ Unified AI v4.1', 'Intent Routing', 'Learning', 'Web Search']),
    'autonomous': ('', ['🤖 Autonomous System', 'VRAM Monitor', 'Load Balancer', 'Smart Training']),
    'audit': ('/audit', ['🧠 Self-Audit', 'Consciousness Level', 'Deep Audit']),
    'government': ('', ['🏛️ Government', 'Deep Cognition', '97.5% TRANSCENDENT', '14 Ministers']),
    'chat_simple': ('', ['💬 Simple Chat', 'Government AI', 'نوغ']),
    'cognition_stream': ('', ['🧠 Live Cognition', 'Real-time Monitoring', 'WebSocket', 'Ministers Activity']),
    'training_auto': ('', ['⚙️ Training Auto-Rotation', 'Model Versioning', 'Scheduled Training', 'Auto-Deploy']),
    'feedback': ('', ['🔄 Self-Feedback Loop', 'Minister Feedback', 'Brain Adjustments', 'Self-Learning']),
    'finance_minister_dashboard': ('/finance_minister_dashboard', ['💰 Finance Minister Dashboard', 'KPIs', 'Performance', 'Fusion Stats']),
    'development': ('/development', ['🎨 Development Minister', 'Code Quality', 'CI/CD', 'Code Generation', 'Refactoring']),
    'communication': ('/communication', ['📡 Communication Minister', 'Translation', 'Conversation', 'Reports', 'WebSocket']),
}

for name, (prefix, tags) in router_prefixes.items():
    if name in routers_loaded:
        app.include_router(routers_loaded[name], prefix=prefix, tags=tags)
        logger.info(f"✅ Registered {name} router at {prefix}")

logger.info(f"📊 {len(routers_loaded)}/{len(router_prefixes)} routers loaded successfully")

# Mount React Dashboard static files
DASHBOARD_DIST = Path(__file__).parent / "dashboard" / "dist"
if DASHBOARD_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DASHBOARD_DIST / "assets")), name="dashboard_assets")
    logger.info(f"✅ Mounted dashboard assets from {DASHBOARD_DIST}")
else:
    logger.warning(f"⚠️  Dashboard not found at {DASHBOARD_DIST}")

# GPT Plugin manifest endpoint
@app.get("/.well-known/ai-plugin.json")
async def get_plugin_manifest():
    """GPT Plugin manifest for ChatGPT integration"""
    return {
        "schema_version": "v1",
        "name_for_model": "noogh_unified_system",
        "name_for_human": "Noogh AI System",
        "description_for_model": "نظام ذكاء اصطناعي موحد يحتوي على: نظام حكومي مع 14 وزير، دماغ عصبي بـ 326 عصبون، تداول العملات الرقمية، إدارة الملفات الذكية، أدوات GPU. استخدمه للحصول على معلومات النظام، حالة الوزراء، التنبؤات، تحليل البيانات، وإدارة الملفات.",
        "description_for_human": "نظام ذكاء اصطناعي شامل مع نظام حكومي، تداول كريبتو، دماغ عصبي، وأدوات GPU",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": "https://plugin.nooogh.com/openapi.json"
        },
        "logo_url": "https://plugin.nooogh.com/static/logo.png",
        "contact_email": "dev@nooogh.com",
        "legal_info_url": "https://plugin.nooogh.com/legal"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """فحص صحة النظام"""
    return {
        "status": "OK",
        "system": "Noogh Unified AI System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "government": "✅ 14 Ministers + President",
            "brain": "✅ 326 Neurons",
            "crypto": "✅ Trading & Prediction",
            "files": "✅ Smart File Manager",
            "gpu": "✅ GPU-Accelerated Tools",
            "api": "✅ FastAPI"
        }
    }


# Root endpoint - Serve React Dashboard
@app.get("/")
async def root():
    """الصفحة الرئيسية - Dashboard"""
    dashboard_index = DASHBOARD_DIST / "index.html"
    if dashboard_index.exists():
        return FileResponse(dashboard_index)
    else:
        # Fallback to JSON if dashboard not built
        return {
            "message": "🏛️ Welcome to Noogh Unified AI System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "government": {
                "ministers": 14,
                "sectors": [
                    "📚 Knowledge (21%)",
                    "🔒 Security (14%)",
                    "🎨 Development (14%)",
                    "📊 Analysis (14%)",
                    "🤖 AI Core (14%)",
                    "💬 Communication (14%)",
                    "💰 Finance (7%)"
                ]
            }
        }

# Catch-all route for React Router (SPA support)
# This MUST be defined after all API routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Catch-all route to serve React SPA for all non-API routes.
    This enables React Router to handle client-side routing.
    """
    # Skip if it's an API endpoint or special path
    if full_path.startswith(('api/', 'docs', 'redoc', 'openapi.json', 'health',
                             'system/', 'chat/', 'brain/', 'trading/', 'data/',
                             'web/', 'ws/', 'models/', 'plugin/', 'training/',
                             'gpu/', 'knowledge/', 'unified/', 'v4.1/', 'simple-qa/',
                             'audit/', 'government/', 'finance_minister_dashboard/',
                             'development/', 'communication/', 'education/', 'security/',
                             '.well-known/', 'assets/')):
        raise HTTPException(status_code=404, detail="Not found")

    # Serve index.html for all other paths (React Router will handle them)
    dashboard_index = DASHBOARD_DIST / "index.html"
    if dashboard_index.exists():
        return FileResponse(dashboard_index)
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🏛️ Starting Noogh Unified AI System...")
    logger.info("=" * 70)
    logger.info("📡 API Server: http://0.0.0.0:8000")
    logger.info("📚 API Docs: http://0.0.0.0:8000/docs")
    logger.info("🏛️ Government: http://0.0.0.0:8000/government/status")
    logger.info("=" * 70)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except Exception as e:
        logger.critical(f"❌ Failed to start server: {e}", exc_info=True)
        raise
