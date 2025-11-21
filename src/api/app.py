from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import os
from datetime import datetime
# Add project root to path
project_root = Path(__file__).parent.parent.parent
# Import core config
try:
    from src.core.config import API_HOST, API_PORT, LOG_LEVEL
except ImportError:
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    LOG_LEVEL = "INFO"

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("noogh_unified")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    إنشاء وتكوين تطبيق FastAPI
    
    Returns:
        Configured FastAPI application instance
    """
    # Initialize Dependency Injection Container
    try:
        from src.core.service_registry import register_all_services
        register_all_services()
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize DI container: {e}")
    
    # Create FastAPI instance
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

    # CORS Configuration
    origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    register_routers(app)

    # Mount Static Files
    mount_static_files(app)

    return app

def register_routers(app: FastAPI):
    """Register all API routers"""
    
    # Router definitions: (module_path, prefix, tags)
    routers = {
        'core': ('src.api.routes.core', '', ['Core']),
        'system': ('src.api.routes.system', '/system', ['System']),
        'dashboard': ('src.api.routes.dashboard', '', ['📊 Dashboard']),
        'system_metrics': ('src.api.routes.system_metrics', '', ['📈 Metrics']),
        'monitoring': ('src.api.routes.monitoring', '', ['📊 Monitoring']),
        'trading': ('src.api.routes.trading', '/trading', ['Trading']),
        'data': ('src.api.routes.data', '/data', ['Data']),
        'web': ('src.api.routes.web', '/web', ['Web']),
        'websocket': ('src.api.routes.websocket', '/ws', ['WebSocket']),
        'models': ('src.api.routes.models', '/models', ['Models']),
        'plugin': ('src.api.routes.plugin', '/plugin', ['Plugin']),
        'training': ('src.api.routes.training', '/training', ['Training']),
        'gpu': ('src.api.routes.gpu', '/gpu', ['GPU']),
        'chat': ('src.api.routes.chat', '/chat', ['💬 Chat']),
        'brain': ('src.api.routes.brain', '/brain', ['🧠 Brain']),
        'knowledge': ('src.api.routes.knowledge', '/knowledge', ['🔍 Knowledge']),
        'unified_v4_1': ('src.api.routes.unified_v4_1', '/v4.1', ['✨ Unified AI v4.1']),
        'autonomous': ('src.api.routes.autonomy', '', ['🤖 Autonomous']),
        'audit': ('src.api.routes.audit_routes', '/audit', ['🧠 Self-Audit']),
        'government': ('src.api.routes.government', '', ['🏛️ Government']),
        'cognition_stream': ('src.api.routes.cognition_stream', '', ['🧠 Live Cognition']),
        'training_auto': ('src.api.routes.training_auto', '', ['⚙️ Training Auto']),
        'feedback': ('src.api.routes.feedback', '', ['🔄 Feedback']),
        'finance_minister_dashboard': ('src.api.routes.finance_minister_dashboard', '/finance_minister_dashboard', ['💰 Finance']),
        'development': ('src.api.routes.development', '/development', ['🎨 Development']),
        'communication': ('src.api.routes.communication', '/communication', ['📡 Communication']),
        'education': ('src.api.routes.education', '', ['Education']),
        'security': ('src.api.routes.security', '', ['Security']),
        'users': ('src.api.routes.users', '/api/v1', ['users']),
        'government_v2': ('src.api.routes.government_v2', '/api/v1', ['🏛️ Government V2']),  # Modern government
    }

    for name, (module_path, prefix, tags) in routers.items():
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=['router'])
            if hasattr(module, 'router'):
                app.include_router(module.router, prefix=prefix, tags=tags)
                logger.info(f"✅ Registered {name} router")
            else:
                logger.warning(f"⚠️ Module {name} has no 'router' object")
        except ImportError as e:
            logger.warning(f"⚠️ Could not load {name} router: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading {name}: {e}")

def mount_static_files(app: FastAPI):
    """Mount static assets for dashboard"""
    dashboard_dist = Path(__file__).parent.parent.parent / "dashboard" / "dist"
    
    # Serve dashboard index at root
    @app.get("/")
    async def dashboard_index():
        """Serve dashboard index page"""
        index_file = dashboard_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Noogh AI System", "docs": "/docs", "health": "/health"}
    
    # Mount static assets
    if dashboard_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(dashboard_dist / "assets")), name="dashboard_assets")
        logger.info(f"✅ Mounted dashboard assets from {dashboard_dist}")
    else:
        logger.warning(f"⚠️ Dashboard assets not found at {dashboard_dist}")
