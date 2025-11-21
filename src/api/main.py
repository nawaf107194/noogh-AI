#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ Noogh Unified AI System - Main Entry Point
"""

import uvicorn
import logging
from src.api.app import create_app
from src.core.settings import settings

# Create the application instance
app = create_app()

if __name__ == "__main__":
    logging.info("🏛️ Starting Noogh Unified AI System...")
    logging.info("=" * 70)
    logging.info(f"📡 API Server: http://{settings.api_host}:{settings.api_port}")
    logging.info(f"📚 API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logging.info(f"🔧 Debug Mode: {settings.debug_mode}")
    logging.info(f"💾 Database: {settings.database_url}")
    logging.info(f"🎮 GPU Enabled: {settings.use_gpu}")
    logging.info("=" * 70)

    try:
        uvicorn.run(
            "src.api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower(),
            reload=settings.debug_mode  # Auto-reload in debug mode
        )
    except Exception as e:
        logging.critical(f"❌ Failed to start server: {e}", exc_info=True)
        raise
