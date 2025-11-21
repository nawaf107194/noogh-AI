#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ Noogh Unified AI System - Main Entry Point
"""

import uvicorn
import logging
from src.api.app import create_app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    logging.info("🏛️ Starting Noogh Unified AI System...")
    logging.info("=" * 70)
    logging.info("📡 API Server: http://0.0.0.0:8000")
    logging.info("📚 API Docs: http://0.0.0.0:8000/docs")
    logging.info("=" * 70)

    try:
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except Exception as e:
        logging.critical(f"❌ Failed to start server: {e}", exc_info=True)
        raise
