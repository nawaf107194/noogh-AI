#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Logger
"""

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from enum import Enum
import sys
# Add parent to path
PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
from .logger import get_logger, LogCategory

def setup_global_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a rotating file handler
    file_handler = RotatingFileHandler('logs/noogh.log', maxBytes=1024*1024*5, backupCount=5)
    file_handler.setFormatter(formatter)

    # Create a timed rotating file handler
    time_handler = TimedRotatingFileHandler('logs/noogh_timed.log', when='midnight', interval=1, backupCount=7)
    time_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(time_handler)
    logger.addHandler(console_handler)

    return logger
