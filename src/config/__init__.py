#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# المشروع الجذر
# src/config/__init__.py -> src/config -> src -> BASE_DIR
BASE_DIR = Path(__file__).resolve().parents[2]

# مسارات مفيدة
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = SRC_DIR / "config"

__all__ = ["BASE_DIR", "SRC_DIR", "CONFIG_DIR"]
