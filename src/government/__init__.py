"""
Noogh Government System - Main Entry Point
"""

# Main system facade
from .noogh_government_system import NooghGovernmentSystem

# Core components
from .president import President
from .base_minister import BaseMinister, MinisterReport, TaskStatus, Priority
from .minister_types_universal import MinisterType, MinisterCategory, TaskCategory

# Active Ministers
from .communication_minister import CommunicationMinister
from .development_minister import DevelopmentMinister
from .education_minister import EducationMinister
from .security_minister import SecurityMinister


__all__ = [
    # Main System
    "NooghGovernmentSystem",

    # Core
    "President",
    "BaseMinister",
    "MinisterType",
    "MinisterCategory",
    "TaskCategory",
    "MinisterReport",
    "TaskStatus",
    "Priority",

    # Ministers
    "CommunicationMinister",
    "DevelopmentMinister",
    "EducationMinister",
    "SecurityMinister",
]
