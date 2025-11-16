import logging
from enum import Enum

class LogCategory(Enum):
    GENERAL = "general"
    API = "api"
    TRADING = "trading"
    BRAIN = "brain"
    SYSTEM = "system"
    MODEL = "model"
    DATA = "data"

def get_logger(name: str, category: LogCategory = LogCategory.GENERAL):
    return logging.getLogger(f"{category.value}.{name}")
