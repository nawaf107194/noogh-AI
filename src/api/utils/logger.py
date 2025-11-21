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

class CategoryAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        category = kwargs.pop('category', None)
        if category:
            if isinstance(category, LogCategory):
                cat_str = category.value
            else:
                cat_str = str(category)
            msg = f"[{cat_str.upper()}] {msg}"
        return msg, kwargs

def get_logger(name: str, category: LogCategory = LogCategory.GENERAL):
    logger = logging.getLogger(name)
    return CategoryAdapter(logger, {})
