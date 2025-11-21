from src.core.database import engine, Base
from src.core.models import SystemLog, Memory, AuditRecord, MinisterAction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize database tables"""
    logger.info("🔄 Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise

if __name__ == "__main__":
    init_db()
