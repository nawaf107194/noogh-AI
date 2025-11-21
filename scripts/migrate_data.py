import sys
import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import SessionLocal, engine, Base
from src.core.models import SystemLog, Memory, AuditRecord, MinisterAction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_memories(session, jsonl_path):
    """Migrate brain_v4_memories.jsonl to Memory table"""
    if not os.path.exists(jsonl_path):
        logger.warning(f"Memory file not found: {jsonl_path}")
        return

    logger.info(f"Migrating memories from {jsonl_path}...")
    count = 0
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    memory = Memory(
                        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                        memory_type=data.get('type', 'general'),
                        content=data.get('content', ''),
                        embedding=data.get('embedding'),
                        importance=data.get('importance', 0.5),
                        context=data.get('context')
                    )
                    session.add(memory)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to parse memory line: {e}")
        
        session.commit()
        logger.info(f"✅ Migrated {count} memories")
    except Exception as e:
        logger.error(f"Error migrating memories: {e}")
        session.rollback()

def migrate_deep_cognition(session, db_path):
    """Migrate deep_cognition.db to AuditRecord table"""
    if not os.path.exists(db_path):
        logger.warning(f"DB not found: {db_path}")
        return

    logger.info(f"Migrating deep cognition from {db_path}...")
    count = 0
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'audit_scores' in tables:
            cursor.execute("SELECT * FROM audit_scores")
            for row in cursor.fetchall():
                audit = AuditRecord(
                    timestamp=datetime.now(), # Timestamp might be in audit_runs, simplified here
                    component=row['dimension'],
                    score=row['score_percentage'],
                    details={'audit_id': row['audit_id'], 'level': row['consciousness_level']},
                    recommendation=None
                )
                session.add(audit)
                count += 1
        
        conn.close()
        session.commit()
        logger.info(f"✅ Migrated {count} deep cognition records")
    except Exception as e:
        logger.error(f"Error migrating deep cognition: {e}")
        session.rollback()

def migrate_subsystem_intelligence(session, db_path):
    """Migrate subsystem_intelligence.db to AuditRecord table"""
    if not os.path.exists(db_path):
        logger.warning(f"DB not found: {db_path}")
        return

    logger.info(f"Migrating subsystem intelligence from {db_path}...")
    count = 0
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'subsystem_tests' in tables:
            cursor.execute("SELECT * FROM subsystem_tests")
            for row in cursor.fetchall():
                audit = AuditRecord(
                    timestamp=datetime.fromisoformat(row['timestamp']) if 'timestamp' in row.keys() else datetime.now(),
                    component=row['subsystem'],
                    score=row['score'],
                    details={
                        'test_id': row['test_id'],
                        'passed': row['passed'],
                        'failed': row['failed'],
                        'details': row['details'],
                        'intelligence_level': row['intelligence_level']
                    }
                )
                session.add(audit)
                count += 1
                
        conn.close()
        session.commit()
        logger.info(f"✅ Migrated {count} subsystem records")
    except Exception as e:
        logger.error(f"Error migrating subsystem intelligence: {e}")
        session.rollback()

def main():
    logger.info("🚀 Starting Data Migration...")
    
    # Initialize DB
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    
    try:
        # Paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'src', 'api', 'data')
        jsonl_path = os.path.join(project_root, 'data', 'brain_v4_memories.jsonl')
        
        # Migrate
        migrate_memories(session, jsonl_path)
        migrate_deep_cognition(session, os.path.join(data_dir, 'deep_cognition.db'))
        migrate_subsystem_intelligence(session, os.path.join(data_dir, 'subsystem_intelligence.db'))
        
        # Reflection and Self Audit could be added similarly if schemas were known/consistent
        
        logger.info("✨ Migration Complete!")
        
    finally:
        session.close()

if __name__ == "__main__":
    main()
