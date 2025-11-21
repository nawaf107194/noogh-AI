from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from src.core.database import Base

class SystemLog(Base):
    """Unified system logs"""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(String, index=True)
    module = Column(String, index=True)
    message = Column(Text)
    metadata_json = Column(JSON, nullable=True)

class Memory(Base):
    """Unified memory storage (replacing brain_v4_memories.jsonl)"""
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    memory_type = Column(String, index=True)  # e.g., 'conversation', 'fact', 'experience'
    content = Column(Text)
    embedding = Column(JSON, nullable=True)  # Store vector embedding if needed
    importance = Column(Float, default=0.5)
    context = Column(JSON, nullable=True)

class AuditRecord(Base):
    """Deep Cognition Audit Records"""
    __tablename__ = "audit_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    component = Column(String, index=True)
    score = Column(Float)
    details = Column(JSON)
    recommendation = Column(Text, nullable=True)

class MinisterAction(Base):
    """Record of actions taken by Ministers"""
    __tablename__ = "minister_actions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    minister_name = Column(String, index=True)
    action_type = Column(String)
    description = Column(Text)
    result = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
