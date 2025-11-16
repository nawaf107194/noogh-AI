#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic Models for the Noogh API
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class APIResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the request was successful")
    message: str = Field(..., description="A message describing the result")
    data: Optional[Dict[str, Any]] = Field(None, description="The data payload of the response")
    timestamp: str = Field(..., description="The timestamp of the response")

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int

class SystemConfig(BaseModel):
    # This is likely just a wrapper for a dict
    config: Dict[str, Any]

class ConfigUpdate(BaseModel):
    section: str
    key: str
    value: Any

class BackupRequest(BaseModel):
    backup_type: str = "full"
    include_config: bool = True
    include_data: bool = True
    include_models: bool = True

class RestoreRequest(BaseModel):
    backup_file: str
    restore_config: bool = True
    restore_data: bool = True
    restore_models: bool = True

class SystemStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    active_users: int
    system_load: float
    memory_usage_percent: float
    disk_usage_percent: float

class TradingSignal(BaseModel):
    symbol: str
    signal_type: str
    confidence: float
    price: float
    timestamp: str
    indicators: Dict[str, Any]

class TradingStrategy(BaseModel):
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_level: str
    expected_return: float

class PortfolioStatus(BaseModel):
    total_value: float
    available_balance: float
    invested_amount: float
    profit_loss: float
    profit_loss_percent: float
    positions: List[Any]

class ModelInfo(BaseModel):
    model_id: str
    name: str
    type: str
    version: str
    size_mb: float
    accuracy: Optional[float]
    created_at: str
    status: str

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    training_time: float

class DataSource(BaseModel):
    source_id: str
    name: str
    type: str
    status: str
    last_update: str

class DataQuery(BaseModel):
    source: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    limit: int = 100
    offset: int = 0
