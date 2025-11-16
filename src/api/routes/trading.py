#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقاط نهاية التداول
Trading Endpoints

يحتوي على endpoints إدارة التداول والاستراتيجيات
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

# استخدام BASE_DIR من config بدلاً من hardcoded path
from config import BASE_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ..models import (
    APIResponse, TradingSignal, TradingStrategy, PortfolioStatus
)
from ..auth import get_current_user, require_permission, Permission, User
from api.utils.logger import get_logger, LogCategory
from api.utils.error_handler_factory import get_error_handler
from api.utils.error_handler import NooghError, ErrorCategory, ErrorSeverity

router = APIRouter()
logger = get_logger("trading_api")
error_handler = get_error_handler()

@router.get("/signals", response_model=APIResponse)
async def get_trading_signals(
    symbol: Optional[str] = None,
    limit: int = 20,
    current_user: User = Depends(require_permission(Permission.TRADING_READ))
):
    """الحصول على إشارات التداول"""
    
    try:
        signals = []
        
        # قراءة الإشارات من ملف البيانات
        signals_file = BASE_DIR / "data" / "live_signals.csv"
        
        if signals_file.exists():
            df = pd.read_csv(signals_file)
            
            # تطبيق المرشح حسب الرمز
            if symbol:
                df = df[df['symbol'] == symbol]
            
            # أحدث الإشارات
            df = df.tail(limit)
            
            for _, row in df.iterrows():
                signal = TradingSignal(
                    symbol=row.get('symbol', 'UNKNOWN'),
                    signal_type=row.get('signal', 'HOLD'),
                    confidence=float(row.get('confidence', 0.5)),
                    price=float(row.get('price', 0.0)),
                    timestamp=row.get('timestamp', datetime.now().isoformat()),
                    indicators={}  # يمكن إضافة المؤشرات من الأعمدة الأخرى
                )
                signals.append(signal)
        
        logger.info(f"تم جلب إشارات التداول بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب إشارات التداول بنجاح",
            data={
                "signals": [signal.dict() for signal in signals],
                "total": len(signals),
                "symbol_filter": symbol
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_trading_signals',
            'user': current_user.username,
            'symbol': symbol
        })
        raise HTTPException(status_code=500, detail="فشل في جلب إشارات التداول")

@router.post("/signals/generate", response_model=APIResponse)
async def generate_trading_signal(
    symbol: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.TRADING_WRITE))
):
    """توليد إشارة تداول جديدة"""
    
    try:
        signal_id = f"signal_{int(datetime.now().timestamp())}"
        
        logger.info(f"بدء توليد إشارة تداول لـ {symbol} بواسطة {current_user.username}", LogCategory.TRADING)
        
        # إضافة مهمة توليد الإشارة في الخلفية
        background_tasks.add_task(_generate_signal_task, symbol, signal_id)
        
        return APIResponse(
            success=True,
            message="تم بدء توليد إشارة التداول",
            data={
                "signal_id": signal_id,
                "symbol": symbol,
                "status": "generating"
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'generate_trading_signal',
            'user': current_user.username,
            'symbol': symbol
        })
        raise HTTPException(status_code=500, detail="فشل في توليد إشارة التداول")

async def _generate_signal_task(symbol: str, signal_id: str):
    """مهمة توليد إشارة التداول"""
    
    try:
        # هنا يمكن إضافة منطق توليد الإشارة الفعلي
        # محاكاة التحليل
        import random
        import asyncio
        
        await asyncio.sleep(2)  # محاكاة وقت المعالجة
        
        # توليد إشارة عشوائية للمحاكاة
        signals = ['BUY', 'SELL', 'HOLD']
        signal_type = random.choice(signals)
        confidence = random.uniform(0.6, 0.95)
        price = random.uniform(100, 1000)
        
        # حفظ الإشارة
        signal_data = {
            'signal_id': signal_id,
            'symbol': symbol,
            'signal': signal_type,
            'confidence': confidence,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        
        # إضافة إلى ملف الإشارات
        signals_file = BASE_DIR / "data" / "live_signals.csv"
        
        if signals_file.exists():
            df = pd.read_csv(signals_file)
            new_row = pd.DataFrame([signal_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([signal_data])
        
        df.to_csv(signals_file, index=False)
        
        logger.info(f"تم توليد إشارة {signal_type} لـ {symbol}", LogCategory.TRADING)
        
    except Exception as e:
        logger.error(f"فشل في توليد الإشارة {signal_id}: {e}", LogCategory.TRADING)

@router.get("/strategies", response_model=APIResponse)
async def get_trading_strategies(current_user: User = Depends(require_permission(Permission.TRADING_READ))):
    """الحصول على استراتيجيات التداول"""
    
    try:
        strategies = []
        
        # قراءة الاستراتيجيات من ملف التكوين
        strategies_file = BASE_DIR / "data" / "trading_strategies.json"
        
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                strategies_data = json.load(f)
            
            for strategy_data in strategies_data:
                strategy = TradingStrategy(
                    strategy_id=strategy_data.get('id', 'unknown'),
                    name=strategy_data.get('name', 'Unknown Strategy'),
                    description=strategy_data.get('description', ''),
                    parameters=strategy_data.get('parameters', {}),
                    risk_level=strategy_data.get('risk_level', 'medium'),
                    expected_return=strategy_data.get('expected_return', 0.0)
                )
                strategies.append(strategy)
        else:
            # إنشاء استراتيجيات افتراضية
            default_strategies = [
                {
                    "id": "moving_average",
                    "name": "استراتيجية المتوسط المتحرك",
                    "description": "استراتيجية تعتمد على تقاطع المتوسطات المتحركة",
                    "parameters": {"short_period": 10, "long_period": 30},
                    "risk_level": "low",
                    "expected_return": 0.08
                },
                {
                    "id": "rsi_strategy",
                    "name": "استراتيجية مؤشر القوة النسبية",
                    "description": "استراتيجية تعتمد على مؤشر RSI",
                    "parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70},
                    "risk_level": "medium",
                    "expected_return": 0.12
                }
            ]
            
            for strategy_data in default_strategies:
                strategy = TradingStrategy(**strategy_data)
                strategies.append(strategy)
        
        logger.info(f"تم جلب استراتيجيات التداول بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب استراتيجيات التداول بنجاح",
            data={
                "strategies": [strategy.dict() for strategy in strategies],
                "total": len(strategies)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_trading_strategies',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب استراتيجيات التداول")

@router.post("/strategies", response_model=APIResponse)
async def create_trading_strategy(
    strategy: TradingStrategy,
    current_user: User = Depends(require_permission(Permission.TRADING_WRITE))
):
    """إنشاء استراتيجية تداول جديدة"""
    
    try:
        strategies_file = BASE_DIR / "data" / "trading_strategies.json"
        
        # قراءة الاستراتيجيات الموجودة
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                strategies_data = json.load(f)
        else:
            strategies_data = []
        
        # إضافة الاستراتيجية الجديدة
        new_strategy = strategy.dict()
        new_strategy['created_by'] = current_user.username
        new_strategy['created_at'] = datetime.now(timezone.utc).isoformat()
        
        strategies_data.append(new_strategy)
        
        # حفظ الاستراتيجيات
        strategies_file.parent.mkdir(parents=True, exist_ok=True)
        with open(strategies_file, 'w', encoding='utf-8') as f:
            json.dump(strategies_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم إنشاء استراتيجية تداول {strategy.name} بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم إنشاء استراتيجية التداول بنجاح",
            data=new_strategy,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'create_trading_strategy',
            'user': current_user.username,
            'strategy_id': strategy.strategy_id
        })
        raise HTTPException(status_code=500, detail="فشل في إنشاء استراتيجية التداول")

@router.get("/portfolio", response_model=APIResponse)
async def get_portfolio_status(current_user: User = Depends(require_permission(Permission.TRADING_READ))):
    """الحصول على حالة المحفظة"""
    
    try:
        # قراءة بيانات المحفظة
        portfolio_file = BASE_DIR / "data" / "portfolio.json"
        
        if portfolio_file.exists():
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                portfolio_data = json.load(f)
        else:
            # إنشاء محفظة افتراضية
            portfolio_data = {
                "total_value": 10000.0,
                "available_balance": 5000.0,
                "invested_amount": 5000.0,
                "profit_loss": 0.0,
                "profit_loss_percent": 0.0,
                "positions": []
            }
        
        portfolio_status = PortfolioStatus(**portfolio_data)
        
        logger.info(f"تم جلب حالة المحفظة بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب حالة المحفظة بنجاح",
            data=portfolio_status.dict(),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_portfolio_status',
            'user': current_user.username
        })
        raise HTTPException(status_code=500, detail="فشل في جلب حالة المحفظة")

@router.post("/execute", response_model=APIResponse)
async def execute_trade(
    symbol: str,
    action: str,  # BUY, SELL
    quantity: float,
    price: Optional[float] = None,
    current_user: User = Depends(require_permission(Permission.TRADING_EXECUTE))
):
    """تنفيذ صفقة تداول"""
    
    try:
        if action not in ['BUY', 'SELL']:
            raise HTTPException(status_code=400, detail="نوع الصفقة يجب أن يكون BUY أو SELL")
        
        if quantity <= 0:
            raise HTTPException(status_code=400, detail="الكمية يجب أن تكون أكبر من صفر")
        
        # محاكاة تنفيذ الصفقة
        trade_id = f"trade_{int(datetime.now().timestamp())}"
        execution_price = price or 100.0  # سعر افتراضي للمحاكاة
        
        trade_data = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": execution_price,
            "total_value": quantity * execution_price,
            "executed_by": current_user.username,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "status": "executed"
        }
        
        # حفظ الصفقة في سجل التداول
        trades_file = BASE_DIR / "data" / "trades_log.json"
        
        if trades_file.exists():
            with open(trades_file, 'r', encoding='utf-8') as f:
                trades_data = json.load(f)
        else:
            trades_data = []
        
        trades_data.append(trade_data)
        
        trades_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trades_file, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم تنفيذ صفقة {action} {quantity} {symbol} بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم تنفيذ الصفقة بنجاح",
            data=trade_data,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'execute_trade',
            'user': current_user.username,
            'symbol': symbol,
            'action': action,
            'quantity': quantity
        })
        raise HTTPException(status_code=500, detail="فشل في تنفيذ الصفقة")

@router.get("/history", response_model=APIResponse)
async def get_trading_history(
    symbol: Optional[str] = None,
    days: int = 30,
    limit: int = 100,
    current_user: User = Depends(require_permission(Permission.TRADING_READ))
):
    """الحصول على تاريخ التداول"""
    
    try:
        trades = []
        
        # قراءة سجل التداول
        trades_file = BASE_DIR / "data" / "trades_log.json"
        
        if trades_file.exists():
            with open(trades_file, 'r', encoding='utf-8') as f:
                trades_data = json.load(f)
            
            # تطبيق المرشحات
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for trade in trades_data:
                trade_date = datetime.fromisoformat(trade['executed_at'].replace('Z', '+00:00'))
                
                # فلترة حسب التاريخ
                if trade_date < cutoff_date:
                    continue
                
                # فلترة حسب الرمز
                if symbol and trade['symbol'] != symbol:
                    continue
                
                trades.append(trade)
            
            # ترتيب حسب التاريخ (الأحدث أولاً)
            trades.sort(key=lambda x: x['executed_at'], reverse=True)
            
            # تطبيق الحد الأقصى
            trades = trades[:limit]
        
        logger.info(f"تم جلب تاريخ التداول بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب تاريخ التداول بنجاح",
            data={
                "trades": trades,
                "total": len(trades),
                "symbol_filter": symbol,
                "days_filter": days
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_trading_history',
            'user': current_user.username,
            'symbol': symbol,
            'days': days
        })
        raise HTTPException(status_code=500, detail="فشل في جلب تاريخ التداول")

@router.get("/market-data/{symbol}", response_model=APIResponse)
async def get_market_data(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
    current_user: User = Depends(require_permission(Permission.TRADING_READ))
):
    """الحصول على بيانات السوق"""
    
    try:
        market_data = []
        
        # قراءة بيانات السوق
        market_file = BASE_DIR / "data" / "market_data.csv"
        
        if market_file.exists():
            df = pd.read_csv(market_file)
            
            # فلترة حسب الرمز إذا كان متاحاً
            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol]
            
            # أحدث البيانات
            df = df.tail(limit)
            
            market_data = df.to_dict('records')
        
        logger.info(f"تم جلب بيانات السوق لـ {symbol} بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب بيانات السوق بنجاح",
            data={
                "symbol": symbol,
                "interval": interval,
                "data": market_data,
                "count": len(market_data)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_market_data',
            'user': current_user.username,
            'symbol': symbol
        })
        raise HTTPException(status_code=500, detail="فشل في جلب بيانات السوق")

@router.get("/performance", response_model=APIResponse)
async def get_trading_performance(
    days: int = 30,
    current_user: User = Depends(require_permission(Permission.TRADING_READ))
):
    """الحصول على أداء التداول"""
    
    try:
        performance_data = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_profit_loss": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
        
        # قراءة سجل التداول
        trades_file = BASE_DIR / "data" / "trades_log.json"
        
        if trades_file.exists():
            with open(trades_file, 'r', encoding='utf-8') as f:
                trades_data = json.load(f)
            
            # فلترة حسب التاريخ
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade in trades_data:
                trade_date = datetime.fromisoformat(trade['executed_at'].replace('Z', '+00:00'))
                if trade_date >= cutoff_date:
                    recent_trades.append(trade)
            
            # حساب الأداء
            if recent_trades:
                performance_data["total_trades"] = len(recent_trades)
                
                # هنا يمكن إضافة حسابات أداء أكثر تفصيلاً
                # للمحاكاة، سنستخدم قيم عشوائية
                import random
                performance_data["winning_trades"] = int(len(recent_trades) * 0.6)
                performance_data["losing_trades"] = len(recent_trades) - performance_data["winning_trades"]
                performance_data["win_rate"] = performance_data["winning_trades"] / len(recent_trades)
                performance_data["total_profit_loss"] = random.uniform(-1000, 2000)
        
        logger.info(f"تم جلب أداء التداول بواسطة {current_user.username}", LogCategory.TRADING)
        
        return APIResponse(
            success=True,
            message="تم جلب أداء التداول بنجاح",
            data=performance_data,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        error_handler.handle_error(e, {
            'endpoint': 'get_trading_performance',
            'user': current_user.username,
            'days': days
        })
        raise HTTPException(status_code=500, detail="فشل في جلب أداء التداول")
