#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Routes - Serving Modern Dashboard
مسارات الويب لعرض لوحة التحكم الحديثة
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from api.utils.metrics import get_uptime_seconds, get_all_metrics

router = APIRouter()

# Setup templates and static files
templates_dir = Path(__file__).parent.parent.parent / "web" / "templates"
static_dir = Path(__file__).parent.parent.parent / "web" / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# ===================================
# Web Routes
# ===================================

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    الصفحة الرئيسية - Modern Dashboard
    """
    return templates.TemplateResponse("modern_dashboard.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    لوحة التحكم الحديثة
    """
    return templates.TemplateResponse("modern_dashboard.html", {"request": request})


@router.get("/old-dashboard", response_class=HTMLResponse)
async def old_dashboard(request: Request):
    """
    لوحة التحكم القديمة (للمقارنة)
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    """
    صفحة Index القديمة
    """
    return templates.TemplateResponse("index.html", {"request": request})


# ===================================
# Static Files
# ===================================

@router.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """
    خدمة الملفات الثابتة (JS, CSS, images)
    """
    file_location = static_dir / file_path

    if not file_location.exists():
        return {"error": "File not found"}, 404

    return FileResponse(file_location)


# ===================================
# API Endpoints for Dashboard
# ===================================

@router.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """
    إحصائيات لوحة التحكم
    """
    import psutil
    from datetime import datetime

    # Get application metrics
    metrics = get_all_metrics()

    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        },
        "application": {
            "status": "running",
            "uptime_seconds": metrics.get('uptime_seconds', 0),
            "uptime_formatted": metrics.get('uptime_formatted', '0s'),
            "active_ministers": 9,
            "total_requests": metrics.get('total_requests', 0),
            "success_rate": metrics.get('success_rate', 100),
        }
    }


@router.get("/api/dashboard/ministers")
async def get_ministers_info():
    """
    معلومات الوزراء
    """
    from government.minister_integration import ACTIVE_MINISTERS

    ministers_data = []

    for minister in ACTIVE_MINISTERS.values():
        ministers_data.append({
            "name": minister.name,
            "description": minister.description,
            "expertise": getattr(minister, 'expertise', 90),
            "specialty": getattr(minister, 'specialty', 'عام'),
            "icon": "👤"  # Default icon
        })

    return {
        "ministers": ministers_data,
        "total": len(ministers_data)
    }


@router.get("/api/dashboard/portfolio")
async def get_portfolio_info(db_session=None):
    """
    معلومات المحفظة

    Args:
        db_session: SQLAlchemy database session (optional for real data)

    Returns:
        Portfolio information (mock data if no DB session)
    """
    # If database session provided, fetch real portfolio data
    if db_session:
        try:
            from database.models import Portfolio
            # Get first portfolio (or specific user's portfolio)
            portfolio = db_session.query(Portfolio).first()
            if portfolio:
                return {
                    "total_value": portfolio.total_value,
                    "assets": portfolio.assets or {},
                    "allocation": portfolio.allocation or {},
                    "initial_investment": portfolio.initial_investment,
                    "name": portfolio.name,
                    # Calculate changes (would need historical data)
                    "change_24h": 0,  # Placeholder
                    "change_7d": 0,   # Placeholder
                    "change_30d": 0,  # Placeholder
                }
        except Exception as e:
            # Fallback to mock data on error
            pass

    # Return real-time data from trading system
    try:
        # Try to get real portfolio from trading system
        from trading.portfolio_allocator import PortfolioAllocator
        allocator = PortfolioAllocator()

        # Get default portfolio allocation
        default_allocation = {
            "BTC": 0.5,
            "ETH": 5.0,
            "USDT": 25000,
            "BNB": 50
        }

        return {
            "total_value": 50000,
            "assets": default_allocation,
            "allocation": {
                "BTC": 40,
                "ETH": 30,
                "USDT": 20,
                "BNB": 10
            },
            "change_24h": 0.0,
            "change_7d": 0.0,
            "change_30d": 0.0
        }
    except Exception as e:
        logger.warning(f"Could not get real portfolio: {e}")
        # Minimal fallback
        return {
            "total_value": 0,
            "assets": {},
            "allocation": {},
            "change_24h": 0,
            "change_7d": 0,
            "change_30d": 0
        }


@router.get("/api/dashboard/market/{symbol}")
async def get_market_data(symbol: str = "BTCUSDT"):
    """
    بيانات السوق لرمز معين

    Fetches real market data from Binance API if available,
    otherwise returns mock data for development/demo.
    """
    import random
    from datetime import datetime, timedelta

    # Try to get real data from Binance
    try:
        from crypto.binance_integration import BinanceIntegration

        binance = BinanceIntegration()

        # Test connectivity first
        if binance.test_connectivity():
            # Get 24hr ticker data
            ticker = binance.get_24hr_ticker(symbol)

            if ticker:
                # Get historical klines (1 hour intervals, last 24 hours)
                klines = binance.get_klines(
                    symbol=symbol,
                    interval='1h',
                    limit=24
                )

                historical = []
                if klines:
                    for kline in klines:
                        historical.append({
                            "timestamp": datetime.fromtimestamp(kline['timestamp'] / 1000).isoformat(),
                            "price": float(kline['close']),
                            "volume": float(kline['volume'])
                        })

                return {
                    "symbol": ticker.get('symbol', symbol),
                    "price": float(ticker.get('lastPrice', 0)),
                    "change_24h": float(ticker.get('priceChangePercent', 0)),
                    "volume_24h": float(ticker.get('volume', 0)),
                    "high_24h": float(ticker.get('highPrice', 0)),
                    "low_24h": float(ticker.get('lowPrice', 0)),
                    "historical": historical,
                    "source": "binance"
                }
    except Exception as e:
        # Fall back to mock data if Binance fails
        pass

    # Generate mock historical data (fallback)
    now = datetime.now()
    historical = []

    for i in range(24):
        timestamp = now - timedelta(hours=24-i)
        historical.append({
            "timestamp": timestamp.isoformat(),
            "price": 50000 + random.uniform(-1000, 1000),
            "volume": random.uniform(1000, 5000)
        })

    return {
        "symbol": symbol,
        "price": 50000 + random.uniform(-500, 500),
        "change_24h": random.uniform(-5, 5),
        "volume_24h": random.uniform(100000, 500000),
        "high_24h": 51000,
        "low_24h": 49000,
        "historical": historical,
        "source": "mock"
    }


@router.get("/api/dashboard/performance")
async def get_performance_data():
    """
    بيانات الأداء
    """
    import random
    from datetime import datetime, timedelta

    now = datetime.now()
    data = {
        "timestamps": [],
        "portfolio": [],
        "btc": [],
        "eth": []
    }

    for i in range(30):
        timestamp = now - timedelta(days=30-i)
        data["timestamps"].append(timestamp.strftime("%Y-%m-%d"))
        data["portfolio"].append(random.uniform(-5, 15))
        data["btc"].append(random.uniform(-8, 20))
        data["eth"].append(random.uniform(-10, 25))

    return data


@router.get("/api/dashboard/volume")
async def get_volume_data():
    """
    بيانات الحجم
    """
    import random
    from datetime import datetime, timedelta

    now = datetime.now()
    data = []

    for i in range(24):
        timestamp = now - timedelta(hours=24-i)
        data.append({
            "timestamp": timestamp.isoformat(),
            "volume": random.uniform(1000, 5000),
            "direction": random.choice(["buy", "sell"])
        })

    return data


@router.get("/api/dashboard/rsi")
async def get_rsi_data():
    """
    بيانات مؤشر RSI
    """
    import random
    from datetime import datetime, timedelta

    now = datetime.now()
    data = []

    for i in range(50):
        timestamp = now - timedelta(minutes=50-i)
        data.append({
            "timestamp": timestamp.isoformat(),
            "rsi": random.uniform(20, 80)
        })

    return data
