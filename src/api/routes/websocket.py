#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket Routes - Real-time Communication
مسارات WebSocket للتواصل الفوري
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ===================================
# Connection Manager
# ===================================

class ConnectionManager:
    """مدير الاتصالات المتعددة"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            'market': [],
            'portfolio': [],
            'ministers': [],
            'system': [],
            'chat': []
        }
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, channel: str = 'general'):
        """قبول اتصال جديد"""
        await websocket.accept()

        if channel not in self.active_connections:
            self.active_connections[channel] = []

        self.active_connections[channel].append(websocket)
        self.connection_metadata[websocket] = {
            'channel': channel,
            'connected_at': datetime.now(),
            'message_count': 0
        }

        logger.info(f"✅ New connection to channel: {channel}")

        # Send welcome message
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'channel': channel,
            'timestamp': datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """قطع الاتصال"""
        metadata = self.connection_metadata.get(websocket, {})
        channel = metadata.get('channel', 'general')

        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        logger.info(f"❌ Disconnected from channel: {channel}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """إرسال رسالة شخصية"""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1
        except Exception as e:
            logger.error(f"❌ Error sending personal message: {e}")

    async def broadcast_to_channel(self, message: Dict[str, Any], channel: str):
        """بث رسالة لجميع المتصلين بقناة معينة"""
        if channel not in self.active_connections:
            return

        disconnected = []

        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]['message_count'] += 1
            except Exception as e:
                logger.error(f"❌ Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """بث رسالة لجميع المتصلين"""
        for channel in self.active_connections.keys():
            await self.broadcast_to_channel(message, channel)

    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات الاتصالات"""
        return {
            'total_connections': sum(len(conns) for conns in self.active_connections.values()),
            'channels': {
                channel: len(conns)
                for channel, conns in self.active_connections.items()
            },
            'total_messages': sum(
                meta.get('message_count', 0)
                for meta in self.connection_metadata.values()
            )
        }


# Global connection manager
manager = ConnectionManager()

# ===================================
# WebSocket Endpoints
# ===================================

@router.websocket("/ws/market")
async def websocket_market(websocket: WebSocket):
    """
    WebSocket للأسعار المباشرة
    يرسل تحديثات الأسعار كل ثانية
    """
    await manager.connect(websocket, 'market')

    try:
        while True:
            # Simulate market data (replace with real Binance stream)
            import random

            market_data = {
                'type': 'price_update',
                'symbol': 'BTCUSDT',
                'price': 50000 + random.uniform(-500, 500),
                'volume': random.uniform(1000, 5000),
                'change_24h': random.uniform(-5, 5),
                'timestamp': datetime.now().isoformat()
            }

            await manager.send_personal_message(market_data, websocket)

            # Check for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                client_msg = json.loads(data)

                # Handle client requests (e.g., change symbol)
                if client_msg.get('action') == 'subscribe':
                    symbol = client_msg.get('symbol', 'BTCUSDT')
                    await manager.send_personal_message({
                        'type': 'subscription',
                        'status': 'success',
                        'symbol': symbol
                    }, websocket)

            except asyncio.TimeoutError:
                pass  # No message from client, continue

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ Market WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/portfolio")
async def websocket_portfolio(websocket: WebSocket):
    """
    WebSocket لتحديثات المحفظة
    يرسل تحديثات عند تغيير المحفظة
    """
    await manager.connect(websocket, 'portfolio')

    try:
        while True:
            # Wait for portfolio updates or client messages
            data = await websocket.receive_text()
            client_msg = json.loads(data)

            if client_msg.get('action') == 'get_portfolio':
                # Simulate portfolio data
                portfolio_data = {
                    'type': 'portfolio_update',
                    'total_value': 100000,
                    'assets': {
                        'BTC': 1.5,
                        'ETH': 10,
                        'USDT': 50000
                    },
                    'change_24h': 2.5,
                    'timestamp': datetime.now().isoformat()
                }

                await manager.send_personal_message(portfolio_data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ Portfolio WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/ministers")
async def websocket_ministers(websocket: WebSocket):
    """
    WebSocket لتحديثات الوزراء
    يرسل إشعارات عند استشارة الوزراء
    """
    await manager.connect(websocket, 'ministers')

    try:
        while True:
            data = await websocket.receive_text()
            client_msg = json.loads(data)

            if client_msg.get('action') == 'consult':
                minister = client_msg.get('minister', 'Prime Minister')
                query = client_msg.get('query', '')

                # Simulate minister consultation
                response = {
                    'type': 'minister_response',
                    'minister': minister,
                    'query': query,
                    'response': f'رد من {minister}: تحليل مفصل للسؤال...',
                    'confidence': 0.95,
                    'timestamp': datetime.now().isoformat()
                }

                await manager.send_personal_message(response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ Ministers WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/system")
async def websocket_system(websocket: WebSocket):
    """
    WebSocket لحالة النظام
    يرسل تحديثات موارد النظام كل 5 ثوانٍ
    """
    await manager.connect(websocket, 'system')

    try:
        import psutil

        while True:
            system_data = {
                'type': 'system_update',
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_connections': manager.get_stats()['total_connections'],
                'timestamp': datetime.now().isoformat()
            }

            await manager.send_personal_message(system_data, websocket)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ System WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket للدردشة التفاعلية
    يدعم الرسائل الثنائية الاتجاه
    """
    await manager.connect(websocket, 'chat')

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Process message (integrate with ministers here)
            user_message = message.get('message', '')

            # Simulate AI response
            ai_response = {
                'type': 'chat_response',
                'message': f'رد على: {user_message}',
                'sender': 'Noogh AI',
                'ministers_consulted': ['Prime Minister', 'Portfolio Minister'],
                'confidence': 0.92,
                'timestamp': datetime.now().isoformat()
            }

            await manager.send_personal_message(ai_response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ Chat WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    الحصول على إحصائيات WebSocket
    """
    return {
        'success': True,
        'stats': manager.get_stats(),
        'timestamp': datetime.now().isoformat()
    }


# ===================================
# Background Tasks for Broadcasting
# ===================================

async def broadcast_market_updates():
    """بث تحديثات السوق الدورية"""
    import random

    while True:
        market_update = {
            'type': 'market_broadcast',
            'btc_price': 50000 + random.uniform(-500, 500),
            'eth_price': 3000 + random.uniform(-100, 100),
            'timestamp': datetime.now().isoformat()
        }

        await manager.broadcast_to_channel(market_update, 'market')
        await asyncio.sleep(2)


async def broadcast_system_alerts():
    """بث تنبيهات النظام"""
    while True:
        import psutil
        cpu = psutil.cpu_percent(interval=1)

        if cpu > 80:
            alert = {
                'type': 'system_alert',
                'severity': 'warning',
                'message': f'استخدام CPU مرتفع: {cpu}%',
                'timestamp': datetime.now().isoformat()
            }

            await manager.broadcast_to_all(alert)

        await asyncio.sleep(30)


# Start background tasks (call from main.py)
async def start_background_tasks():
    """بدء المهام الخلفية"""
    asyncio.create_task(broadcast_market_updates())
    asyncio.create_task(broadcast_system_alerts())
