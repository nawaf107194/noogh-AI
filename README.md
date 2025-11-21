# 🧠 Noogh Unified AI System

**نظام نوح الموحد للذكاء الاصطناعي**

A production-ready, modular AI system with autonomous government, self-learning capabilities, and comprehensive API infrastructure.

## 🌟 Features

### Core Capabilities
- **🏛️ Government System**: 13 autonomous ministers managing different aspects of the system
- **🧠 Brain v4.0**: Advanced memory and cognition system with 3000+ memories
- **💾 Unified Database**: SQLAlchemy ORM with PostgreSQL/SQLite support
- **🔌 Dependency Injection**: Centralized service management for better testability
- **📊 Real-time Monitoring**: System metrics, performance tracking, and alerts
- **🔄 Auto-Training**: Automatic model versioning and rollback system
- **⚡ GPU Acceleration**: CUDA support for faster inference

### API Features
- **241 REST API Endpoints**: Comprehensive API coverage
- **🔐 Authentication**: JWT-based auth with role-based permissions
- **📡 WebSocket Support**: Real-time communication
- **📝 Auto-Documentation**: OpenAPI/Swagger UI at `/docs`
- **🌐 CORS Enabled**: Configurable cross-origin support

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Initialize database
make init-db

# Start server
make run
```

Access the system:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🏗️ Architecture

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Database**: SQLite/PostgreSQL
- **Caching**: Redis
- **AI/ML**: PyTorch, Transformers
- **DI Container**: Centralized service management

## 📚 Key Components

### DI Container
```python
from src.core.di import Container
cache = Container.resolve("cache_manager")
```

### Government Ministers
13 autonomous ministers managing different system aspects

### Database Models
- `Memory`: Brain memories (3000+ records)
- `SystemLog`: System logs
- `AuditRecord`: Audit trails
- `MinisterAction`: Minister actions

## 📊 System Status

- ✅ 241 API routes registered
- ✅ 13 government ministers active
- ✅ 9 services in DI Container
- ✅ 3000+ memories in database

## 🧪 Testing

```bash
# Run tests
make test

# Verify system
python verify_app.py
python tests/test_di_services.py
```

## 📝 License

Proprietary software. All rights reserved.

---

**Built with ❤️ by the Noogh AI Team**
