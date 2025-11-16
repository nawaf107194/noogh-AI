#!/bin/bash
# Noogh Unified System - Development Server Startup Script
# نظام نوغ الموحد - سكريبت تشغيل الخادم

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🏛️  Noogh Unified AI System                       ║${NC}"
echo -e "${BLUE}║         نظام نوغ الموحد للذكاء الاصطناعي                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${YELLOW}📂 Project Root: ${PROJECT_ROOT}${NC}"
echo ""

# Load environment variables
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Loading .env file...${NC}"
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}⚠️  No .env file found, using defaults${NC}"
    export NOOGH_ENV=development
    export NOOGH_API_KEYS_JSON='{"dev-test-key":{"permissions":["system:*"],"name":"Development Key"}}'
    export CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p logs data models backups

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
    source venv/bin/activate
fi

# Check PyTorch installation
echo -e "${YELLOW}🔍 Checking PyTorch installation...${NC}"
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✅ PyTorch ${TORCH_VERSION} installed${NC}"
else
    echo -e "${YELLOW}⚠️  PyTorch not found, installing...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Check for running instances
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}⚠️  Port 8000 is already in use!${NC}"
    echo -e "${YELLOW}Do you want to kill the existing process? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        PID=$(lsof -ti:8000)
        kill $PID
        echo -e "${GREEN}✅ Killed process $PID${NC}"
        sleep 2
    else
        echo -e "${RED}❌ Exiting...${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🚀 Starting Noogh Unified AI System...                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📡 API Server:     http://localhost:8000${NC}"
echo -e "${BLUE}📚 Documentation:  http://localhost:8000/docs${NC}"
echo -e "${BLUE}🏛️  Government:     http://localhost:8000/government/status${NC}"
echo -e "${BLUE}💚 Health Check:   http://localhost:8000/health${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
python src/api/main.py
