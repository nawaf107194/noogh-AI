#!/bin/bash
# ==============================================================================
# 🦅 NOOGH SOVEREIGN SYSTEM - MASTER LAUNCH SCRIPT
# ==============================================================================
# Chief SRE: Production-Grade Launch Sequence
# Version: LIVE-1.1 (Network Unlocked)
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        🦅 NOOGH SOVEREIGN SYSTEM - LIVE OPERATION v5.0           ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ==============================================================================
# PATH CONFIGURATION (Explicit Virtual Environment)
# ==============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_CMD="./venv/bin/python"
STREAMLIT_CMD="./venv/bin/streamlit"

# Verify executables exist
if [ ! -f "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ ERROR: Python not found at $PYTHON_CMD${NC}"
    echo -e "${YELLOW}Run: python3 -m venv venv${NC}"
    exit 1
fi

if [ ! -f "$STREAMLIT_CMD" ]; then
    echo -e "${RED}❌ ERROR: Streamlit not found at $STREAMLIT_CMD${NC}"
    echo -e "${YELLOW}Run: ./venv/bin/pip install streamlit${NC}"
    exit 1
fi

# ==============================================================================
# DIRECTORY SETUP
# ==============================================================================
mkdir -p logs
mkdir -p data/charts

# Clear old PIDs
rm -f .pids/*
mkdir -p .pids

# ==============================================================================
# SIGNAL TRAPPING (Clean Shutdown)
# ==============================================================================
cleanup() {
    echo ""
    echo -e "${RED}🛑 INITIATING SOVEREIGN SHUTDOWN SEQUENCE...${NC}"
    
    # Kill all background processes
    if [ -f .pids/api.pid ]; then
        kill $(cat .pids/api.pid) 2>/dev/null || true
        rm -f .pids/api.pid
    fi
    
    if [ -f .pids/hunter.pid ]; then
        kill $(cat .pids/hunter.pid) 2>/dev/null || true
        rm -f .pids/hunter.pid
    fi
    
    # Streamlit runs in foreground, will be killed by Ctrl+C
    
    echo -e "${GREEN}✅ All systems halted. Sovereignty preserved.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==============================================================================
# PHASE 1: NEURAL CORE API (Background)
# ==============================================================================
echo -e "${YELLOW}[PHASE 1]${NC} Initializing Neural Core API..."
$PYTHON_CMD -m src.api.main > logs/core.log 2>&1 &
API_PID=$!
echo $API_PID > .pids/api.pid
echo -e "${GREEN}✓ API launched (PID: $API_PID)${NC}"

# ==============================================================================
# PHASE 2: WAIT FOR BRAIN LOAD (Critical!)
# ==============================================================================
echo -e "${YELLOW}[PHASE 2]${NC} Waiting for Meta-Llama-3-8B to load into VRAM..."
echo -e "${CYAN}   (This can take 15-30 seconds depending on your RTX 5070)${NC}"

# Wait 10 seconds for basic initialization
sleep 10

# Check if API is responding
MAX_RETRIES=12
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Neural Core is ONLINE${NC}"
        break
    fi
    echo -e "${CYAN}   ... still loading (attempt $((RETRY+1))/$MAX_RETRIES)${NC}"
    sleep 5
    RETRY=$((RETRY+1))
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo -e "${RED}⚠️  WARNING: API health check timed out. Proceeding anyway...${NC}"
fi

# ==============================================================================
# PHASE 3: AUTONOMOUS HUNTER (Background)
# ==============================================================================
echo -e "${YELLOW}[PHASE 3]${NC} Releasing the Autonomous Hunter..."
$PYTHON_CMD scripts/run_autonomous_hunter.py > logs/hunter.log 2>&1 &
HUNTER_PID=$!
echo $HUNTER_PID > .pids/hunter.pid
echo -e "${GREEN}✓ Hunter deployed (PID: $HUNTER_PID)${NC}"

sleep 2

# ==============================================================================
# PHASE 4: SOVEREIGN COMMAND CENTER (Foreground)
# ==============================================================================
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ SYSTEM FULLY OPERATIONAL - NETWORK UNLOCKED${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "   ${GREEN}●${NC} Dashboard:  ${CYAN}http://0.0.0.0:8501${NC} (Accessible via Network IP)"
echo -e "   ${GREEN}●${NC} API Docs:   ${CYAN}http://localhost:8000/docs${NC}"
echo -e "   ${GREEN}●${NC} Health:     ${CYAN}http://localhost:8000/health${NC}"
echo ""
echo -e "   ${YELLOW}📁 Logs:${NC}"
echo -e "      - Core API:  ${CYAN}tail -f logs/core.log${NC}"
echo -e "      - Hunter:    ${CYAN}tail -f logs/hunter.log${NC}"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Press Ctrl+C to initiate shutdown sequence${NC}"
echo ""

# Launch Dashboard (Foreground)
echo -e "${YELLOW}[PHASE 4]${NC} Launching Sovereign Command Center..."
$STREAMLIT_CMD run src/interface/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --theme.base "dark" \
    --server.headless true \
    --browser.gatherUsageStats false

# If we reach here, dashboard was closed
cleanup
