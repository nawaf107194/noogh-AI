#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🐕 Process Watchdog - Auto-restart crashed components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON="${VENV_DIR}/bin/python"
UVICORN="${VENV_DIR}/bin/uvicorn"

API_PORT="${API_PORT:-8000}"
MCP_PORT="${MCP_PORT:-8001}"
CHECK_INTERVAL="${WATCHDOG_INTERVAL:-60}"  # seconds
MAX_RESTARTS="${MAX_RESTARTS:-5}"
RESTART_WINDOW="${RESTART_WINDOW:-300}"  # 5 minutes

LOG_DIR="${PROJECT_ROOT}/logs"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"

mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Restart counters
declare -A restart_count
declare -A last_restart_time

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$WATCHDOG_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$WATCHDOG_LOG"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$WATCHDOG_LOG"
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

should_restart() {
    local component=$1
    local current_time=$(date +%s)

    # Initialize if not exists
    if [ -z "${restart_count[$component]}" ]; then
        restart_count[$component]=0
        last_restart_time[$component]=0
    fi

    # Check if we're in the restart window
    local time_diff=$((current_time - last_restart_time[$component]))

    if [ $time_diff -gt $RESTART_WINDOW ]; then
        # Outside window, reset counter
        restart_count[$component]=0
    fi

    # Check if we've hit max restarts
    if [ ${restart_count[$component]} -ge $MAX_RESTARTS ]; then
        log_error "$component has been restarted ${restart_count[$component]} times in ${RESTART_WINDOW}s - giving up"
        return 1
    fi

    return 0
}

restart_api() {
    log_warn "🔄 Attempting to restart API server..."

    if ! should_restart "api"; then
        return 1
    fi

    # Kill existing process
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    sleep 2

    # Start API server
    nohup bash -c "source '$VENV_DIR/bin/activate' && $UVICORN src.api.main:app --host 0.0.0.0 --port $API_PORT --workers 2" > "$LOG_DIR/api.log" 2>&1 &

    sleep 5

    if check_port "$API_PORT"; then
        restart_count["api"]=$((restart_count["api"] + 1))
        last_restart_time["api"]=$(date +%s)
        log_info "✅ API server restarted successfully (restart #${restart_count["api"]})"
        return 0
    else
        log_error "❌ Failed to restart API server"
        return 1
    fi
}

restart_mcp() {
    log_warn "🔄 Attempting to restart MCP server..."

    if ! should_restart "mcp"; then
        return 1
    fi

    # Kill existing process
    lsof -ti:$MCP_PORT | xargs kill -9 2>/dev/null || true
    sleep 2

    # Start MCP server
    nohup bash -c "source '$VENV_DIR/bin/activate' && export MCP_PORT=$MCP_PORT && $PYTHON scripts/mcp_server.py" > "$LOG_DIR/mcp.log" 2>&1 &

    sleep 5

    if check_port "$MCP_PORT"; then
        restart_count["mcp"]=$((restart_count["mcp"] + 1))
        last_restart_time["mcp"]=$(date +%s)
        log_info "✅ MCP server restarted successfully (restart #${restart_count["mcp"]})"
        return 0
    else
        log_error "❌ Failed to restart MCP server"
        return 1
    fi
}

check_and_restart() {
    # Check API
    if ! check_port "$API_PORT"; then
        log_error "🚨 API server is DOWN on port $API_PORT"
        restart_api
    else
        log_info "✅ API server is UP on port $API_PORT"
    fi

    # Check MCP
    if ! check_port "$MCP_PORT"; then
        log_error "🚨 MCP server is DOWN on port $MCP_PORT"
        restart_mcp
    else
        log_info "✅ MCP server is UP on port $MCP_PORT"
    fi
}

monitor() {
    log_info "🐕 Starting process watchdog..."
    log_info "   Check interval: ${CHECK_INTERVAL}s"
    log_info "   Max restarts: ${MAX_RESTARTS} in ${RESTART_WINDOW}s"
    log_info "   API Port: ${API_PORT}"
    log_info "   MCP Port: ${MCP_PORT}"

    while true; do
        check_and_restart
        sleep $CHECK_INTERVAL
    done
}

# Main
case "${1:-monitor}" in
    monitor)
        monitor
        ;;
    check)
        check_and_restart
        ;;
    *)
        echo "Usage: $0 {monitor|check}"
        exit 1
        ;;
esac
