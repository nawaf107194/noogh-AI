#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Noogh Unified System - Production Run Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified script to run all components of the Noogh system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Default environment variables
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export MCP_PORT="${MCP_PORT:-8001}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-3000}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Data directories
export DATA_DIR="${PROJECT_ROOT}/data"
export MODELS_DIR="${PROJECT_ROOT}/models"
export LOGS_DIR="${PROJECT_ROOT}/logs"

# Create necessary directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$LOGS_DIR"

# Virtual environment
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON="${VENV_DIR}/bin/python"
UVICORN="${VENV_DIR}/bin/uvicorn"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_info "Installing dependencies..."
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install -r requirements.txt
        log_info "Virtual environment ready!"
    fi
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component Runners
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

run_tests() {
    log_section "🧪 Running Smoke Tests"
    check_venv
    "$PYTHON" tests/smoke_test.py
}

run_api() {
    log_section "🚀 Starting FastAPI Server"
    check_venv

    if check_port "$API_PORT"; then
        log_warn "Port $API_PORT is already in use"
        log_info "Stopping existing process..."
        lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    log_info "Starting API server on http://$API_HOST:$API_PORT"
    log_info "Logs: $LOGS_DIR/api.log"

    "$UVICORN" src.api.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --workers 2 \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --log-config logging.conf 2>&1 | tee "$LOGS_DIR/api.log"
}

run_mcp() {
    log_section "🔌 Starting MCP Server"
    check_venv

    if check_port "$MCP_PORT"; then
        log_warn "Port $MCP_PORT is already in use"
        log_info "Stopping existing process..."
        lsof -ti:$MCP_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    log_info "Starting MCP server on port $MCP_PORT"
    log_info "Logs: $LOGS_DIR/mcp.log"

    export MCP_PORT
    "$PYTHON" scripts/mcp_server.py 2>&1 | tee "$LOGS_DIR/mcp.log"
}

run_dashboard() {
    log_section "📊 Starting Dashboard"

    cd src/dashboard

    if [ ! -d "node_modules" ]; then
        log_info "Installing dashboard dependencies..."
        npm install
    fi

    if check_port "$DASHBOARD_PORT"; then
        log_warn "Port $DASHBOARD_PORT is already in use"
    fi

    log_info "Starting dashboard on http://localhost:$DASHBOARD_PORT"
    npm run dev -- --port "$DASHBOARD_PORT"
}

run_all() {
    log_section "🚀 Starting All Components"

    # Start API in background
    log_info "Starting API server in background..."
    nohup bash -c "source '$VENV_DIR/bin/activate' && $UVICORN src.api.main:app --host $API_HOST --port $API_PORT --workers 2" > "$LOGS_DIR/api.log" 2>&1 &
    API_PID=$!
    log_info "API server started (PID: $API_PID)"

    # Wait for API to be ready
    sleep 5

    # Start MCP in background
    log_info "Starting MCP server in background..."
    nohup bash -c "source '$VENV_DIR/bin/activate' && export MCP_PORT=$MCP_PORT && $PYTHON scripts/mcp_server.py" > "$LOGS_DIR/mcp.log" 2>&1 &
    MCP_PID=$!
    log_info "MCP server started (PID: $MCP_PID)"

    log_info ""
    log_info "All components started!"
    log_info "  - API Server: http://$API_HOST:$API_PORT"
    log_info "  - MCP Server: port $MCP_PORT"
    log_info "  - API Logs: $LOGS_DIR/api.log"
    log_info "  - MCP Logs: $LOGS_DIR/mcp.log"
    log_info ""
    log_info "To stop all components: ./run.sh stop"
}

stop_all() {
    log_section "🛑 Stopping All Components"

    # Stop API server
    if check_port "$API_PORT"; then
        log_info "Stopping API server on port $API_PORT..."
        lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    fi

    # Stop MCP server
    if check_port "$MCP_PORT"; then
        log_info "Stopping MCP server on port $MCP_PORT..."
        lsof -ti:$MCP_PORT | xargs kill -9 2>/dev/null || true
    fi

    # Stop Dashboard
    if check_port "$DASHBOARD_PORT"; then
        log_info "Stopping Dashboard on port $DASHBOARD_PORT..."
        lsof -ti:$DASHBOARD_PORT | xargs kill -9 2>/dev/null || true
    fi

    log_info "All components stopped!"
}

show_status() {
    log_section "📊 System Status"

    log_info "Component Status:"

    if check_port "$API_PORT"; then
        echo -e "  ${GREEN}✓${NC} API Server (port $API_PORT) - Running"
    else
        echo -e "  ${RED}✗${NC} API Server (port $API_PORT) - Stopped"
    fi

    if check_port "$MCP_PORT"; then
        echo -e "  ${GREEN}✓${NC} MCP Server (port $MCP_PORT) - Running"
    else
        echo -e "  ${RED}✗${NC} MCP Server (port $MCP_PORT) - Stopped"
    fi

    if check_port "$DASHBOARD_PORT"; then
        echo -e "  ${GREEN}✓${NC} Dashboard (port $DASHBOARD_PORT) - Running"
    else
        echo -e "  ${RED}✗${NC} Dashboard (port $DASHBOARD_PORT) - Stopped"
    fi

    echo ""
    log_info "Recent logs:"
    echo "  API: tail -f $LOGS_DIR/api.log"
    echo "  MCP: tail -f $LOGS_DIR/mcp.log"
}

show_help() {
    cat << EOF

🚀 Noogh Unified System - Run Script

Usage: ./run.sh [COMMAND]

Commands:
  test         Run smoke tests
  api          Run FastAPI server only
  mcp          Run MCP server only
  dashboard    Run Dashboard only
  all          Run all components in background
  stop         Stop all running components
  status       Show status of all components
  help         Show this help message

Environment Variables:
  API_HOST        API server host (default: 0.0.0.0)
  API_PORT        API server port (default: 8000)
  MCP_PORT        MCP server port (default: 8001)
  DASHBOARD_PORT  Dashboard port (default: 3000)
  LOG_LEVEL       Logging level (default: INFO)

Examples:
  ./run.sh test          # Run tests
  ./run.sh api           # Start API server
  ./run.sh all           # Start all components
  ./run.sh stop          # Stop everything
  ./run.sh status        # Check status

  API_PORT=9000 ./run.sh api    # Run API on custom port

EOF
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

main() {
    case "${1:-help}" in
        test)
            run_tests
            ;;
        api)
            run_api
            ;;
        mcp)
            run_mcp
            ;;
        dashboard)
            run_dashboard
            ;;
        all)
            run_all
            ;;
        stop)
            stop_all
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
