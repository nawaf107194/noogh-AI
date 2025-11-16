#!/bin/bash
# Cloudflare Tunnel Management Script
# سكريبت إدارة نفق Cloudflare

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TUNNEL_CONFIG="$HOME/.cloudflared/noogh-api-config.yml"
TUNNEL_ID="fd213a4a-6275-44e0-a4d4-721a0b542bf3"
LOG_FILE="$PROJECT_ROOT/logs/cloudflare_tunnel.log"
PID_FILE="$PROJECT_ROOT/logs/cloudflare_tunnel.pid"

# Functions
start_tunnel() {
    echo -e "${BLUE}🚀 Starting Cloudflare Tunnel...${NC}"

    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Tunnel is already running (PID: $(cat "$PID_FILE"))${NC}"
        return 0
    fi

    cloudflared tunnel --config "$TUNNEL_CONFIG" run "$TUNNEL_ID" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 3

    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo -e "${GREEN}✅ Tunnel started successfully (PID: $(cat "$PID_FILE"))${NC}"
    else
        echo -e "${RED}❌ Failed to start tunnel${NC}"
        tail -20 "$LOG_FILE"
        return 1
    fi
}

stop_tunnel() {
    echo -e "${YELLOW}🛑 Stopping Cloudflare Tunnel...${NC}"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            sleep 2
            echo -e "${GREEN}✅ Tunnel stopped${NC}"
            rm -f "$PID_FILE"
        else
            echo -e "${YELLOW}⚠️  Tunnel was not running${NC}"
            rm -f "$PID_FILE"
        fi
    else
        # Try to kill by process name
        pkill -f "cloudflared tunnel.*$TUNNEL_ID" 2>/dev/null && echo -e "${GREEN}✅ Tunnel stopped${NC}" || echo -e "${YELLOW}⚠️  No tunnel process found${NC}"
    fi
}

restart_tunnel() {
    echo -e "${BLUE}🔄 Restarting Cloudflare Tunnel...${NC}"
    stop_tunnel
    sleep 2
    start_tunnel
}

status_tunnel() {
    echo -e "${BLUE}📊 Cloudflare Tunnel Status${NC}"
    echo ""

    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}✅ Status: Running${NC}"
        echo -e "${BLUE}   PID: $PID${NC}"
        echo -e "${BLUE}   Config: $TUNNEL_CONFIG${NC}"
        echo -e "${BLUE}   Tunnel ID: $TUNNEL_ID${NC}"
        echo ""
        echo -e "${YELLOW}Recent logs:${NC}"
        tail -10 "$LOG_FILE" | grep "Registered tunnel connection" || tail -5 "$LOG_FILE"
    else
        echo -e "${RED}❌ Status: Not Running${NC}"
    fi
}

show_logs() {
    echo -e "${BLUE}📋 Tunnel Logs (last 30 lines)${NC}"
    echo ""
    tail -30 "$LOG_FILE"
}

validate_config() {
    echo -e "${BLUE}🔍 Validating tunnel configuration...${NC}"
    cloudflared tunnel validate "$TUNNEL_CONFIG" 2>&1 || true
    echo -e "${GREEN}✅ Configuration file: $TUNNEL_CONFIG${NC}"
}

# Main
case "${1:-}" in
    start)
        start_tunnel
        ;;
    stop)
        stop_tunnel
        ;;
    restart)
        restart_tunnel
        ;;
    status)
        status_tunnel
        ;;
    logs)
        show_logs
        ;;
    validate)
        validate_config
        ;;
    *)
        echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║  Cloudflare Tunnel Management Script          ║${NC}"
        echo -e "${BLUE}║  سكريبت إدارة نفق Cloudflare                  ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|validate}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the tunnel"
        echo "  stop      - Stop the tunnel"
        echo "  restart   - Restart the tunnel"
        echo "  status    - Show tunnel status"
        echo "  logs      - Show recent logs"
        echo "  validate  - Validate configuration"
        echo ""
        exit 1
        ;;
esac
