#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧪 Complete Endpoint Testing - Noogh Unified System
# Tests all Grafana-optimized and legacy endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_BASE="http://localhost:8000"
PASS=0
FAIL=0

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🧪 Testing Noogh Unified System - All Endpoints${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test function
test_endpoint() {
    local endpoint=$1
    local expected=$2
    local description=$3

    echo -n "Testing $description... "

    response=$(curl -s "$API_BASE$endpoint" 2>/dev/null)
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE$endpoint" 2>/dev/null)

    if [ "$http_code" == "200" ]; then
        if [ -n "$expected" ]; then
            if echo "$response" | jq -e "$expected" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++))
                return 0
            else
                echo -e "${RED}❌ FAIL${NC} (missing: $expected)"
                ((FAIL++))
                return 1
            fi
        else
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++))
            return 0
        fi
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $http_code)"
        ((FAIL++))
        return 1
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo -e "${YELLOW}📈 New Grafana-Optimized Endpoints:${NC}"
echo "─────────────────────────────────────"

test_endpoint "/api/system/metrics" ".[0].metric" "Prometheus Metrics (23 metrics)"
test_endpoint "/api/system/ministers/metrics" ".[0].minister_name" "Ministers Metrics (14 ministers)"
test_endpoint "/api/system/overview" ".system_health" "System Overview"
test_endpoint "/api/system/logs/recent" "" "Recent Logs (may be empty)"
test_endpoint "/api/system/metrics/timeseries" ".[0].target" "Time-series Data"
test_endpoint "/api/system/metrics/test" ".status" "Test Endpoint"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo -e "${YELLOW}📊 Legacy Dashboard Endpoints:${NC}"
echo "─────────────────────────────────────"

test_endpoint "/api/system/health" ".health_percent" "Health Check"
# test_endpoint "/api/system/overview" ".data.health_percent" "System Overview (old format)"
test_endpoint "/api/system/brain" ".data.status" "Brain Status"
test_endpoint "/api/system/knowledge" ".data.total_chunks" "Knowledge Index"
test_endpoint "/api/system/ministers" ".ministers[0]" "Ministers List"
test_endpoint "/api/system/mcp" ".data.status" "MCP Server"
test_endpoint "/api/system/training/history" ".data.latest_run" "Training History"
test_endpoint "/api/system/logs/summary" ".data" "Logs Summary"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Test Results:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Passed: $PASS${NC}"
echo -e "${RED}❌ Failed: $FAIL${NC}"
TOTAL=$((PASS + FAIL))
echo -e "${BLUE}📈 Total: $TOTAL${NC}"

if [ $FAIL -eq 0 ]; then
    SUCCESS_RATE=100
else
    SUCCESS_RATE=$((PASS * 100 / TOTAL))
fi
echo -e "${YELLOW}📊 Success Rate: ${SUCCESS_RATE}%${NC}"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 All endpoints working correctly!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "✅ Ready to import Grafana dashboard!"
    echo ""
    echo "Next steps:"
    echo "1. Open Grafana: http://localhost:3000"
    echo "2. Import dashboard:"
    echo "   /home/noogh/projects/noogh_unified_system/grafana/noogh_unified_system_dashboard.json"
    echo "3. Enjoy real-time monitoring! 🚀"
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}⚠️  Some endpoints failed${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if API server is running:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "2. Restart API server:"
    echo "   cd /home/noogh/projects/noogh_unified_system"
    echo "   ./run.sh stop"
    echo "   ./run.sh api"
    echo ""
    echo "3. Check logs:"
    echo "   tail -f logs/api.log"
    echo ""
    echo "4. Re-run this test:"
    echo "   ./grafana/test_all_endpoints.sh"
    echo ""
    exit 1
fi
