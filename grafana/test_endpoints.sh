#!/bin/bash

# 🧪 Test All Grafana-Optimized API Endpoints
# This script verifies that all endpoints are working correctly before importing the dashboard

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing Noogh Unified System API Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

API_BASE="http://localhost:8000"
PASS=0
FAIL=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local endpoint=$1
    local expected_field=$2
    local description=$3

    echo -n "Testing $description... "

    response=$(curl -s "$API_BASE$endpoint")
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE$endpoint")

    if [ "$http_code" == "200" ]; then
        if [ -n "$expected_field" ]; then
            # Check if expected field exists in JSON
            if echo "$response" | jq -e "$expected_field" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++))
                return 0
            else
                echo -e "${RED}❌ FAIL${NC} (missing field: $expected_field)"
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

# Test legacy endpoints
echo "📊 Legacy Endpoints:"
echo "─────────────────────"
test_endpoint "/api/system/health" ".health_percent" "Health Check"
test_endpoint "/api/system/overview" ".data.health_percent" "System Overview"
test_endpoint "/api/system/brain" ".data.status" "Brain Status"
test_endpoint "/api/system/knowledge" ".data.total_chunks" "Knowledge Index"
test_endpoint "/api/system/ministers" ".ministers[0]" "Ministers List"
test_endpoint "/api/system/mcp" ".data.status" "MCP Server"
test_endpoint "/api/system/training/history" ".data.latest_run" "Training History"
test_endpoint "/api/system/logs/summary" ".data" "Logs Summary"

echo ""
echo "🎯 Grafana-Optimized Endpoints:"
echo "────────────────────────────────"
test_endpoint "/api/system/metrics" ".[0].metric" "Prometheus Metrics"
test_endpoint "/api/system/ministers/metrics" ".[0].minister_name" "Ministers Metrics"
test_endpoint "/api/system/logs/recent" ".[0].timestamp" "Recent Logs"
test_endpoint "/api/system/metrics/timeseries" ".[0].target" "Time-Series Data"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Test Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Passed: $PASS${NC}"
echo -e "${RED}❌ Failed: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}🎉 All endpoints working correctly!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Open Grafana: http://localhost:3000"
    echo "2. Import dashboard from:"
    echo "   /home/noogh/projects/noogh_unified_system/grafana/noogh_unified_system_dashboard.json"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  Some endpoints failed. Check API server.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Restart API server:"
    echo "   cd /home/noogh/projects/noogh_unified_system"
    echo "   export PYTHONPATH=/home/noogh/projects/noogh_unified_system/src"
    echo "   ./venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "2. Re-run this test script"
    echo ""
    exit 1
fi
