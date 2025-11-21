#!/bin/bash

# ألوان للتنسيق
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 INITIALIZING NOOGH UNIFIED SYSTEM v5.0...${NC}"
echo "=================================================="

# 1. تفعيل البيئة الافتراضية (تأكد من مسارها)
source venv/bin/activate || echo "⚠️  Warning: Could not activate venv, assuming global python."

# وظيفة لتنظيف العمليات عند الإغلاق (Cleanup Function)
cleanup() {
    echo -e "\n${RED}🛑 Shutting down all Noogh Systems...${NC}"
    kill $API_PID
    kill $HUNTER_PID
    kill $DASHBOARD_PID
    exit
}

# التقاط أمر الإغلاق (Ctrl+C)
trap cleanup SIGINT

# 2. تشغيل العقل (API)
echo -e "${GREEN}🧠 Starting Neural Core (API)...${NC}"
python -m src.api.main > logs/api.log 2>&1 &
API_PID=$!
echo "   -> API running (PID: $API_PID)"

# انتظار بسيط ليتأكد أن الـ API يعمل
sleep 5

# 3. تشغيل الصياد (Autonomous Hunter)
echo -e "${GREEN}🦅 Releasing the Hunter...${NC}"
python scripts/run_autonomous_hunter.py > logs/hunter.log 2>&1 &
HUNTER_PID=$!
echo "   -> Hunter active (PID: $HUNTER_PID)"

# 4. تشغيل الواجهة (Dashboard)
echo -e "${GREEN}🖥️  Launching Command Center...${NC}"
streamlit run src/interface/dashboard.py --server.port 8501 --theme.base "dark" &
DASHBOARD_PID=$!

echo "=================================================="
echo -e "${CYAN}✅ SYSTEM FULLY OPERATIONAL!${NC}"
echo "   - Dashboard: http://localhost:8501"
echo "   - API Docs:  http://localhost:8000/docs"
echo "   - Logs:      tail -f logs/api.log"
echo "=================================================="
echo "Press Ctrl+C to stop the system."

# إبقاء السكربت يعمل
wait
