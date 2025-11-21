#!/bin/bash
# Noogh Sovereign Dashboard Launcher
# Professional Bloomberg-style interface

echo "🏛️ Launching Noogh Sovereign Dashboard..."
echo "=========================================="
echo ""
echo "Dashboard will be available at: http://localhost:8501"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Launch Streamlit with custom theme
streamlit run src/interface/dashboard.py \
  --server.port 8501 \
  --server.address localhost \
  --theme.base "dark" \
  --theme.primaryColor "#2962FF" \
  --theme.backgroundColor "#1e1e1e" \
  --theme.secondaryBackgroundColor "#2a2a2a" \
  --theme.textColor "#ffffff" \
  --theme.font "sans serif" \
  --server.headless true \
  --browser.gatherUsageStats false
