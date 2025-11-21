#!/bin/bash
# =============================================================================
# Noogh OS Quick Start
# =============================================================================
# Simply run: ./start_noogh_os.sh
# =============================================================================

cd "$(dirname "$0")"

echo "🚀 Starting Noogh Sovereign OS..."
echo ""

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No venv found. Using system Python."
fi

# Create logs directory
mkdir -p logs

# Start both services
echo ""
echo "Starting services..."
./venv/bin/python3 start_noogh_os.sh 2>&1 | tee logs/startup.log

echo ""
echo "To view logs:"
echo "  Backend: tail -f logs/backend.log"
echo "  Dashboard: tail -f logs/dashboard.log"
