#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🕒 Setup Cron Job for Daily Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/train_daily.py"

echo "🕒 Setting up cron job for daily training..."
echo "   Project: $PROJECT_ROOT"
echo "   Script: $SCRIPT_PATH"

# Cron job configuration (runs daily at 2 AM)
CRON_SCHEDULE="0 2 * * *"
CRON_COMMAND="cd $PROJECT_ROOT && $VENV_PYTHON $SCRIPT_PATH >> $PROJECT_ROOT/logs/cron_train.log 2>&1"

# Full cron entry
CRON_ENTRY="$CRON_SCHEDULE $CRON_COMMAND"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH" >/dev/null 2>&1; then
    echo "⚠️  Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
fi

# Add new cron job
echo "✅ Adding cron job..."
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "Schedule: Daily at 2:00 AM"
echo "Command: $CRON_COMMAND"
echo ""
echo "Current crontab:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
crontab -l | grep "$SCRIPT_PATH" || echo "(no matching entries)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Logs will be written to: $PROJECT_ROOT/logs/cron_train.log"
echo ""
echo "To view cron logs: tail -f $PROJECT_ROOT/logs/cron_train.log"
echo "To disable cron job: crontab -e (then remove the line)"
echo "To test manually: $VENV_PYTHON $SCRIPT_PATH"
echo ""
