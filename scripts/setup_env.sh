#!/bin/bash

# Setup Environment Script for Noogh Unified System

echo "🚀 Setting up Noogh Unified System Environment..."

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found. Please install Python 3."
    exit 1
fi

# Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate Virtual Environment
source venv/bin/activate

# Install Dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found!"
fi

# Create necessary directories
echo "📂 Creating system directories..."
mkdir -p logs data models backups

# Set Environment Variables (Example)
export NOOGH_ENV=development
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "✅ Setup Complete! Run './run.sh' to start the system."
