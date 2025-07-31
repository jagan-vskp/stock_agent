#!/bin/bash

# Azure Web App startup script for Indian Stock Recommendation System
# Optimized for Azure App Service deployment

echo "🚀 Starting Indian Stock Recommendation System on Azure..."

# Set environment variables for Azure
export HOST=0.0.0.0
export PORT=8000
export PYTHONPATH="/tmp/8ddd041d399ed14:$PYTHONPATH"

# Change to the app directory
cd /tmp/8ddd041d399ed14 || cd /home/site/wwwroot

echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Checking installed packages..."

# Check if required packages are available
python -c "import fastapi, uvicorn; print('✅ FastAPI and Uvicorn available')" || {
    echo "❌ Missing required packages"
    pip install fastapi uvicorn
}

# Create a simple health check
echo "🏥 Setting up health check..."

# Start the FastAPI application directly (not in background)
echo "📡 Starting FastAPI backend on 0.0.0.0:8000..."
echo "🌐 Application will be accessible via Azure Web App URL"

# Use uvicorn to start the application (this keeps the process alive)
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1