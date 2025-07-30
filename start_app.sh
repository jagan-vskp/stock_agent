#!/bin/bash

# Startup script for Indian Stock Recommendation System
# This script starts both FastAPI backend and Streamlit frontend

echo "🚀 Starting Indian Stock Recommendation System..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    source .venv/bin/activate
fi

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
python -m app.main &
FASTAPI_PID=$!

# Wait a bit for FastAPI to start
sleep 3

# Check if FastAPI started successfully
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ FastAPI backend is running (PID: $FASTAPI_PID)"
else
    echo "❌ Failed to start FastAPI backend"
    kill $FASTAPI_PID 2>/dev/null
    exit 1
fi

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port 8501..."
echo "🌐 Access the web app at: http://localhost:8501"
streamlit run streamlit_app.py

# Cleanup: Kill FastAPI when Streamlit exits
echo "🛑 Shutting down services..."
kill $FASTAPI_PID 2>/dev/null
echo "👋 Goodbye!"
