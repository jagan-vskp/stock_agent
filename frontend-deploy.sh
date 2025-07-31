#!/bin/bash

# Frontend-only deployment script for development
# Use this when you want to run frontend in Docker but API locally

set -e

echo "🎨 Starting Frontend Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing frontend container
echo "🛑 Stopping existing frontend container..."
docker-compose stop stock_agent_frontend 2>/dev/null || true

# Start only the frontend
echo "🔨 Starting frontend service..."
docker-compose up --build -d stock_agent_frontend

# Wait for service
echo "⏳ Waiting for frontend..."
sleep 5

# Check frontend health
echo "🏥 Checking frontend health..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is ready"
else
    echo "⚠️ Frontend may still be starting up"
    echo "   Check status: docker-compose logs stock_agent_frontend"
fi

echo ""
echo "🎨 Frontend service ready!"
echo "   - Frontend: http://localhost:8501"
echo ""
echo "💡 Make sure your API is running on:"
echo "   http://localhost:8000"
