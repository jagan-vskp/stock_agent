#!/bin/bash

# Development deployment script - runs only the essential services

set -e

echo "🔧 Starting Development Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start Redis, API, and Frontend for development
echo "🔨 Starting development services..."
docker-compose up --build -d redis stock_agent_api stock_agent_frontend

# Wait for services
echo "⏳ Waiting for services..."
sleep 10

# Check health
echo "🏥 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is ready for development"
else
    echo "❌ API failed to start"
    docker-compose logs stock_agent_api
    exit 1
fi

# Check frontend
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is ready for development"
else
    echo "⚠️ Frontend may still be starting up"
fi

echo ""
echo "🎯 Development environment ready!"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8501"
echo "   - Redis: localhost:6379"
