#!/bin/bash

# Build and deploy the Stock Recommendation System using Docker Compose

set -e  # Exit on any error

echo "🚀 Starting Stock Recommendation System Deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down --remove-orphans

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

# Check Redis
if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis health check failed"
fi

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
fi

# Check Frontend (if enabled)
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "⚠️ Frontend may still be starting up"
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📊 Services are available at:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - API Health: http://localhost:8000/health"
echo "   - Streamlit Frontend: http://localhost:8501"
echo "   - Redis: localhost:6379"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
echo "   - View service status: docker-compose ps"
echo ""
echo "🔍 To monitor logs in real-time:"
echo "   docker-compose logs -f stock_agent_api"
echo "   docker-compose logs -f stock_agent_frontend"
