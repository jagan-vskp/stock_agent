# Docker Deployment Guide

This guide explains how to deploy the Stock Recommendation System using Docker and Docker Compose.

## Prerequisites

1. **Docker**: Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. **Docker Compose**: Usually included with Docker Desktop
3. **Git**: To clone the repository

## Quick Start

### 1. Full Production Deployment

Deploy all services (API, Frontend, Redis):

```bash
# Make scripts executable (first time only)
chmod +x deploy.sh dev-deploy.sh

# Deploy all services
./deploy.sh
```

This will start:
- ✅ **FastAPI Backend** on http://localhost:8000
- ✅ **Streamlit Frontend** on http://localhost:8501
- ✅ **Redis Cache** on localhost:6379

### 2. Development Deployment

For development, you might want to run only the backend services:

```bash
# Deploy only API and Redis (no frontend container)
./dev-deploy.sh

# Run Streamlit separately for development
python streamlit_app.py
```

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │     Redis       │
│   Frontend      │────│   Backend       │────│     Cache       │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Available Endpoints

### API Endpoints (http://localhost:8000)
- 📚 **API Documentation**: `/docs` - Interactive Swagger UI
- 🏥 **Health Check**: `/health` - Service status
- 📊 **Stock Analysis**: `/api/v1/analysis/{symbol}` - Comprehensive analysis
- 💹 **Stock Data**: `/api/v1/stocks/{symbol}` - Basic stock information
- 🎯 **Recommendations**: `/api/v1/recommendations/{symbol}` - Investment recommendations

### Frontend (http://localhost:8501)
- 🎨 **Interactive Dashboard** - Stock analysis interface
- 📈 **Charts and Visualizations** - Technical indicators and trends
- 📋 **Analysis Reports** - Comprehensive stock analysis

## Configuration

### Environment Variables

The application uses these key environment variables:

```bash
# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Database Configuration  
DATABASE_URL=sqlite:///./data/stocks.db

# Cache Configuration
CACHE_TTL=300  # 5 minutes

# Logging
LOG_LEVEL=INFO
```

### Custom Configuration

1. **Copy environment template**:
   ```bash
   cp .env.docker .env
   ```

2. **Edit configuration** as needed:
   ```bash
   nano .env
   ```

3. **Restart services**:
   ```bash
   docker-compose restart
   ```

## Management Commands

### Service Management
```bash
# View all services status
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs for specific service
docker-compose logs -f stock_agent_api

# Restart specific service
docker-compose restart stock_agent_api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Development Commands
```bash
# Build without cache
docker-compose build --no-cache

# Scale API service (multiple instances)
docker-compose up --scale stock_agent_api=3

# Enter container shell
docker-compose exec stock_agent_api bash

# View resource usage
docker stats
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   
   # Kill the process or change port in docker-compose.yml
   ```

2. **Redis Connection Failed**
   ```bash
   # Check Redis container
   docker-compose logs redis
   
   # Test Redis connection
   docker-compose exec redis redis-cli ping
   ```

3. **API Not Responding**
   ```bash
   # Check API logs
   docker-compose logs stock_agent_api
   
   # Check API health
   curl http://localhost:8000/health
   ```

4. **Frontend Not Loading**
   ```bash
   # Check frontend logs
   docker-compose logs stock_agent_frontend
   
   # Restart frontend
   docker-compose restart stock_agent_frontend
   ```

### Performance Optimization

1. **Memory Limits**: Add memory limits to docker-compose.yml
   ```yaml
   deploy:
     resources:
       limits:
         memory: 512M
   ```

2. **CPU Limits**: Limit CPU usage
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '0.5'
   ```

## Production Deployment

### Security Considerations

1. **Change Default Secrets**:
   ```bash
   # Generate new secret key
   openssl rand -hex 32
   ```

2. **Use Environment Files**:
   ```bash
   # Create production environment file
   cp .env.docker .env.production
   # Edit with production values
   ```

3. **Enable HTTPS** (reverse proxy required):
   - Use nginx or traefik
   - Configure SSL certificates
   - Update CORS origins

### Scaling

1. **Horizontal Scaling**:
   ```bash
   # Scale API instances
   docker-compose up --scale stock_agent_api=3 -d
   ```

2. **Load Balancer**: Add nginx for load balancing multiple API instances

3. **External Redis**: Use managed Redis service for production

### Monitoring

1. **Health Checks**: Built-in health endpoints
2. **Logging**: Centralized logging with ELK stack
3. **Metrics**: Add Prometheus metrics
4. **Alerts**: Configure alerts for service failures

## Development Workflow

1. **Make Changes**: Edit code locally
2. **Rebuild**: `docker-compose build stock_agent_api`
3. **Restart**: `docker-compose restart stock_agent_api`
4. **Test**: Access http://localhost:8000/docs

## Data Persistence

- **Database**: SQLite file persisted in `./data/` volume
- **Redis Data**: Persisted in `redis_data` volume
- **Logs**: Available in `./logs/` directory

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8000/health`
3. Check service status: `docker-compose ps`
