#!/bin/bash

# Pre-deployment validation script
# This script checks if all components are ready for deployment

set -e
cd "$(dirname "$0")"

echo "🔍 PRE-DEPLOYMENT VALIDATION"
echo "=============================="

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
WARNINGS=0
ERRORS=0

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((SUCCESS++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((ERRORS++))
}

echo ""
echo "📋 CHECKING SYSTEM REQUIREMENTS..."

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    log_success "Docker installed: $DOCKER_VERSION"
else
    log_error "Docker not installed"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version)
        log_success "Docker Compose installed: $COMPOSE_VERSION"
    else
        COMPOSE_VERSION=$(docker-compose --version)
        log_success "Docker Compose installed: $COMPOSE_VERSION"
    fi
else
    log_error "Docker Compose not installed"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python installed: $PYTHON_VERSION"
else
    log_error "Python 3 not installed"
fi

echo ""
echo "📁 CHECKING PROJECT FILES..."

# Check critical files
CRITICAL_FILES=(
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
    "app/main.py"
    ".env"
    "data/stocks.db"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        log_success "Found: $file"
    else
        log_error "Missing: $file"
    fi
done

# Check directories
CRITICAL_DIRS=(
    "app"
    "app/api"
    "app/core"
    "app/services"
    "app/models"
    "data"
)

for dir in "${CRITICAL_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        log_success "Directory: $dir"
    else
        log_error "Missing directory: $dir"
    fi
done

echo ""
echo "🐍 CHECKING PYTHON DEPENDENCIES..."

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    log_success "Virtual environment found"
    
    # Activate venv and check key packages
    source .venv/bin/activate 2>/dev/null || true
    
    KEY_PACKAGES=("fastapi" "uvicorn" "streamlit" "pandas" "yfinance" "redis" "sqlalchemy")
    
    for package in "${KEY_PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "Package: $package"
        else
            log_error "Missing package: $package"
        fi
    done
else
    log_warning "Virtual environment not found - using system Python"
fi

echo ""
echo "🔧 CHECKING APPLICATION MODULES..."

# Test application imports
if python3 -c "
import sys
sys.path.append('.')
try:
    from app.main import app
    print('SUCCESS: Main application')
except Exception as e:
    print(f'ERROR: Main application - {e}')
    
try:
    from app.services.comprehensive_analyzer import ComprehensiveAnalyzer
    print('SUCCESS: Comprehensive analyzer')
except Exception as e:
    print(f'ERROR: Comprehensive analyzer - {e}')
    
try:
    from app.core.database import engine
    print('SUCCESS: Database connection')
except Exception as e:
    print(f'ERROR: Database connection - {e}')
" 2>/dev/null | while read line; do
    if [[ $line == SUCCESS:* ]]; then
        log_success "${line#SUCCESS: }"
    elif [[ $line == ERROR:* ]]; then
        log_error "${line#ERROR: }"
    fi
done

echo ""
echo "🌐 CHECKING CONFIGURATION..."

# Check environment variables
if [[ -f ".env" ]]; then
    log_success "Environment file exists"
    
    # Check critical env vars
    if grep -q "REDIS_URL" .env; then
        log_success "Redis URL configured"
    else
        log_warning "Redis URL not configured"
    fi
    
    if grep -q "DATABASE_URL" .env; then
        log_success "Database URL configured"
    else
        log_warning "Database URL not configured"
    fi
else
    log_error "Environment file missing"
fi

# Check Docker files
if [[ -f "Dockerfile" ]]; then
    if grep -q "FROM python:" Dockerfile; then
        log_success "Dockerfile properly configured"
    else
        log_warning "Dockerfile may have issues"
    fi
fi

if [[ -f "docker-compose.yml" ]]; then
    if grep -q "redis:" docker-compose.yml && grep -q "stock_agent_api:" docker-compose.yml; then
        log_success "Docker Compose properly configured"
    else
        log_warning "Docker Compose may have issues"
    fi
fi

echo ""
echo "📊 VALIDATION SUMMARY"
echo "===================="
echo -e "${GREEN}✅ Successful checks: $SUCCESS${NC}"
echo -e "${YELLOW}⚠️  Warnings: $WARNINGS${NC}"
echo -e "${RED}❌ Errors: $ERRORS${NC}"

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}🎉 READY FOR DEPLOYMENT!${NC}"
    echo ""
    echo "🚀 To deploy your application:"
    echo "   ./deploy.sh"
    echo ""
    echo "🔧 For development deployment:"
    echo "   ./dev-deploy.sh"
    echo ""
    echo "📚 For detailed instructions, see:"
    echo "   cat DEPLOYMENT.md"
    
    exit 0
else
    echo -e "${RED}❌ DEPLOYMENT NOT RECOMMENDED${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    
    if [[ $WARNINGS -gt 0 ]]; then
        echo -e "${YELLOW}Note: Warnings should be reviewed but won't prevent deployment.${NC}"
    fi
    
    exit 1
fi
