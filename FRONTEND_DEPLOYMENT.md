# 🚀 FRONTEND DEPLOYMENT OPTIONS

Your system has multiple ways to run the frontend. Here are all the options:

## 🎯 **Option 1: Full Docker Deployment (Recommended for Production)**
```bash
./deploy.sh
```
**What it does:**
- ✅ Starts Redis cache
- ✅ Starts FastAPI backend
- ✅ Starts Streamlit frontend in Docker
- ✅ Sets up networking between services

**Access:**
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔧 **Option 2: Development with All Services in Docker**
```bash
./dev-deploy.sh
```
**What it does:**
- ✅ Starts Redis cache  
- ✅ Starts FastAPI backend
- ✅ Starts Streamlit frontend in Docker (Updated!)
- ⚡ Optimized for development

**Access:**
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🎨 **Option 3: Frontend Only (Docker)**
```bash
./frontend-deploy.sh
```
**What it does:**
- ✅ Starts only Streamlit frontend in Docker
- 💡 Use when API is running locally or elsewhere

**Requirements:**
- API must be running on http://localhost:8000

---

## 🖥️ **Option 4: Frontend Locally (Outside Docker)**
```bash
./run-streamlit.sh
```
**What it does:**
- ✅ Runs Streamlit using local Python environment
- ⚡ Fastest for frontend development (hot reload)
- 📦 Automatically installs dependencies if needed

**Requirements:**
- API must be running (use `./dev-deploy.sh` but ignore frontend instructions)

---

## 🛠️ **Option 5: Manual Mixed Setup**

### Step 1: Start Backend Services
```bash
# Start only Redis and API
docker-compose up -d redis stock_agent_api
```

### Step 2: Run Frontend Locally
```bash
# Run Streamlit locally for development
./run-streamlit.sh
```

---

## 📊 **Comparison Table**

| Option | Backend | Frontend | Best For | Speed |
|--------|---------|----------|----------|-------|
| Full Docker | Docker | Docker | Production | ⭐⭐⭐ |
| Dev Docker | Docker | Docker | Testing | ⭐⭐⭐⭐ |
| Frontend Only | External | Docker | Hybrid | ⭐⭐⭐ |
| Local Frontend | Docker | Local | Development | ⭐⭐⭐⭐⭐ |
| Manual Mixed | Docker | Local | Custom | ⭐⭐⭐⭐⭐ |

---

## 🎯 **Recommended Workflow**

### For Development:
```bash
# Option 1: Everything in Docker (easier)
./dev-deploy.sh

# Option 2: Mixed (faster frontend iteration)
docker-compose up -d redis stock_agent_api
./run-streamlit.sh
```

### For Production:
```bash
./deploy.sh
```

### For Testing:
```bash
# Start everything
./deploy.sh

# Or start step by step
docker-compose up -d redis
docker-compose up -d stock_agent_api  
docker-compose up -d stock_agent_frontend
```

---

## 🔍 **Service Status Check**

```bash
# Check all services
docker-compose ps

# Check specific service
docker-compose logs stock_agent_frontend

# Check health
curl http://localhost:8000/health      # API
curl http://localhost:8501/_stcore/health  # Frontend
```

---

## 🚨 **Troubleshooting Frontend Issues**

### Frontend not starting?
```bash
# Check logs
docker-compose logs stock_agent_frontend

# Rebuild frontend
docker-compose build stock_agent_frontend
docker-compose up -d stock_agent_frontend
```

### Can't access frontend?
1. Check if port 8501 is available: `lsof -i :8501`
2. Check if container is running: `docker-compose ps`
3. Check network connectivity: `docker network ls`

### API connection issues?
1. Verify API is running: `curl http://localhost:8000/health`
2. Check API_BASE_URL environment variable
3. Verify Docker network configuration

---

**Choose the option that best fits your development workflow!** 🚀
