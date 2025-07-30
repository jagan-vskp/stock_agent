# Indian Stock Recommendation System

A comprehensive stock recommendation system for Indian markets that analyzes:
- Fundamental data (P/E, EPS, debt ratios, etc.)
- Technical indicators (RSI, MACD, moving averages)
- News sentiment analysis
- Market trends and external factors

## Features

- 🌐 **Streamlit Web Interface** - User-friendly web dashboard
- 📡 **RESTful API** built with FastAPI
- 📊 **Real-time data collection** from multiple sources
- 🤖 **Advanced ML models** for price prediction
- 📈 **Interactive charts** and visualizations
- 💾 **SQLite database** for data persistence
- ⚡ **Redis caching** for performance
- 📰 **News sentiment analysis**
- 📋 **Top stock recommendations**
- 🎯 **Comprehensive analysis** (Fundamental + Technical + Sentiment)

## Installation

1. Install system dependencies (macOS):
```bash
# Install homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system packages
brew install python@3.11
brew install redis
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 🚀 Quick Start (Web Interface)
```bash
# Start both FastAPI backend and Streamlit frontend
./start_app.sh
```
Then open your browser to: **http://localhost:8501**

### 🔧 Manual Start
1. Start the API server:
```bash
python -m app.main
```

2. Start Streamlit web interface (in another terminal):
```bash
streamlit run streamlit_app.py
```

3. Access the web app at: **http://localhost:8501**
4. Access the API documentation at: **http://localhost:8000/docs**

### 📡 Direct API Usage
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/analyze" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "RELIANCE.NS", "analysis_type": "comprehensive"}'
```

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   ├── database.py      # Database connection
│   │   └── cache.py         # Redis cache
│   ├── models/              # Database models
│   ├── services/            # Business logic
│   ├── api/                 # API routes
│   └── utils/               # Utility functions
├── streamlit_app.py         # Streamlit web interface
├── start_app.sh             # Startup script for both services
├── .streamlit/              # Streamlit configuration
├── data/                    # Data storage
├── logs/                    # Application logs
└── tests/                   # Test files
```

## API Endpoints

- `POST /api/v1/analyze` - Get comprehensive stock analysis
- `GET /api/v1/stocks/{symbol}` - Get stock basic info
- `GET /api/v1/recommendations` - Get top recommendations
- `POST /api/v1/watchlist` - Add stock to watchlist
