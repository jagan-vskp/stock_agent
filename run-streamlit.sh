#!/bin/bash

# Run Streamlit frontend locally (outside Docker)
# Use this for faster development iteration

set -e

echo "🎨 Starting Streamlit Frontend Locally..."

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    echo "📦 Using virtual environment"
    source .venv/bin/activate
else
    echo "⚠️ No virtual environment found, using system Python"
fi

# Check if required packages are installed
echo "🔍 Checking Streamlit installation..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📥 Installing Streamlit requirements..."
    pip install -r streamlit_requirements.txt
fi

# Set environment variable for API URL
export API_BASE_URL="http://localhost:8000"

echo "🚀 Starting Streamlit application..."
echo "   Frontend will be available at: http://localhost:8501"
echo "   Make sure your API is running at: http://localhost:8000"
echo ""
echo "   Press Ctrl+C to stop"

# Run Streamlit
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
