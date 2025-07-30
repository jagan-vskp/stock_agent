import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Indian Stock Recommendation System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}

.recommendation-buy {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}

.recommendation-sell {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}

.recommendation-hold {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "http://localhost:8000/api/v1"

class StockAnalysisApp:
    def __init__(self):
        self.session = requests.Session()
    
    def check_api_health(self):
        """Check if the FastAPI backend is running"""
        try:
            response = self.session.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_stock_analysis(self, symbol, analysis_type):
        """Get comprehensive stock analysis"""
        try:
            url = f"{API_BASE_URL}/analysis/analyze"
            payload = {
                "symbol": symbol,
                "analysis_type": analysis_type
            }
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            url = f"{API_BASE_URL}/stocks/info/{symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recommendations(self, limit=10):
        """Get top stock recommendations"""
        try:
            url = f"{API_BASE_URL}/recommendations/top-picks?limit={limit}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_historical_data(self, symbol, period="1y"):
        """Get historical stock data"""
        try:
            url = f"{API_BASE_URL}/stocks/historical/{symbol}?period={period}"
            response = self.session.get(url, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    app = StockAnalysisApp()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Recommendation System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not app.check_api_health():
        st.error("🚨 **FastAPI Backend is not running!**")
        st.info("Please start the FastAPI server by running: `python -m app.main`")
        st.stop()
    
    st.success("✅ **Backend API is running**")
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Configuration")
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input(
        "Stock Symbol", 
        value="RELIANCE.NS",
        help="Enter Indian stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)"
    ).upper()
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["comprehensive", "fundamental", "technical", "sentiment"],
        help="Choose the type of analysis to perform"
    )
    
    # Period for historical data
    period = st.sidebar.selectbox(
        "Historical Data Period",
        ["1y", "6mo", "3mo", "1mo", "5d"],
        help="Select time period for historical data"
    )
    
    # Main content
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if stock_symbol:
            analyze_stock(app, stock_symbol, analysis_type, period)
        else:
            st.error("Please enter a stock symbol")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🏆 Top Recommendations", "📈 Market Overview"])
    
    with tab1:
        st.header("Stock Analysis")
        if stock_symbol:
            display_quick_info(app, stock_symbol)
    
    with tab2:
        st.header("Top Stock Recommendations")
        display_top_recommendations(app)
    
    with tab3:
        st.header("Market Overview")
        display_market_overview(app)

def analyze_stock(app, symbol, analysis_type, period):
    """Perform comprehensive stock analysis"""
    
    with st.spinner(f"Analyzing {symbol}..."):
        # Get analysis
        analysis_result = app.get_stock_analysis(symbol, analysis_type)
        
        if not analysis_result.get("success"):
            st.error(f"Error: {analysis_result.get('error', 'Unknown error')}")
            return
        
        data = analysis_result["data"]
        
        # Display stock information
        display_stock_header(data)
        
        # Create columns for different analysis types
        col1, col2 = st.columns(2)
        
        with col1:
            # Fundamental analysis
            if "fundamental" in data:
                display_fundamental_analysis(data["fundamental"])
            
            # Technical analysis
            if "technical" in data:
                display_technical_analysis(data["technical"])
        
        with col2:
            # Sentiment analysis
            if "sentiment" in data:
                display_sentiment_analysis(data["sentiment"])
            
            # Historical chart
            display_historical_chart(app, symbol, period)

def display_stock_header(data):
    """Display stock basic information"""
    stock_info = data.get("stock_info", {})
    
    st.subheader(f"🏢 {stock_info.get('name', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{stock_info.get('current_price', 0):,.2f}",
            delta=None
        )
    
    with col2:
        market_cap = stock_info.get('market_cap', 0)
        if market_cap:
            market_cap_cr = market_cap / 10000000  # Convert to crores
            st.metric(
                label="Market Cap",
                value=f"₹{market_cap_cr:,.0f} Cr"
            )
    
    with col3:
        st.metric(
            label="Sector",
            value=stock_info.get('sector', 'N/A')
        )
    
    with col4:
        st.metric(
            label="Exchange",
            value=stock_info.get('exchange', 'N/A')
        )

def display_fundamental_analysis(fundamental_data):
    """Display fundamental analysis results"""
    st.subheader("📊 Fundamental Analysis")
    
    overall_score = fundamental_data.get("overall_score", 0)
    recommendation = fundamental_data.get("recommendation", "HOLD")
    
    # Recommendation badge
    if recommendation == "BUY":
        st.markdown(f'<div class="recommendation-buy">🟢 {recommendation}</div>', unsafe_allow_html=True)
    elif recommendation == "SELL":
        st.markdown(f'<div class="recommendation-sell">🔴 {recommendation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="recommendation-hold">🟡 {recommendation}</div>', unsafe_allow_html=True)
    
    # Overall score
    st.metric("Overall Score", f"{overall_score:.1f}/100")
    
    # Progress bar for score
    st.progress(overall_score / 100)
    
    # Key metrics
    fund_data = fundamental_data.get("fundamental_data", {})
    scores = fundamental_data.get("scores", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Key Ratios:**")
        if fund_data.get("pe_ratio"):
            st.write(f"• P/E Ratio: {fund_data['pe_ratio']:.2f}")
        if fund_data.get("eps"):
            st.write(f"• EPS: ₹{fund_data['eps']:.2f}")
        if fund_data.get("debt_to_equity"):
            st.write(f"• Debt/Equity: {fund_data['debt_to_equity']:.2f}")
    
    with col2:
        st.write("**Scores:**")
        for metric, score in scores.items():
            metric_name = metric.replace("_score", "").replace("_", " ").title()
            st.write(f"• {metric_name}: {score:.0f}/100")
    
    # Insights
    insights = fundamental_data.get("insights", [])
    if insights:
        st.write("**💡 Key Insights:**")
        for insight in insights:
            st.write(f"• {insight}")

def display_technical_analysis(technical_data):
    """Display technical analysis results"""
    st.subheader("📈 Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if technical_data.get("rsi"):
            st.metric("RSI", f"{technical_data['rsi']:.1f}")
    
    with col2:
        if technical_data.get("macd"):
            st.metric("MACD", f"{technical_data['macd']:.2f}")
    
    if "message" in technical_data:
        st.info(technical_data["message"])

def display_sentiment_analysis(sentiment_data):
    """Display sentiment analysis results"""
    st.subheader("📰 Sentiment Analysis")
    
    sentiment_score = sentiment_data.get("sentiment_score", 0)
    news_count = sentiment_data.get("news_count", 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
    
    with col2:
        st.metric("News Articles", news_count)
    
    # Sentiment gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_historical_chart(app, symbol, period):
    """Display historical price chart"""
    st.subheader("📊 Price History")
    
    with st.spinner("Loading historical data..."):
        hist_result = app.get_historical_data(symbol, period)
        
        if hist_result.get("success") and hist_result.get("data", {}).get("data"):
            hist_data = hist_result["data"]["data"]
            df = pd.DataFrame(hist_data)
            
            if not df.empty and 'Close' in df.columns:
                # Convert Date column to datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df.index)
                
                # Create candlestick chart
                fig = go.Figure(data=go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart ({period})",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current", f"₹{df['Close'].iloc[-1]:.2f}")
                
                with col2:
                    change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                    change_pct = (change / df['Close'].iloc[-2]) * 100
                    st.metric("Change", f"₹{change:.2f}", f"{change_pct:+.2f}%")
                
                with col3:
                    st.metric("High", f"₹{df['High'].max():.2f}")
                
                with col4:
                    st.metric("Low", f"₹{df['Low'].min():.2f}")
            else:
                st.warning("No historical data available")
        else:
            st.error("Unable to load historical data")

def display_quick_info(app, symbol):
    """Display quick stock information"""
    if symbol:
        with st.spinner(f"Loading {symbol} info..."):
            info_result = app.get_stock_info(symbol)
            
            if info_result.get("success"):
                stock_data = info_result["data"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Price", f"₹{stock_data.get('current_price', 0):,.2f}")
                
                with col2:
                    market_cap = stock_data.get('market_cap', 0)
                    if market_cap:
                        market_cap_cr = market_cap / 10000000
                        st.metric("Market Cap", f"₹{market_cap_cr:,.0f} Cr")
                
                with col3:
                    st.metric("Sector", stock_data.get('sector', 'N/A'))

def display_top_recommendations(app):
    """Display top stock recommendations"""
    with st.spinner("Loading top recommendations..."):
        reco_result = app.get_recommendations(10)
        
        if reco_result.get("success") and reco_result.get("data"):
            recommendations = reco_result["data"]
            
            for reco in recommendations:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                    
                    with col1:
                        st.write(f"**{reco.get('name', reco.get('symbol', 'N/A'))}**")
                        st.write(f"({reco.get('symbol', 'N/A')})")
                    
                    with col2:
                        recommendation = reco.get('recommendation', 'HOLD')
                        if recommendation == 'BUY':
                            st.success(f"🟢 {recommendation}")
                        elif recommendation == 'SELL':
                            st.error(f"🔴 {recommendation}")
                        else:
                            st.warning(f"🟡 {recommendation}")
                    
                    with col3:
                        st.metric("Current", f"₹{reco.get('current_price', 0):,.2f}")
                    
                    with col4:
                        st.metric("Target", f"₹{reco.get('target_price', 0):,.2f}")
                    
                    with col5:
                        confidence = reco.get('confidence_score', 0) * 100
                        st.metric("Confidence", f"{confidence:.0f}%")
                    
                    st.divider()
        else:
            st.error("Unable to load recommendations")

def display_market_overview(app):
    """Display market overview"""
    st.info("Market overview will be implemented with real-time market data")
    
    # Placeholder market data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NIFTY 50")
        st.metric("Index", "19,500", "+150 (+0.77%)")
    
    with col2:
        st.subheader("SENSEX")
        st.metric("Index", "65,800", "+450 (+0.69%)")

if __name__ == "__main__":
    main()
